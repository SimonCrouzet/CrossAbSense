"""Attention-based decoder for property prediction with multiple cross-attention strategies."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from .base_decoder import BaseDecoder

logger = logging.getLogger(__name__)


class AttentionDecoder(BaseDecoder):
    """
    Attention-based decoder with configurable cross-attention strategies.

    Uses pre-normalization (LayerNorm before attention/FFN) for better gradient flow.

    Supports three biologically-motivated attention strategies:
    1. Bidirectional Cross-Attention: VH ↔ VL (for interface-dependent properties)
    2. Self + Cross-Attention: Self-attention within chains, then cross-attention (hierarchical)
    3. Self-Attention Only: Independent chain processing (for chain-independent properties)
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 8,
        hidden_dim: int = None,
        n_layers: int = 3,
        dropout: float = 0.3,
        activation: str = "gelu",
        pooling_strategy: str = "mean",
        attention_strategy: str = "bidirectional_cross",
        use_output_norm: bool = False,  # LayerNorm between output MLP layers
        output_activation: str = "none",
        n_output_layers: int = 3,
        use_learnable_chain_fusion: str = "none",
        antibody_features_dim: int = 0,  # Dimension of antibody features (0 = disabled)
        antibody_features_normalized: bool = False,  # Whether features are z-score normalized
        antibody_features_projection_dim: Optional[int] = None,  # Project to this dim (None = no projection)
        antibody_features_injection_layer: Optional[str] = None,  # Injection layer ("first", "second", "third", "last")
    ):
        """
        Args:
            input_dim: Dimension of input embeddings PER CHAIN (from encoder.get_embedding_dim())
            n_heads: Number of attention heads
            hidden_dim: Hidden dimension for attention. If None, uses input_dim
                       (no projection, preserves all information from frozen encoder).
                       For ESM-C 6B: input_dim=2560 per chain
            n_layers: Number of attention layers
            dropout: Dropout probability
            activation: Activation function (relu, gelu, silu)
            pooling_strategy: How to pool sequences:
                            - 'mean' (default): Average pooling
                            - 'attention': Learned attention weights for position importance
            attention_strategy: Cross-attention strategy (bidirectional_cross, self_cross, self_only)
                              Uses pre-normalization (LayerNorm before attention/FFN)
            use_output_norm: Whether to use LayerNorm between output MLP layers
                           Note: LayerNorm at entrance of output head (after chain combination) is always present
            output_activation: Final activation for output layer (none, softplus, sigmoid, exp)
                             - none: Linear output (default, for normalized targets)
                             - softplus: Smooth strictly positive outputs (good for Titer)
                             - sigmoid: Bounded [0, 1] (good for probabilities)
                             - exp: Exponential e^x (strictly positive, handles large ranges)
            n_output_layers: Number of layers in output MLP head (default: 3)
                           - Smooth compression: divide by 2 at each step
                           - Warning issued if > 3 (risk of overfitting)
            use_learnable_chain_fusion: How to combine VH and VL chains (default: "none")
                           - "none": Simple 50/50 average (no learnable parameters)
                           - "per_chain": Single scalar weight for VH vs VL balance (1 parameter)
                           - "per_dim": Per-dimension weights for fine-grained control (hidden_dim parameters)
            antibody_features_dim: Dimension of antibody features to concatenate (default: 0 = disabled)
            antibody_features_normalized: Whether antibody features are z-score normalized (default: False)
                           - True: Features are normalized → inject right after pooling (early integration)
                           - False: Features are raw counts → inject after first FFN layer (late integration)
            antibody_features_projection_dim: Project features to this dimension before injection (default: None)
                           - None: No projection, concatenate raw/normalized features directly
                           - int: Linear projection layer antibody_features_dim → projection_dim
            antibody_features_injection_layer: Where to inject features ("first", "second", "third", "last")
                                              Overrides antibody_features_normalized for injection point.
        """
        super().__init__(input_dim)

        # Backward compatibility: convert old boolean values to new string format
        if isinstance(use_learnable_chain_fusion, bool):
            use_learnable_chain_fusion = "per_chain" if use_learnable_chain_fusion else "none"

        # Validate n_output_layers
        if n_output_layers > 3:
            logger.warning(
                f"n_output_layers={n_output_layers} is quite deep. "
                f"Consider using 1-3 layers to avoid overfitting. "
                f"Output MLP will progressively compress by ÷2 each layer (min 32d) down to 1."
            )

        # input_dim is now the dimension PER CHAIN (from encoder's get_embedding_dim())
        # For ESM-C 6B: input_dim=2560 (per chain)
        # For multi-encoder (esmc_6b+prott5): input_dim=3584 (fused dimension per chain)
        chain_dim = input_dim

        # Default: work directly on native dimensions (no projection, zero information loss!)
        self.hidden_dim = hidden_dim if hidden_dim is not None else chain_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_output_layers = n_output_layers
        self.pooling_strategy = pooling_strategy
        self.attention_strategy = attention_strategy
        self.chain_dim = chain_dim
        self.use_output_norm = use_output_norm  # LayerNorm between MLP layers
        self.use_learnable_chain_fusion = use_learnable_chain_fusion
        self.antibody_features_dim = antibody_features_dim
        self.antibody_features_normalized = antibody_features_normalized
        self.antibody_features_projection_dim = antibody_features_projection_dim
        self.antibody_features_injection_layer = antibody_features_injection_layer

        # Set up antibody features projection and injection point
        if antibody_features_dim > 0:
            # Optional projection layer
            if antibody_features_projection_dim is not None:
                self.antibody_features_projection = nn.Linear(
                    antibody_features_dim, antibody_features_projection_dim
                )
                self.antibody_features_effective_dim = antibody_features_projection_dim
            else:
                self.antibody_features_projection = None
                self.antibody_features_effective_dim = antibody_features_dim

            # Determine injection strategy
            if antibody_features_injection_layer:
                self.inject_strategy = antibody_features_injection_layer
            else:
                # Fallback to legacy normalization-based strategy
                # Normalized features → inject early ("first")
                # Raw features → inject late ("second")
                self.inject_strategy = "first" if antibody_features_normalized else "second"
        else:
            self.antibody_features_projection = None
            self.antibody_features_effective_dim = 0
            self.inject_strategy = None

        # Select activation function
        if activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Projection layers (only if hidden_dim != chain_dim)
        if self.hidden_dim != chain_dim:
            # Project from chain_dim to hidden_dim
            self.vh_projection = nn.Linear(chain_dim, self.hidden_dim)
            self.vl_projection = nn.Linear(chain_dim, self.hidden_dim)
        else:
            # No projection needed (native dimensions)
            self.vh_projection = None
            self.vl_projection = None

        # Build attention layers based on strategy
        # Use self.hidden_dim (which has default value applied) instead of hidden_dim parameter
        if attention_strategy == "bidirectional_cross":
            self._build_bidirectional_cross_attention(n_layers, n_heads, self.hidden_dim, dropout)
        elif attention_strategy == "self_cross":
            self._build_self_cross_attention(n_layers, n_heads, self.hidden_dim, dropout)
        elif attention_strategy == "self_only":
            self._build_self_only_attention(n_layers, n_heads, self.hidden_dim, dropout)
        else:
            raise ValueError(f"Unknown attention strategy: {attention_strategy}")

        # Attention-based pooling (learn which positions are important)
        if self.pooling_strategy == "attention":
            self.vh_pooling_attn = self._build_attention_pooling(self.hidden_dim)
            self.vl_pooling_attn = self._build_attention_pooling(self.hidden_dim)

        # Learnable chain fusion weights (learn VH/VL balance)
        if self.use_learnable_chain_fusion == "per_chain":
            # Single scalar weight for VH vs VL (1 parameter)
            self.chain_weight = nn.Parameter(torch.tensor(0.5))
        elif self.use_learnable_chain_fusion == "per_dim":
            # Per-dimension weights (hidden_dim parameters)
            self.chain_weight = nn.Parameter(torch.full((self.hidden_dim,), 0.5))
        elif self.use_learnable_chain_fusion == "none":
            self.chain_weight = None
        else:
            raise ValueError(
                f"Invalid use_learnable_chain_fusion: {self.use_learnable_chain_fusion}. "
                f"Choose from: 'none', 'per_chain', 'per_dim'"
            )

        # LayerNorm at entrance of output head (always present)
        self.output_entrance_norm = nn.LayerNorm(self.hidden_dim)

        # Build output MLP head with antibody features injection
        # Progressive compression: hidden_dim → hidden_dim//2 → hidden_dim//4 → ... → 1
        self.output_layers = nn.ModuleList()
        current_dim = self.hidden_dim

        for layer_idx in range(n_output_layers):
            # Determine input dim for this layer
            input_dim = current_dim

            # Inject antibody features at appropriate layer
            if antibody_features_dim > 0:
                inject = False
                if self.inject_strategy == "first" and layer_idx == 0:
                    inject = True
                elif self.inject_strategy == "second" and layer_idx == 1:
                    inject = True
                elif self.inject_strategy == "third" and layer_idx == 2:
                    inject = True
                elif self.inject_strategy == "last" and layer_idx == max(0, n_output_layers - 1):
                    # Only inject if not already injected
                    if not ((self.inject_strategy == "first" and layer_idx == 0) or
                            (self.inject_strategy == "second" and layer_idx == 1) or
                            (self.inject_strategy == "third" and layer_idx == 2)):
                        inject = True
                
                if inject:
                    input_dim += self.antibody_features_effective_dim
                    logger.info(
                        f"🧬 Injecting {antibody_features_dim}d antibody features "
                        f"(strategy={self.inject_strategy}, proj={antibody_features_projection_dim}) "
                        f"at layer {layer_idx}: {current_dim} + {self.antibody_features_effective_dim} = {input_dim}"
                    )

            # Determine output dim
            is_last_layer = (layer_idx == n_output_layers - 1)
            if is_last_layer:
                output_dim = 1  # Final output
            else:
                # Progressive compression: divide by 2, minimum 32
                output_dim = max(current_dim // 2, 32)

            # Build layer
            layer_modules = [
                nn.Linear(input_dim, output_dim),
            ]

            # Add activation + dropout for non-final layers
            if not is_last_layer:
                layer_modules.extend([
                    self.act_fn,
                    nn.Dropout(dropout),
                ])
                # Optional LayerNorm
                if use_output_norm:
                    layer_modules.append(nn.LayerNorm(output_dim))

            self.output_layers.append(nn.Sequential(*layer_modules))

            # Update current_dim for next iteration
            current_dim = output_dim

        # Property-specific output activation
        self.output_activation_name = output_activation
        if output_activation == "softplus":
            self.output_activation = nn.Softplus()
        elif output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "exp":
            self.output_activation = torch.exp
        elif output_activation == "none" or output_activation is None:
            self.output_activation = None
        else:
            raise ValueError(
                f"Unknown output activation: {output_activation}. "
                f"Choose from: none, softplus, sigmoid, exp"
            )

    def _xavier_init_weights(self, gain: float = 1.0):
        """
        Apply Xavier uniform initialization to all Linear layers.

        Args:
            gain: Scaling factor for Xavier init (default: 1.0)
                 For pre-norm, use small gain (0.1-0.3) to prevent exploding activations
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    init.zeros_(module.bias)
        logger.info(f"Applied Xavier uniform initialization with gain={gain}")

    def _build_attention_pooling(self, hidden_dim):
        """Build attention-based pooling network."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def _build_bidirectional_cross_attention(self, n_layers, n_heads, hidden_dim, dropout):
        """Build bidirectional cross-attention layers (VH ↔ VL)."""
        self.vh_cross_attn = nn.ModuleList()
        self.vl_cross_attn = nn.ModuleList()
        self.vh_norm1 = nn.ModuleList()
        self.vl_norm1 = nn.ModuleList()
        self.vh_ffn = nn.ModuleList()
        self.vl_ffn = nn.ModuleList()
        self.vh_norm2 = nn.ModuleList()
        self.vl_norm2 = nn.ModuleList()

        for _ in range(n_layers):
            # VH attends to VL
            self.vh_cross_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vh_norm1.append(nn.LayerNorm(hidden_dim))

            # VL attends to VH
            self.vl_cross_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vl_norm1.append(nn.LayerNorm(hidden_dim))

            # Feed-forward networks
            self.vh_ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    self.act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.vh_norm2.append(nn.LayerNorm(hidden_dim))

            self.vl_ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    self.act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.vl_norm2.append(nn.LayerNorm(hidden_dim))

    def _build_self_cross_attention(self, n_layers, n_heads, hidden_dim, dropout):
        """Build self + cross-attention layers."""
        # First: self-attention within each chain
        self.vh_self_attn = nn.ModuleList()
        self.vl_self_attn = nn.ModuleList()
        self.vh_self_norm = nn.ModuleList()
        self.vl_self_norm = nn.ModuleList()

        # Then: cross-attention between chains
        self.vh_cross_attn = nn.ModuleList()
        self.vl_cross_attn = nn.ModuleList()
        self.vh_cross_norm = nn.ModuleList()
        self.vl_cross_norm = nn.ModuleList()

        # Finally: feed-forward
        self.vh_ffn = nn.ModuleList()
        self.vl_ffn = nn.ModuleList()
        self.vh_norm2 = nn.ModuleList()
        self.vl_norm2 = nn.ModuleList()

        for _ in range(n_layers):
            # Self-attention
            self.vh_self_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vh_self_norm.append(nn.LayerNorm(hidden_dim))

            self.vl_self_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vl_self_norm.append(nn.LayerNorm(hidden_dim))

            # Cross-attention
            self.vh_cross_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vh_cross_norm.append(nn.LayerNorm(hidden_dim))

            self.vl_cross_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vl_cross_norm.append(nn.LayerNorm(hidden_dim))

            # Feed-forward
            self.vh_ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    self.act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.vh_norm2.append(nn.LayerNorm(hidden_dim))

            self.vl_ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    self.act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.vl_norm2.append(nn.LayerNorm(hidden_dim))

    def _build_self_only_attention(self, n_layers, n_heads, hidden_dim, dropout):
        """Build self-attention only layers (independent chains)."""
        self.vh_self_attn = nn.ModuleList()
        self.vl_self_attn = nn.ModuleList()
        self.vh_norm1 = nn.ModuleList()
        self.vl_norm1 = nn.ModuleList()
        self.vh_ffn = nn.ModuleList()
        self.vl_ffn = nn.ModuleList()
        self.vh_norm2 = nn.ModuleList()
        self.vl_norm2 = nn.ModuleList()

        for _ in range(n_layers):
            # Self-attention
            self.vh_self_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vh_norm1.append(nn.LayerNorm(hidden_dim))

            self.vl_self_attn.append(
                nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            )
            self.vl_norm1.append(nn.LayerNorm(hidden_dim))

            # Feed-forward
            self.vh_ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    self.act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.vh_norm2.append(nn.LayerNorm(hidden_dim))

            self.vl_ffn.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    self.act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.vl_norm2.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        vh_embeddings: Optional[torch.Tensor] = None,
        vl_embeddings: Optional[torch.Tensor] = None,
        antibody_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict property value from embeddings.

        Args:
            vh_embeddings: Heavy chain embeddings of shape:
                          - (batch_size, hidden_dim) for pooled
                          - (batch_size, seq_len, hidden_dim) for unpooled
            vl_embeddings: Light chain embeddings of shape:
                          - (batch_size, hidden_dim) for pooled
                          - (batch_size, seq_len, hidden_dim) for unpooled
            antibody_features: Antibody features tensor of shape (batch_size, features_dim)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Validate at least one chain present
        if vh_embeddings is None and vl_embeddings is None:
            raise ValueError("Must provide at least one chain (VH or VL embeddings)")

        # Get batch size from whichever chain is present
        batch_size = (vh_embeddings if vh_embeddings is not None else vl_embeddings).shape[0]

        # Process VH embeddings
        if vh_embeddings is not None:
            if vh_embeddings.dim() == 2:
                # Pooled: (batch, chain_dim)
                vh_emb = vh_embeddings
                # Project to hidden_dim if needed
                if self.vh_projection is not None:
                    vh_emb = self.vh_projection(vh_emb)
                # Add sequence dimension for attention
                vh = vh_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
            elif vh_embeddings.dim() == 3:
                # Unpooled: (batch, seq_len, chain_dim)
                # Project to hidden_dim if needed
                if self.vh_projection is not None:
                    vh = self.vh_projection(vh_embeddings)
                else:
                    vh = vh_embeddings
            else:
                raise ValueError(f"VH embeddings must be 2D or 3D, got {vh_embeddings.dim()}D")
        else:
            vh = None

        # Process VL embeddings
        if vl_embeddings is not None:
            if vl_embeddings.dim() == 2:
                # Pooled: (batch, chain_dim)
                vl_emb = vl_embeddings
                # Project to hidden_dim if needed
                if self.vl_projection is not None:
                    vl_emb = self.vl_projection(vl_emb)
                # Add sequence dimension for attention
                vl = vl_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
            elif vl_embeddings.dim() == 3:
                # Unpooled: (batch, seq_len, chain_dim)
                # Project to hidden_dim if needed
                if self.vl_projection is not None:
                    vl = self.vl_projection(vl_embeddings)
                else:
                    vl = vl_embeddings
            else:
                raise ValueError(f"VL embeddings must be 2D or 3D, got {vl_embeddings.dim()}D")
        else:
            vl = None

        # Apply attention strategy
        if self.attention_strategy == "bidirectional_cross":
            vh, vl = self._forward_bidirectional_cross(vh, vl)
        elif self.attention_strategy == "self_cross":
            vh, vl = self._forward_self_cross(vh, vl)
        elif self.attention_strategy == "self_only":
            vh, vl = self._forward_self_only(vh, vl)

        # Pool sequences to single vectors if needed
        if vh.shape[1] > 1:  # Sequence length > 1
            if self.pooling_strategy == "attention":
                # Attention-based pooling (learn position importance)
                vh_scores = self.vh_pooling_attn(vh)  # (batch, seq_len, 1)
                vh_weights = torch.softmax(vh_scores, dim=1)
                vh_pooled = (vh * vh_weights).sum(dim=1)  # (batch, hidden_dim)
                
                vl_scores = self.vl_pooling_attn(vl)  # (batch, seq_len, 1)
                vl_weights = torch.softmax(vl_scores, dim=1)
                vl_pooled = (vl * vl_weights).sum(dim=1)  # (batch, hidden_dim)
            else:
                # Mean pool across sequence dimension
                vh_pooled = vh.mean(dim=1)  # (batch, hidden_dim)
                vl_pooled = vl.mean(dim=1)  # (batch, hidden_dim)
        else:
            # Already pooled (seq_len=1)
            vh_pooled = vh.squeeze(1)  # (batch, hidden_dim)
            vl_pooled = vl.squeeze(1)  # (batch, hidden_dim)

        # Combine chains for final prediction
        if self.chain_weight is not None:
            # Learnable weighted combination
            w_vh = torch.sigmoid(self.chain_weight)  # Scalar or vector ∈ [0, 1]
            w_vl = 1 - w_vh
            combined = w_vh * vh_pooled + w_vl * vl_pooled  # (batch, hidden_dim)
        else:
            # Simple average (50/50)
            combined = (vh_pooled + vl_pooled) / 2  # (batch, hidden_dim)

        # Apply entrance LayerNorm before output head
        combined = self.output_entrance_norm(combined)

        # Apply antibody features projection if specified
        if antibody_features is not None and self.antibody_features_projection is not None:
            antibody_features = self.antibody_features_projection(antibody_features)

        # Pass through output layers, injecting antibody features at appropriate layer
        x = combined
        num_output_layers = len(self.output_layers)
        for layer_idx, layer in enumerate(self.output_layers):
            # Inject antibody features based on chosen strategy
            if antibody_features is not None:
                inject = False
                if self.inject_strategy == "first" and layer_idx == 0:
                    inject = True
                elif self.inject_strategy == "second" and layer_idx == 1:
                    inject = True
                elif self.inject_strategy == "third" and layer_idx == 2:
                    inject = True
                elif self.inject_strategy == "last" and layer_idx == max(0, num_output_layers - 1):
                    # Only inject if not already injected
                    if not ((self.inject_strategy == "first" and layer_idx == 0) or
                            (self.inject_strategy == "second" and layer_idx == 1) or
                            (self.inject_strategy == "third" and layer_idx == 2)):
                        inject = True
                
                if inject:
                    x = torch.cat([x, antibody_features], dim=1)

            x = layer(x)

        output = x

        # Apply property-specific activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def _forward_bidirectional_cross(self, vh, vl):
        """Forward pass with bidirectional cross-attention (pre-norm)."""
        for i in range(self.n_layers):
            # Pre-norm: LayerNorm BEFORE attention
            vh_normed = self.vh_norm1[i](vh)
            vl_normed = self.vl_norm1[i](vl)

            # VH attends to VL
            vh_cross, _ = self.vh_cross_attn[i](vh_normed, vl_normed, vl_normed)
            vh = vh + vh_cross

            # VL attends to VH
            vl_cross, _ = self.vl_cross_attn[i](vl_normed, vh_normed, vh_normed)
            vl = vl + vl_cross

            # Pre-norm: LayerNorm BEFORE FFN
            vh = vh + self.vh_ffn[i](self.vh_norm2[i](vh))
            vl = vl + self.vl_ffn[i](self.vl_norm2[i](vl))

        return vh, vl

    def _forward_self_cross(self, vh, vl):
        """Forward pass with self + cross-attention (pre-norm)."""
        for i in range(self.n_layers):
            # Pre-norm: Self-attention within each chain
            vh_normed = self.vh_self_norm[i](vh)
            vh_self, _ = self.vh_self_attn[i](vh_normed, vh_normed, vh_normed)
            vh = vh + vh_self

            vl_normed = self.vl_self_norm[i](vl)
            vl_self, _ = self.vl_self_attn[i](vl_normed, vl_normed, vl_normed)
            vl = vl + vl_self

            # Pre-norm: Cross-attention between chains
            vh_normed = self.vh_cross_norm[i](vh)
            vl_normed = self.vl_cross_norm[i](vl)

            vh_cross, _ = self.vh_cross_attn[i](vh_normed, vl_normed, vl_normed)
            vh = vh + vh_cross

            vl_cross, _ = self.vl_cross_attn[i](vl_normed, vh_normed, vh_normed)
            vl = vl + vl_cross

            # Pre-norm: Feed-forward
            vh = vh + self.vh_ffn[i](self.vh_norm2[i](vh))
            vl = vl + self.vl_ffn[i](self.vl_norm2[i](vl))

        return vh, vl

    def _forward_self_only(self, vh, vl):
        """Forward pass with self-attention only (pre-norm)."""
        for i in range(self.n_layers):
            # Pre-norm: Self-attention within each chain independently
            vh_normed = self.vh_norm1[i](vh)
            vh_self, _ = self.vh_self_attn[i](vh_normed, vh_normed, vh_normed)
            vh = vh + vh_self

            vl_normed = self.vl_norm1[i](vl)
            vl_self, _ = self.vl_self_attn[i](vl_normed, vl_normed, vl_normed)
            vl = vl + vl_self

            # Pre-norm: Feed-forward
            vh = vh + self.vh_ffn[i](self.vh_norm2[i](vh))
            vl = vl + self.vl_ffn[i](self.vl_norm2[i](vl))

        return vh, vl
