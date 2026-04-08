"""Multi-encoder that combines multiple protein language models."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .antiberty_encoder import AntiBERTyEncoder
from .base_encoder import BaseEncoder
from .esmc_encoder import ESMCEncoder
from .prott5_encoder import ProtT5Encoder


class MultiEncoder(BaseEncoder):
    """
    Flexible multi-encoder that combines multiple protein language models.

    Supports different fusion strategies:
    - concat: Simple concatenation of embeddings
    - weighted: Learned weighted sum of embeddings
    - attention: Cross-attention between embeddings
    - gated: Gated fusion with learned gates
    """

    # Reference: Native embedding dimensions for each encoder type
    # Note: Encoders now use their native dimensions directly (no projection)
    # These are provided for reference only - actual dimensions come from get_embedding_dim()
    DEFAULT_EMBEDDING_DIMS = {
        "antiberty": 512,      # AntiBERTy native hidden size
        "esmc": 2560,          # ESM-C 6B native hidden size (most common)
        "esmc_300m": 960,      # ESM-C 300M native hidden size
        "esmc_600m": 1152,     # ESM-C 600M native hidden size
        "prott5": 1024,        # ProtT5-XL native hidden size
    }

    def __init__(
        self,
        encoder_types: List[str],
        encoder_configs: Optional[Dict[str, dict]] = None,
        fusion_strategy: str = "concat",
        embedding_dim: Optional[int] = None,
        use_heavy: bool = True,
        use_light: bool = True,
        pooling: str = "mean",
        freeze_epochs: int = 0,
    ):
        """
        Args:
            encoder_types: List of encoder types to use (e.g., ["antiberty", "esmc_300m", "prott5"])
            encoder_configs: Dict of encoder-specific configs (optional)
            fusion_strategy: How to combine embeddings (concat, weighted, attention, gated)
            embedding_dim: Final output embedding dimension (optional, only used for fusion output)
                          If None, uses native dimensions (concat strategy sums them, others use first encoder's dim)
            use_heavy: Whether to encode heavy chain
            use_light: Whether to encode light chain
            pooling: Pooling strategy (mean, cls, max, sliced_wasserstein)
            freeze_epochs: Epochs to keep frozen during fine-tuning
        """
        super().__init__(use_heavy, use_light, pooling, freeze_epochs)

        if not encoder_types:
            raise ValueError("Must specify at least one encoder type")

        self.encoder_types = encoder_types
        self.fusion_strategy = fusion_strategy
        encoder_configs = encoder_configs or {}

        # Initialize encoders
        self.encoders = nn.ModuleDict()
        raw_embedding_dims = []

        for enc_type in encoder_types:
            enc_config = encoder_configs.get(enc_type, {}).copy()  # Copy to avoid modifying original

            # Remove keys that are explicitly passed to encoder constructors to avoid duplicate arguments
            for key in ['embedding_dim', 'use_heavy', 'use_light', 'pooling', 'freeze_epochs']:
                enc_config.pop(key, None)

            # Create encoder (no embedding_dim parameter - they use native dimensions)
            if enc_type == "antiberty":
                encoder = AntiBERTyEncoder(
                    use_heavy=use_heavy,
                    use_light=use_light,
                    pooling=pooling,
                    freeze_epochs=freeze_epochs,
                    **enc_config,
                )
            elif enc_type in ["esmc", "esmc_300m", "esmc_600m", "esmc_6b"]:
                # Map encoder type to model_name if it's a specific variant
                model_name_map = {
                    "esmc": "esmc_600m",  # Default to 600M
                    "esmc_300m": "esmc_300m",
                    "esmc_600m": "esmc_600m",
                    "esmc_6b": "esmc_6b",
                }
                if "model_name" not in enc_config:
                    enc_config["model_name"] = model_name_map.get(enc_type, "esmc_600m")

                encoder = ESMCEncoder(
                    use_heavy=use_heavy,
                    use_light=use_light,
                    pooling=pooling,
                    freeze_epochs=freeze_epochs,
                    **enc_config,
                )
            elif enc_type == "prott5":
                encoder = ProtT5Encoder(
                    use_heavy=use_heavy,
                    use_light=use_light,
                    pooling=pooling,
                    freeze_epochs=freeze_epochs,
                    **enc_config,
                )
            else:
                raise ValueError(f"Unknown encoder type: {enc_type}")

            self.encoders[enc_type] = encoder

            # Get encoder's native output dimension (handles VH+VL concatenation internally)
            encoder_output_dim = encoder.get_embedding_dim()
            raw_embedding_dims.append(encoder_output_dim)

        # Initialize fusion module
        # For single encoder: use native dimension (no projection)
        # For multiple encoders: use specified embedding_dim or auto-detect based on strategy
        if len(raw_embedding_dims) == 1:
            # Single encoder: output dimension is the encoder's native dimension
            fusion_output_dim = raw_embedding_dims[0]
        else:
            # Multiple encoders
            if embedding_dim is None:
                # Auto-detect based on fusion strategy:
                # - concat: sum all native dimensions
                # - other strategies: use first encoder's dimension (default)
                if fusion_strategy == "concat":
                    fusion_output_dim = sum(raw_embedding_dims)
                else:
                    fusion_output_dim = raw_embedding_dims[0]
            else:
                # Use explicitly specified output dimension
                fusion_output_dim = embedding_dim

        # Store the actual output dimension
        self.embedding_dim = fusion_output_dim

        self.fusion = self._create_fusion_module(
            raw_embedding_dims, fusion_output_dim, fusion_strategy
        )

    def _create_fusion_module(
        self,
        input_dims: List[int],
        output_dim: int,
        strategy: str
    ) -> nn.Module:
        """
        Create the fusion module based on the strategy.

        Args:
            input_dims: List of input dimensions from each encoder
            output_dim: Desired output dimension
            strategy: Fusion strategy (concat, weighted, attention, gated)

        Returns:
            Fusion module
        """
        # Concat strategy: simple concatenation without projection
        # Dimension reduction is handled by decoder's projection layers
        if strategy == "concat":
            return ConcatFusion(input_dims, output_dim)
        elif strategy == "weighted":
            return WeightedFusion(input_dims, output_dim)
        elif strategy == "attention":
            return AttentionFusion(input_dims, output_dim)
        elif strategy == "gated":
            return GatedFusion(input_dims, output_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def forward(
        self,
        heavy_sequences: Optional[list] = None,
        light_sequences: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode antibody sequences using multiple encoders.

        Args:
            heavy_sequences: List of heavy chain sequences (batch_size,)
            light_sequences: List of light chain sequences (batch_size,)

        Returns:
            Tuple of (vh_embeddings, vl_embeddings):
            - Each encoder returns (vh, vl) tuples
            - Fusion combines all VH embeddings and all VL embeddings separately
        """
        # Get embeddings from each encoder (each returns tuple)
        vh_embeddings = []
        vl_embeddings = []

        for enc_type in self.encoder_types:
            vh_emb, vl_emb = self.encoders[enc_type](heavy_sequences, light_sequences)
            if vh_emb is not None:
                vh_embeddings.append(vh_emb)
            if vl_emb is not None:
                vl_embeddings.append(vl_emb)

        # Fuse embeddings for each chain separately
        fused_vh = self.fusion(vh_embeddings) if vh_embeddings else None
        fused_vl = self.fusion(vl_embeddings) if vl_embeddings else None

        return (fused_vh, fused_vl)

    def get_embedding_dim(self) -> int:
        """
        Get output embedding dimension per chain after fusion.

        Returns:
            embedding_dim (dimension of each chain's fused embeddings)
        """
        return self.embedding_dim

    def freeze(self):
        """Freeze all sub-encoders."""
        for encoder in self.encoders.values():
            encoder.freeze()
        self._frozen = True

    def unfreeze(self):
        """Unfreeze all sub-encoders."""
        for encoder in self.encoders.values():
            encoder.unfreeze()
        self._frozen = False


# ============================================================================
# Fusion Modules
# ============================================================================


class ConcatFusion(nn.Module):
    """
    Simple concatenation of encoder embeddings without projection.

    The decoder handles all dimension reduction through its projection layers,
    so fusion modules only need to combine embeddings.
    """

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        # No projection needed - decoder handles dimension reduction

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate embeddings along feature dimension."""
        # Single encoder case: return as-is
        if len(embeddings) == 1:
            return embeddings[0]

        # Check if we're dealing with sequence embeddings (3D) or pooled embeddings (2D)
        if embeddings[0].dim() == 3:
            # Sequence embeddings: (batch, seq_len, hidden_dim)
            # Need to pad to same sequence length before concatenating along feature dim
            batch_size = embeddings[0].shape[0]
            max_seq_len = max(emb.shape[1] for emb in embeddings)

            padded_embeddings = []
            for emb in embeddings:
                seq_len = emb.shape[1]
                if seq_len < max_seq_len:
                    # Pad sequence dimension with zeros
                    padding = torch.zeros(
                        batch_size, max_seq_len - seq_len, emb.shape[2],
                        dtype=emb.dtype, device=emb.device
                    )
                    emb = torch.cat([emb, padding], dim=1)
                padded_embeddings.append(emb)

            # Concatenate along feature dimension: (batch, max_seq_len, total_hidden_dim)
            return torch.cat(padded_embeddings, dim=-1)
        else:
            # Pooled embeddings: (batch, hidden_dim) - direct concatenation
            return torch.cat(embeddings, dim=-1)


class WeightedFusion(nn.Module):
    """Learned weighted sum of embeddings."""

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.num_encoders = len(input_dims)

        # Project each embedding to the same dimension first
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Learned weights for each encoder (normalized via softmax)
        self.weights = nn.Parameter(torch.ones(self.num_encoders))

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of embeddings."""
        # Project each embedding to output_dim
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings)]

        # Stack and weight
        stacked = torch.stack(projected, dim=0)  # (num_encoders, batch_size, output_dim)

        # Compute softmax weights
        weights = torch.softmax(self.weights, dim=0)
        weights = weights.view(-1, 1, 1)  # (num_encoders, 1, 1)

        # Weighted sum
        weighted_sum = (stacked * weights).sum(dim=0)  # (batch_size, output_dim)

        return self.norm(weighted_sum)


class AttentionFusion(nn.Module):
    """Cross-attention between embeddings for fusion."""

    def __init__(self, input_dims: List[int], output_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_encoders = len(input_dims)

        # Project each embedding to the same dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings using cross-attention."""
        # Project each embedding to output_dim
        projected = [proj(emb) for proj, emb in zip(self.projections, embeddings)]

        # Stack as sequence: (batch_size, num_encoders, output_dim)
        stacked = torch.stack(projected, dim=1)

        # Self-attention over encoder embeddings
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Mean pool across encoders
        pooled = attended.mean(dim=1)  # (batch_size, output_dim)

        return self.output_proj(pooled)


class GatedFusion(nn.Module):
    """Gated fusion with learned gates for each encoder."""

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.num_encoders = len(input_dims)

        # Project each embedding to output_dim
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Gate networks for each encoder
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.Sigmoid(),
            )
            for dim in input_dims
        ])

        # Final normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Fuse embeddings using learned gates."""
        gated_embeddings = []

        for emb, proj, gate in zip(embeddings, self.projections, self.gates):
            # Project embedding
            proj_emb = proj(emb)

            # Compute gate
            gate_value = gate(emb)

            # Apply gate
            gated_emb = proj_emb * gate_value
            gated_embeddings.append(gated_emb)

        # Sum gated embeddings
        fused = torch.stack(gated_embeddings, dim=0).sum(dim=0)

        return self.norm(fused)
