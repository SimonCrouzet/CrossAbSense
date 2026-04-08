"""Multi-Layer Perceptron decoder."""

import logging
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.init as init

from .base_decoder import BaseDecoder
from .output_activations import get_output_activation

logger = logging.getLogger(__name__)


class MLPDecoder(BaseDecoder):
    """
    Multi-Layer Perceptron (MLP) decoder.

    Simple but effective feed-forward neural network with:
    - Sequence pooling (mean pooling by default)
    - Multiple hidden layers
    - Activation functions (ReLU, GELU, SiLU)
    - Dropout for regularization
    - Optional batch normalization
    - Antibody features injection support
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        activation: str = "relu",
        dropout: float = 0.3,
        batch_norm: bool = True,
        pooling: str = "mean",
        output_activation: str = "none",
        target_stats: Optional[Dict] = None,
        antibody_features_dim: int = 0,
        antibody_features_projection_dim: Optional[int] = None,
        antibody_features_injection_layer: str = "second",
    ):
        """
        Args:
            input_dim: Dimension of input embeddings (already concatenated if using MultiEncoder concat)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function (relu, gelu, silu)
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            pooling: Pooling strategy for sequences (mean, max, cls)
            output_activation: Final activation for output layer (none, softplus, sigmoid, exp)
            target_stats: Target statistics for normalization
            antibody_features_dim: Dimension of antibody features to concatenate (0 = disabled)
            antibody_features_projection_dim: Project features to this dimension before injection
            antibody_features_injection_layer: Where to inject features ("first", "second", "third", "last")
        """
        super().__init__(input_dim)

        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.pooling = pooling
        self.antibody_features_dim = antibody_features_dim
        self.antibody_features_injection_layer = antibody_features_injection_layer

        # Select activation function
        if activation == "relu":
            self.act_fn_class = nn.ReLU
        elif activation == "gelu":
            self.act_fn_class = nn.GELU
        elif activation == "silu":
            self.act_fn_class = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Set up antibody features projection
        if antibody_features_dim > 0:
            if antibody_features_projection_dim is not None:
                self.antibody_features_projection = nn.Linear(
                    antibody_features_dim, antibody_features_projection_dim
                )
                self.antibody_features_effective_dim = antibody_features_projection_dim
            else:
                self.antibody_features_projection = None
                self.antibody_features_effective_dim = antibody_features_dim
        else:
            self.antibody_features_projection = None
            self.antibody_features_effective_dim = 0

        # Build MLP layers as a ModuleList for flexible injection
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        num_hidden_layers = len(hidden_dims)

        for i, hidden_dim in enumerate(hidden_dims):
            # Determine if we should inject features at this layer
            inject_here = False
            if antibody_features_dim > 0:
                if antibody_features_injection_layer == "first" and i == 0:
                    inject_here = True
                elif antibody_features_injection_layer == "second" and i == 1:
                    inject_here = True
                elif antibody_features_injection_layer == "third" and i == 2:
                    inject_here = True
                elif antibody_features_injection_layer == "last" and i == num_hidden_layers - 1:
                    # Special case: if "last" is also "first" or "second", it will be handled there
                    # But if we have say 5 layers, "last" will inject at layer 4
                    if not ((antibody_features_injection_layer == "first" and i == 0) or
                            (antibody_features_injection_layer == "second" and i == 1) or
                            (antibody_features_injection_layer == "third" and i == 2)):
                        inject_here = True
                
                # Fallback: if user specified a layer beyond what we have, inject at first available or last
                elif i == num_hidden_layers - 1 and antibody_features_injection_layer not in ["first", "second", "third"]:
                     inject_here = True

            if inject_here:
                logger.info(f"🧬 MLPDecoder: Injecting {self.antibody_features_effective_dim}d features at layer {i}")
                prev_dim += self.antibody_features_effective_dim

            # Build layer block
            layer_block = nn.Sequential()
            layer_block.add_module("linear", nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layer_block.add_module("bn", nn.BatchNorm1d(hidden_dim))
            
            layer_block.add_module("activation", self.act_fn_class())
            
            if dropout > 0:
                layer_block.add_module("dropout", nn.Dropout(dropout))
            
            self.layers.append(layer_block)
            prev_dim = hidden_dim

        # Output layer
        # Check if we should inject at "last" but only had one hidden layer (handled at i=0)
        # Or if we want to inject JUST before the output layer (not supported by "last" definition above, 
        # but let's make it robust)
        if antibody_features_dim > 0 and antibody_features_injection_layer == "last" and num_hidden_layers == 0:
             prev_dim += self.antibody_features_effective_dim
             logger.info(f"🧬 MLPDecoder: Injecting {self.antibody_features_effective_dim}d features at output layer")

        self.output_layer = nn.Linear(prev_dim, 1)

        # Property-specific output activation
        self.output_activation_name = output_activation
        self.output_activation = get_output_activation(
            output_activation, target_stats
        )

    def _xavier_init_weights(self, gain: float = 1.0):
        """Apply Xavier uniform initialization to all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    init.zeros_(module.bias)
        logger.info(f"Applied Xavier uniform initialization with gain={gain}")

    def forward(
        self,
        vh_embeddings: Optional[torch.Tensor] = None,
        vl_embeddings: Optional[torch.Tensor] = None,
        antibody_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict property value from embeddings.

        Args:
            vh_embeddings: Heavy chain embeddings (batch, dim) or (batch, seq, dim)
            vl_embeddings: Light chain embeddings (batch, dim) or (batch, seq, dim)
            antibody_features: Optional antibody features (batch, features_dim)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Validate at least one chain present
        if vh_embeddings is None and vl_embeddings is None:
            raise ValueError("Must provide at least one chain (VH or VL embeddings)")

        # Pool each chain separately if needed
        vh_pooled = self._pool_if_needed(vh_embeddings) if vh_embeddings is not None else None
        vl_pooled = self._pool_if_needed(vl_embeddings) if vl_embeddings is not None else None

        # Concatenate pooled embeddings
        if vh_pooled is not None and vl_pooled is not None:
            x = torch.cat([vh_pooled, vl_pooled], dim=-1)
        elif vh_pooled is not None:
            x = vh_pooled
        else:
            x = vl_pooled

        # Prepare antibody features
        if antibody_features is not None and self.antibody_features_projection is not None:
            antibody_features = self.antibody_features_projection(antibody_features)

        # Forward through layers with injection
        num_hidden_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            # Check for injection
            if antibody_features is not None:
                inject = False
                if self.antibody_features_injection_layer == "first" and i == 0:
                    inject = True
                elif self.antibody_features_injection_layer == "second" and i == 1:
                    inject = True
                elif self.antibody_features_injection_layer == "third" and i == 2:
                    inject = True
                elif self.antibody_features_injection_layer == "last" and i == max(0, num_hidden_layers - 1):
                    # Only inject here if it wasn't already injected at first/second/third
                    if not ((self.antibody_features_injection_layer == "first" and i == 0) or
                            (self.antibody_features_injection_layer == "second" and i == 1) or
                            (self.antibody_features_injection_layer == "third" and i == 2)):
                        inject = True
                
                if inject:
                    x = torch.cat([x, antibody_features], dim=-1)

            x = layer(x)

        # Final check for injection just before output layer if no hidden layers
        if antibody_features is not None and num_hidden_layers == 0 and self.antibody_features_injection_layer == "last":
             x = torch.cat([x, antibody_features], dim=-1)

        # Output layer
        output = self.output_layer(x)

        # Apply property-specific output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def _pool_if_needed(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pool embeddings if they are 3D sequences."""
        if embeddings.dim() == 3:
            # Pool sequences: (batch, seq_len, dim) -> (batch, dim)
            if self.pooling == "mean":
                return embeddings.mean(dim=1)
            elif self.pooling == "max":
                return embeddings.max(dim=1)[0]
            elif self.pooling == "cls":
                return embeddings[:, 0, :]  # Use first token
            else:
                raise ValueError(
                    f"Unknown pooling strategy: {self.pooling}. "
                    f"Choose from: mean, max, cls"
                )
        elif embeddings.dim() == 2:
            # Already pooled
            return embeddings
        else:
            raise ValueError(
                f"Expected 2D or 3D input, got {embeddings.dim()}D "
                f"with shape {embeddings.shape}"
            )