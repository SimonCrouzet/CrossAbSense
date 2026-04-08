"""Base decoder class for property prediction."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for property decoders.

    All decoders must:
    1. Accept fixed-size embeddings from encoder (VH and/or VL)
    2. Return scalar predictions for a single property
    3. Support optional antibody features
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimension of input embeddings from encoder
        """
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(
        self,
        vh_embeddings: Optional[torch.Tensor] = None,
        vl_embeddings: Optional[torch.Tensor] = None,
        antibody_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict property value from embeddings.

        Args:
            vh_embeddings: Heavy chain embeddings
            vl_embeddings: Light chain embeddings
            antibody_features: Optional antibody features

        Returns:
            Predictions of shape (batch_size, 1)
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters in decoder."""
        return sum(p.numel() for p in self.parameters())