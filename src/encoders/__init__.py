"""Encoder modules for antibody sequence representation."""

from .base_encoder import BaseEncoder
from .antiberty_encoder import AntiBERTyEncoder
from .esmc_encoder import ESMCEncoder
from .prott5_encoder import ProtT5Encoder
from .multi_encoder import MultiEncoder

__all__ = [
    "BaseEncoder",
    "AntiBERTyEncoder",
    "ESMCEncoder",
    "ProtT5Encoder",
    "MultiEncoder",
]
