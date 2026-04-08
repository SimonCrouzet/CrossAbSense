"""
Inference dataset for antibody sequences without targets.

This dataset is designed for prediction/inference tasks where no ground truth
labels are available. Used for generating submissions and making predictions
on new data.
"""

import logging
from typing import Optional, List

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AntibodyInferenceDataset(Dataset):
    """Dataset for inference on antibody sequences (no targets required)."""

    def __init__(
        self,
        heavy_seqs: List[str],
        light_seqs: List[str],
        precomputed_embeddings: Optional[dict] = None,
    ):
        """
        Args:
            heavy_seqs: List of heavy chain sequences
            light_seqs: List of light chain sequences
            precomputed_embeddings: Dict mapping (heavy_seq, light_seq) -> (heavy_emb, light_emb) tensors
        """
        assert len(heavy_seqs) == len(light_seqs), "Heavy and light sequences must have same length"
        
        self.heavy_seqs = heavy_seqs
        self.light_seqs = light_seqs
        self.precomputed_embeddings = precomputed_embeddings
        
        # Filter out rows with missing sequences
        self.valid_indices = []
        for i in range(len(heavy_seqs)):
            if heavy_seqs[i] and light_seqs[i]:
                self.valid_indices.append(i)
        
        n_filtered = len(heavy_seqs) - len(self.valid_indices)
        if n_filtered > 0:
            logger.warning(f"Filtered out {n_filtered} samples with missing sequences")

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        heavy_col: str = "vh_protein_sequence",
        light_col: str = "vl_protein_sequence",
        precomputed_embeddings: Optional[dict] = None,
    ):
        """Create dataset from DataFrame.
        
        Args:
            data: DataFrame with antibody sequences
            heavy_col: Column name for heavy chain sequences
            light_col: Column name for light chain sequences
            precomputed_embeddings: Optional precomputed embeddings dict
        """
        heavy_seqs = data[heavy_col].fillna("").tolist()
        light_seqs = data[light_col].fillna("").tolist()
        
        return cls(heavy_seqs, light_seqs, precomputed_embeddings)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        """Get a single antibody sample."""
        actual_idx = self.valid_indices[idx]
        
        heavy_seq = self.heavy_seqs[actual_idx]
        light_seq = self.light_seqs[actual_idx]

        result = {
            "heavy_sequence": heavy_seq,
            "light_sequence": light_seq,
        }

        # Add precomputed embeddings if available
        if self.precomputed_embeddings is not None:
            key = (heavy_seq, light_seq)
            if key in self.precomputed_embeddings:
                vh_emb, vl_emb = self.precomputed_embeddings[key]
                result["vh_embedding"] = vh_emb
                result["vl_embedding"] = vl_emb

        return result
