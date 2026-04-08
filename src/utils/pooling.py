"""Pooling strategies for sequence embeddings."""

import torch
import torch.nn as nn


class SlicedWassersteinPooling(nn.Module):
    """
    Sliced Wasserstein pooling for variable-length sequence embeddings.

    Uses optimal transport theory to aggregate sequence representations
    while preserving distributional structure.

    Reference:
    - Optimal Transport-based Pooling (OT-Pool)
    - Approximates Wasserstein distance via random projections
    """

    def __init__(self, num_projections: int = 100, temperature: float = 1.0):
        """
        Args:
            num_projections: Number of random projections for approximation
            temperature: Temperature for soft matching (lower = more selective)
        """
        super().__init__()
        self.num_projections = num_projections
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Pool sequence embeddings using Sliced Wasserstein distance.

        Args:
            embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask of shape (batch_size, seq_len)

        Returns:
            Pooled embeddings of shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        device = embeddings.device

        # Handle masking
        if mask is not None:
            # Zero out masked positions
            embeddings = embeddings * mask.unsqueeze(-1)
            # Count valid tokens per sequence
            valid_counts = mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
        else:
            valid_counts = torch.full((batch_size, 1), seq_len, device=device)

        # Generate random projection directions
        # Shape: (num_projections, hidden_dim)
        projections = torch.randn(
            self.num_projections, hidden_dim, device=device, dtype=embeddings.dtype
        )
        projections = projections / projections.norm(dim=1, keepdim=True)

        # Project embeddings onto random directions
        # Shape: (batch_size, seq_len, num_projections)
        projected = torch.matmul(embeddings, projections.T)

        # Sort projections (computes 1D Wasserstein in closed form)
        # Shape: (batch_size, seq_len, num_projections)
        sorted_proj, _ = torch.sort(projected, dim=1)

        # Compute weighted centroid (approximation of Wasserstein barycenter)
        # Use exponential weighting to emphasize central values
        positions = torch.arange(seq_len, device=device, dtype=embeddings.dtype)
        positions = positions.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)

        # Gaussian-like weighting centered at median
        center = valid_counts.unsqueeze(-1) / 2  # (batch_size, 1, 1)
        weights = torch.exp(-((positions - center) ** 2) / (self.temperature * valid_counts.unsqueeze(-1)))

        # Normalize weights
        if mask is not None:
            weights = weights * mask.unsqueeze(-1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)

        # Weighted average of sorted projections
        # Shape: (batch_size, num_projections)
        weighted_proj = (sorted_proj * weights).sum(dim=1)

        # Project back to original space (pseudo-inverse via least squares)
        # This is an approximation: find point in original space whose projections match
        # We use a simple weighted mean with projection-based attention

        # Compute attention weights based on alignment with target projections
        # Shape: (batch_size, seq_len, num_projections)
        alignment = projected - weighted_proj.unsqueeze(1)
        alignment_scores = -torch.norm(alignment, dim=-1)  # (batch_size, seq_len)

        # Convert to attention weights
        if mask is not None:
            alignment_scores = alignment_scores.masked_fill(~mask.bool(), float('-inf'))
        attention_weights = torch.softmax(alignment_scores / self.temperature, dim=1)

        # Apply attention to pool embeddings
        # Shape: (batch_size, hidden_dim)
        pooled = torch.matmul(attention_weights.unsqueeze(1), embeddings).squeeze(1)

        return pooled


def sliced_wasserstein_pool(
    embeddings: torch.Tensor,
    mask: torch.Tensor = None,
    num_projections: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Functional interface for Sliced Wasserstein pooling.

    Args:
        embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        mask: Optional attention mask of shape (batch_size, seq_len)
        num_projections: Number of random projections
        temperature: Temperature for soft matching

    Returns:
        Pooled embeddings of shape (batch_size, hidden_dim)
    """
    pooler = SlicedWassersteinPooling(num_projections, temperature)
    return pooler(embeddings, mask)
