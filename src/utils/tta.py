"""Test-Time Augmentation (TTA) utilities for antibody sequences."""

import numpy as np
import torch
from typing import Optional


def add_embedding_noise(
    embeddings: torch.Tensor,
    noise_std: float = 0.01
) -> torch.Tensor:
    """
    Add Gaussian noise to embeddings for TTA.
    
    Args:
        embeddings: Input embeddings (batch, seq_len, dim) or (batch, dim)
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Embeddings with added noise
    """
    noise = torch.randn_like(embeddings) * noise_std
    return embeddings + noise


def predict_with_tta(
    model,
    embeddings: torch.Tensor,
    n_augmentations: int = 5,
    noise_std: float = 0.01,
    target_transform=None,
    device: str = "cuda"
) -> tuple:
    """
    Make predictions with test-time augmentation.
    
    Applies multiple forward passes with slightly perturbed embeddings,
    then averages predictions for robustness and uncertainty estimation.
    
    Args:
        model: Trained DevelopabilityModel
        embeddings: Input embeddings to augment
        n_augmentations: Number of augmented predictions to make
        noise_std: Standard deviation of noise to add to embeddings
        target_transform: Optional transform to denormalize predictions
        device: Device to run on
    
    Returns:
        (mean_prediction, std_prediction): Mean and std across augmentations
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_augmentations):
            # Add noise to embeddings
            if noise_std > 0:
                aug_embeddings = add_embedding_noise(embeddings, noise_std)
            else:
                aug_embeddings = embeddings
            
            # Predict
            aug_embeddings = aug_embeddings.to(device)
            pred = model(aug_embeddings)
            
            # Apply inverse transform if provided
            if target_transform is not None:
                pred = target_transform.inverse_transform(
                    pred.cpu().numpy()
                )
                pred = torch.from_numpy(pred)
            
            predictions.append(pred.cpu().numpy())
    
    predictions = np.array(predictions)  # (n_aug, batch, 1)
    
    # Compute mean and std across augmentations
    mean_pred = predictions.mean(axis=0)  # (batch, 1)
    std_pred = predictions.std(axis=0)    # (batch, 1)
    
    return mean_pred, std_pred


def predict_batch_with_tta(
    model,
    batch: dict,
    n_augmentations: int = 5,
    noise_std: float = 0.01,
    target_transform=None,
    device: str = "cuda"
) -> tuple:
    """
    Make predictions on a batch with TTA.
    
    Args:
        model: Trained DevelopabilityModel
        batch: Batch dict with 'embeddings' key
        n_augmentations: Number of augmented predictions
        noise_std: Std of noise to add
        target_transform: Optional transform to denormalize
        device: Device to run on
    
    Returns:
        (mean_predictions, std_predictions): Arrays of shape (batch_size,)
    """
    embeddings = batch['embeddings']
    mean_pred, std_pred = predict_with_tta(
        model=model,
        embeddings=embeddings,
        n_augmentations=n_augmentations,
        noise_std=noise_std,
        target_transform=target_transform,
        device=device
    )
    
    # Flatten to (batch_size,)
    return mean_pred.squeeze(-1), std_pred.squeeze(-1)
