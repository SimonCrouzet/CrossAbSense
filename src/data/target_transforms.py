"""
Target transformations for antibody developability properties.

Supports:
- Min-max normalization: Scale to [0, 1]
- Z-score normalization: Zero mean, unit variance
- Log-transform: For highly skewed distributions
- Composition: Log + Z-score for heavy-tailed properties
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TargetTransform:
    """Base class for target transformations."""

    def __init__(self, name: str = "identity"):
        """
        Args:
            name: Name of the transformation (for logging)
        """
        self.name = name
        self.fitted = False

    def fit(self, targets: np.ndarray) -> "TargetTransform":
        """
        Fit transformation parameters from training data.

        Args:
            targets: Training target values, shape (N,)

        Returns:
            self (for chaining)
        """
        raise NotImplementedError

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """
        Transform target values.

        Args:
            targets: Target values to transform, shape (N,)

        Returns:
            Transformed targets, shape (N,)
        """
        raise NotImplementedError

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transformation to recover original scale.

        Args:
            transformed: Transformed target values, shape (N,)

        Returns:
            Original scale targets, shape (N,)
        """
        raise NotImplementedError

    def transform_tensor(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Transform PyTorch tensor (for use during training).

        Args:
            targets: Target tensor, shape (N,)

        Returns:
            Transformed tensor, shape (N,)
        """
        numpy_targets = targets.cpu().numpy()
        transformed = self.transform(numpy_targets)
        return torch.from_numpy(transformed).to(targets.device).float()

    def inverse_transform_tensor(
        self, transformed: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse transform PyTorch tensor (for predictions).

        Args:
            transformed: Transformed tensor, shape (N,)

        Returns:
            Original scale tensor, shape (N,)
        """
        numpy_transformed = transformed.cpu().numpy()
        original = self.inverse_transform(numpy_transformed)
        return torch.from_numpy(original).to(transformed.device).float()

    def get_stats(self) -> Dict[str, float]:
        """Get transformation statistics for logging."""
        return {"name": self.name, "fitted": self.fitted}


class IdentityTransform(TargetTransform):
    """No transformation (pass-through)."""

    def __init__(self):
        super().__init__(name="identity")
        self.fitted = True  # Always fitted

    def fit(self, targets: np.ndarray) -> "IdentityTransform":
        """No-op fit."""
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Return targets unchanged."""
        return targets

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """Return targets unchanged."""
        return transformed


class MinMaxTransform(TargetTransform):
    """
    Min-max normalization: Scale to [0, 1].

    Formula: (x - min) / (max - min)
    """

    def __init__(self, clip: bool = True):
        """
        Args:
            clip: Whether to clip predictions to [0, 1] during inverse
                transform
        """
        super().__init__(name="min_max")
        self.min = None
        self.max = None
        self.clip = clip

    def fit(self, targets: np.ndarray) -> "MinMaxTransform":
        """Fit min and max from training data."""
        self.min = float(np.min(targets))
        self.max = float(np.max(targets))
        self.fitted = True
        range_val = self.max - self.min
        logger.info(
            f"MinMax fitted: min={self.min:.4f}, "
            f"max={self.max:.4f}, range={range_val:.4f}"
        )
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        range_val = self.max - self.min
        if range_val == 0:
            logger.warning("Range is zero, returning zeros")
            return np.zeros_like(targets)
        return (targets - self.min) / range_val

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """Denormalize from [0, 1] to original range."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        result = transformed * (self.max - self.min) + self.min
        if self.clip:
            result = np.clip(result, self.min, self.max)
        return result

    def get_stats(self) -> Dict[str, float]:
        """Get min-max statistics."""
        stats = super().get_stats()
        if self.fitted:
            stats.update({
                "min": self.min,
                "max": self.max,
                "range": self.max - self.min,
                "clip": self.clip,
            })
        return stats


class ZScoreTransform(TargetTransform):
    """
    Z-score normalization: Zero mean, unit variance.

    Formula: (x - mean) / std
    """

    def __init__(self, eps: float = 1e-8):
        """
        Args:
            eps: Small constant to prevent division by zero
        """
        super().__init__(name="z_score")
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, targets: np.ndarray) -> "ZScoreTransform":
        """Fit mean and std from training data."""
        self.mean = float(np.mean(targets))
        self.std = float(np.std(targets))
        self.fitted = True
        logger.info(
            f"Z-Score fitted: mean={self.mean:.4f}, "
            f"std={self.std:.4f}"
        )
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Standardize to mean=0, std=1."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        return (targets - self.mean) / (self.std + self.eps)

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """Destandardize to original scale."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        return transformed * self.std + self.mean

    def get_stats(self) -> Dict[str, float]:
        """Get z-score statistics."""
        stats = super().get_stats()
        if self.fitted:
            stats.update({
                "mean": self.mean,
                "std": self.std,
            })
        return stats


class LogTransform(TargetTransform):
    """
    Log transformation for skewed distributions.

    Formula: log(x + offset)
    Handles negative values by adding offset to shift minimum to
    positive range.
    """

    def __init__(self, offset: Optional[float] = None, base: str = "natural"):
        """
        Args:
            offset: Value to add before log to ensure positivity.
                If None, auto-computed as -min(x) + 1
            base: Logarithm base - "natural" (ln), "10" (log10),
                or "2" (log2)
        """
        super().__init__(name=f"log_{base}")
        self.offset = offset
        self.base = base
        self.auto_offset = offset is None

    def fit(self, targets: np.ndarray) -> "LogTransform":
        """Fit offset from training data if not provided."""
        min_val = float(np.min(targets))

        if self.auto_offset:
            # Automatically compute offset to make all values positive
            if min_val <= 0:
                self.offset = -min_val + 1.0
                logger.info(
                    f"Log auto-offset computed: {self.offset:.4f} "
                    f"(min was {min_val:.4f})"
                )
            else:
                self.offset = 0.0
                logger.info(
                    f"Log offset set to 0 "
                    f"(min is positive: {min_val:.4f})"
                )
        else:
            # Use provided offset
            if min_val + self.offset <= 0:
                logger.warning(
                    f"Provided offset {self.offset} is insufficient "
                    f"for min value {min_val}. "
                    f"Some values may be <= 0 after offset!"
                )

        self.fitted = True
        logger.info(
            f"Log transform fitted: offset={self.offset:.4f}, "
            f"base={self.base}"
        )
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Apply log transformation."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")

        shifted = targets + self.offset

        # Ensure all values are positive
        if np.any(shifted <= 0):
            n_negative = np.sum(shifted <= 0)
            logger.warning(
                f"Found {n_negative} non-positive values after offset, "
                f"clipping to small positive"
            )
            shifted = np.maximum(shifted, 1e-8)

        if self.base == "natural":
            return np.log(shifted)
        elif self.base == "10":
            return np.log10(shifted)
        elif self.base == "2":
            return np.log2(shifted)
        else:
            raise ValueError(f"Unknown log base: {self.base}")

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """Apply inverse log (exponential)."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")

        if self.base == "natural":
            result = np.exp(transformed)
        elif self.base == "10":
            result = np.power(10, transformed)
        elif self.base == "2":
            result = np.power(2, transformed)
        else:
            raise ValueError(f"Unknown log base: {self.base}")

        return result - self.offset

    def get_stats(self) -> Dict[str, float]:
        """Get log transform statistics."""
        stats = super().get_stats()
        if self.fitted:
            stats.update({
                "offset": self.offset,
                "base": self.base,
                "auto_offset": self.auto_offset,
            })
        return stats


class ComposedTransform(TargetTransform):
    """
    Compose multiple transformations sequentially.

    Example: Log + Z-score for heavy-tailed distributions
    """

    def __init__(self, transforms: Tuple[TargetTransform, ...]):
        """
        Args:
            transforms: Tuple of transforms to apply in order
        """
        names = " -> ".join([t.name for t in transforms])
        super().__init__(name=f"composed({names})")
        self.transforms = transforms

    def fit(self, targets: np.ndarray) -> "ComposedTransform":
        """Fit all transforms sequentially."""
        current = targets
        for transform in self.transforms:
            transform.fit(current)
            current = transform.transform(current)
        self.fitted = True
        logger.info(f"Composed transform fitted: {self.name}")
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Apply all transforms in order."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        current = targets
        for transform in self.transforms:
            current = transform.transform(current)
        return current

    def inverse_transform(self, transformed: np.ndarray) -> np.ndarray:
        """Apply inverse transforms in reverse order."""
        if not self.fitted:
            raise ValueError("Transform must be fitted before use")
        current = transformed
        for transform in reversed(self.transforms):
            current = transform.inverse_transform(current)
        return current

    def get_stats(self) -> Dict[str, float]:
        """Get statistics from all transforms."""
        stats = super().get_stats()
        for i, transform in enumerate(self.transforms):
            t_stats = transform.get_stats()
            for key, val in t_stats.items():
                stats[f"transform_{i}_{key}"] = val
        return stats


def create_transform(transform_type: str, **kwargs) -> TargetTransform:
    """
    Factory function to create target transforms.

    Args:
        transform_type: Type of transform
            - "identity": No transformation
            - "min_max": Min-max normalization to [0, 1]
            - "z_score": Z-score standardization
            - "log": Log transformation
            - "log_zscore": Log + Z-score (recommended for heavy-tailed)
        **kwargs: Additional arguments for the transform

    Returns:
        Configured TargetTransform instance
    """
    if transform_type == "identity":
        return IdentityTransform()

    elif transform_type == "min_max":
        return MinMaxTransform(**kwargs)

    elif transform_type == "z_score":
        return ZScoreTransform(**kwargs)

    elif transform_type == "log":
        return LogTransform(**kwargs)

    elif transform_type == "log_zscore":
        # Composed: Log then Z-score
        log_transform = LogTransform(**kwargs)
        zscore_transform = ZScoreTransform()
        return ComposedTransform((log_transform, zscore_transform))

    else:
        raise ValueError(
            f"Unknown transform type: {transform_type}. "
            f"Choose from: identity, min_max, z_score, log, log_zscore"
        )


# Property-specific recommendations based on statistical analysis
PROPERTY_TRANSFORM_RECOMMENDATIONS = {
    "HIC": "log_zscore",  # Skewness 2.027, log improves distribution
    "AC-SINS_pH7.4": "z_score",  # Skewness 1.120, can be negative (no log)
    "Titer": "log_zscore",  # Skewness 1.208, 22.8x range, log improves
    "Tm2": "z_score",  # Skewness -0.626 (slight left skew), narrow range
    # Skewness 0.427, bounded [0, 0.55], z-score preferred
    "PR_CHO": "z_score",
}


def get_recommended_transform(property_name: str) -> str:
    """
    Get recommended transformation for a property.

    Args:
        property_name: Property name (e.g., "HIC", "Titer")

    Returns:
        Recommended transform type
    """
    return PROPERTY_TRANSFORM_RECOMMENDATIONS.get(property_name, "z_score")
