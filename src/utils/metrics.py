"""Evaluation metrics for antibody developability prediction."""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics for regression predictions.

    Args:
        predictions: Array of predicted values
        targets: Array of ground truth values

    Returns:
        Dictionary of metric names and values
    """
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    if len(predictions) == 0:
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "rmse": float("inf"),
            "mae": float("inf"),
            "n_samples": 0,
        }

    # Correlation metrics
    pearson_corr, pearson_pval = pearsonr(predictions, targets)
    spearman_corr, spearman_pval = spearmanr(predictions, targets)

    # Error metrics
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {
        "pearson": float(pearson_corr),
        "pearson_pval": float(pearson_pval),
        "spearman": float(spearman_corr),
        "spearman_pval": float(spearman_pval),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_samples": len(predictions),
    }


def aggregate_fold_metrics(fold_metrics: list) -> dict:
    """
    Aggregate metrics across cross-validation folds.

    Args:
        fold_metrics: List of metric dictionaries from each fold

    Returns:
        Dictionary with mean and std of each metric
    """
    aggregated = {}

    # Get all metric names from first fold
    metric_names = [k for k in fold_metrics[0].keys() if k != "n_samples"]

    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        aggregated[f"{metric_name}_mean"] = float(np.mean(values))
        aggregated[f"{metric_name}_std"] = float(np.std(values))

    # Total samples
    aggregated["total_samples"] = sum(fold["n_samples"] for fold in fold_metrics)

    return aggregated
