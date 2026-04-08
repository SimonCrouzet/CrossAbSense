"""Visualization utilities for model predictions and analysis."""

import matplotlib.pyplot as plt
import numpy as np
import wandb
from typing import Optional


def create_prediction_scatter(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    title: str = "Ground Truth vs Predictions",
    show_error_lines: bool = True,
    show_metrics: bool = True,
) -> plt.Figure:
    """
    Create a scatter plot comparing ground truth and predictions with residual lines.

    This plot shows both the scatter points AND vertical lines connecting each
    prediction to its corresponding ground truth value, visualizing the residual error.

    Args:
        ground_truth: Array of ground truth values
        predictions: Array of predicted values
        title: Plot title
        show_error_lines: Whether to show residual lines (vertical lines from GT to prediction)
        show_metrics: Whether to show correlation metrics on plot

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Add residual lines FIRST (so they appear behind scatter points)
    if show_error_lines:
        for gt, pred in zip(ground_truth, predictions):
            # Vertical line from (gt, gt) to (gt, pred) - shows residual error
            color = 'red' if abs(pred - gt) > np.std(predictions - ground_truth) else 'lightblue'
            ax.plot([gt, gt], [gt, pred], color=color, alpha=0.5, linewidth=1.5, zorder=1)

    # Add diagonal line (perfect predictions)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5,
            label='Perfect prediction', zorder=2)

    # Scatter plot of predictions vs ground truth (on top of lines)
    ax.scatter(ground_truth, predictions, alpha=0.7, s=80, c='darkblue',
               edgecolors='white', linewidths=1, zorder=3, label='Predictions')

    # Add metrics if requested
    if show_metrics:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, _ = pearsonr(ground_truth, predictions)
        spearman_r, _ = spearmanr(ground_truth, predictions)
        rmse = np.sqrt(np.mean((predictions - ground_truth) ** 2))
        mae = np.mean(np.abs(predictions - ground_truth))

        metrics_text = (f'Pearson: {pearson_r:.3f}\n'
                       f'Spearman: {spearman_r:.3f}\n'
                       f'RMSE: {rmse:.3f}\n'
                       f'MAE: {mae:.3f}\n'
                       f'N samples: {len(ground_truth)}')
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black'))

    ax.set_xlabel('Ground Truth', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predictions', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Equal aspect ratio to make diagonal truly diagonal
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig


def create_residual_plot(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    title: str = "Residual Plot",
) -> plt.Figure:
    """
    Create a residual plot showing prediction errors.

    Args:
        ground_truth: Array of ground truth values
        predictions: Array of predicted values
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    residuals = predictions - ground_truth

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of residuals
    ax.scatter(ground_truth, residuals, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Add horizontal line at 0 (perfect predictions)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero error')

    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Residual (Prediction - Ground Truth)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_error_distribution(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    title: str = "Error Distribution",
) -> plt.Figure:
    """
    Create a histogram of prediction errors.

    Args:
        ground_truth: Array of ground truth values
        predictions: Array of predicted values
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    errors = predictions - ground_truth

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of errors
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')

    # Add vertical line at 0
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')

    # Add mean and std
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax.axvline(x=mean_error, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')

    ax.set_xlabel('Error (Prediction - Ground Truth)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{title}\nStd: {std_error:.3f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def log_prediction_plots_to_wandb(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    prefix: str = "val",
    epoch: Optional[int] = None,
    show_error_lines: bool = True,
):
    """
    Create and log prediction visualization plots to WandB.

    Args:
        ground_truth: Array of ground truth values
        predictions: Array of predicted values
        prefix: Prefix for WandB logging (e.g., 'train', 'val', 'test')
        epoch: Current epoch number (optional, for title)
        show_error_lines: Whether to show error lines on scatter plot
    """
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""

    # 1. Scatter plot: GT vs Predictions
    scatter_fig = create_prediction_scatter(
        ground_truth,
        predictions,
        title=f"{prefix.capitalize()} Predictions{epoch_str}",
        show_error_lines=show_error_lines,
        show_metrics=True,
    )
    wandb.log({f"{prefix}/prediction_scatter": wandb.Image(scatter_fig)})
    plt.close(scatter_fig)

    # 2. Residual plot
    residual_fig = create_residual_plot(
        ground_truth,
        predictions,
        title=f"{prefix.capitalize()} Residuals{epoch_str}",
    )
    wandb.log({f"{prefix}/residual_plot": wandb.Image(residual_fig)})
    plt.close(residual_fig)

    # 3. Error distribution
    error_fig = create_error_distribution(
        ground_truth,
        predictions,
        title=f"{prefix.capitalize()} Error Distribution{epoch_str}",
    )
    wandb.log({f"{prefix}/error_distribution": wandb.Image(error_fig)})
    plt.close(error_fig)


def log_wandb_scatter_table(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    prefix: str = "val",
):
    """
    Log an interactive WandB scatter plot table.

    Args:
        ground_truth: Array of ground truth values
        predictions: Array of predicted values
        prefix: Prefix for WandB logging
    """
    # Create WandB Table for interactive plotting
    data = [[gt, pred, abs(pred - gt)] for gt, pred in zip(ground_truth, predictions)]
    table = wandb.Table(data=data, columns=["ground_truth", "prediction", "absolute_error"])

    # Log as scatter plot
    wandb.log({
        f"{prefix}/prediction_scatter_interactive": wandb.plot.scatter(
            table,
            "ground_truth",
            "prediction",
            title=f"{prefix.capitalize()} Predictions vs Ground Truth"
        )
    })
