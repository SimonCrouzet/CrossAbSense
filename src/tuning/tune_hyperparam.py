#!/usr/bin/env python3
"""
Generic hyperparameter tuning script for all phases.

Supports:
- Phase 4.1: Sequence representation comparison (seq_rep)
- Phase 4.2: Encoder comparison (encoder + antibody features)
- Phase 4.3: Antibody features integration with dependency filtering
- Future phases: Easily extensible

Run via: wandb agent <sweep-id>
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# PARAMETER DEPENDENCIES CONFIGURATION
# ============================================================================
# Define dependencies between parameters to automatically skip invalid sweep runs.
#
# PARAMETER_DEPENDENCIES: Specifies which parameters depend on others
#   Format: {dependent_param: {required_param: required_value, ...}}
#
# PARAMETER_DEFAULTS: Default values when dependencies are not met
#   If dependency NOT met AND parameter != default → SKIP run
#   If dependency NOT met AND parameter == default → ALLOW run (baseline)
#
# Example: normalize_antibody_features only matters when use_antibody_features=True
#   - use=False, normalize=False → ALLOW (baseline)
#   - use=False, normalize=True → SKIP (invalid combination)
# ============================================================================

PARAMETER_DEPENDENCIES = {
    'normalize_antibody_features': {
        'use_antibody_features': True
    },
    'antibody_features_projection_dim': {
        'use_antibody_features': True
    }
}

PARAMETER_DEFAULTS = {
    'normalize_antibody_features': False,
    'antibody_features_projection_dim': None
}

# Add project root to path for WandB execution
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import pytorch_lightning as pl

# Custom type parsers for argparse
def nullable_int(value):
    """Parse integer or None/null string."""
    if value is None or str(value).lower() in ('none', 'null', ''):
        return None
    return int(value)

# Set matmul precision for better performance on Tensor Cores
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger

from src.data import GDPa1DataModule
from src.models import DevelopabilityModel
from src.utils import load_config, get_embeddings_config, setup_logger, get_property_config
from src.callbacks import GradientMonitor

# Setup logger with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger("tune_hyperparam", log_file=f"hyperparam_tuning_{timestamp}.log")
logger.info(f"Command: python {' '.join(sys.argv)}")

# Validation epoch frequency - must be consistent throughout
VALIDATION_EPOCH_FREQUENCY = 2


class EarlyStoppingWithLogging(EarlyStopping):
    """Early stopping callback that logs when triggered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stopped = False

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if trainer.should_stop and not self.stopped:
            self.stopped = True
            logger.info(f"Early stopping triggered at epoch {trainer.current_epoch}")


class MetricsCollectorCallback(pl.Callback):
    """Callback to collect and log fold metrics live to WandB run."""

    def __init__(self, fold_idx: int, wandb_run):
        """
        Args:
            fold_idx: Current fold index (0-indexed)
            wandb_run: WandB run to log to
        """
        super().__init__()
        self.fold_idx = fold_idx
        self.wandb_run = wandb_run
        self.epoch_metrics = []  # Store for CV aggregation (one entry per epoch)
        self.validation_epochs = set()  # Track which epochs actually had validation

    def on_train_epoch_end(self, trainer, pl_module):
        """Collect and log training metrics after each epoch."""
        epoch = trainer.current_epoch

        # Find or create entry for this epoch
        epoch_data = None
        for entry in self.epoch_metrics:
            if entry["epoch"] == epoch:
                epoch_data = entry
                break

        if epoch_data is None:
            # Create new entry
            epoch_data = {
                "epoch": epoch,
                "train_loss": float(trainer.callback_metrics.get("train_loss_epoch", 0.0)),
                "train_spearman": float(trainer.callback_metrics.get("train_spearman", 0.0)),
            }
            self.epoch_metrics.append(epoch_data)
        else:
            # Update existing entry
            epoch_data.update({
                "train_loss": float(trainer.callback_metrics.get("train_loss_epoch", 0.0)),
                "train_spearman": float(trainer.callback_metrics.get("train_spearman", 0.0)),
            })

        # Log training metrics to WandB run with fold prefix
        metrics_to_log = {
            "epoch": epoch,
            f"fold_{self.fold_idx}/train_loss": epoch_data["train_loss"],
            f"fold_{self.fold_idx}/train_spearman": epoch_data["train_spearman"],
        }

        # Add gradient norm if available
        if "grad_norm" in trainer.callback_metrics:
            metrics_to_log[f"fold_{self.fold_idx}/grad_norm"] = float(trainer.callback_metrics["grad_norm"])

        # Add learning rate
        if len(trainer.optimizers) > 0:
            optimizer = trainer.optimizers[0]
            if len(optimizer.param_groups) > 0:
                metrics_to_log[f"fold_{self.fold_idx}/learning_rate"] = optimizer.param_groups[0]['lr']

        # Log to WandB run
        if self.wandb_run is not None and metrics_to_log:
            self.wandb_run.log(metrics_to_log)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Update epoch metrics with validation results after validation."""
        # Skip during sanity check (runs before training starts)
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch

        # Track that validation ran at this epoch
        self.validation_epochs.add(epoch)

        # Find the epoch_metrics entry for this epoch
        epoch_entry = None
        for entry in self.epoch_metrics:
            if entry["epoch"] == epoch:
                epoch_entry = entry
                break

        # If entry doesn't exist, create it (defensive handling for callback order)
        if epoch_entry is None:
            epoch_entry = {"epoch": epoch}
            self.epoch_metrics.append(epoch_entry)

        # Update with validation metrics
        val_metrics = {
            "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
            "val_spearman": float(trainer.callback_metrics.get("val_spearman", 0.0)),
            "val_pearson": float(trainer.callback_metrics.get("val_pearson", 0.0)),
            "val_rmse": float(trainer.callback_metrics.get("val_rmse", 0.0)),
            "val_mae": float(trainer.callback_metrics.get("val_mae", 0.0)),
        }
        epoch_entry.update(val_metrics)

        # Verify metrics were actually collected
        val_metrics_in_callback = {
            k: float(v) if hasattr(v, 'item') else float(v)
            for k, v in trainer.callback_metrics.items()
            if k.startswith('val_')
        }

        if not val_metrics_in_callback:
            # No validation metrics at all - model's on_validation_epoch_end returned early
            logger.info(
                f"Fold {self.fold_idx}, Epoch {epoch}: Validation callback triggered but no metrics found. "
                f"Model's on_validation_epoch_end should have logged why (empty validation_step_outputs). "
                f"Skipping this epoch - not tracking as a validation epoch."
            )
            # Don't store this epoch's metrics since validation didn't actually run
            if epoch_entry in self.epoch_metrics:
                self.epoch_metrics.remove(epoch_entry)
            return

        # Log successful metric collection with values for verification
        logger.debug(
            f"Fold {self.fold_idx}, Epoch {epoch}: Collected validation metrics: "
            f"loss={val_metrics_in_callback.get('val_loss', 0.0):.4f}, "
            f"spearman={val_metrics_in_callback.get('val_spearman', 0.0):.4f}"
        )

        # Track that validation actually ran at this epoch (with metrics)
        self.validation_epochs.add(epoch)

        # Update epoch_entry with validation metrics
        epoch_entry.update(val_metrics_in_callback)

        # Log validation metrics to WandB run with fold prefix
        if not trainer.sanity_checking:
            metrics_to_log = {
                "epoch": epoch,
                f"fold_{self.fold_idx}/val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
                f"fold_{self.fold_idx}/val_spearman": float(trainer.callback_metrics.get("val_spearman", 0.0)),
                f"fold_{self.fold_idx}/val_pearson": float(trainer.callback_metrics.get("val_pearson", 0.0)),
                f"fold_{self.fold_idx}/val_rmse": float(trainer.callback_metrics.get("val_rmse", 0.0)),
                f"fold_{self.fold_idx}/val_mae": float(trainer.callback_metrics.get("val_mae", 0.0)),
            }

            if self.wandb_run is not None:
                self.wandb_run.log(metrics_to_log)


def run_single_config(
    config: dict,
    property_name: str,
    phase: str,
    # Architecture parameters
    hidden_dim: int,
    n_layers: int,
    n_output_layers: int,
    n_heads: int,
    dropout: float,
    pooling_strategy: str,
    use_learnable_chain_fusion: str,
    attention_strategy: str,
    use_output_norm: bool,
    # Training parameters
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    max_epochs: int,
    gradient_clip_val: float,
    # SWA parameters
    use_swa: bool,
    swa_epoch_start: float,
    swa_lr_factor: float,
    # Data/Loss parameters
    target_transform: str,
    loss: str,
    use_xavier_init: bool,
    xavier_gain: float,
    # Phase-specific parameters
    encoder_type: str,
    sequence_representation: str,
    decoder_type: str,
    n_folds: int,
    use_antibody_features: bool,
    normalize_antibody_features: bool,
    antibody_features_projection_dim: int,
):
    """Run training for a single configuration across CV folds."""

    # Parse encoder_type: can be single "esmc_600m" or multi "esmc_6b+prott5"
    encoder_types = [e.strip() for e in encoder_type.split('+')]
    is_multi_encoder = len(encoder_types) > 1

    # Get sequence representation: use sweep parameter if provided, otherwise read from merged config
    if sequence_representation is not None:
        use_aho_aligned = (sequence_representation == "aho_aligned")
        use_full_chain = (sequence_representation == "full_chain")
    else:
        # Correctly look in the 'data' section of the merged config
        use_full_chain = config.get("data", {}).get("use_full_chain", False)
        use_aho_aligned = config.get("data", {}).get("use_aho_aligned", False)

    seq_rep = "aho_aligned" if use_aho_aligned else ("full_chain" if use_full_chain else "normal")
    seq_rep_source = "sweep param" if sequence_representation is not None else "config"

    logger.info(f"\n{'='*80}")
    logger.info(f"Phase {phase} Hyperparameter Tuning - Property: {property_name}")
    logger.info(f"ENCODER(S): {' + '.join(encoder_types)} {'(multi-encoder)' if is_multi_encoder else '(single)'}")
    logger.info(f"SEQUENCE: {seq_rep} (from {seq_rep_source})")
    logger.info(f"ANTIBODY FEATURES: {'ENABLED (33-dim)' if use_antibody_features else 'DISABLED'}")
    logger.info(f"Architecture: hidden_dim={hidden_dim}, n_layers={n_layers}, "
                f"n_output_layers={n_output_layers}, dropout={dropout}")
    logger.info(f"Pooling: {pooling_strategy}, Chain Fusion: {use_learnable_chain_fusion}, "
                f"Strategy: {attention_strategy}, Output Norm: {use_output_norm}")
    logger.info(f"Training: LR={learning_rate}, BS={batch_size}, WD={weight_decay}, "
                f"Max Epochs={max_epochs}, Grad Clip={gradient_clip_val}")
    logger.info(f"SWA: use={use_swa}, start={swa_epoch_start}, lr_factor={swa_lr_factor}")
    logger.info(f"Data: Transform={target_transform}, Loss={loss}")
    logger.info(f"{'='*80}")

    # Build encoder config
    encoder_config = config["encoder"].copy()
    encoder_config["encoder_types"] = encoder_types
    encoder_config["freeze_encoder"] = True
    encoder_config["pooling"] = "none"  # Decoder handles pooling

    # Auto-detect precomputed embeddings for each encoder
    source_csv = config["data"]["gdpa1_path"]
    for enc_type in encoder_types:
        embeddings_config = get_embeddings_config(
            source_csv,
            encoder_type=enc_type,
            use_aho_aligned=use_aho_aligned,
            use_full_chain=use_full_chain
        )
        if embeddings_config:
            if enc_type not in encoder_config["encoder_configs"]:
                encoder_config["encoder_configs"][enc_type] = {}
            encoder_config["encoder_configs"][enc_type].update(embeddings_config)
            logger.info(f"✓ Using precomputed {enc_type} embeddings")

    # Configure antibody features if enabled
    antibody_features_config = None
    if use_antibody_features:
        antibody_features_config = config["antibody_features"].copy()
        antibody_features_config["enabled"] = True
        logger.info("✓ Antibody features enabled (33-dim)")

    # Build decoder config (attention decoder with all tunable parameters)
    decoder_config = {
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_output_layers": n_output_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "attention_strategy": attention_strategy,
        "pooling_strategy": pooling_strategy,
        "use_output_norm": use_output_norm,
        "use_learnable_chain_fusion": use_learnable_chain_fusion,
        "output_activation": "none",  # Fixed for Phase 3
    }

    # Add antibody features configuration if enabled
    if use_antibody_features:
        from src.features.antibody_features import AntibodyFeatures
        ab_features = AntibodyFeatures(**{k: v for k, v in antibody_features_config.items() if k not in ["enabled", "normalize_antibody_features", "projection_dim"]})
        antibody_features_dim = ab_features.get_feature_dim()
        decoder_config["antibody_features_dim"] = antibody_features_dim
        decoder_config["antibody_features_normalized"] = normalize_antibody_features
        decoder_config["antibody_features_projection_dim"] = antibody_features_projection_dim

        injection_point = "after pooling" if normalize_antibody_features else "after first FFN"
        proj_info = f"proj={antibody_features_projection_dim}" if antibody_features_projection_dim else "no projection"
        logger.info(
            f"✓ Decoder configured for {antibody_features_dim}-dim antibody features "
            f"({proj_info}, normalized={normalize_antibody_features}, inject {injection_point})"
        )

    logger.info(f"Decoder config: {decoder_config}")

    # Cross-validation results
    fold_results = []
    all_epoch_metrics = []  # Store per-epoch metrics from each fold
    all_validation_epochs = []  # Store validation epochs from each fold

    # Run name components
    encoder_name = '+'.join(encoder_types)
    seq_rep_suffix = f"_{seq_rep}" if seq_rep != "normal" else ""
    ab_features_suffix = "_abfeat" if use_antibody_features else ""
    run_name_base = f"P{phase}_{property_name}_{encoder_name}{seq_rep_suffix}{ab_features_suffix}"

    # Create single WandbLogger for all folds
    wandb_logger = WandbLogger(
        project=config["logging"]["wandb_project"],
        name=run_name_base
    )

    # Define custom step metrics to avoid conflicts with global step counter
    wandb_logger.experiment.define_metric("epoch")
    wandb_logger.experiment.define_metric("fold*", step_metric="epoch")
    wandb_logger.experiment.define_metric("cv*", step_metric="epoch")

    logger.info(f"Created WandB logger: {run_name_base}")

    for fold in range(n_folds):
        # Set seed for reproducibility (same seed for all folds for consistency)
        seed = 42
        pl.seed_everything(seed, workers=True)
        logger.info(f"\n\n{'='*80}\nFold {fold+1}/{n_folds}, Seed: {seed}")

        # Data module with target transform
        data_module = GDPa1DataModule(
            data_path=config["data"]["gdpa1_path"],
            target_property=property_name,
            batch_size=batch_size,
            fold_idx=fold,
            cv_fold_col=config["data"]["cv_fold_column"],
            target_transform=target_transform,
            use_aho_aligned=use_aho_aligned,
            use_full_chain=use_full_chain,
            antibody_features_config=antibody_features_config,
            normalize_antibody_features=normalize_antibody_features,
        )

        # Model
        freeze_encoder = encoder_config.pop("freeze_encoder", True)

        model = DevelopabilityModel(
            encoder_config=encoder_config,
            decoder_type=decoder_type,
            decoder_config=decoder_config,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler=config["training"]["finetune"]["scheduler"],
            warmup_epochs=2,
            max_epochs=max_epochs,
            loss_fn=loss,
            freeze_encoder=freeze_encoder,
            target_transform=data_module.target_transform,
            use_xavier_init=use_xavier_init,
            xavier_gain=xavier_gain,
        )

        # Run name for this fold
        run_name = f"{run_name_base}_F{fold+1}"

        # Checkpoint directory using phase in path
        checkpoint_dir = f"models/tuning/{phase}/{run_name}"

        # Check for existing checkpoint to resume from
        checkpoint_path = Path(checkpoint_dir) / "last.ckpt"
        resume_from_checkpoint = str(checkpoint_path) if checkpoint_path.exists() else None

        if resume_from_checkpoint:
            logger.info(f"✓ Resuming from checkpoint: {resume_from_checkpoint}")

        # Callbacks (pass fold index and wandb logger experiment for live logging)
        metrics_collector = MetricsCollectorCallback(fold_idx=fold, wandb_run=wandb_logger.experiment)
        callbacks = [
            EarlyStoppingWithLogging(
                monitor="val_spearman",
                patience=config["training"]["finetune"]["early_stopping_patience"],
                mode="max"
            ),
            GradientMonitor(log_frequency=50),
            metrics_collector,
        ]

        # Add SWA if enabled
        if use_swa:
            swa_epoch_start_abs = int(swa_epoch_start * max_epochs)
            swa_lrs = learning_rate * swa_lr_factor
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=swa_lrs,
                    swa_epoch_start=swa_epoch_start_abs,
                    annealing_epochs=int(max_epochs * 0.1),  # 10% annealing
                    annealing_strategy="cos",
                )
            )
            logger.info(f"✓ SWA enabled: start_epoch={swa_epoch_start_abs}, lr={swa_lrs:.2e}")

        # Trainer (use common wandb_logger for all folds)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=wandb_logger,  # Single logger for all folds
            accelerator="auto",
            devices=1,
            gradient_clip_val=gradient_clip_val,
            check_val_every_n_epoch=VALIDATION_EPOCH_FREQUENCY,
            log_every_n_steps=10,
            precision="16-mixed",
            enable_checkpointing=False,
            deterministic=True,
        )

        # Train (with checkpoint resumption if available)
        trainer.fit(model, data_module, ckpt_path=resume_from_checkpoint)

        # Get last validation metrics (not best - no checkpoint restoration)
        last_spearman = trainer.callback_metrics.get("val_spearman", 0.0).item()
        last_pearson = trainer.callback_metrics.get("val_pearson", 0.0).item()

        fold_results.append({
            "fold": fold + 1,
            "val_spearman": last_spearman,
            "val_pearson": last_pearson,
        })

        # Collect per-epoch metrics and validation epochs from this fold
        if metrics_collector.epoch_metrics:
            all_epoch_metrics.append(metrics_collector.epoch_metrics)
            all_validation_epochs.append(metrics_collector.validation_epochs)
            logger.debug(f"Collected {len(metrics_collector.epoch_metrics)} epoch metrics from fold {fold+1}, "
                        f"validation ran at {len(metrics_collector.validation_epochs)} epochs")

        if fold == n_folds - 1:
            # On last fold, log final CV summary metrics and per-epoch aggregates
            import numpy as np
            spearman_scores = [r["val_spearman"] for r in fold_results]
            pearson_scores = [r["val_pearson"] for r in fold_results]

            # Log final summary metrics (for sweep optimization)
            wandb_logger.experiment.log({
                "cv_val_spearman": float(np.mean(spearman_scores)),  # For sweep optimization
                "cv_val_spearman_mean": float(np.mean(spearman_scores)),
                "cv_val_spearman_std": float(np.std(spearman_scores)),
                "cv_val_pearson": float(np.mean(pearson_scores)),  # For sweep optimization
                "cv_val_pearson_mean": float(np.mean(pearson_scores)),
                "cv_val_pearson_std": float(np.std(pearson_scores)),
            })

            # Aggregate per-epoch metrics across folds if available
            if all_epoch_metrics and len(all_epoch_metrics) == n_folds:
                logger.info(f"Aggregating per-epoch CV metrics from {len(all_epoch_metrics)} folds")

                # Collect all validation epochs across all folds
                # Note: validation_epochs from callbacks may be incomplete due to checkpoint resumption
                # So we also reconstruct from actual metrics
                all_val_epochs = set()

                # First, collect from callback tracking
                for val_epochs in all_validation_epochs:
                    all_val_epochs.update(val_epochs)

                # Then, reconstruct from actual metrics (more reliable than callback tracking)
                for fold_idx, fold_metrics in enumerate(all_epoch_metrics):
                    fold_val_epochs = set()
                    for epoch_entry in fold_metrics:
                        # If this epoch has validation metrics, it was a validation epoch
                        if "val_loss" in epoch_entry or "val_spearman" in epoch_entry:
                            fold_val_epochs.add(epoch_entry["epoch"])
                            all_val_epochs.add(epoch_entry["epoch"])
                    logger.debug(f"Fold {fold_idx+1}: {len(fold_val_epochs)} validation epochs, "
                                f"last epoch: {max([e['epoch'] for e in fold_metrics]) if fold_metrics else 'N/A'}")

                sorted_val_epochs = sorted(all_val_epochs)
                logger.debug(f"Validation ran at {len(sorted_val_epochs)} epochs: {sorted_val_epochs}")

                # Check if validation epochs make sense given the frequency
                if sorted_val_epochs:
                    min_val_epoch = min(sorted_val_epochs)
                    max_val_epoch = max(sorted_val_epochs)
                    expected_count = (max_val_epoch - min_val_epoch) // VALIDATION_EPOCH_FREQUENCY + 1

                    if len(sorted_val_epochs) < expected_count * 0.5:  # Less than 50% of expected
                        logger.warning(
                            f"⚠️  Validation epochs seem incomplete: got {len(sorted_val_epochs)} epochs "
                            f"but expected ~{expected_count} between epochs {min_val_epoch}-{max_val_epoch}. "
                            f"This may be due to checkpoint resumption. Validation metrics will only be "
                            f"aggregated for epochs where data was actually collected."
                        )

                    # If validation starts late (after epoch 10), likely resumed from checkpoint
                    if min_val_epoch > 10:
                        logger.warning(
                            f"⚠️  First validation epoch is {min_val_epoch}. Early validation epochs "
                            f"(0-{min_val_epoch-1}) were not captured, possibly due to checkpoint resumption. "
                            f"CV metrics will only cover epochs {min_val_epoch} onwards."
                        )

                # Forward-fill metrics for each fold up to max_epochs (for early stopping cases)
                filled_epoch_metrics = []
                for fold_idx, fold_metrics in enumerate(all_epoch_metrics):
                    filled_fold = []
                    last_metrics = None

                    for epoch in range(max_epochs):
                        # Try to find metric for this epoch
                        epoch_metric = next(
                            (m for m in fold_metrics if m["epoch"] == epoch),
                            None
                        )

                        if epoch_metric:
                            # Use actual metric
                            filled_fold.append(epoch_metric)
                            last_metrics = epoch_metric
                        elif last_metrics:
                            # Forward-fill with last known metrics
                            filled_fold.append({
                                "epoch": epoch,
                                **{k: v for k, v in last_metrics.items() if k != "epoch"}
                            })

                    filled_epoch_metrics.append(filled_fold)

                # Aggregate metrics for each epoch and log to WandB
                for epoch in range(max_epochs):
                    epoch_data = {"epoch": epoch}

                    # Training metrics: available every epoch
                    train_values = {"train_loss": [], "train_spearman": []}
                    for fold_metrics in filled_epoch_metrics:
                        if epoch < len(fold_metrics):
                            epoch_metric = fold_metrics[epoch]
                            for metric_name in ["train_loss", "train_spearman"]:
                                if metric_name in epoch_metric:
                                    train_values[metric_name].append(epoch_metric[metric_name])

                    # Add training metrics to epoch_data
                    for metric_name, values in train_values.items():
                        if values:
                            if len(values) != n_folds:
                                logger.warning(f"Epoch {epoch}, {metric_name}: Expected {n_folds} values, got {len(values)}")
                            epoch_data[f"cv/{metric_name}"] = float(np.mean(values))

                    # Validation metrics: only check on epochs where validation actually ran
                    if epoch in all_val_epochs:
                        val_values = {"val_loss": [], "val_spearman": [], "val_pearson": [], "val_rmse": [], "val_mae": []}
                        for fold_metrics in filled_epoch_metrics:
                            if epoch < len(fold_metrics):
                                epoch_metric = fold_metrics[epoch]
                                for metric_name in ["val_loss", "val_spearman", "val_pearson", "val_rmse", "val_mae"]:
                                    if metric_name in epoch_metric:
                                        val_values[metric_name].append(epoch_metric[metric_name])

                        # Add validation metrics to epoch_data if available
                        for metric_name, values in val_values.items():
                            if values:
                                if len(values) != n_folds:
                                    logger.info(f"Epoch {epoch}, {metric_name}: {len(values)}/{n_folds} folds reported")
                                epoch_data[f"cv/{metric_name}"] = float(np.mean(values))

                    # Log all metrics for this epoch in a single call
                    if len(epoch_data) > 1:  # More than just "epoch" key
                        wandb_logger.experiment.log(epoch_data)

                logger.info(f"Logged per-epoch CV metrics for {max_epochs} epochs to WandB run")

    # Aggregate results
    import numpy as np
    spearman_scores = [r["val_spearman"] for r in fold_results]
    pearson_scores = [r["val_pearson"] for r in fold_results]

    return {
        "cv_spearman_mean": float(np.mean(spearman_scores)),
        "cv_spearman_std": float(np.std(spearman_scores)),
        "cv_pearson_mean": float(np.mean(pearson_scores)),
        "cv_pearson_std": float(np.std(pearson_scores)),
        "fold_results": fold_results,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    """Main function for WandB sweep execution - runs SINGLE configuration."""
    import wandb

    parser = argparse.ArgumentParser(
        description="Generic Hyperparameter Tuning (WandB Agent)"
    )
    parser.add_argument("--config", type=str, default="src/config/default_config.yaml")
    parser.add_argument("--phase", type=str, required=True,
                       help="Phase identifier (e.g., '4.1', '4.2')")

    # Property
    parser.add_argument("--property", type=str, required=True,
                       choices=["HIC", "PR_CHO", "AC-SINS_pH7.4", "Tm2", "Titer"])
    parser.add_argument("--n_folds", type=int, default=None)

    # Phase-specific parameters (all optional - will load from config if not provided)
    parser.add_argument("--encoder_type", type=str, default=None,
                       help="Single encoder (e.g., 'esmc_600m') or multi-encoder (e.g., 'esmc_6b+prott5')")
    parser.add_argument("--sequence_representation", type=str, default=None,
                       choices=["normal", "aho_aligned", "full_chain"],
                       help="Optional: override sequence representation from config")
    parser.add_argument("--use_antibody_features", type=lambda x: x.lower() == 'true', default=None,
                       help="Enable antibody features (33-dim engineered features)")
    parser.add_argument("--normalize_antibody_features", type=lambda x: x.lower() == 'true', default=None,
                       help="Z-score normalize antibody features from training set")
    parser.add_argument("--antibody_features_projection_dim", type=nullable_int, default=None,
                       help="Project antibody features to this dimension (None = no projection)")
    parser.add_argument("--decoder_type", type=str, default=None)

    # Architecture parameters (optional - will load from config if not provided)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_output_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--pooling_strategy", type=str, default=None,
                       choices=["mean", "attention"])
    parser.add_argument("--use_learnable_chain_fusion", type=str, default=None,
                       choices=["none", "per_chain", "per_dim"])
    parser.add_argument("--attention_strategy", type=str, default=None,
                       choices=["self_only", "self_cross", "bidirectional_cross"])
    parser.add_argument("--use_output_norm", type=lambda x: x.lower() == 'true', default=None)

    # Training parameters (optional - will load from config if not provided)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=None)

    # SWA parameters (optional - will load from config if not provided)
    parser.add_argument("--use_swa", type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument("--swa_epoch_start", type=float, default=None)
    parser.add_argument("--swa_lr_factor", type=float, default=None)

    # Data/Loss parameters (optional - will load from config if not provided)
    parser.add_argument("--target_transform", type=str, default=None,
                       choices=["none", "log", "log_zscore", "z_score", "min_max", "robust"])
    parser.add_argument("--loss", type=str, default=None,
                       choices=["mse", "mae", "huber", "smooth_l1"])

    # Weight initialization parameters
    parser.add_argument("--use_xavier_init", type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument("--xavier_gain", type=float, default=None)

    # Dependency checking (disabled by default)
    parser.add_argument("--enable-dependency-check", action="store_true", default=False,
                       help="Enable parameter dependency checking to skip invalid combinations (e.g., normalize_antibody_features=true when use_antibody_features=false)")

    # Use parse_known_args to allow unknown arguments
    args, unknown_args = parser.parse_known_args()

    # Check parameter dependencies and skip invalid combinations (only if enabled)
    invalid_params = []
    if args.enable_dependency_check:
        for param_name, dependencies in PARAMETER_DEPENDENCIES.items():
            # Check if all dependencies are met
            dependencies_met = all(
                getattr(args, dep_param, None) == dep_value
                for dep_param, dep_value in dependencies.items()
            )

            if not dependencies_met:
                # Dependencies not met - check if parameter has non-default value
                current_value = getattr(args, param_name, None)
                default_value = PARAMETER_DEFAULTS.get(param_name)

                if current_value != default_value:
                    # Non-default value with unmet dependency = invalid combination
                    invalid_params.append({
                        'param': param_name,
                        'value': current_value,
                        'default': default_value,
                        'dependencies': dependencies
                    })

    # Skip run if any invalid parameter combinations detected
    if invalid_params:
        logger.info("=" * 80)
        logger.info("SKIPPING INVALID PARAMETER COMBINATION")
        logger.info("=" * 80)

        for inv in invalid_params:
            logger.info(f"Parameter '{inv['param']}' = {inv['value']} (default: {inv['default']})")
            logger.info(f"  Dependencies not met: {inv['dependencies']}")
            logger.info(f"  Current values:")
            for dep_param, dep_value in inv['dependencies'].items():
                current = getattr(args, dep_param, None)
                logger.info(f"    {dep_param} = {current} (required: {dep_value})")

        logger.info("")
        logger.info("This combination is invalid - only default values allowed when dependencies not met")

        # If in wandb sweep, mark run as skipped
        if wandb.run is not None:
            wandb.log({"skipped": True, "reason": "invalid_parameter_combination"})
            wandb.finish()

        return  # Exit early

    # ========================================================================
    # AUTOMATIC PARAMETER RESOLUTION & RESEARCH INTEGRITY CHECK
    # ========================================================================
    # This mapping defines every modeling parameter and its path in default_config.yaml
    # Any parameter not in this map or not in sweep/config will trigger an error.
    # ========================================================================
    PARAM_MAP = {
        # Architecture
        'hidden_dim': 'decoder.attention.hidden_dim',
        'n_layers': 'decoder.attention.n_layers',
        'n_output_layers': 'decoder.attention.n_output_layers',
        'n_heads': 'decoder.attention.n_heads',
        'dropout': 'decoder.attention.dropout',
        'pooling_strategy': 'decoder.attention.pooling_strategy',
        'use_learnable_chain_fusion': 'decoder.attention.use_learnable_chain_fusion',
        'attention_strategy': 'decoder.attention.attention_strategy',
        'use_output_norm': 'decoder.attention.use_output_norm',
        
        # Training
        'learning_rate': 'training.finetune.learning_rate',
        'batch_size': 'training.finetune.batch_size',
        'weight_decay': 'training.finetune.weight_decay',
        'max_epochs': 'training.finetune.max_epochs',
        'gradient_clip_val': 'training.gradient_clip_val',
        'use_swa': 'training.use_swa',
        'swa_epoch_start': 'training.swa_epoch_start',
        'swa_lr_factor': 'training.swa_lr_factor',
        'use_xavier_init': 'training.use_xavier_init',
        'xavier_gain': 'training.xavier_gain',
        
        # Data/Loss
        'target_transform': 'training.target_transform',
        'loss': 'training.loss',
        
        # Antibody features
        'use_antibody_features': 'antibody_features.enabled',
        'normalize_antibody_features': 'antibody_features.normalize_antibody_features',
        'antibody_features_projection_dim': 'antibody_features.projection_dim',
        
        # Core Components
        'encoder_type': 'encoder_type',
        'decoder_type': 'decoder.type',
        'sequence_representation': 'data.use_full_chain',  # Special handling in resolution
        'n_folds': 'data.n_folds',
    }

    # Load config and merge with property-specific
    base_config = load_config(args.config)
    config = get_property_config(base_config, args.property)

    # 1. Identify which arguments were explicitly passed via command line (swept)
    # Use the parser's own knowledge of its arguments for robust detection
    passed_args = set()
    
    # Check all arguments defined in the parser
    for action in parser._actions:
        # Check if any of the option strings (e.g., '--dropout') appear in sys.argv
        # We check both '--name' and '--name=' variants
        for opt in action.option_strings:
            if any(arg == opt or arg.startswith(f"{opt}=") for arg in sys.argv):
                passed_args.add(action.dest)
                break
    
    # Also check unknown_args for any other --params
    for arg in unknown_args:
        if arg.startswith('--'):
            name = arg[2:].split('=')[0].replace('-', '_')
            passed_args.add(name)

    # 2. Resolve every parameter and verify integrity
    sweep_params = {}
    config_params = {}
    integrity_errors = []
    
    logger.info("")
    logger.info("┌" + "─"*78 + "┐")
    logger.info(f"│ {'RESEARCH INTEGRITY VERIFICATION':^76} │")
    logger.info(f"│ {'Phase: ' + args.phase + ' | Property: ' + args.property:^76} │")
    logger.info("├" + "─"*40 + "┬" + "─"*15 + "┬" + "─"*21 + "┤")
    logger.info(f"│ {'Parameter':<38} │ {'Status':^13} │ {'Value':^19} │")
    logger.info("├" + "─"*40 + "┼" + "─"*15 + "┼" + "─"*21 + "┤")

    for arg_name, config_path in PARAM_MAP.items():
        # Get value from loaded config
        val = config
        for key in config_path.split('.'):
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                val = None
                break
        
        default_val = val
        current_val = getattr(args, arg_name)
        
        if arg_name in passed_args:
            # Parameter is being swept
            sweep_params[arg_name] = current_val
            status = "SWEPT"
            display_val = str(current_val)
            if len(display_val) > 17: display_val = display_val[:14] + "..."
            logger.info(f"│ {arg_name:<38} │ {status:^13} │ {display_val:^19} │")
        else:
            # Parameter is NOT in sweep - MUST match default config
            if default_val is None and arg_name != 'antibody_features_projection_dim':
                integrity_errors.append(f"MISSING: '{arg_name}' not in sweep AND not in config path '{config_path}'")
                status = "!! MISSING !!"
            else:
                # If current value is None (from argparse), populate it from config
                if current_val is None:
                    setattr(args, arg_name, default_val)
                    config_params[arg_name] = default_val
                    status = "FIXED"
                else:
                    # If it's already set but not in sweep, it must match default
                    if current_val != default_val:
                        integrity_errors.append(
                            f"MISMATCH: '{arg_name}' has value '{current_val}' but default is '{default_val}'. "
                            f"It was NOT passed in sweep, so it must match default config!"
                        )
                        status = "!! MISMATCH !!"
                    else:
                        status = "FIXED"
            
            display_val = str(getattr(args, arg_name))
            if len(display_val) > 17: display_val = display_val[:14] + "..."
            logger.info(f"│ {arg_name:<38} │ {status:^13} │ {display_val:^19} │")

    logger.info("└" + "─"*40 + "┴" + "─"*15 + "┴" + "─"*21 + "┘")

    # 3. Final Validation
    if integrity_errors:
        logger.error("\n" + "!"*80)
        logger.error(f"{'CRITICAL RESEARCH INTEGRITY ERROR':^80}")
        logger.error("!"*80)
        logger.error("The experimental controlled state has been compromised:")
        for err in integrity_errors:
            logger.error(f"  - {err}")
        logger.error("-" * 80)
        logger.error("All parameters NOT explicitly in sweep must match default_config.yaml.")
        logger.error("!"*80 + "\n")
        raise ValueError("Research Flow Violation: Non-swept parameters deviate from default config.")
    
    logger.info(f"✓ INTEGRITY CHECK PASSED: {len(sweep_params)} swept, {len(config_params)} pinned to defaults.")
    logger.info("")

    # Run training with this configuration (detailed logging happens in run_single_config)
    try:
        result = run_single_config(
            config=config,
            property_name=args.property,
            phase=args.phase,
            # Architecture
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_output_layers=args.n_output_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            pooling_strategy=args.pooling_strategy,
            use_learnable_chain_fusion=args.use_learnable_chain_fusion,
            attention_strategy=args.attention_strategy,
            use_output_norm=args.use_output_norm,
            # Training
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            gradient_clip_val=args.gradient_clip_val,
            # SWA
            use_swa=args.use_swa,
            swa_epoch_start=args.swa_epoch_start,
            swa_lr_factor=args.swa_lr_factor,
            # Data/Loss
            target_transform=args.target_transform,
            loss=args.loss,
            use_xavier_init=args.use_xavier_init,
            xavier_gain=args.xavier_gain,
            # Phase-specific
            encoder_type=args.encoder_type,
            sequence_representation=args.sequence_representation,
            decoder_type=args.decoder_type,
            n_folds=args.n_folds,
            use_antibody_features=args.use_antibody_features,
            normalize_antibody_features=args.normalize_antibody_features,
            antibody_features_projection_dim=args.antibody_features_projection_dim,
        )

        # Log final metrics to WandB sweep run if available
        if wandb.run is not None:
            wandb.log({
                "cv_val_spearman": result["cv_spearman_mean"],  # Main metric for sweep optimization
                "cv_val_spearman_std": result["cv_spearman_std"],
                "cv_val_pearson": result["cv_pearson_mean"],
                "cv_val_pearson_std": result["cv_pearson_std"],
            })

        logger.info(f"\n✓ Completed!")
        logger.info(f"  Spearman: {result['cv_spearman_mean']:.4f} ± {result['cv_spearman_std']:.4f}")
        logger.info(f"  Pearson: {result['cv_pearson_mean']:.4f} ± {result['cv_pearson_std']:.4f}")

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
