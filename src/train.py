#!/usr/bin/env python3
"""
CrossAbSense: Antibody Developability Prediction
Training script for antibody developability prediction.

Models are saved under: models/<property>_<config_name>_<config_checksum>/

Usage:
    python -m src.train --property HIC [--fold all|0-4|final]
    python -m src.train --all

Examples:
    # Full CV + final model (default)
    python -m src.train --property HIC
    python -m src.train --property HIC --fold all

    # Single fold (quick check / smoke test)
    python -m src.train --property HIC --fold 0

    # Final model only (train on all data)
    python -m src.train --property HIC --fold final

    # Train all 5 properties sequentially
    python -m src.train --all

    # Custom config
    python -m src.train --property Tm2 --config experiments/my_config.yaml
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger

from src.data import GDPa1DataModule
from src.models import DevelopabilityModel
from src.callbacks import GradientMonitor
from src.utils import (
    load_config,
    get_embeddings_config,
    get_property_config,
    setup_logger,
    compute_file_checksum,
)

# Set matmul precision for better performance
torch.set_float32_matmul_precision('medium')

# Setup logger
logger = setup_logger("train", log_file="train.log")


class MetricsCollectorCallback(pl.Callback):
    """Callback to collect validation metrics at each epoch for CV aggregation."""
    
    def __init__(self):
        super().__init__()
        self.epoch_metrics = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Collect metrics after each validation epoch."""
        # Get metrics from trainer and convert tensors to floats
        metrics = {
            "epoch": trainer.current_epoch,
            "val_spearman": float(trainer.callback_metrics.get("val_spearman", 0.0)),
            "val_pearson": float(trainer.callback_metrics.get("val_pearson", 0.0)),
            "val_rmse": float(trainer.callback_metrics.get("val_rmse", 0.0)),
            "val_mae": float(trainer.callback_metrics.get("val_mae", 0.0)),
            "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
        }
        self.epoch_metrics.append(metrics)


class CVMetricsLoggerCallback(pl.Callback):
    """Callback to log CV metrics during final model training."""
    
    def __init__(self, cv_metrics):
        super().__init__()
        self.cv_metrics = cv_metrics  # List of epoch_metrics dicts
        self.max_cv_epochs = len(cv_metrics)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log corresponding CV metrics after each training epoch."""
        current_epoch = trainer.current_epoch
        # Log CV metrics for this epoch if available
        if current_epoch < self.max_cv_epochs:
            cv_data = self.cv_metrics[current_epoch].copy()
            # KEEP epoch in the data - it's needed for plotting!
            # Log to WandB
            if trainer.logger:
                trainer.logger.log_metrics(cv_data, step=trainer.global_step)


class BestWeightsCallback(pl.Callback):
    """Store best model weights in memory without disk I/O."""
    
    def __init__(self, monitor='val_loss', mode='min'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.best_metrics = {}
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Check if current model is best and store weights in memory."""
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
        
        try:
            current_score = float(current)
        except (TypeError, ValueError):
            return

        is_better = (
            (self.mode == 'min' and current_score < self.best_score) or
            (self.mode == 'max' and current_score > self.best_score)
        )
        
        if is_better:
            self.best_score = current_score
            # Store model state on CPU to save GPU memory and avoid potential deadlocks during deepcopy
            self.best_weights = {k: v.cpu().clone() for k, v in pl_module.state_dict().items()}
            
            # Capture all relevant validation metrics at this best point
            self.best_metrics = {
                "val_spearman": float(trainer.callback_metrics.get("val_spearman", 0.0)),
                "val_pearson": float(trainer.callback_metrics.get("val_pearson", 0.0)),
                "val_r2": float(trainer.callback_metrics.get("val_r2", 0.0)),
                "val_rmse": float(trainer.callback_metrics.get("val_rmse", 0.0)),
                "val_mae": float(trainer.callback_metrics.get("val_mae", 0.0)),
                "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
                "best_epoch": trainer.current_epoch
            }
    
    def restore_best_weights(self, pl_module):
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            logger.info(f"Restoring best weights from epoch {self.best_metrics.get('best_epoch', 'unknown')}")
            pl_module.load_state_dict(self.best_weights)
        else:
            logger.warning("No best weights found in memory to restore!")


def train_single_fold(
    base_config: dict,
    property_name: str,
    fold_idx: int,
    model_dir: Path,
    resume_checkpoint: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """Train a single fold (or final model on all data).

    Args:
        base_config: Base configuration dict (will be merged with property-specific)
        property_name: Target property (e.g., "HIC", "Titer")
        fold_idx: Fold index (0-4) or None for final model on all data
        model_dir: Model directory (e.g., Path("models/HIC_abc123"))
        resume_checkpoint: Path to checkpoint to resume from (optional)
        debug: Enable gradient monitoring and detailed logging (optional)

    Returns:
        tuple: (dict of validation metrics, wandb_run_id)
    """
    logger.info(f"\n{'='*60}")
    if fold_idx is None:
        logger.info(f"Training {property_name}, Final Model (all data)")
    else:
        logger.info(f"Training {property_name}, Fold {fold_idx+1}/5")
    logger.info(f"{'='*60}")

    # Merge base config with property-specific overrides
    config = get_property_config(base_config, property_name)

    # Set seed for reproducibility (same seed for all folds, matching tune_hyperparam.py)
    seed = config.get("random_seed", 42)
    pl.seed_everything(seed, workers=True)
    logger.info(f"Random seed: {seed}")
    
    # Build encoder config (already merged with property-specific)
    encoder_config = config["encoder"].copy()

    # Get encoder_type (required, already merged from property_specific)
    encoder_type = config["encoder_type"]  # This is now at top level after merge
    encoder_types = [e.strip() for e in encoder_type.split('+')]
    encoder_type_str = "+".join(encoder_types)

    # Set encoder_types in encoder_config (required by DevelopabilityModel)
    encoder_config["encoder_types"] = encoder_types
    encoder_config["freeze_encoder"] = encoder_config.get("freeze_encoder", True)
    encoder_config["pooling"] = "none"  # Decoder handles pooling

    # Auto-detect precomputed embeddings for each encoder
    source_csv = config["data"]["gdpa1_path"]
    use_full_chain = config["data"].get("use_full_chain", False)
    use_aho_aligned = config["data"].get("use_aho_aligned", False)
    
    for enc_type in encoder_types:
        embeddings_cfg = get_embeddings_config(
            source_csv, 
            encoder_type=enc_type,
            use_full_chain=use_full_chain,
            use_aho_aligned=use_aho_aligned
        )
        if embeddings_cfg:
            encoder_config["encoder_configs"][enc_type].update(embeddings_cfg)
            logger.info(f"✓ Using precomputed {enc_type} embeddings")

    # Build decoder config (already merged with property-specific)
    decoder_type = config["decoder"]["type"]
    decoder_config = config["decoder"].get(decoder_type, {}).copy()

    # Auto-configure antibody features if enabled
    if config.get("antibody_features", {}).get("enabled", False):
        from src.features.antibody_features import AntibodyFeatures
        ab_cfg = config["antibody_features"]
        
        # Only pass arguments that AntibodyFeatures.__init__ accepts
        valid_args = [
            "use_abnumber", "use_biophi", "use_scalop", 
            "use_sequence_features", "cdr_definition", "cache_abnumber"
        ]
        ab_features_cfg = {k: v for k, v in ab_cfg.items() if k in valid_args}
        ab_extractor = AntibodyFeatures(**ab_features_cfg)
        antibody_features_dim = ab_extractor.get_feature_dim()
        
        # Set in decoder config
        decoder_config["antibody_features_dim"] = antibody_features_dim
        decoder_config["antibody_features_normalized"] = ab_cfg.get("normalize_antibody_features", True)
        decoder_config["antibody_features_projection_dim"] = ab_cfg.get("projection_dim")
        
        # Set injection layer if not already specified
        if "antibody_features_injection_layer" not in decoder_config:
            # Match AttentionDecoder's default behavior if not specified
            # For MLP, we also default to "second" layer
            decoder_config["antibody_features_injection_layer"] = "second"
            
        logger.info(f"🧬 Auto-configured decoder for {antibody_features_dim}d antibody features")
        logger.info(f"   Normalized: {decoder_config['antibody_features_normalized']}")
        logger.info(f"   Projection: {decoder_config['antibody_features_projection_dim']}")
        logger.info(f"   Injection layer: {decoder_config['antibody_features_injection_layer']}")

    logger.info(f"Encoder: {encoder_type_str}")
    logger.info(f"Decoder: {decoder_type}")
    logger.info(f"Decoder config: {decoder_config}")

    # Get training params (already merged with property-specific)
    target_transform = config["training"]["target_transform"]  # Required
    target_transform_kwargs = config["training"].get("target_transform_kwargs", {})
    use_xavier_init = config["training"]["use_xavier_init"]
    xavier_gain = config["training"]["xavier_gain"]
    
    logger.info(f"Target transformation: {target_transform}")
    
    # Initialize data module
    data_module = GDPa1DataModule(
        data_path=config["data"]["gdpa1_path"],
        target_property=property_name,
        batch_size=config["training"]["finetune"]["batch_size"],
        fold_idx=fold_idx,
        cv_fold_col=config["data"]["cv_fold_column"],
        target_transform=target_transform,
        target_transform_kwargs=target_transform_kwargs,
        use_full_chain=use_full_chain,
        use_aho_aligned=use_aho_aligned,
        antibody_features_config=config.get("antibody_features"),
        normalize_antibody_features=config.get("antibody_features", {}).get("normalize_antibody_features", True),
    )
    
    # Initialize model
    freeze_encoder = encoder_config.pop("freeze_encoder", True)
    
    # Ensure learning_rate and weight_decay are floats (handle singleton lists from sweeps)
    lr = config["training"]["finetune"]["learning_rate"]
    if isinstance(lr, (list, tuple)) and len(lr) == 1:
        lr = lr[0]
    wd = config["training"]["finetune"]["weight_decay"]
    if isinstance(wd, (list, tuple)) and len(wd) == 1:
        wd = wd[0]

    model = DevelopabilityModel(
        encoder_config=encoder_config,
        decoder_type=decoder_type,
        decoder_config=decoder_config,
        learning_rate=float(lr),
        weight_decay=float(wd),
        scheduler=config["training"]["finetune"]["scheduler"],
        warmup_epochs=config["training"]["finetune"]["warmup_epochs"],
        max_epochs=config["training"]["finetune"]["max_epochs"],
        loss_fn=config["training"]["loss"],
        freeze_encoder=freeze_encoder,
        target_transform=data_module.target_transform,
        use_xavier_init=use_xavier_init,
        xavier_gain=xavier_gain,
    )
    
    # Setup checkpoint directory for this fold
    # Use temp directory for training, will save final model to model_dir
    fold_dir_name = f"fold{fold_idx}" if fold_idx is not None else "final"
    checkpoint_dir = Path("models/.tmp") / model_dir.name / fold_dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run name
    encoder_model = encoder_types[0] if len(encoder_types) == 1 else encoder_type_str
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_display = fold_idx if fold_idx is not None else "final"
    run_name = f"{property_name}_{encoder_model}_{decoder_type}_fold{fold_display}_{timestamp}"
    
    logger.info(f"Run name: {run_name}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    
    # Setup callbacks
    # In-memory best weights tracking (no disk I/O)
    best_weights_callback = BestWeightsCallback(
        monitor=config["logging"]["monitor_metric"],
        mode=config["logging"]["monitor_mode"]
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config["logging"]["monitor_metric"],
        patience=config["training"]["finetune"]["early_stopping_patience"],
        mode=config["logging"]["monitor_mode"],
        verbose=True,
        check_finite=False,  # Don't stop on NaN (can happen with small val sets)
    )
    
    # Prepare callbacks list
    callbacks = [best_weights_callback, early_stop_callback]
    
    # Add metrics collector callback for CV
    metrics_collector = None
    if fold_idx is not None:
        metrics_collector = MetricsCollectorCallback()
        callbacks.append(metrics_collector)

    # Add gradient monitor callback if debug mode enabled
    if debug:
        gradient_monitor = GradientMonitor(log_frequency=10)
        callbacks.append(gradient_monitor)
        logger.info("🐛 Debug mode: Gradient monitoring enabled (logging every 10 steps)")

    # Add SWA callback if enabled
    if config["training"].get("use_swa", False):
        from pytorch_lightning.callbacks import StochasticWeightAveraging
        
        # Calculate SWA learning rate as fraction of base LR
        base_lr = config["training"]["finetune"]["learning_rate"]
        # Handle cases where base_lr might be a sequence (e.g. from some sweep configs)
        if isinstance(base_lr, (list, tuple)) and len(base_lr) == 1:
            base_lr = base_lr[0]
        
        swa_lr_factor = config["training"].get("swa_lr_factor", 0.1)
        swa_lrs = float(base_lr) * swa_lr_factor
        
        swa_callback = StochasticWeightAveraging(
            swa_lrs=swa_lrs,
            swa_epoch_start=config["training"].get(
                "swa_epoch_start", 0.75
            ),
            annealing_epochs=config["training"].get(
                "swa_annealing_epochs", 10
            ),
            annealing_strategy=config["training"].get(
                "swa_annealing_strategy", "cos"
            ),
            device=None,  # Auto-detect
        )
        callbacks.append(swa_callback)
        logger.info(
            f"SWA enabled: swa_lrs={swa_lrs:.2e} "
            f"(factor={swa_lr_factor})"
        )
    
    # Setup W&B logger
    wandb_logger = WandbLogger(
        project=config["logging"].get("wandb_project", "CrossAbSense"),
        entity=config["logging"].get("wandb_entity"),
        name=run_name,
        save_dir="models/checkpoints",  # Save WandB logs to models/checkpoints
        config={
            "property": property_name,
            "fold": fold_idx,
            "encoder_type": encoder_type_str,
            "decoder_type": decoder_type,
            "batch_size": config["training"]["finetune"]["batch_size"],
            "learning_rate": config["training"]["finetune"]["learning_rate"],
            "freeze_encoder": freeze_encoder,
            **config,
        },
        tags=[property_name, encoder_type_str, decoder_type, f"fold{fold_idx}"],
        log_model=False,
    )
    
    # Initialize trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    precision = "16-mixed"  # Match tune_hyperparam.py

    logger.info(f"Accelerator: {accelerator}, Devices: {devices}, Precision: {precision}")
    
    # For final model (fold_idx=None), disable validation since there's no val set
    enable_validation = fold_idx is not None
    
    trainer = pl.Trainer(
        max_epochs=config["training"]["finetune"]["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        precision=precision,
        log_every_n_steps=config["logging"].get("wandb_log_every_n_steps", 10),
        check_val_every_n_epoch=2 if enable_validation else None,  # Match tune_hyperparam.py
        num_sanity_val_steps=2 if enable_validation else 0,  # Run validation at epoch 0
        limit_val_batches=1.0 if enable_validation else 0,  # Disable val for final model
        deterministic=True,
    )
    
    # Train
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
    else:
        logger.info("Starting training from scratch")
    
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)
    
    # Restore best weights from memory (no disk I/O) and use best captured metrics
    if best_weights_callback.best_weights is not None:
        logger.info(
            f"Restoring best weights "
            f"(best {config['logging']['monitor_metric']}: "
            f"{best_weights_callback.best_score:.4f})"
        )
        best_weights_callback.restore_best_weights(model)
        logger.info("Best model weights restored from memory")
        # USE THE BEST METRICS CAPTURED
        metrics = best_weights_callback.best_metrics
    else:
        # Fallback to current if no "best" was ever found
        metrics = {
            "val_spearman": float(trainer.callback_metrics.get("val_spearman", 0.0)),
            "val_pearson": float(trainer.callback_metrics.get("val_pearson", 0.0)),
            "val_r2": float(trainer.callback_metrics.get("val_r2", 0.0)),
            "val_rmse": float(trainer.callback_metrics.get("val_rmse", 0.0)),
            "val_mae": float(trainer.callback_metrics.get("val_mae", 0.0)),
            "val_loss": float(trainer.callback_metrics.get("val_loss", 0.0)),
        }
    
    # SAVE MODEL IMMEDIATELY after restoring best weights
    # This ensures we don't lose the trained model if post-training steps fail
    if fold_idx is None:
        final_model_path = model_dir / "final.ckpt"
    else:
        final_model_path = model_dir / f"fold{fold_idx}.ckpt"
    
    logger.info(f"Saving model to: {final_model_path}")
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'property': property_name,
        'fold': fold_idx,
        'metrics': metrics,
        'target_transform_stats': data_module.target_transform.get_stats(),
    }, final_model_path)
    
    # Get per-epoch metrics from collector callback
    epoch_metrics = []
    if metrics_collector and metrics_collector.epoch_metrics:
        epoch_metrics = metrics_collector.epoch_metrics
        logger.info(
            f"Collected {len(epoch_metrics)} epoch metrics during training"
        )
    
    # Save metrics
    metrics_file = checkpoint_dir / "best_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save per-epoch metrics for CV aggregation to model_dir (not checkpoint_dir)
    if epoch_metrics and fold_idx is not None:
        epoch_metrics_file = model_dir / f"fold{fold_idx}_epoch_metrics.json"
        with open(epoch_metrics_file, "w") as f:
            json.dump(epoch_metrics, f, indent=2)
        logger.info(f"Saved epoch metrics to: {epoch_metrics_file}")
    
    if fold_idx is None:
        logger.info(f"Final model results:")
    else:
        logger.info(f"Fold {fold_idx+1} results:")
    logger.info(f"  Spearman: {metrics.get('val_spearman', 0.0):.4f}")
    logger.info(f"  Pearson:  {metrics.get('val_pearson', 0.0):.4f}")
    logger.info(f"  RMSE:     {metrics.get('val_rmse', 0.0):.4f}")
    logger.info(f"  MAE:      {metrics.get('val_mae', 0.0):.4f}")
    
    # Save WandB run ID before closing (for CV metrics logging)
    wandb_run_id = wandb_logger.experiment.id if wandb_logger else None
    
    # Close W&B
    wandb_logger.experiment.finish()
    
    # Cleanup temp directory
    import shutil
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    
    logger.info("Training complete")
    
    return metrics, wandb_run_id


def train_final_model(
    base_config: dict,
    property_name: str,
    model_dir: Path,
    resume_checkpoint: Optional[str] = None,
    cv_results: Optional[dict] = None,
) -> None:
    """
    Train final model on ALL data (no CV split).

    Args:
        base_config: Base configuration dict (will be merged with property-specific)
        property_name: Target property
        model_dir: Model directory
        resume_checkpoint: Path to checkpoint to resume from (optional)
        cv_results: CV results dict to add test metrics (optional)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training FINAL model on ALL data: {property_name}")
    logger.info(f"{'='*60}")

    # Merge base config with property-specific overrides
    config = get_property_config(base_config, property_name)

    # Build encoder config (already merged with property-specific)
    encoder_config = config["encoder"].copy()

    # Get encoder_type (required, already merged from property_specific)
    encoder_type = config["encoder_type"]  # This is now at top level after merge
    encoder_types = [e.strip() for e in encoder_type.split('+')]
    encoder_type_str = "+".join(encoder_types)

    # Set encoder_types in encoder_config (required by DevelopabilityModel)
    encoder_config["encoder_types"] = encoder_types
    encoder_config["freeze_encoder"] = encoder_config.get("freeze_encoder", True)
    encoder_config["pooling"] = "none"  # Decoder handles pooling

    # Auto-detect precomputed embeddings
    source_csv = config["data"]["gdpa1_path"]
    use_full_chain = config["data"].get("use_full_chain", False)
    use_aho_aligned = config["data"].get("use_aho_aligned", False)

    for enc_type in encoder_types:
        embeddings_cfg = get_embeddings_config(
            source_csv, 
            encoder_type=enc_type,
            use_full_chain=use_full_chain,
            use_aho_aligned=use_aho_aligned
        )
        if embeddings_cfg:
            encoder_config["encoder_configs"][enc_type].update(embeddings_cfg)
            logger.info(f"✓ Using precomputed {enc_type} embeddings")

    # Build decoder config (already merged with property-specific)
    decoder_type = config["decoder"]["type"]
    decoder_config = config["decoder"].get(decoder_type, {}).copy()

    # Auto-configure antibody features if enabled
    if config.get("antibody_features", {}).get("enabled", False):
        from src.features.antibody_features import AntibodyFeatures
        ab_cfg = config["antibody_features"]
        
        # Only pass arguments that AntibodyFeatures.__init__ accepts
        valid_args = [
            "use_abnumber", "use_biophi", "use_scalop", 
            "use_sequence_features", "cdr_definition", "cache_abnumber"
        ]
        ab_features_cfg = {k: v for k, v in ab_cfg.items() if k in valid_args}
        ab_extractor = AntibodyFeatures(**ab_features_cfg)
        antibody_features_dim = ab_extractor.get_feature_dim()
        
        # Set in decoder config
        decoder_config["antibody_features_dim"] = antibody_features_dim
        decoder_config["antibody_features_normalized"] = ab_cfg.get("normalize_antibody_features", True)
        decoder_config["antibody_features_projection_dim"] = ab_cfg.get("projection_dim")
        
        # Set injection layer if not already specified
        if "antibody_features_injection_layer" not in decoder_config:
            # Match AttentionDecoder's default behavior if not specified
            # For MLP, we also default to "second" layer
            decoder_config["antibody_features_injection_layer"] = "second"
            
        logger.info(f"🧬 Auto-configured decoder for {antibody_features_dim}d antibody features")
        logger.info(f"   Normalized: {decoder_config['antibody_features_normalized']}")
        logger.info(f"   Projection: {decoder_config['antibody_features_projection_dim']}")
        logger.info(f"   Injection layer: {decoder_config['antibody_features_injection_layer']}")

    # Get training params (already merged with property-specific, same as CV folds)
    target_transform = config["training"]["target_transform"]  # Required
    target_transform_kwargs = config["training"].get("target_transform_kwargs", {})
    use_xavier_init = config["training"]["use_xavier_init"]
    xavier_gain = config["training"]["xavier_gain"]

    logger.info(f"Target transformation: {target_transform}")

    # Initialize data module WITHOUT fold split (uses all data for training)
    data_module = GDPa1DataModule(
        data_path=config["data"]["gdpa1_path"],
        target_property=property_name,
        batch_size=config["training"]["finetune"]["batch_size"],
        fold_idx=None,  # None = use all data
        cv_fold_col=config["data"]["cv_fold_column"],
        target_transform=target_transform,
        target_transform_kwargs=target_transform_kwargs,
        use_full_chain=use_full_chain,
        use_aho_aligned=use_aho_aligned,
        antibody_features_config=config.get("antibody_features"),
        normalize_antibody_features=config.get("antibody_features", {}).get("normalize_antibody_features", True),
    )
    
    # Initialize model
    freeze_encoder = encoder_config.pop("freeze_encoder", True)
    
    # Ensure learning_rate and weight_decay are floats (handle singleton lists from sweeps)
    lr = config["training"]["finetune"]["learning_rate"]
    if isinstance(lr, (list, tuple)) and len(lr) == 1:
        lr = lr[0]
    wd = config["training"]["finetune"]["weight_decay"]
    if isinstance(wd, (list, tuple)) and len(wd) == 1:
        wd = wd[0]

    model = DevelopabilityModel(
        encoder_config=encoder_config,
        decoder_type=decoder_type,
        decoder_config=decoder_config,
        learning_rate=float(lr),
        weight_decay=float(wd),
        scheduler=config["training"]["finetune"]["scheduler"],
        warmup_epochs=config["training"]["finetune"]["warmup_epochs"],
        max_epochs=config["training"]["finetune"]["max_epochs"],
        loss_fn=config["training"]["loss"],
        freeze_encoder=freeze_encoder,
        target_transform=data_module.target_transform,
        use_xavier_init=use_xavier_init,
        xavier_gain=xavier_gain,
    )
    
    # Setup checkpoint directory
    checkpoint_dir = Path("models/.tmp") / model_dir.name / "final"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run name
    encoder_model = encoder_types[0] if len(encoder_types) == 1 else encoder_type_str
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{property_name}_{encoder_model}_{decoder_type}_FINAL_{timestamp}"
    
    # Setup callbacks (no checkpoints to avoid I/O overhead)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="train_loss",
        mode="min",
        save_top_k=0,  # Don't save checkpoints
        enable_version_counter=False,
        save_last=False,
        verbose=False,
    )
    
    # Setup CV metrics logger callback if CV results available
    callbacks = [checkpoint_callback]
    if cv_results and "epoch_metrics" in cv_results:
        cv_logger_callback = CVMetricsLoggerCallback(
            cv_metrics=cv_results["epoch_metrics"]
        )
        callbacks.append(cv_logger_callback)

    # Add SWA callback if enabled
    if config["training"].get("use_swa", False):
        from pytorch_lightning.callbacks import StochasticWeightAveraging
        
        # Calculate SWA learning rate as fraction of base LR
        base_lr = config["training"]["finetune"]["learning_rate"]
        # Handle cases where base_lr might be a sequence (e.g. from some sweep configs)
        if isinstance(base_lr, (list, tuple)):
            base_lr = base_lr[0]
            
        swa_lr_factor = config["training"].get("swa_lr_factor", 0.1)
        swa_lrs = float(base_lr) * swa_lr_factor
        
        swa_callback = StochasticWeightAveraging(
            swa_lrs=swa_lrs,
            swa_epoch_start=config["training"].get(
                "swa_epoch_start", 0.75
            ),
            annealing_epochs=config["training"].get(
                "swa_annealing_epochs", 10
            ),
            annealing_strategy=config["training"].get(
                "swa_annealing_strategy", "cos"
            ),
            device=None,  # Auto-detect
        )
        callbacks.append(swa_callback)
        logger.info(
            f"SWA enabled: swa_lrs={swa_lrs:.2e} "
            f"(factor={swa_lr_factor})"
        )
    
    # Setup W&B logger
    wandb_logger = WandbLogger(
        project=config["logging"].get("wandb_project", "CrossAbSense"),
        entity=config["logging"].get("wandb_entity"),
        name=run_name,
        save_dir="models/checkpoints",  # Save WandB logs to models/checkpoints
        config={
            "property": property_name,
            "mode": "final_all_data",
            "encoder_type": encoder_type_str,
            "decoder_type": decoder_type,
            **config,
        },
        tags=[property_name, encoder_type_str, decoder_type, "final_model"],
        log_model=False,
    )
    
    # Initialize trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    precision = "16-mixed"  # Match tune_hyperparam.py

    trainer = pl.Trainer(
        max_epochs=config["training"]["finetune"]["max_epochs"],
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=config["training"]["gradient_clip_val"],
        precision=precision,
        log_every_n_steps=config["logging"].get("wandb_log_every_n_steps", 10),
        limit_val_batches=0,  # No validation for final model
        deterministic=True,
    )
    
    # Train on all data (CV metrics will be logged during training via callback)
    logger.info("Training final model on ALL data (no validation split)")
    if cv_results and "epoch_metrics" in cv_results:
        logger.info(
            f"Will log CV metrics during training "
            f"({len(cv_results['epoch_metrics'])} epochs)"
        )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)
    
    # Manually save final model state immediately after training
    import shutil
    final_model_path = model_dir / "final.ckpt"
    logger.info(f"Saving final model to: {final_model_path}")
    
    # Save model state dict with transformation stats
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'property': property_name,
        'mode': 'final_all_data',
        'target_transform_stats': data_module.target_transform.get_stats(),
    }, final_model_path)
    
    # Log CV summary statistics
    if cv_results and "summary" in cv_results:
        for key, value in cv_results["summary"].items():
            wandb_logger.experiment.summary[key] = value
        
        # Also add summary metrics to the saved model
        try:
            checkpoint = torch.load(final_model_path, weights_only=False)
            checkpoint['cv_summary'] = cv_results["summary"]
            torch.save(checkpoint, final_model_path)
        except Exception as e:
            logger.warning(f"Could not update final model with CV summary: {e}")
    
    # Close W&B (this will sync all logged metrics)
    wandb_logger.experiment.finish()
    
    logger.info(f"Final model saved: {final_model_path}")
    
    # Cleanup temp directory
    shutil.rmtree(checkpoint_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train antibody developability prediction model"
    )
    parser.add_argument(
        "--property",
        type=str,
        required=False,
        help="Target property (e.g., HIC, Titer, Tm2, PR_CHO, AC-SINS_pH7.4)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all properties sequentially",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default_config.yaml",
        help="Path to config file (default: src/config/default_config.yaml)",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        choices=["all", "0", "1", "2", "3", "4", "final"],
        help="Fold to train: 0-4 for a single fold, 'all' for full CV + final model (default), 'final' for final model only.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from most recent checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if model already exists",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "mae", "huber", "smooth_l1"],
        help="Override loss function to use for training (overrides config)",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        help="Override metric to monitor for early stopping and checkpointing (e.g. val_loss)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed gradient logging to WandB",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if not args.property and not args.all:
        parser.error("Either --property or --all must be specified")
    if args.property and args.all:
        parser.error("Cannot specify both --property and --all")
    
    # Define all properties
    all_properties = ["HIC", "Titer", "PR_CHO", "AC-SINS_pH7.4", "Tm2"]
    
    # Determine which properties to train
    if args.all:
        properties_to_train = all_properties
        print(f"\n{'='*60}")
        print(f"Training ALL properties: {', '.join(properties_to_train)}")
        print(f"{'='*60}\n")
    else:
        properties_to_train = [args.property]
    
    # Train each property
    for property_name in properties_to_train:
        if len(properties_to_train) > 1:
            print(f"\n{'#'*60}")
            print(f"# Starting training for: {property_name}")
            print(f"{'#'*60}\n")
        
        train_property(property_name, args)
        
        if len(properties_to_train) > 1:
            print(f"\n{'#'*60}")
            print(f"# Completed training for: {property_name}")
            print(f"{'#'*60}\n")


def train_property(property_name: str, args):
    """Train a single property."""
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    # Override loss function if requested via CLI
    if getattr(args, "loss", None):
        logger.info(f"Overriding loss function from config: {config['training'].get('loss')} -> {args.loss}")
        config["training"]["loss"] = args.loss

    # Override early stopping / checkpoint monitor if requested via CLI
    if getattr(args, "monitor", None):
        old_monitor = config.get("logging", {}).get("monitor_metric")
        logger.info(f"Overriding monitor metric from config: {old_monitor} -> {args.monitor}")
        # Ensure logging section exists
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["monitor_metric"] = args.monitor
        # Set monitor mode automatically: minimize losses, maximize metrics
        if "loss" in args.monitor or args.monitor.endswith("_loss"):
            config["logging"]["monitor_mode"] = "min"
        else:
            config["logging"]["monitor_mode"] = "max"
    
    # Note: Random seed is set per-fold in train_single_fold() for proper CV diversity
    # Base seed: config.get("random_seed", 42)
    # Per-fold seeds will be: 42, 43, 44, 45, 46 (base_seed + fold_idx)
    
    # property_name is passed as parameter, don't override it
    n_folds = config["data"]["n_folds"]
    
    # Compute config checksum for model directory naming
    config_path = Path(args.config)
    config_checksum = compute_file_checksum(str(config_path))[:8]
    
    # Determine model directory name
    if config_path == Path("src/config/default_config.yaml"):
        # Default config: use property name + checksum
        model_dir_name = f"{property_name}_{config_checksum}"
    else:
        # Custom config: use property + filename + checksum
        config_filename = config_path.stem
        model_dir_name = f"{property_name}_{config_filename}_{config_checksum}"
    
    # Create model directory
    model_dir = Path("models") / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Config checksum: {config_checksum}")

    # Save config copy and property name for reproducibility (needed by predict.py)
    config_copy = model_dir / "config.yaml"
    if not config_copy.exists():
        import shutil
        shutil.copy(args.config, config_copy)
        logger.info(f"Config saved to: {config_copy}")
    property_file = model_dir / "property.txt"
    if not property_file.exists():
        property_file.write_text(property_name)
        logger.info(f"Property saved to: {property_file}")

    # Single-fold or final-model shortcut
    fold = getattr(args, "fold", "all")
    if fold != "all":
        print(f"\n{'='*60}")
        if fold == "final":
            print(f"Training FINAL model on ALL data for {property_name}")
            train_final_model(config, property_name, model_dir, None, None)
        else:
            fold_idx = int(fold)
            print(f"Training single fold {fold_idx+1}/{n_folds} for {property_name}")
            train_single_fold(config, property_name, fold_idx, model_dir, None, debug=args.debug)
        print(f"{'='*60}\n")
        return

    # Check if model already exists (unless --force)
    if not args.force and model_dir.exists():
        fold_models_exist = all(
            (model_dir / f"fold{i}.ckpt").exists() for i in range(n_folds)
        )
        final_model_exists = (model_dir / "final.ckpt").exists()

        if fold_models_exist and final_model_exists:
            print(f"\n{'='*60}")
            print(f"Model already exists: {model_dir}")
            print(f"All {n_folds} fold models + final model found.")
            print(f"\nUse --force to retrain.")
            print(f"{'='*60}\n")
            return
    
    # Cross-validation (all folds)
    print(f"\n{'='*60}")
    print(f"Running {n_folds}-fold cross-validation for {property_name}")
    print(f"Model directory: {model_dir}")
    print(f"{'='*60}\n")
    
    all_metrics = []
    all_epoch_metrics = []  # Store per-epoch metrics from each fold
    
    for fold_idx in range(n_folds):  # 0-based: 0,1,2,3,4
        print(f"\n--- Fold {fold_idx+1}/{n_folds} ---")
        
        # Check if this fold is already trained
        fold_checkpoint = model_dir / f"fold{fold_idx}.ckpt"
        if fold_checkpoint.exists() and not args.force:
            print(f"Fold {fold_idx+1} already trained, skipping...")
            # Load metrics from checkpoint
            try:
                checkpoint = torch.load(
                    fold_checkpoint, map_location='cpu', weights_only=False
                )
                if 'metrics' in checkpoint:
                    metrics = checkpoint['metrics']
                    all_metrics.append(metrics)
                    print(
                        f"  Loaded metrics: "
                        f"Spearman={metrics.get('val_spearman', 0):.4f}"
                    )
                else:
                    # Old checkpoint format without metrics
                    all_metrics.append({
                        "val_spearman": 0.0,
                        "val_pearson": 0.0,
                        "val_rmse": 0.0,
                        "val_mae": 0.0
                    })
                    print(
                        "  Warning: Checkpoint exists but "
                        "no metrics found (old format)"
                    )
                
                # Try to load per-epoch metrics
                epoch_metrics_file = model_dir / f"fold{fold_idx}_epoch_metrics.json"
                if epoch_metrics_file.exists():
                    with open(epoch_metrics_file) as f:
                        epoch_metrics = json.load(f)
                        all_epoch_metrics.append(epoch_metrics)
                        print(
                            f"  Loaded {len(epoch_metrics)} "
                            f"epoch metrics for fold {fold_idx+1}"
                        )
            except Exception as e:
                print(
                    f"  Warning: Could not load metrics "
                    f"from checkpoint: {e}"
                )
                all_metrics.append({
                    "val_spearman": 0.0,
                    "val_pearson": 0.0,
                    "val_rmse": 0.0,
                    "val_mae": 0.0
                })
            continue
        
        # Note: Resume not supported since checkpoints are disabled
        # If you need to resume, enable checkpointing in train_single_fold
        resume_checkpoint = None
        
        try:
            metrics, _ = train_single_fold(
                config, property_name, fold_idx, model_dir, resume_checkpoint,
                debug=args.debug
            )
            all_metrics.append(metrics)
            
            # Load per-epoch metrics that were just saved
            epoch_metrics_file = model_dir / f"fold{fold_idx}_epoch_metrics.json"
            if epoch_metrics_file.exists():
                with open(epoch_metrics_file) as f:
                    epoch_metrics = json.load(f)
                    all_epoch_metrics.append(epoch_metrics)
                    logger.info(
                        f"Loaded {len(epoch_metrics)} epoch metrics "
                        f"for fold {fold_idx+1}"
                    )
        except Exception as e:
            logger.error(f"ERROR in fold {fold_idx+1}: {e}")
            print(f"ERROR in fold {fold_idx+1}: {e}")
            continue
    
    # Aggregate CV results
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results for {property_name}")
        print(f"{'='*60}")
        
        for metric_name in [
            "val_spearman", "val_pearson", "val_rmse", "val_mae"
        ]:
            values = [m[metric_name] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name:20s}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"{'='*60}\n")
        
        # Save aggregated CV results
        cv_results = {
            "property": property_name,
            "timestamp": datetime.now().isoformat(),
            "n_folds": len(all_metrics),
            "folds": all_metrics,
            "summary": {
                f"cv_{metric}_mean": float(
                    np.mean([m.get(f"val_{metric}", 0.0) for m in all_metrics])
                )
                for metric in [
                    "spearman", "pearson", "r2", "rmse", "mae", "loss"
                ]
            },
        }
        
        # Add std to summary
        for metric in ["spearman", "pearson", "r2", "rmse", "mae", "loss"]:
            cv_results["summary"][f"cv_{metric}_std"] = float(
                np.std([m.get(f"val_{metric}", 0.0) for m in all_metrics])
            )
        
        # Aggregate per-epoch metrics across folds if available
        if all_epoch_metrics and len(all_epoch_metrics) == n_folds:
            logger.info(
                f"Aggregating per-epoch CV metrics "
                f"from {len(all_epoch_metrics)} folds"
            )
            
            # Get max_epochs from config
            max_epochs = config["training"]["finetune"]["max_epochs"]
            
            # Forward-fill metrics for each fold up to max_epochs
            # (for early stopping cases)
            filled_epoch_metrics = []
            for fold_metrics in all_epoch_metrics:
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
            
            # Aggregate metrics for each epoch (means only for per-epoch)
            cv_epoch_metrics = []
            for epoch in range(max_epochs):
                epoch_data = {"epoch": epoch}
                
                # Collect values from all folds for this epoch
                for metric_name in [
                    "val_spearman", "val_pearson",
                    "val_rmse", "val_mae", "val_loss"
                ]:
                    values = []
                    for fold_metrics in filled_epoch_metrics:
                        if epoch < len(fold_metrics):
                            epoch_metric = fold_metrics[epoch]
                            if metric_name in epoch_metric:
                                values.append(epoch_metric[metric_name])
                    
                    # Calculate mean only for per-epoch metrics
                    if values:
                        epoch_data[f"cv_{metric_name}"] = float(
                            np.mean(values)
                        )
                
                if len(epoch_data) > 1:  # Has metrics beyond epoch number
                    cv_epoch_metrics.append(epoch_data)
            
            # Add to cv_results
            cv_results["epoch_metrics"] = cv_epoch_metrics
            logger.info(
                f"Aggregated per-epoch CV metrics "
                f"from {len(all_epoch_metrics)} folds "
                f"({len(cv_epoch_metrics)} epochs)"
            )
        
        # Save to model directory
        results_file = model_dir / "cv_results.json"
        
        with open(results_file, "w") as f:
            json.dump(cv_results, f, indent=2)
        
        logger.info(f"CV results saved to: {results_file}")
        print(f"Results saved to: {results_file}")
        print(f"\nCV Model files:")
        for i in range(n_folds):  # 0-based fold indexing
            fold_file = model_dir / f"fold{i}.ckpt"
            if fold_file.exists():
                print(f"  {fold_file}")
        
        # Train final model on ALL data (no CV)
        print(f"\n{'='*60}")
        print("Training final model on ALL data")
        print(f"{'='*60}\n")
        
        final_checkpoint = model_dir / "final.ckpt"
        resume_final = None
        if args.resume and final_checkpoint.exists():
            resume_final = str(final_checkpoint)
            print(f"Resuming final model from: {resume_final}")
        
        try:
            train_final_model(
                config, property_name, model_dir, resume_final, cv_results
            )
            print(f"\nFinal model saved to: {final_checkpoint}")
        except Exception as e:
            logger.error(f"ERROR training final model: {e}")
            print(f"ERROR training final model: {e}")
    else:
        print("\n⚠️  No successful fold completions!")


if __name__ == "__main__":
    main()
