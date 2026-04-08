"""PyTorch Lightning model for antibody developability prediction."""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, ConstantInputWarning

from ..encoders import MultiEncoder
from ..decoders import AttentionDecoder, MLPDecoder
from ..utils.visualization import log_prediction_plots_to_wandb

logger = logging.getLogger(__name__)


class DevelopabilityModel(pl.LightningModule):
    """
    Lightning module for antibody developability prediction.

    Combines an encoder (AntiBERTy, ESM-C, ProtT5, Combined, or Multi) with
    a decoder (MLP or Attention) to predict developability properties.
    """

    def __init__(
        self,
        encoder_type: str = "esmc_300m",  # Deprecated: use encoder_config["encoder_types"] instead
        encoder_config: Optional[Dict] = None,
        decoder_type: str = "mlp",
        decoder_config: Optional[Dict] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        loss_fn: str = "mse",
        freeze_encoder: bool = True,
        target_transform: Optional[Any] = None,
        use_xavier_init: bool = False,
        xavier_gain: float = 0.1,
    ):
        """
        Args:
            encoder_type: DEPRECATED - kept for backward compatibility. Use encoder_config["encoder_types"] instead.
            encoder_config: Configuration dict for encoder (must contain "encoder_types" list)
            decoder_type: Type of decoder (mlp, attention)
            decoder_config: Configuration dict for decoder
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            scheduler: Learning rate scheduler (cosine, plateau, step)
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum training epochs (for scheduler)
            loss_fn: Loss function (mse, mae, huber, smooth_l1)
            freeze_encoder: Whether to freeze encoder (feature extraction mode)
            target_transform: Target transformation object for inverse transform (optional)
            use_xavier_init: Apply Xavier uniform initialization to decoder weights (default: False)
            xavier_gain: Gain for Xavier init, use small values for pre-norm (default: 0.1)
        """
        super().__init__()

        logger.info("="*60)
        logger.info("Initializing DevelopabilityModel")
        logger.info("="*60)
        logger.info(f"Encoder type: {encoder_type}")
        logger.info(f"Decoder type: {decoder_type}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Scheduler: {scheduler}")
        logger.info(f"Loss function: {loss_fn}")
        logger.info(f"Freeze encoder: {freeze_encoder}")
        
        # Store target transform for metric computation on original scale
        self.target_transform = target_transform

        # Initialize encoder - ALWAYS use MultiEncoder
        encoder_config = encoder_config or {}

        # Determine encoder_types from config
        if "encoder_types" in encoder_config:
            # New unified format: encoder_types list is directly in config
            encoder_types_list = encoder_config["encoder_types"]
            logger.info(f"Using MultiEncoder with encoders: {encoder_types_list}")
            self.encoder = MultiEncoder(**encoder_config)
        else:
            # Backward compatibility: convert old encoder_type format
            logger.warning(f"⚠️  DEPRECATED: encoder_type='{encoder_type}' is deprecated. Use encoder_config['encoder_types'] instead.")

            if encoder_type in ["antiberty", "esmc", "esmc_300m", "esmc_600m", "esmc_6b", "prott5"]:
                # Convert single encoder to MultiEncoder format
                logger.info(f"Converting to MultiEncoder with single encoder: {encoder_type}")
                self.encoder = MultiEncoder(
                    encoder_types=[encoder_type],
                    encoder_configs={encoder_type: encoder_config},
                    fusion_strategy="concat",
                    embedding_dim=encoder_config.get("embedding_dim", 1024),
                    use_heavy=encoder_config.get("use_heavy", True),
                    use_light=encoder_config.get("use_light", True),
                    pooling=encoder_config.get("pooling", "mean"),
                    freeze_epochs=encoder_config.get("freeze_epochs", 0),
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Freeze encoder immediately if specified (BEFORE save_hyperparameters for accurate model summary)
        if freeze_encoder:
            logger.info("🔒 Freezing encoder (feature extraction mode)")
            self.encoder.freeze()
        else:
            logger.info("🔓 Encoder unfrozen (full fine-tuning mode)")

        # Save hyperparameters AFTER freezing so model summary is accurate
        # Ignore target_transform as it's not serializable and will conflict with datamodule
        self.save_hyperparameters(ignore=['target_transform'])

        # Initialize decoder
        decoder_config = decoder_config or {}
        embedding_dim = self.encoder.get_embedding_dim()
        logger.info(f"Encoder embedding dimension: {embedding_dim}")
        logger.info(f"Initializing {decoder_type} decoder with config: {decoder_config}")

        if decoder_type == "mlp":
            self.decoder = MLPDecoder(input_dim=embedding_dim, **decoder_config)
        elif decoder_type == "attention":
            self.decoder = AttentionDecoder(input_dim=embedding_dim, **decoder_config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Update hyperparameters with actual hidden_dim used by decoder (if it was None)
        if hasattr(self.decoder, 'hidden_dim') and decoder_config.get('hidden_dim') is None:
            actual_hidden_dim = self.decoder.hidden_dim
            logger.info(f"Decoder hidden_dim was None, using actual value: {actual_hidden_dim}")
            # Update hparams.decoder_config with actual value for wandb logging
            self.hparams.decoder_config['hidden_dim'] = actual_hidden_dim

        # Apply Xavier initialization if requested
        if use_xavier_init and hasattr(self.decoder, '_xavier_init_weights'):
            logger.info(f"Applying Xavier initialization to decoder with gain={xavier_gain}")
            self.decoder._xavier_init_weights(gain=xavier_gain)

        # Loss function
        logger.info(f"Initializing loss function: {loss_fn}")
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        elif loss_fn == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Track metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        logger.info("Model initialization complete")
        logger.info("="*60)

    def forward(self, heavy_sequences: Optional[list] = None, light_sequences: Optional[list] = None,
                heavy_embeddings: Optional[torch.Tensor] = None,
                light_embeddings: Optional[torch.Tensor] = None,
                antibody_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through encoder and decoder.

        Args:
            heavy_sequences: List of heavy chain sequences (only needed if embeddings not provided)
            light_sequences: List of light chain sequences (only needed if embeddings not provided)
            heavy_embeddings: Optional precomputed heavy chain embeddings (batch, hidden_dim) - frozen
            light_embeddings: Optional precomputed light chain embeddings (batch, hidden_dim) - frozen
            antibody_features: Optional antibody features tensor (batch, features_dim)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Use cached embeddings if provided (frozen, no gradients)
        if heavy_embeddings is not None or light_embeddings is not None:
            # Move embeddings to model device (they come from DataLoader on CPU)
            device = next(self.parameters()).device
            vh_emb = heavy_embeddings.to(device) if heavy_embeddings is not None else None
            vl_emb = light_embeddings.to(device) if light_embeddings is not None else None
        else:
            # Fallback: encode sequences using encoder (for inference on new data)
            if heavy_sequences is None and light_sequences is None:
                raise ValueError("Must provide either embeddings or sequences")
            # Encoder returns tuple (vh_embeddings, vl_embeddings)
            vh_emb, vl_emb = self.encoder(heavy_sequences, light_sequences)

        # Move antibody features to device if provided
        if antibody_features is not None:
            device = next(self.parameters()).device
            antibody_features = antibody_features.to(device)

        # Decode to property prediction (decoder accepts VH, VL, and optional antibody features)
        predictions = self.decoder(vh_emb, vl_emb, antibody_features=antibody_features)

        return predictions

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        heavy_seqs = batch["heavy_sequences"]
        light_seqs = batch["light_sequences"]
        targets = batch["targets"].unsqueeze(1)  # (batch_size, 1)
        batch_size = len(targets)

        # Get cached embeddings if available
        heavy_embs = batch.get("heavy_embeddings", None)
        light_embs = batch.get("light_embeddings", None)

        # Get antibody features if available
        antibody_features = batch.get("antibody_features", None)

        # Forward pass
        predictions = self(heavy_seqs, light_seqs, heavy_embs, light_embs, antibody_features)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Log loss (on_step for detailed tracking, on_epoch for aggregated view)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Store predictions and targets for epoch-end correlation metrics
        self.training_step_outputs.append(
            {"predictions": predictions.detach(), "targets": targets}
        )

        return loss

    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Compute training metrics
        if self.training_step_outputs:
            # Gather all predictions and targets
            all_preds = torch.cat([x["predictions"] for x in self.training_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.training_step_outputs])
            batch_size = len(all_preds)

            # Compute correlation metrics (on CPU)
            # Convert to float32 for NumPy compatibility (NumPy doesn't support bfloat16)
            preds_np = all_preds.detach().cpu().float().numpy().flatten()
            targets_np = all_targets.detach().cpu().float().numpy().flatten()

            spearman_corr = spearmanr(preds_np, targets_np)[0]

            # Log train spearman
            self.log("train_spearman", spearman_corr, prog_bar=True, sync_dist=True, batch_size=batch_size)

            # Log learnable chain fusion weights (Exp 5)
            if hasattr(self.decoder, 'use_learnable_chain_fusion') and \
               self.decoder.use_learnable_chain_fusion in ["per_chain", "per_dim"] and \
               self.decoder.chain_weight is not None:
                
                # Get weight and apply sigmoid to get [0, 1] range for w_VH
                with torch.no_grad():
                    w_vh = torch.sigmoid(self.decoder.chain_weight)
                    
                    if self.decoder.use_learnable_chain_fusion == "per_chain":
                        # Log single scalar
                        self.log("fusion/w_vh", w_vh.item(), sync_dist=True)
                    else:
                        # Log mean and std for per_dim
                        self.log("fusion/w_vh_mean", w_vh.mean().item(), sync_dist=True)
                        self.log("fusion/w_vh_std", w_vh.std().item(), sync_dist=True)

            # Clear outputs
            self.training_step_outputs.clear()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """Validation step."""
        # Log first validation step for debugging Epoch 1 issue
        if batch_idx == 0 and len(self.validation_step_outputs) == 0:
            logger.info(f"Epoch {self.current_epoch}: validation_step called (first batch, {len(batch['targets'])} samples)")

        heavy_seqs = batch["heavy_sequences"]
        light_seqs = batch["light_sequences"]
        targets = batch["targets"].unsqueeze(1)

        # Get cached embeddings if available
        heavy_embs = batch.get("heavy_embeddings", None)
        light_embs = batch.get("light_embeddings", None)

        # Get antibody features if available
        antibody_features = batch.get("antibody_features", None)

        # Forward pass
        predictions = self(heavy_seqs, light_seqs, heavy_embs, light_embs, antibody_features)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Store for epoch-end metrics
        self.validation_step_outputs.append(
            {"predictions": predictions.detach(), "targets": targets.detach(), "loss": loss.detach()}
        )

        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        num_outputs = len(self.validation_step_outputs)
        logger.info(f"Epoch {self.current_epoch}: on_validation_epoch_end called with {num_outputs} validation batches")

        if not self.validation_step_outputs:
            logger.warning(
                f"Epoch {self.current_epoch}: validation_step_outputs is empty! "
                f"validation_step() was never called or outputs were cleared. "
                f"Skipping validation metric computation."
            )
            return

        # Gather all predictions and targets
        all_preds = torch.cat([x["predictions"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
        all_losses = torch.stack([x["loss"] for x in self.validation_step_outputs])

        # Compute average loss
        avg_loss = all_losses.mean()

        # Compute correlation metrics (requires CPU for scipy)
        # Convert to float32 for NumPy compatibility (NumPy doesn't support bfloat16)
        preds_np = all_preds.detach().cpu().float().numpy().flatten()
        targets_np = all_targets.detach().cpu().float().numpy().flatten()
        
        # Apply inverse transform for metrics on original scale
        if self.target_transform is not None:
            # Clip predictions to reasonable range to prevent overflow in exp()
            # For log transforms: clip to ±10 in log-z-score space (~e^10 = 22k range)
            preds_np_clipped = np.clip(preds_np, -10, 10)
            if not np.allclose(preds_np, preds_np_clipped):
                n_clipped = np.sum(np.abs(preds_np) > 10)
                logger.warning(
                    f"Clipped {n_clipped}/{len(preds_np)} extreme predictions "
                    f"(range: [{preds_np.min():.2f}, {preds_np.max():.2f}]) "
                    f"to prevent overflow"
                )
            preds_np_original = self.target_transform.inverse_transform(preds_np_clipped)
            targets_np_original = self.target_transform.inverse_transform(targets_np)
        else:
            preds_np_original = preds_np
            targets_np_original = targets_np

        # Check for overflow/extreme values
        preds_min, preds_max = preds_np_original.min(), preds_np_original.max()
        targets_min, targets_max = targets_np_original.min(), targets_np_original.max()

        if np.abs(preds_max) > 1e6 or np.abs(preds_min) > 1e6:
            logger.warning(
                f"⚠️  Extreme predictions detected: min={preds_min:.2e}, max={preds_max:.2e}. "
                f"This may indicate numerical instability."
            )

        # Compute RMSE, MAE and R2 on original scale (with overflow protection)
        with np.errstate(over='warn', invalid='warn'):
            rmse = np.sqrt(np.mean(np.clip((preds_np_original - targets_np_original) ** 2, 0, 1e12)))
            mae = np.mean(np.abs(preds_np_original - targets_np_original))
            
            # Simple R2 calculation
            ss_res = np.sum((targets_np_original - preds_np_original) ** 2)
            ss_tot = np.sum((targets_np_original - np.mean(targets_np_original)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

        # Check for constant arrays (correlation undefined)
        preds_std = np.std(preds_np_original)
        targets_std = np.std(targets_np_original)

        if preds_std < 1e-6 or targets_std < 1e-6:
            # Constant predictions or targets - correlation undefined
            logger.warning(
                f"Constant array detected in validation (preds_std={preds_std:.2e}, "
                f"targets_std={targets_std:.2e}). Setting correlations to 0.0."
            )
            pearson_corr = 0.0
            spearman_corr = 0.0
        else:
            # Suppress scipy warnings about constant input (we already checked)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConstantInputWarning)
                pearson_corr = pearsonr(preds_np_original, targets_np_original)[0]
                spearman_corr = spearmanr(preds_np_original, targets_np_original)[0]

        # Log metrics (don't use batch_size parameter - it's misleading for epoch-level metrics)
        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_pearson", pearson_corr, prog_bar=True, sync_dist=True)
        self.log("val_spearman", spearman_corr, sync_dist=True)
        self.log("val_rmse", rmse, sync_dist=True)
        self.log("val_mae", mae, sync_dist=True)
        self.log("val_r2", r2, sync_dist=True)

        # DISABLED: Plots are generated only after all folds complete (in CV summary)
        # This saves significant I/O time during validation
        # if self.logger is not None and hasattr(self.logger, 'experiment'):
        #     try:
        #         log_prediction_plots_to_wandb(
        #             ground_truth=targets_np,
        #             predictions=preds_np,
        #             prefix="val",
        #             epoch=self.current_epoch,
        #             show_error_lines=True,
        #         )
        #     except Exception as e:
        #         logger.warning(f"Failed to log prediction plots: {e}")

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """Test step."""
        heavy_seqs = batch["heavy_sequences"]
        light_seqs = batch["light_sequences"]
        targets = batch["targets"].unsqueeze(1)

        # Get cached embeddings if available
        heavy_embs = batch.get("heavy_embeddings", None)
        light_embs = batch.get("light_embeddings", None)

        # Get antibody features if available
        antibody_features = batch.get("antibody_features", None)

        # Forward pass
        predictions = self(heavy_seqs, light_seqs, heavy_embs, light_embs, antibody_features)

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Store for epoch-end metrics
        self.test_step_outputs.append(
            {
                "predictions": predictions.detach(),
                "targets": targets.detach(),
                "loss": loss.detach(),
                "antibody_ids": batch["antibody_ids"],
            }
        )

        return loss

    def on_test_epoch_end(self):
        """Compute test metrics at epoch end."""
        if not self.test_step_outputs:
            return

        # Gather all predictions and targets
        all_preds = torch.cat([x["predictions"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
        all_losses = torch.stack([x["loss"] for x in self.test_step_outputs])
        batch_size = len(all_preds)

        # Compute metrics
        avg_loss = all_losses.mean()
        # Convert to float32 for NumPy compatibility (NumPy doesn't support bfloat16)
        preds_np = all_preds.detach().cpu().float().numpy().flatten()
        targets_np = all_targets.detach().cpu().float().numpy().flatten()
        
        # Apply inverse transform for metrics on original scale
        if self.target_transform is not None:
            # Clip predictions to reasonable range to prevent overflow in exp()
            preds_np_clipped = np.clip(preds_np, -10, 10)
            if not np.allclose(preds_np, preds_np_clipped):
                n_clipped = np.sum(np.abs(preds_np) > 10)
                logger.warning(
                    f"Clipped {n_clipped}/{len(preds_np)} extreme predictions "
                    f"in test step to prevent overflow"
                )
            preds_np_original = self.target_transform.inverse_transform(preds_np_clipped)
            targets_np_original = self.target_transform.inverse_transform(targets_np)
        else:
            preds_np_original = preds_np
            targets_np_original = targets_np

        # Check for overflow/extreme values
        preds_min, preds_max = preds_np_original.min(), preds_np_original.max()
        targets_min, targets_max = targets_np_original.min(), targets_np_original.max()

        if np.abs(preds_max) > 1e6 or np.abs(preds_min) > 1e6:
            logger.warning(
                f"⚠️  Extreme predictions detected: min={preds_min:.2e}, max={preds_max:.2e}. "
                f"This may indicate numerical instability."
            )

        # Compute RMSE, MAE and R2 on original scale (with overflow protection)
        with np.errstate(over='warn', invalid='warn'):
            rmse = np.sqrt(np.mean(np.clip((preds_np_original - targets_np_original) ** 2, 0, 1e12)))
            mae = np.mean(np.abs(preds_np_original - targets_np_original))
            
            # Simple R2 calculation
            ss_res = np.sum((targets_np_original - preds_np_original) ** 2)
            ss_tot = np.sum((targets_np_original - np.mean(targets_np_original)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))

        # Check for constant arrays (correlation undefined)
        preds_std = np.std(preds_np_original)
        targets_std = np.std(targets_np_original)

        if preds_std < 1e-6 or targets_std < 1e-6:
            # Constant predictions or targets - correlation undefined
            logger.warning(
                f"Constant array detected in test (preds_std={preds_std:.2e}, "
                f"targets_std={targets_std:.2e}). Setting correlations to 0.0."
            )
            pearson_corr = 0.0
            spearman_corr = 0.0
        else:
            # Suppress scipy warnings about constant input (we already checked)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConstantInputWarning)
                pearson_corr = pearsonr(preds_np_original, targets_np_original)[0]
                spearman_corr = spearmanr(preds_np_original, targets_np_original)[0]

        # Log metrics (will automatically go to WandB if Trainer has a logger)
        self.log("test_loss", avg_loss, sync_dist=True, batch_size=batch_size)
        self.log("test_pearson", pearson_corr, sync_dist=True, batch_size=batch_size)
        self.log("test_spearman", spearman_corr, sync_dist=True, batch_size=batch_size)
        self.log("test_rmse", rmse, sync_dist=True, batch_size=batch_size)
        self.log("test_mae", mae, sync_dist=True, batch_size=batch_size)
        self.log("test_r2", r2, sync_dist=True, batch_size=batch_size)

        # Save predictions and targets for final plot generation (after all folds)
        # This is used by the CV summary to generate plots after all folds complete
        self.test_predictions = preds_np
        self.test_targets = targets_np

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        elif self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_pearson",
                    "interval": "epoch",
                },
            }

        elif self.hparams.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        else:
            return optimizer
