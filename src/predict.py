#!/usr/bin/env python3
"""
Prediction script for antibody developability properties.

Loads trained models and makes predictions on new antibody sequences.

Usage:
    # Predict with final model (default - single model trained on all data)
    python -m src.predict --input sequences.csv --model models/HIC_51318e6b --output predictions.csv

    # Predict with CV ensemble (uses mean of 5 folds with uncertainty estimates)
    python -m src.predict --input sequences.csv --model models/HIC_51318e6b --use-cv --output predictions.csv

    # Predict multiple properties
    python -m src.predict --input sequences.csv --models models/HIC_* models/Titer_* --output predictions.csv

Input CSV format:
    Must contain columns:
    - vh_protein_sequence: Heavy chain variable region sequence
    - vl_protein_sequence: Light chain variable region sequence
    - antibody_id (optional): Identifier for each antibody
    - hc_subtype: Heavy chain isotype (IgG1, IgG2, IgG4). Required for full-chain
                  models (HIC, Titer, Tm2, PR_CHO). Can be omitted if
                  --default-subtype is provided on the command line.
    - lc_subtype: Light chain isotype (Kappa, Lambda). Same rules as hc_subtype.

Output CSV format:
    Contains all input columns plus:
    - <property>_prediction: Prediction from final model (or mean across CV folds with --use-cv)
    - <property>_std: Standard deviation across CV folds (only with --use-cv)
    - <property>_fold1, <property>_fold2, ...: Individual fold predictions (only with --use-cv)
"""

import argparse
import json
import warnings
import os
import getpass

# Avoid CUDA memory fragmentation when loading large encoder checkpoints
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data import GDPa1Dataset, create_transform
from src.models import DevelopabilityModel
from src.utils import load_config, setup_logger, get_property_config
from src.utils.tta import predict_batch_with_tta

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logger
logger = setup_logger("predict", log_file="predict.log")


def prompt_for_forge_token_if_needed(encoder_config: dict) -> dict:
    """
    Check if ESMC-6B encoder is being used and prompt for forge token if not set.
    
    Args:
        encoder_config: Encoder configuration dictionary
    
    Returns:
        Updated encoder_config with forge_token if needed
    
    Note:
        Token is NOT stored - only used for this session
    """
    encoder_types = encoder_config.get("encoder_types", [])
    
    # Check if esmc_6b is in the encoder types
    if "esmc_6b" not in encoder_types:
        return encoder_config
    
    # Check if forge_token is already in config or environment
    esmc_config = encoder_config.get("encoder_configs", {}).get("esmc_6b", {})
    if "forge_token" in esmc_config or os.environ.get("FORGE_TOKEN"):
        return encoder_config
    
    # Prompt user for token
    print("\n" + "="*70)
    print("FORGE TOKEN REQUIRED FOR ESMC-6B MODEL")
    print("="*70)
    print("The model uses ESMC-6B encoder which requires a EvolutionaryScale Forge token.")
    print("\n⚠️  SECURITY NOTE: Your token will NOT be stored anywhere.")
    print("   It will only be used for this prediction session.")
    print("\nTo avoid this prompt in the future, set the FORGE_TOKEN environment variable.")
    print("="*70)
    
    forge_token = getpass.getpass("Enter your Forge token (input hidden): ").strip()
    
    if not forge_token:
        raise ValueError(
            "Forge token is required for ESMC-6B models. "
            "Set FORGE_TOKEN environment variable or provide when prompted."
        )
    
    # Inject token into config (only for this session)
    if "encoder_configs" not in encoder_config:
        encoder_config["encoder_configs"] = {}
    if "esmc_6b" not in encoder_config["encoder_configs"]:
        encoder_config["encoder_configs"]["esmc_6b"] = {}
    
    encoder_config["encoder_configs"]["esmc_6b"]["forge_token"] = forge_token
    
    # Clear token variable from memory for extra security
    del forge_token
    
    print("✓ Token received and will be used for this session only.\n")
    
    return encoder_config


def load_target_transform_from_checkpoint(checkpoint_path: str):
    """
    Load and recreate target transform from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        TargetTransform object or None if no transform in checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    transform_stats = checkpoint.get('target_transform_stats', None)
    
    if transform_stats is None:
        logger.warning("No target transform found in checkpoint - predictions will be unnormalized")
        return None
    
    # Get transform type
    transform_name = transform_stats.get('name', 'identity')
    logger.info(f"Loading target transform: {transform_name}")
    
    # Handle composed transforms
    if 'composed' in transform_name:
        # For composed transforms, we need to recreate each component
        from src.data.target_transforms import ComposedTransform, LogTransform, ZScoreTransform
        
        # Extract component transforms from stats
        transforms = []
        i = 0
        while f'transform_{i}_name' in transform_stats:
            comp_name = transform_stats[f'transform_{i}_name']
            if comp_name == 'log_natural':
                t = LogTransform(base='natural')
                t.offset = transform_stats[f'transform_{i}_offset']
                t.base = transform_stats[f'transform_{i}_base']
                t.auto_offset = transform_stats[f'transform_{i}_auto_offset']
                t.fitted = True
            elif comp_name == 'z_score':
                t = ZScoreTransform()
                t.mean = transform_stats[f'transform_{i}_mean']
                t.std = transform_stats[f'transform_{i}_std']
                t.fitted = True
            else:
                raise ValueError(f"Unknown component transform: {comp_name}")
            transforms.append(t)
            i += 1
        
        transform = ComposedTransform(tuple(transforms))
        transform.fitted = True
    
    else:
        # Simple transform
        transform = create_transform(transform_name)
        
        # Set fitted parameters
        if 'mean' in transform_stats:
            transform.mean = transform_stats['mean']
        if 'std' in transform_stats:
            transform.std = transform_stats['std']
        if 'min' in transform_stats:
            transform.min = transform_stats['min']
        if 'max' in transform_stats:
            transform.max = transform_stats['max']
        if 'offset' in transform_stats:
            transform.offset = transform_stats['offset']
        if 'base' in transform_stats:
            transform.base = transform_stats['base']
        
        transform.fitted = True
    
    logger.info(f"✓ Target transform loaded: {transform.get_stats()}")
    return transform


def load_antibody_features_stats_from_checkpoint(checkpoint_path: str):
    """
    Load antibody features normalization stats from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Tuple of (antibody_features_mean, antibody_features_std) or (None, None)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ab_mean = checkpoint.get('antibody_features_mean', None)
    ab_std = checkpoint.get('antibody_features_std', None)
    
    if ab_mean is not None and ab_std is not None:
        logger.info(f"✓ Loaded antibody features normalization stats (dim={len(ab_mean)})")
    else:
        logger.warning(
            "⚠️  Antibody features normalization stats not found in checkpoint. "
            "This is expected for models trained before this feature was added. "
            "Predictions will proceed without feature normalization."
        )
    
    return ab_mean, ab_std


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_precomputed_embeddings: bool = False
) -> DevelopabilityModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Config dict used for training (already merged with property-specific)
        device: Device to load model on
        use_precomputed_embeddings: If True, loads with strict=False to allow missing encoder weights
    
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint on CPU first to inspect state dict, then move to device
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Auto-detect whether checkpoint contains encoder weights.
    # Frozen encoders are excluded from checkpoints to save storage; they will be
    # re-initialized from the ESM package at predict time with the correct weights.
    state_dict = checkpoint.get("state_dict", {})
    has_encoder_weights = any(k.startswith("encoder.encoders.") and ".client." in k for k in state_dict)
    strict_load = has_encoder_weights
    if not strict_load:
        logger.info("Checkpoint has no encoder weights (frozen encoder — will reload from ESM package)")

    # Build encoder config
    encoder_config = config["encoder"].copy()
    
    # Get encoder_type and set encoder_types in encoder_config (exactly like train.py)
    encoder_type = config.get("encoder_type")
    
    # Handle missing encoder_type (for older models or when using precomputed embeddings only)
    if not encoder_type:
        # Try loading encoder_type from checkpoint (older models may have it stored there)
        if "config" in checkpoint and "encoder_type" in checkpoint["config"]:
            encoder_type = checkpoint["config"]["encoder_type"]
            logger.info(f"✓ Loaded encoder_type from checkpoint: {encoder_type}")
        elif use_precomputed_embeddings:
            # Use a dummy encoder since weights won't be loaded anyway
            logger.warning("⚠️  No encoder_type in config or checkpoint. Using dummy encoder (precomputed embeddings mode).")
            encoder_types = ["esmc_600m"]  # Dummy value, won't be used
        else:
            raise ValueError(
                "encoder_type not found in config or checkpoint, and precomputed embeddings not enabled. "
                "Cannot load model without encoder configuration."
            )
    
    if encoder_type:
        encoder_types = [e.strip() for e in encoder_type.split('+')]
    
    encoder_config["encoder_types"] = encoder_types
    encoder_config["pooling"] = "none"  # Decoder handles pooling
    
    # Check if forge token is needed and prompt if not available
    encoder_config = prompt_for_forge_token_if_needed(encoder_config)
    
    # Build decoder config (already merged with property-specific)
    decoder_type = config["decoder"]["type"]
    decoder_config = config["decoder"].get(decoder_type, {}).copy()
    
    # WORKAROUND: Saved configs are incomplete (train.py saves unmerged config)
    # Load critical architectural params from checkpoint if available
    if "config" in checkpoint and "decoder" in checkpoint["config"]:
        checkpoint_decoder = checkpoint["config"]["decoder"]
        if decoder_type in checkpoint_decoder:
            checkpoint_decoder_config = checkpoint_decoder[decoder_type]
            # Override with checkpoint values for architectural params
            # (these must match exactly for weight loading to work)
            critical_params = ["hidden_dim", "n_heads", "n_layers", "n_output_layers", 
                              "pooling_strategy", "attention_strategy", "use_learnable_chain_fusion"]
            for param in critical_params:
                if param in checkpoint_decoder_config:
                    decoder_config[param] = checkpoint_decoder_config[param]
            logger.info(f"✓ Loaded decoder architecture from checkpoint: hidden_dim={decoder_config.get('hidden_dim')}, "
                       f"n_layers={decoder_config.get('n_layers')}, n_heads={decoder_config.get('n_heads')}")
    
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

    # Get training params (already merged with property-specific)
    use_xavier_init = config["training"]["use_xavier_init"]
    xavier_gain = config["training"]["xavier_gain"]
    
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
        use_xavier_init=use_xavier_init,
        xavier_gain=xavier_gain,
    )
    
    # SECURITY: Immediately delete forge token from config after model initialization
    # to prevent it from being logged or exposed in error messages
    if "encoder_configs" in encoder_config:
        if "esmc_6b" in encoder_config["encoder_configs"]:
            if "forge_token" in encoder_config["encoder_configs"]["esmc_6b"]:
                del encoder_config["encoder_configs"]["esmc_6b"]["forge_token"]
    
    # Load state dict — use strict=False when encoder weights are absent (frozen encoder,
    # weights come from ESM package initialization above)
    if not strict_load:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
        non_encoder_missing = [k for k in missing_keys if not k.startswith('encoder.')]
        if non_encoder_missing:
            logger.warning(f"Missing non-encoder keys: {non_encoder_missing[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
        logger.info(f"✓ Model loaded (decoder weights + ESM encoder from package, {len(missing_keys)} checkpoint keys skipped)")
    else:
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"✓ Model loaded successfully")
    
    model.to(device)
    model.eval()
    
    return model


def _get_embedding_dims_from_pt(pt_path: Path):
    """Read the .pt cache to find max VH and VL sequence lengths used during training."""
    if not pt_path.exists():
        return None, None
    cache = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    vh_max = max((v.shape[0] for k, v in cache.items() if k.startswith("VH:")), default=None)
    vl_max = max((v.shape[0] for k, v in cache.items() if k.startswith("VL:")), default=None)
    return vh_max, vl_max


def _find_training_embedding_dims(config: dict):
    """Find the .pt cache used during training and return max VH/VL seq lengths."""
    from src.utils.precompute_utils import get_embeddings_config as _get_emb_cfg
    data_path = config.get("data", {}).get("gdpa1_path")
    encoder_type = config.get("encoder_type", "")
    if not data_path or not encoder_type:
        return None, None
    use_full_chain = "fullchain" in encoder_type
    base_enc = encoder_type.replace("_fullchain", "")
    emb_cfg = _get_emb_cfg(data_path, base_enc, use_full_chain=use_full_chain)
    pt_path = emb_cfg.get("precomputed_embeddings_path")
    if not pt_path:
        return None, None
    return _get_embedding_dims_from_pt(Path(pt_path))


def _pad_to(emb: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad a (seq_len, hidden_dim) embedding to (target_len, hidden_dim)."""
    if emb.shape[0] >= target_len:
        return emb[:target_len]
    pad = torch.zeros(target_len - emb.shape[0], emb.shape[1], dtype=emb.dtype, device=emb.device)
    return torch.cat([emb, pad], dim=0)


def _encode_and_pad(encoder, vh_seqs, vl_seqs, device,
                    vh_max_len: int = None, vl_max_len: int = None):
    """Encode one antibody at a time (like precompute_embeddings.py), pad to training dims,
    then batch. Mirrors the GDPa1DataModule precomputed embeddings path used during training."""
    vh_embs, vl_embs = [], []
    for vh, vl in zip(vh_seqs or [], vl_seqs or []):
        vh_emb, vl_emb = encoder([vh], [vl])      # each: (1, seq_len, hidden_dim)
        if vh_emb is not None:
            e = vh_emb.squeeze(0).cpu()            # (seq_len, hidden_dim)
            if vh_max_len is not None:
                e = _pad_to(e, vh_max_len)
            vh_embs.append(e)
        if vl_emb is not None:
            e = vl_emb.squeeze(0).cpu()
            if vl_max_len is not None:
                e = _pad_to(e, vl_max_len)
            vl_embs.append(e)

    def _stack(embs, max_len):
        if not embs:
            return None
        target = max_len if max_len is not None else max(e.shape[0] for e in embs)
        return torch.stack([_pad_to(e, target) for e in embs]).to(device)

    return _stack(vh_embs, vh_max_len), _stack(vl_embs, vl_max_len)


def predict_with_model(
    model: DevelopabilityModel,
    dataset: GDPa1Dataset,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_transform=None,
    use_tta: bool = False,
    tta_n_augmentations: int = 5,
    tta_noise_std: float = 0.01,
    vh_max_len: int = None,
    vl_max_len: int = None,
) -> np.ndarray:
    """
    Make predictions with a single model.
    
    Args:
        model: Trained model
        dataset: Dataset to predict on
        batch_size: Batch size for inference
        device: Device to run inference on
        target_transform: Target transform to apply inverse (optional)
        use_tta: Use test-time augmentation (default: False)
        tta_n_augmentations: Number of TTA augmentations (default: 5)
        tta_noise_std: Std of noise for TTA (default: 0.01)
    
    Returns:
        Array of predictions (N,) in original scale
        If use_tta=True, returns tuple (mean_predictions, std_predictions)
    """
    from torch.utils.data import DataLoader
    from src.utils.tta import predict_batch_with_tta
    
    # Define custom collate function to handle variable-length embeddings
    def collate_fn(batch):
        """Custom collate function to handle cached embeddings (now pooled to fixed size)."""
        heavy_seqs = [item["heavy_sequence"] for item in batch]
        light_seqs = [item["light_sequence"] for item in batch]
        targets = torch.stack([item["target"] for item in batch])
        antibody_ids = [item["antibody_id"] for item in batch]

        result = {
            "heavy_sequences": heavy_seqs,
            "light_sequences": light_seqs,
            "targets": targets,
            "antibody_ids": antibody_ids,
        }

        # Include cached embeddings if available (now fixed-size after pooling)
        if "heavy_embedding" in batch[0]:
            heavy_embs = torch.stack([item["heavy_embedding"] for item in batch])
            light_embs = torch.stack([item["light_embedding"] for item in batch])
            result["heavy_embeddings"] = heavy_embs
            result["light_embeddings"] = light_embs

        # Include antibody features if available
        if "antibody_features" in batch[0]:
            antibody_features = torch.stack([item["antibody_features"] for item in batch])
            result["antibody_features"] = antibody_features

        return result
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        collate_fn=collate_fn
    )
    
    predictions = []
    tta_stds = [] if use_tta else None
    
    with torch.no_grad():
        for batch in dataloader:
            # Check if using precomputed embeddings (collate_fn uses plural keys)
            use_precomputed = "heavy_embeddings" in batch
            
            # Get sequences (collate_fn outputs heavy_sequences/light_sequences as lists)
            vh_seq = batch.get("heavy_sequences", None)
            vl_seq = batch.get("light_sequences", None)
            
            # Get antibody features if available
            antibody_features = batch.get("antibody_features", None)
            if antibody_features is not None:
                antibody_features = antibody_features.to(device)
            
            if use_tta:
                # Use TTA for predictions
                # Handle precomputed embeddings
                if use_precomputed:
                    vh_emb = batch["heavy_embeddings"].to(device)
                    vl_emb = batch["light_embeddings"].to(device)
                    # Concatenate pooled embeddings
                    embeddings = torch.cat([vh_emb, vl_emb], dim=-1)
                    batch_dict = {'embeddings': embeddings}
                    if antibody_features is not None:
                        batch_dict['antibody_features'] = antibody_features
                else:
                    # Get embeddings from encoder
                    with torch.no_grad():
                        embeddings = model.encoder(vh_seq, vl_seq)
                        embeddings = embeddings.to(device)
                    batch_dict = {'embeddings': embeddings}
                    if antibody_features is not None:
                        batch_dict['antibody_features'] = antibody_features
                
                # Apply TTA
                mean_pred, std_pred = predict_batch_with_tta(
                    model=model,
                    batch=batch_dict,
                    n_augmentations=tta_n_augmentations,
                    noise_std=tta_noise_std,
                    target_transform=target_transform,
                    device=device
                )
                predictions.append(mean_pred)
                tta_stds.append(std_pred)
            else:
                # Standard prediction (no TTA)
                # Handle precomputed embeddings
                if use_precomputed:
                    vh_emb = batch["heavy_embeddings"].to(device)
                    vl_emb = batch["light_embeddings"].to(device)
                    # Forward with precomputed embeddings (using keyword args to match signature)
                    pred = model(
                        heavy_sequences=vh_seq,
                        light_sequences=vl_seq,
                        heavy_embeddings=vh_emb,
                        light_embeddings=vl_emb,
                        antibody_features=antibody_features
                    )
                else:
                    # Encode one sequence at a time (same as precompute_embeddings.py),
                    # then pad and batch before passing to the decoder.
                    logger.debug(f"Encoding {len(vh_seq or vl_seq)} sequence(s) on-the-fly with ESM")
                    vh_emb_batch, vl_emb_batch = _encode_and_pad(
                        model.encoder, vh_seq, vl_seq, device,
                        vh_max_len=vh_max_len, vl_max_len=vl_max_len
                    )
                    pred = model(
                        heavy_embeddings=vh_emb_batch,
                        light_embeddings=vl_emb_batch,
                        antibody_features=antibody_features
                    )
                
                predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0).flatten()
    
    # Apply inverse transform if not using TTA (TTA already applies it)
    if target_transform is not None and not use_tta:
        predictions = target_transform.inverse_transform(predictions)
        logger.debug(f"Applied inverse transform to {len(predictions)} predictions")
    
    if use_tta:
        tta_stds = np.concatenate(tta_stds, axis=0).flatten()
        return predictions, tta_stds
    else:
        return predictions


def predict_cv_ensemble(
    model_dir: Path,
    config: dict,
    dataset: GDPa1Dataset,
    property_name: str,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_precomputed_embeddings: bool = False
) -> Dict[str, np.ndarray]:
    """
    Make predictions using CV ensemble (all 5 folds).
    
    Args:
        model_dir: Directory containing fold checkpoints
        config: Config dict
        dataset: Dataset to predict on
        property_name: Name of property being predicted
        batch_size: Batch size for inference
        device: Device to run inference on
        use_precomputed_embeddings: If True, loads models without encoder weights
    
    Returns:
        Dict with keys:
        - 'mean': Mean prediction across folds
        - 'std': Standard deviation across folds
        - 'fold1', 'fold2', ...: Individual fold predictions
    """
    logger.info(f"\nPredicting {property_name} with CV ensemble")
    logger.info(f"Model directory: {model_dir}")
    
    # Find training .pt file to get the max seq lengths used during training
    vh_max_len, vl_max_len = _find_training_embedding_dims(config)
    if vh_max_len:
        logger.info(f"Padding embeddings to training dims: VH={vh_max_len}, VL={vl_max_len}")

    fold_predictions = []
    target_transform = None  # Will be loaded from first checkpoint

    for fold_idx in range(5):  # 0-based: 0,1,2,3,4
        checkpoint_path = model_dir / f"fold{fold_idx}.ckpt"

        if not checkpoint_path.exists():
            logger.warning(f"⚠️  Fold {fold_idx+1} checkpoint not found: {checkpoint_path}")
            continue

        logger.info(f"Predicting with fold {fold_idx+1}...")

        # Load transform from first checkpoint only (same for all folds)
        if target_transform is None:
            target_transform = load_target_transform_from_checkpoint(
                str(checkpoint_path)
            )

        model = load_model_from_checkpoint(
            str(checkpoint_path), config, device, use_precomputed_embeddings
        )
        predictions = predict_with_model(
            model, dataset, batch_size, device, target_transform,
            vh_max_len=vh_max_len, vl_max_len=vl_max_len,
        )
        fold_predictions.append(predictions)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    if len(fold_predictions) == 0:
        raise ValueError(f"No fold checkpoints found in {model_dir}")
    
    # Stack predictions
    fold_predictions = np.stack(fold_predictions, axis=0)  # (n_folds, n_samples)
    
    results = {
        "mean": np.mean(fold_predictions, axis=0),
        "std": np.std(fold_predictions, axis=0),
    }
    
    # Add individual fold predictions
    for fold_idx in range(fold_predictions.shape[0]):
        results[f"fold{fold_idx}"] = fold_predictions[fold_idx]
    
    logger.info(f"✓ CV ensemble prediction complete ({len(fold_predictions)} folds)")
    
    return results


def predict_final_model(
    model_dir: Path,
    config: dict,
    dataset: GDPa1Dataset,
    property_name: str,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_tta: bool = False,
    tta_n_augmentations: int = 5,
    tta_noise_std: float = 0.01,
    use_precomputed_embeddings: bool = False,
    checkpoint_name: str = "final.ckpt",
):
    """
    Make predictions using a single model checkpoint.

    Args:
        model_dir: Directory containing the checkpoint
        config: Config dict
        dataset: Dataset to predict on
        property_name: Name of property being predicted
        batch_size: Batch size for inference
        device: Device to run inference on
        use_tta: Use test-time augmentation
        tta_n_augmentations: Number of TTA augmentations
        tta_noise_std: Std of noise for TTA
        use_precomputed_embeddings: If True, loads model without encoder weights
        checkpoint_name: Checkpoint filename to load (default: "final.ckpt")

    Returns:
        Array of predictions (N,) or tuple (predictions, stds) if TTA enabled
    """
    logger.info(f"\nPredicting {property_name} with {checkpoint_name}")
    logger.info(f"Model directory: {model_dir}")
    if use_tta:
        logger.info(f"TTA enabled: {tta_n_augmentations} augmentations")

    checkpoint_path = model_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load target transform
    target_transform = load_target_transform_from_checkpoint(
        str(checkpoint_path)
    )
    
    model = load_model_from_checkpoint(
        str(checkpoint_path), config, device, use_precomputed_embeddings
    )

    # Find training .pt file to get the max seq lengths used during training,
    # so on-the-fly encoded sequences are padded to the exact same shape.
    vh_max_len, vl_max_len = _find_training_embedding_dims(config)
    if vh_max_len:
        logger.info(f"Padding embeddings to training dims: VH={vh_max_len}, VL={vl_max_len}")
    else:
        logger.info("Training .pt dims not found — will pad to batch max length")

    results = predict_with_model(
        model, dataset, batch_size, device, target_transform,
        use_tta=use_tta,
        tta_n_augmentations=tta_n_augmentations,
        tta_noise_std=tta_noise_std,
        vh_max_len=vh_max_len,
        vl_max_len=vl_max_len,
    )
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    logger.info(f"✓ Final model prediction complete")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions with trained developability models"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file: CSV with antibody sequences or .pt cache file from precompute_prediction_cache.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model directory (e.g., models/HIC_51318e6b)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Paths to multiple model directories (alternative to --model)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file with predictions",
    )
    parser.add_argument(
        "--default-subtype",
        type=str,
        nargs=2,
        metavar=("HC_SUBTYPE", "LC_SUBTYPE"),
        help=(
            "Default heavy and light chain isotypes to use when hc_subtype/lc_subtype columns are "
            "absent from the input CSV. Required for full-chain models (HIC, Titer, Tm2, PR_CHO) "
            "if the CSV does not have those columns. "
            "Example: --default-subtype IgG1 Kappa"
        ),
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use CV ensemble (mean of 5 folds) instead of final model (provides uncertainty estimates)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Use a specific CV fold checkpoint instead of final model (0-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )
    parser.add_argument(
        "--use-tta",
        action="store_true",
        help="Use test-time augmentation for robustness and uncertainty estimation",
    )
    parser.add_argument(
        "--tta-n-augmentations",
        type=int,
        default=5,
        help="Number of TTA augmentations (default: 5)",
    )
    parser.add_argument(
        "--tta-noise-std",
        type=float,
        default=0.01,
        help="Std of Gaussian noise for TTA (default: 0.01)",
    )
    args = parser.parse_args()
    
    # Detect input type from extension
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    
    is_cache_file = input_path.suffix == '.pt'
    
    # Validate inputs
    if args.model and args.models:
        parser.error("Cannot specify both --model and --models")
    if not args.model and not args.models:
        parser.error("Must specify either --model or --models")
    if args.use_cv and args.fold is not None:
        parser.error("Cannot specify both --use-cv and --fold")
    
    # Get list of model directories
    if args.model:
        model_dirs = [Path(args.model)]
    else:
        model_dirs = [Path(m) for m in args.models]
    
    # Validate model directories
    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logger.info("="*60)
    logger.info("Antibody Developability Prediction")
    logger.info("="*60)
    if is_cache_file:
        logger.info(f"Precomputed cache: {args.input}")
    else:
        logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Number of models: {len(model_dirs)}")
    if args.use_cv:
        mode_str = "CV ensemble (all folds)"
    elif args.fold is not None:
        mode_str = f"CV fold {args.fold}"
    else:
        mode_str = "Final model"
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Load input data
    if is_cache_file:
        logger.info(f"\nLoading precomputed cache from: {args.input}")
        cache = torch.load(args.input, map_location='cpu', weights_only=False)
        
        # Extract sequences DataFrame from cache
        df = pd.DataFrame(cache['sequences'])
        logger.info(f"Loaded {len(df)} sequences from cache")
        
        # Log cache metadata (check for discrepancies)
        cache_metadata = cache['metadata']
        logger.info(f"Cache created for {cache_metadata.get('num_sequences')} sequences")
        logger.info(f"Cache encoder types: {cache_metadata.get('encoder_types')}")
        logger.info(f"Cache has antibody features: {cache_metadata.get('has_antibody_features')}")
        
        # Warn if sequence count mismatch
        if len(df) != cache_metadata.get('num_sequences'):
            logger.warning(f"⚠️  Sequence count mismatch: df has {len(df)}, cache metadata says {cache_metadata.get('num_sequences')}")
        
        # Extract precomputed data
        precomputed_embeddings = cache.get('embeddings', None)
        precomputed_antibody_features = cache.get('antibody_features', None)
        
        if precomputed_embeddings:
            logger.info(f"✓ Precomputed embeddings: {len(precomputed_embeddings)} pairs")
        if precomputed_antibody_features:
            logger.info(f"✓ Precomputed antibody features: {len(precomputed_antibody_features)} pairs")
    else:
        logger.info(f"\nLoading input data from: {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} sequences")
        precomputed_embeddings = None
        precomputed_antibody_features = None
    
    # Validate required columns
    required_cols = ["vh_protein_sequence", "vl_protein_sequence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create output dataframe
    output_df = df.copy()
    
    # Process each model
    for model_dir in tqdm(model_dirs, desc="Models"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_dir}")
        logger.info(f"{'='*60}")
        
        # Load config
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            logger.warning(f"⚠️  Config not found: {config_path}, skipping")
            continue
        
        config = load_config(str(config_path))
        
        # Extract property name from model directory
        property_file = model_dir / "property.txt"
        if property_file.exists():
            property_name = property_file.read_text().strip()
        else:
            # Fallback for legacy models: first token of dir name (breaks on e.g. PR_CHO)
            property_name = model_dir.name.split("_")[0]
            logger.warning(f"property.txt not found in {model_dir.name} — guessed '{property_name}' from dir name")
        
        logger.info(f"Property: {property_name}")
        
        # Merge base config with property-specific overrides (exactly like train.py)
        config = get_property_config(config, property_name)
        
        # Extract encoder type for this property (to filter precomputed embeddings)
        encoder_type = config.get("encoder_type")
        if not encoder_type:
            # WORKAROUND: Saved model configs are incomplete (bug in train.py saves unmerged config)
            # The property-specific encoder_type was never merged to top level during training
            # Try to recover it from checkpoint or cache
            logger.warning(f"⚠️  No encoder_type found in config for {property_name}")
            
            # Try to infer encoder_type from checkpoint or cache
            if precomputed_embeddings is not None:
                # Try loading from checkpoint first
                checkpoint_path = model_dir / "fold0.ckpt"
                if checkpoint_path.exists():
                    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
                    if "config" in ckpt and "encoder_type" in ckpt["config"]:
                        encoder_type = ckpt["config"]["encoder_type"]
                        logger.info(f"✓ Inferred encoder_type from checkpoint: {encoder_type}")
                    del ckpt
                
                # If still not found, use first encoder from cache
                if not encoder_type and precomputed_embeddings:
                    # Get first available encoder from cache
                    sample_pair = next(iter(precomputed_embeddings.values()))
                    available_encoders = list(sample_pair.keys())
                    if available_encoders:
                        encoder_type = available_encoders[0]
                        logger.info(f"✓ Using first available encoder from cache: {encoder_type}")
        
        # Convert precomputed embeddings to format expected by GDPa1Dataset
        # Cache format: cache['embeddings'][pair_id][encoder_type]['vh'/'vl']
        # Expected format: precomputed_embeddings[(heavy_seq, light_seq)] = (heavy_emb, light_emb)
        # Note: Must pool per-residue embeddings to match training behavior
        property_precomputed_embeddings = None
        if precomputed_embeddings is not None and encoder_type:
            logger.info(f"Converting precomputed embeddings for encoder: {encoder_type}")
            
            # Get pooling strategy from decoder config
            decoder_config = config.get("decoder", {})
            decoder_type = decoder_config.get("type", "attention")
            if decoder_type in decoder_config:
                pooling = decoder_config[decoder_type].get("pooling_strategy", "mean")
            else:
                pooling = "mean"
            logger.info(f"Using pooling strategy: {pooling}")
            
            # Warn once if attention pooling is requested (will use mean for cache)
            actual_pooling = pooling
            if pooling == "attention":
                logger.warning("Attention pooling will be done by decoder, using mean for cache")
                actual_pooling = "mean"
            
            property_precomputed_embeddings = {}
            
            for idx, row in df.iterrows():
                heavy_seq = row["vh_protein_sequence"]
                light_seq = row["vl_protein_sequence"]
                pair_id = f"pair_{idx}"
                
                # Check if embeddings exist for this pair and encoder
                if pair_id in precomputed_embeddings:
                    if encoder_type in precomputed_embeddings[pair_id]:
                        enc_data = precomputed_embeddings[pair_id][encoder_type]
                        # Pool per-residue embeddings to fixed-size vectors
                        # Shape: (seq_len, hidden_dim) -> (hidden_dim,)
                        heavy_emb = enc_data['vh']
                        light_emb = enc_data['vl']
                        
                        # Apply pooling
                        if pooling == "mean":
                            heavy_emb_pooled = heavy_emb.mean(dim=0)
                            light_emb_pooled = light_emb.mean(dim=0)
                        elif pooling == "max":
                            heavy_emb_pooled = heavy_emb.max(dim=0)[0]
                            light_emb_pooled = light_emb.max(dim=0)[0]
                        elif pooling == "cls":
                            heavy_emb_pooled = heavy_emb[0]
                            light_emb_pooled = light_emb[0]
                        elif pooling == "attention":
                            # Attention pooling needs to be done by decoder, use mean for cache
                            heavy_emb_pooled = heavy_emb.mean(dim=0)
                            light_emb_pooled = light_emb.mean(dim=0)
                        else:
                            heavy_emb_pooled = heavy_emb.mean(dim=0)
                            light_emb_pooled = light_emb.mean(dim=0)
                        
                        property_precomputed_embeddings[(heavy_seq, light_seq)] = (heavy_emb_pooled, light_emb_pooled)
                    else:
                        logger.warning(f"⚠️  Missing encoder {encoder_type} for {pair_id}")
                else:
                    logger.warning(f"⚠️  Missing embeddings for {pair_id}")
            
            logger.info(f"✓ Converted {len(property_precomputed_embeddings)} embedding pairs (pooled with {actual_pooling})")
        
        # Load antibody features normalization stats from checkpoint
        # Use final.ckpt if exists, otherwise fold0.ckpt
        checkpoint_path = model_dir / "final.ckpt"
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / "fold0.ckpt"
        
        antibody_features_mean = None
        antibody_features_std = None
        if checkpoint_path.exists():
            antibody_features_mean, antibody_features_std = \
                load_antibody_features_stats_from_checkpoint(str(checkpoint_path))
        
        # Determine sequence columns — full-chain models need HC/LC, not VH/VL
        use_full_chain = config.get("data", {}).get("use_full_chain", False)
        heavy_col = "vh_protein_sequence"
        light_col = "vl_protein_sequence"
        predict_df = df.copy()

        if use_full_chain:
            if "hc_protein_sequence" in predict_df.columns and "lc_protein_sequence" in predict_df.columns:
                logger.info("Full-chain sequences found in input CSV — using hc/lc_protein_sequence directly")
                heavy_col = "hc_protein_sequence"
                light_col = "lc_protein_sequence"
            else:
                # Fill subtype columns from --default-subtype if not present in CSV
                if "hc_subtype" not in predict_df.columns or "lc_subtype" not in predict_df.columns:
                    if args.default_subtype is None:
                        raise ValueError(
                            f"Model '{property_name}' uses full-chain sequences "
                            f"(hc_protein_sequence / lc_protein_sequence) but neither "
                            f"'hc_subtype'/'lc_subtype' columns were found in the input CSV "
                            f"nor was --default-subtype provided.\n"
                            f"Add hc_subtype and lc_subtype columns to your CSV (e.g. IgG1, Kappa), "
                            f"or pass --default-subtype IgG1 Kappa on the command line."
                        )
                    default_hc, default_lc = args.default_subtype
                    logger.warning(
                        f"⚠  Using --default-subtype ({default_hc}, {default_lc}) for all rows. "
                        f"Add hc_subtype/lc_subtype columns to your CSV for per-row control."
                    )
                    if "hc_subtype" not in predict_df.columns:
                        predict_df["hc_subtype"] = default_hc
                    if "lc_subtype" not in predict_df.columns:
                        predict_df["lc_subtype"] = default_lc

                # Reconstruct full-chain sequences from VH/VL + constant regions
                from src.data.gdpa1_datamodule import (
                    extract_constant_regions_from_reference,
                    reconstruct_full_chains,
                )
                reference_csv = config.get("data", {}).get("gdpa1_path", "inputs/GDPa1_complete.csv")
                logger.info(f"Reconstructing HC/LC full-chain sequences using constant regions from {reference_csv}")
                constant_regions = extract_constant_regions_from_reference(reference_csv)
                predict_df = reconstruct_full_chains(predict_df, constant_regions, inplace=False)
                heavy_col = "hc_protein_sequence"
                light_col = "lc_protein_sequence"

        # Create dataset for prediction (use dummy targets)
        # Add a dummy target column for prediction mode
        df_with_dummy = predict_df
        df_with_dummy["_dummy_target"] = 0.0

        dataset = GDPa1Dataset(
            data=df_with_dummy,
            heavy_col=heavy_col,
            light_col=light_col,
            target_col="_dummy_target",
            precomputed_embeddings=property_precomputed_embeddings,
            precomputed_antibody_features=precomputed_antibody_features,
            antibody_features_config=config.get("antibody_features"),
            antibody_features_mean=antibody_features_mean,
            antibody_features_std=antibody_features_std,
        )
        
        # Make predictions
        if args.use_cv:
            # Use CV ensemble (mean of 5 folds)
            results = predict_cv_ensemble(
                model_dir, config, dataset, property_name,
                args.batch_size, args.device,
                use_precomputed_embeddings=property_precomputed_embeddings is not None
            )

            # Add predictions to output
            output_df[f"{property_name}_prediction"] = results["mean"]
            output_df[f"{property_name}_std"] = results["std"]

            # Add individual fold predictions
            for fold_key in sorted([k for k in results.keys() if k.startswith("fold")]):
                output_df[f"{property_name}_{fold_key}"] = results[fold_key]
        elif args.fold is not None:
            # Use a specific CV fold checkpoint
            checkpoint_path = model_dir / f"fold{args.fold}.ckpt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Fold {args.fold} checkpoint not found: {checkpoint_path}")
            results = predict_final_model(
                model_dir, config, dataset, property_name,
                args.batch_size, args.device,
                use_tta=args.use_tta,
                tta_n_augmentations=args.tta_n_augmentations,
                tta_noise_std=args.tta_noise_std,
                use_precomputed_embeddings=property_precomputed_embeddings is not None,
                checkpoint_name=f"fold{args.fold}.ckpt",
            )
            output_df[f"{property_name}_prediction"] = results
        else:
            # Use final model (default)
            results = predict_final_model(
                model_dir, config, dataset, property_name,
                args.batch_size, args.device,
                use_tta=args.use_tta,
                tta_n_augmentations=args.tta_n_augmentations,
                tta_noise_std=args.tta_noise_std,
                use_precomputed_embeddings=property_precomputed_embeddings is not None
            )
            
            if args.use_tta:
                # Unpack predictions and stds from TTA
                predictions, tta_stds = results
                output_df[f"{property_name}_prediction"] = predictions
                output_df[f"{property_name}_tta_std"] = tta_stds
            else:
                output_df[f"{property_name}_prediction"] = results
    
    # Save predictions
    logger.info(f"\n{'='*60}")
    logger.info("Saving predictions")
    logger.info(f"{'='*60}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Predictions saved to: {output_path}")
    logger.info(f"Output shape: {output_df.shape}")
    
    # Print summary
    prediction_cols = [col for col in output_df.columns if col.endswith("_prediction")]
    if prediction_cols:
        logger.info(f"\nPrediction columns: {len(prediction_cols)}")
        for col in prediction_cols:
            values = output_df[col].values
            logger.info(f"  {col}:")
            logger.info(f"    Mean: {np.mean(values):.4f}")
            logger.info(f"    Std:  {np.std(values):.4f}")
            logger.info(f"    Min:  {np.min(values):.4f}")
            logger.info(f"    Max:  {np.max(values):.4f}")
    
    logger.info("\n✓ Prediction complete!")


if __name__ == "__main__":
    main()
