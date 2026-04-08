"""
End-to-end integration test for training pipeline.

Tests both single encoder and multi-encoder configurations with full training loops.
Also validates antibody features can be extracted alongside training.
"""

import os
import sys
import yaml
import torch
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.developability_model import DevelopabilityModel
from src.data.gdpa1_datamodule import GDPa1DataModule
from src.features.antibody_features import AntibodyFeatures


def test_single_encoder_training():
    """Test single encoder (esmc_6b) with attention decoder - full training loop."""
    print("\n" + "="*70)
    print("TEST 1: Single Encoder (esmc_6b) + AttentionDecoder - Training")
    print("="*70)

    # Load default config
    config_path = Path(__file__).parent.parent / "src" / "config" / "default_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Configure for single encoder test
    config["encoder"]["encoder_types"] = ["esmc_6b"]
    config["encoder"]["pooling"] = "mean"  # Use mean pooling for simpler testing
    config["decoder"]["type"] = "attention"
    config["training"]["finetune"]["max_epochs"] = 3
    config["training"]["finetune"]["batch_size"] = 8
    config["training"]["finetune"]["gradient_clip_val"] = 1.0

    # Enable antibody features (will auto-detect precomputed features)
    config["antibody_features"]["enabled"] = True
    print("📊 Antibody features ENABLED (auto-detecting precomputed)")

    # Use precomputed embeddings
    config["encoder"]["encoder_configs"]["esmc_6b"]["precomputed_embeddings_path"] = "inputs/embeddings/GDPa1_complete_esmc_6b_72626a47.pt"
    config["encoder"]["encoder_configs"]["esmc_6b"]["source_csv_path"] = config["data"]["gdpa1_path"]

    # Create data module
    print("\n📦 Creating data module...")
    datamodule = GDPa1DataModule(
        data_path=config["data"]["gdpa1_path"],
        target_property="HIC",
        batch_size=config["training"]["finetune"]["batch_size"],
        fold_idx=0,
        cv_fold_col=config["data"]["cv_fold_column"],
        antibody_features_config=config["antibody_features"],  # Auto-detects precomputed, falls back to on-the-fly
    )

    # Create model
    print("\n🏗️  Creating model...")
    # Extract freeze_encoder from encoder_config
    encoder_cfg = config["encoder"].copy()
    freeze_encoder = encoder_cfg.pop("freeze_encoder", True)

    # Auto-set antibody_features_dim in decoder config if features enabled
    decoder_cfg = config["decoder"][config["decoder"]["type"]].copy()
    if config["antibody_features"].get("enabled", False):
        from src.features.antibody_features import AntibodyFeatures
        # Only pass keys that AntibodyFeatures.__init__ accepts
        ab_init_keys = {"use_abnumber", "use_biophi", "use_scalop", "use_sequence_features", "cdr_definition", "cache_abnumber"}
        ab_features_config = {k: v for k, v in config["antibody_features"].items() if k in ab_init_keys}
        ab_features = AntibodyFeatures(**ab_features_config)
        antibody_features_dim = ab_features.get_feature_dim()
        decoder_cfg["antibody_features_dim"] = antibody_features_dim
        # Also set injection layer (default "second" falls back to "last" with n_output_layers=1)
        if "antibody_features_injection_layer" not in decoder_cfg:
            decoder_cfg["antibody_features_injection_layer"] = "last"
        print(f"🧬 Auto-configured decoder for antibody features:")
        print(f"   antibody_features_dim: {antibody_features_dim}")
        print(f"   injection_layer: {decoder_cfg['antibody_features_injection_layer']}")

    model = DevelopabilityModel(
        encoder_config=encoder_cfg,
        decoder_type=config["decoder"]["type"],
        decoder_config=decoder_cfg,
        learning_rate=config["training"]["finetune"]["learning_rate"],
        weight_decay=config["training"]["finetune"].get("weight_decay", 0.01),
        freeze_encoder=freeze_encoder,
        max_epochs=config["training"]["finetune"]["max_epochs"],
    )

    print(f"\n📊 Model architecture:")
    print(f"   Encoder: {config['encoder']['encoder_types']}")
    print(f"   Encoder output dim (per chain): {model.encoder.get_embedding_dim()}")
    print(f"   Decoder: {config['decoder']['type']}")
    print(f"   Decoder input_dim (per chain): {model.decoder.input_dim}")

    # Setup trainer
    print("\n🚂 Starting training for 3 epochs...")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision="bf16-mixed",  # Match precomputed embeddings dtype (bfloat16)
    )

    # Train
    trainer.fit(model, datamodule)

    print("\n✅ Single encoder training test PASSED!")
    return True


def test_multi_encoder_training():
    """Test multi-encoder (esmc_6b + prott5) with attention decoder - full training loop."""
    print("\n" + "="*70)
    print("TEST 2: Multi-Encoder (esmc_6b + prott5) + AttentionDecoder - Training")
    print("="*70)

    # Load default config
    config_path = Path(__file__).parent.parent / "src" / "config" / "default_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Configure for multi-encoder test
    config["encoder"]["encoder_types"] = ["esmc_6b", "prott5"]
    config["encoder"]["pooling"] = "mean"  # Use mean pooling for simpler testing
    config["encoder"]["fusion_strategy"] = "concat"
    config["decoder"]["type"] = "attention"
    config["training"]["finetune"]["max_epochs"] = 3
    config["training"]["finetune"]["batch_size"] = 8
    config["training"]["finetune"]["gradient_clip_val"] = 1.0

    # Enable antibody features (will auto-detect precomputed features)
    config["antibody_features"]["enabled"] = True
    print("📊 Antibody features ENABLED (auto-detecting precomputed)")

    # Use precomputed embeddings
    config["encoder"]["encoder_configs"]["esmc_6b"]["precomputed_embeddings_path"] = "inputs/embeddings/GDPa1_complete_esmc_6b_72626a47.pt"
    config["encoder"]["encoder_configs"]["esmc_6b"]["source_csv_path"] = config["data"]["gdpa1_path"]
    config["encoder"]["encoder_configs"]["prott5"]["precomputed_embeddings_path"] = "inputs/embeddings/GDPa1_complete_prott5_72626a47.pt"
    config["encoder"]["encoder_configs"]["prott5"]["source_csv_path"] = config["data"]["gdpa1_path"]

    # Create data module
    print("\n📦 Creating data module...")
    datamodule = GDPa1DataModule(
        data_path=config["data"]["gdpa1_path"],
        target_property="HIC",
        batch_size=config["training"]["finetune"]["batch_size"],
        fold_idx=0,
        cv_fold_col=config["data"]["cv_fold_column"],
        antibody_features_config=config["antibody_features"],  # Auto-detects precomputed, falls back to on-the-fly
    )

    # Create model
    print("\n🏗️  Creating model...")
    # Extract freeze_encoder from encoder_config
    encoder_cfg = config["encoder"].copy()
    freeze_encoder = encoder_cfg.pop("freeze_encoder", True)

    # Auto-set antibody_features_dim in decoder config if features enabled
    decoder_cfg = config["decoder"][config["decoder"]["type"]].copy()
    if config["antibody_features"].get("enabled", False):
        from src.features.antibody_features import AntibodyFeatures
        # Only pass keys that AntibodyFeatures.__init__ accepts
        ab_init_keys = {"use_abnumber", "use_biophi", "use_scalop", "use_sequence_features", "cdr_definition", "cache_abnumber"}
        ab_features_config = {k: v for k, v in config["antibody_features"].items() if k in ab_init_keys}
        ab_features = AntibodyFeatures(**ab_features_config)
        antibody_features_dim = ab_features.get_feature_dim()
        decoder_cfg["antibody_features_dim"] = antibody_features_dim
        # Also set injection layer (default "second" falls back to "last" with n_output_layers=1)
        if "antibody_features_injection_layer" not in decoder_cfg:
            decoder_cfg["antibody_features_injection_layer"] = "last"
        print(f"🧬 Auto-configured decoder for antibody features:")
        print(f"   antibody_features_dim: {antibody_features_dim}")
        print(f"   injection_layer: {decoder_cfg['antibody_features_injection_layer']}")

    model = DevelopabilityModel(
        encoder_config=encoder_cfg,
        decoder_type=config["decoder"]["type"],
        decoder_config=decoder_cfg,
        learning_rate=config["training"]["finetune"]["learning_rate"],
        weight_decay=config["training"]["finetune"].get("weight_decay", 0.01),
        freeze_encoder=freeze_encoder,
        max_epochs=config["training"]["finetune"]["max_epochs"],
    )

    print(f"\n📊 Model architecture:")
    print(f"   Encoders: {config['encoder']['encoder_types']}")
    print(f"   Fusion strategy: {config['encoder']['fusion_strategy']}")
    print(f"   Fused output dim (per chain): {model.encoder.get_embedding_dim()}")
    print(f"      (esmc_6b: 2560 + prott5: 1024 = 3584 per chain)")
    print(f"   Decoder: {config['decoder']['type']}")
    print(f"   Decoder input_dim (per chain): {model.decoder.input_dim}")

    # Setup trainer
    print("\n🚂 Starting training for 3 epochs...")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision="bf16-mixed",  # Match precomputed embeddings dtype (bfloat16)
    )

    # Train
    trainer.fit(model, datamodule)

    print("\n✅ Multi-encoder training test PASSED!")
    return True


if __name__ == "__main__":
    try:
        # Test single encoder
        test_single_encoder_training()

        # Test multi-encoder
        test_multi_encoder_training()

        print("\n" + "="*70)
        print("🎉 ALL TRAINING TESTS PASSED!")
        print("="*70)
        print("\nEnd-to-end training validated with:")
        print("  ✓ Single encoder (esmc_6b)")
        print("  ✓ Multi-encoder (esmc_6b + prott5)")
        print("  ✓ AttentionDecoder")
        print("  ✓ Full training loop (3 epochs each)")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
