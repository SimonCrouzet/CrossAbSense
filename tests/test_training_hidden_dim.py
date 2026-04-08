#!/usr/bin/env python3
"""Test FULL training with hidden_dim=1024 to verify it works"""

import os
os.environ["WANDB_MODE"] = "disabled"

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data import GDPa1DataModule
from src.models import DevelopabilityModel
from src.utils import load_config, get_embeddings_config

def test_training_with_hidden_dim(hidden_dim=1024):
    """Test actual training with specific hidden_dim"""

    print(f"\n{'='*60}")
    print(f"FULL TRAINING TEST: hidden_dim={hidden_dim}")
    print(f"{'='*60}\n")

    # Load config
    config = load_config("src/config/default_config.yaml")

    # Build encoder config
    encoder_config = config["encoder"].copy()
    encoder_config["encoder_types"] = ["esmc"]
    encoder_config["pooling"] = "mean"

    # Auto-detect precomputed embeddings
    source_csv = config["data"]["gdpa1_path"]
    embeddings_config = get_embeddings_config(source_csv, encoder_type="esmc")
    if embeddings_config:
        encoder_config["encoder_configs"]["esmc"].update(embeddings_config)
        print(f"✓ Using precomputed embeddings")

    # Build decoder config with specific hidden_dim
    decoder_config = {
        "n_heads": 4,
        "hidden_dim": hidden_dim,
        "n_layers": 2,
        "dropout": 0.2,
        "pooling_strategy": "sliced_wasserstein",
        "attention_strategy": "bidirectional_cross"
    }

    print(f"Decoder config: {decoder_config}\n")

    # Data module (just fold 0, small subset for quick test)
    data_module = GDPa1DataModule(
        data_path=config["data"]["gdpa1_path"],
        target_property="HIC",
        batch_size=16,
        fold_idx=0,
        cv_fold_col=config["data"]["cv_fold_column"],
    )

    # Model
    freeze_encoder = encoder_config.pop("freeze_encoder", True)

    model = DevelopabilityModel(
        encoder_config=encoder_config,
        decoder_type="attention",
        decoder_config=decoder_config,
        learning_rate=0.001,
        weight_decay=0.01,
        scheduler="cosine",
        warmup_epochs=1,
        max_epochs=3,  # Just 3 epochs for testing
        loss_fn="huber",
        freeze_encoder=freeze_encoder,
    )

    print(f"✓ Model created")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Trainer with precision
    print(f"Creating trainer with precision='16-mixed'...\n")

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        precision="16-mixed",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=5
    )

    # Train
    print(f"Starting training (3 epochs)...\n")
    try:
        trainer.fit(model, data_module)
        print(f"\n{'='*60}")
        print(f"✅ TRAINING SUCCESSFUL with hidden_dim={hidden_dim}")
        print(f"{'='*60}")

        # Get final metrics
        if "val_spearman" in trainer.callback_metrics:
            val_spearman = trainer.callback_metrics["val_spearman"].item()
            val_pearson = trainer.callback_metrics.get("val_pearson", 0.0).item()
            print(f"✓ Final Val Spearman: {val_spearman:.4f}")
            print(f"✓ Final Val Pearson: {val_pearson:.4f}")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ TRAINING FAILED with hidden_dim={hidden_dim}")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test with hidden_dim=1024
    success_1024 = test_training_with_hidden_dim(1024)

    # Also test with None (native dimensions) for comparison
    print(f"\n\n{'='*60}")
    print(f"Now testing with hidden_dim=None (native)")
    print(f"{'='*60}\n")
    success_native = test_training_with_hidden_dim(None)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"hidden_dim=1024: {'✅ PASS' if success_1024 else '❌ FAIL'}")
    print(f"hidden_dim=None:  {'✅ PASS' if success_native else '❌ FAIL'}")
    print(f"{'='*60}\n")
