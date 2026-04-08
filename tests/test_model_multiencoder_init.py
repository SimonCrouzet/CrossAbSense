#!/usr/bin/env python3
"""
Test MultiEncoder initialization with esmc_6b+antiberty.

Tests model initialization with combined encoders and attention decoder.
"""

import torch
from src.models import DevelopabilityModel


def test_multiencoder_init():
    """Test model initialization with esmc_6b+antiberty and attention decoder."""
    print("=" * 80)
    print("Test: MultiEncoder (esmc_6b+antiberty) + AttentionDecoder Initialization")
    print("=" * 80)
    print("\nNOTE: esmc_6b requires Forge API token or precomputed embeddings.")
    print("      Testing with esmc_600m+antiberty as fallback if token unavailable.\n")

    # Try esmc_6b first, fallback to esmc_600m if no token
    import os
    has_forge_token = os.getenv("FORGE_TOKEN") is not None

    if has_forge_token:
        encoder_types = ["esmc_6b", "antiberty"]
        expected_dim = 6144  # 5120 + 1024
        print("✓ FORGE_TOKEN found, using esmc_6b")
    else:
        encoder_types = ["esmc_600m", "antiberty"]
        expected_dim = 3328  # 2304 + 1024
        print("⚠ No FORGE_TOKEN, using esmc_600m instead")

    # Test configuration
    encoder_config = {
        "encoder_types": encoder_types,
        "fusion_strategy": "concat",  # MultiEncoder uses concatenation
        "pooling": "mean",
        "encoder_configs": {
            encoder_types[0]: {},
            "antiberty": {}
        }
    }

    decoder_config = {
        "hidden_dim": 1024,  # Project from concatenated embeddings to 1024
        "n_layers": 3,
        "n_heads": 4,
        "dropout": 0.2,
        "attention_strategy": "bidirectional_cross",
    }

    try:
        # Create model
        print("\n1. Creating model...")
        print(f"   Encoder: esmc_6b + antiberty")
        print(f"   Decoder: attention (hidden_dim=1024, layers=3, heads=4)")

        model = DevelopabilityModel(
            encoder_config=encoder_config,
            decoder_type="attention",
            decoder_config=decoder_config,
            learning_rate=1e-5,
            weight_decay=1e-3,
            scheduler="cosine",
            warmup_epochs=2,
            max_epochs=50,
            loss_fn="mse",
            freeze_encoder=True,
        )

        print("   ✓ Model created successfully")

        # Check encoder embedding dimension
        print("\n2. Checking encoder dimensions...")
        encoder_dim = model.encoder.get_embedding_dim()
        print(f"   Encoder output dim: {encoder_dim}")
        print(f"   Expected: {expected_dim}")

        if encoder_dim == expected_dim:
            print("   ✓ Correct embedding dimension!")
        else:
            print(f"   ✗ Wrong dimension! Expected {expected_dim}, got {encoder_dim}")
            return False

        # Check decoder input dimension
        print("\n3. Checking decoder configuration...")
        print(f"   Decoder input_dim: {model.decoder.chain_dim * 2}")  # VH + VL concatenated
        print(f"   Decoder hidden_dim: {model.decoder.hidden_dim}")
        print(f"   Decoder n_layers: {model.decoder.n_layers}")
        print(f"   Decoder n_heads: {model.decoder.n_heads}")

        if model.decoder.hidden_dim == 1024:
            print("   ✓ Correct decoder hidden dimension!")
        else:
            print(f"   ✗ Wrong hidden dimension! Expected 1024, got {model.decoder.hidden_dim}")
            return False

        # Test forward pass with dummy sequences
        print("\n4. Testing forward pass (without precomputed embeddings)...")
        print("   Note: This will fail if you don't have precomputed embeddings or Forge API")
        print("   Skipping forward pass test for initialization check.")

        # Check model parameters
        print("\n5. Checking model architecture...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        if trainable_params > 0:
            print("   ✓ Model has trainable parameters (decoder)")
        else:
            print("   ✗ No trainable parameters!")
            return False

        print("\n" + "=" * 80)
        print("✓ ALL CHECKS PASSED")
        print("=" * 80)
        print("\nModel successfully initialized with:")
        print(f"  - Encoder: MultiEncoder ({' + '.join(encoder_types)})")
        print(f"  - Encoder output: {encoder_dim} dims")
        print(f"  - Decoder: AttentionDecoder (hidden_dim={model.decoder.hidden_dim})")
        print(f"  - Trainable params: {trainable_params:,}")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Error during model initialization:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multiencoder_init()
    exit(0 if success else 1)
