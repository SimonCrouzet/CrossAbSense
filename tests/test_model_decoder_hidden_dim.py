#!/usr/bin/env python3
"""Test AttentionDecoder hidden_dim settings."""

import torch
from src.decoders import AttentionDecoder


def test_decoder_hidden_dim_none():
    """Test that hidden_dim=None uses input_dim (no compression)."""
    input_dim = 2560  # ESM-C 6B per-chain dim
    decoder = AttentionDecoder(
        input_dim=input_dim,
        hidden_dim=None,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    )

    assert decoder.hidden_dim == input_dim, \
        f"Expected hidden_dim={input_dim}, got {decoder.hidden_dim}"

    # Forward pass with separate VH/VL
    batch_size = 4
    vh = torch.randn(batch_size, input_dim)
    vl = torch.randn(batch_size, input_dim)
    output = decoder(vh_embeddings=vh, vl_embeddings=vl)

    assert output.shape == (batch_size, 1), f"Wrong output shape: {output.shape}"


def test_decoder_hidden_dim_explicit():
    """Test that explicit hidden_dim projects to specified dimension."""
    input_dim = 2560
    hidden_dim = 512
    decoder = AttentionDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    )

    assert decoder.hidden_dim == hidden_dim, \
        f"Expected hidden_dim={hidden_dim}, got {decoder.hidden_dim}"

    # Forward pass
    batch_size = 4
    vh = torch.randn(batch_size, input_dim)
    vl = torch.randn(batch_size, input_dim)
    output = decoder(vh_embeddings=vh, vl_embeddings=vl)

    assert output.shape == (batch_size, 1), f"Wrong output shape: {output.shape}"


def test_decoder_hidden_dim_small_input():
    """Test with smaller input dimensions (AntiBERTy-sized)."""
    input_dim = 512  # AntiBERTy per-chain dim
    decoder = AttentionDecoder(
        input_dim=input_dim,
        hidden_dim=None,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    )

    assert decoder.hidden_dim == input_dim, \
        f"Expected hidden_dim={input_dim}, got {decoder.hidden_dim}"


if __name__ == "__main__":
    test_decoder_hidden_dim_none()
    test_decoder_hidden_dim_explicit()
    test_decoder_hidden_dim_small_input()
    print("ALL TESTS PASSED")
