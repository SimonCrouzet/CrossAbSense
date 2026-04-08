#!/usr/bin/env python3
"""Test device handling for AttentionDecoder with projections."""

import pytest
import torch
from src.decoders import AttentionDecoder


def test_device_cpu():
    """Test forward pass on CPU with projection."""
    input_dim = 2560
    decoder = AttentionDecoder(
        input_dim=input_dim,
        hidden_dim=512,  # Triggers projection
        n_layers=2,
        n_heads=4,
    )

    assert decoder.vh_projection is not None
    assert decoder.vl_projection is not None

    vh = torch.randn(4, input_dim)
    vl = torch.randn(4, input_dim)
    output = decoder(vh_embeddings=vh, vl_embeddings=vl)

    assert output.device.type == "cpu"
    assert output.shape == (4, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    """Test forward pass on CUDA with projection."""
    input_dim = 2560
    decoder = AttentionDecoder(
        input_dim=input_dim,
        hidden_dim=512,
        n_layers=2,
        n_heads=4,
    ).cuda()

    vh = torch.randn(4, input_dim).cuda()
    vl = torch.randn(4, input_dim).cuda()
    output = decoder(vh_embeddings=vh, vl_embeddings=vl)

    assert output.device.type == "cuda"
    assert output.shape == (4, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_native_dims():
    """Test CUDA with native dims (no projection)."""
    input_dim = 2560
    decoder = AttentionDecoder(
        input_dim=input_dim,
        hidden_dim=None,
        n_layers=2,
        n_heads=4,
    ).cuda()

    assert decoder.vh_projection is None
    assert decoder.vl_projection is None

    vh = torch.randn(4, input_dim).cuda()
    vl = torch.randn(4, input_dim).cuda()
    output = decoder(vh_embeddings=vh, vl_embeddings=vl)

    assert output.device.type == "cuda"
    assert output.shape == (4, 1)


if __name__ == "__main__":
    test_device_cpu()
    if torch.cuda.is_available():
        test_device_cuda()
        test_device_native_dims()
    print("ALL TESTS PASSED")
