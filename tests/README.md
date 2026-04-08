# Test Suite

Test suite for the CrossAbSense antibody property prediction framework.

## Test Files

### Setup test battery (`test_setup_checks.py`)
Full end-to-end validation run from `setup.sh --test`. Checks in order:
1. GPU / CUDA functional
2. Config files valid and complete
3. All local encoders load and encode (antiberty, esmc_300m, esmc_600m, prott5)
4. ESM-C 6B API connectivity (optional — needs `FORGE_TOKEN`)
5. Antibody feature extraction
6. Train smoke (2 epochs on HIC)
7. Predict smoke (on public mAbs)
8. Sweep script startup

### Model Architecture Tests
- **`test_model_decoder_hidden_dim.py`**: Decoder hidden dimension configurations
- **`test_model_device_handling.py`**: Device placement and movement (CPU/CUDA)
- **`test_model_multiencoder_init.py`**: Multi-encoder initialization and configuration

### Training Pipeline Tests
- **`test_training_hidden_dim.py`**: Hidden dimension handling during training
- **`test_training_integration.py`**: End-to-end training with single/multi-encoder configs

### Feature Tests
- **`test_antibody_features.py`**: Antibody feature extraction (33 descriptors)

### API Tests
- **`test_esmc_6b_api.py`**: ESM-C 6B API integration (requires `FORGE_TOKEN`)

### Smoke Tests
- **`test_smoke_pipelines.py`**: End-to-end train / predict / sweep pipeline (pytest-compatible)

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# By category
python -m pytest tests/test_model_*.py -v
python -m pytest tests/test_training_*.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Requirements

```bash
pip install pytest pytest-cov
```

## Notes

- Some tests require precomputed embeddings in `inputs/embeddings/`
- ESM-C 6B tests require `FORGE_TOKEN` environment variable
- GPU tests require a CUDA-compatible device
