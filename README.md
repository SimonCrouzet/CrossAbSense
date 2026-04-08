# CrossAbSense

Biologically-Grounded Multi-Encoder Architectures as Developability Oracles for Antibody Design

**Paper:** [ICLR 2026 GEM Workshop](https://openreview.net/forum?id=UPUoa6mcdZ)

## Overview

This repository contains a modular framework for predicting antibody developability properties using transformer-based protein language models. The framework uses a two-stage architecture:

1. **Encoder**: Generates embeddings from antibody sequences (AntiBERTy, ESM-2, ESM-C, ProtT5, or MultiEncoder)
2. **Decoder**: Predicts developability properties from embeddings (MLP or AttentionDecoder)

## Features

- **Multiple Encoders**:
  - AntiBERTy (antibody-specific, 512 dims)
  - ESM-2 (general protein LM, 1280 dims)
  - ESM-C (latest from Meta/FAIR: 300M/600M/6B variants)
  - ProtT5 (T5-based protein LM, 1024 dims)
  - MultiEncoder (combines multiple encoders with fusion strategies)

- **Advanced Decoders**:
  - MLP with batch norm and dropout
  - AttentionDecoder with VH-VL cross-attention strategies

- **Antibody Features** (optional):
  - 33 developability features extracted from VH/VL sequences
  - **abnumber**: Germline identity, mutations, V gene family, CDR lengths (13 features)
  - **BioPhi**: Humanness scores via OASis database (2 features)
  - **ScaLoP**: Canonical structure classes for CDR loops (5 features)
  - **Sequence-based**: Liabilities, CDR-H3 properties, pI, charge (13 features)
  - Robust sentinel values when tools unavailable
  - Concatenates with embeddings for enhanced predictions

- **Baseline Models**:
  - Random baseline (fitted to training distribution)
  - XGBoost on ESM-2 embeddings

- **PyTorch Lightning Integration**:
  - Automatic mixed precision
  - Gradient clipping
  - Learning rate scheduling
  - Early stopping

- **Weights & Biases Support**:
  - Experiment tracking
  - Hyperparameter logging
  - Metric visualization

## Dataset

This framework uses the **GDPa1** antibody developability benchmark by [Ginkgo Bioworks](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1) — 242 antibodies with 5 biophysical assays (HIC, PR_CHO, AC-SINS, Tm2, Titer).

```bash
python scripts/download_dataset.py  # Downloads GDPa1 from HuggingFace to inputs/
```

A small set of public therapeutic antibodies (9 approved mAbs not in GDPa1) is included in `inputs/public_mabs_not_in_gdpa1.csv` for prediction examples.

## Installation

```bash
# Clone repository
git clone https://github.com/SimonCrouzet/CrossAbSense
cd CrossAbSense

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train a single model

```bash
python -m src.train \
  --config src/config/oracle_efficient_config.yaml \
  --property HIC \
  --fold 0 \
  --gpus 1
```

### 2. Run cross-validation

```bash
python -m src.run_cv \
  --config src/config/default_config.yaml \
  --properties HIC Tm2 PR_CHO \
  --gpus 1 \
  --n_folds 5
```

### 3. Hyperparameter tuning

```bash
# Launch a W&B sweep for a specific property
wandb sweep config/tuning/example_HIC.yaml
wandb agent <sweep-id> --count 60
```

### 4. Use antibody features

Extract 33 developability features from antibody sequences and inject into the decoder alongside embeddings:

#### Step 1: Precompute features (recommended)

```bash
# Precompute features for faster training
python scripts/precompute_antibody_features.py inputs/GDPa1_complete.csv
# Output: inputs/antibody_features/GDPa1_complete_antibody_features_<checksum>.pt
```

#### Step 2: Enable in config

```yaml
# Data module - load precomputed features
antibody_features_path: "inputs/antibody_features/GDPa1_complete_antibody_features_72626a47.pt"

# Decoder - configure injection
decoder:
  attention:
    antibody_features_dim: 33
    antibody_features_injection_layer: "second"  # Options: "first", "second", "third", "last"
```

**Injection layers** (configurable, affects where features enter the decoder FFN):
- `"first"`: After pooling, before 1st FFN layer (e.g., 768 + 33 → 384)
- `"second"` (default): After 1st FFN layer (e.g., 384 + 33 → 192)
- `"third"`: After 2nd FFN layer (e.g., 192 + 33 → 96)
- `"last"`: Just before final output projection

**Fallback**: If precomputed features are unavailable, they can be computed on-the-fly (slower):
```yaml
antibody_features_config:
  enabled: true
  use_abnumber: true
  use_biophi: true
  use_scalop: true
  use_sequence_features: true
```

**Requirements:**
- `abnumber`: `conda install -c bioconda abnumber`
- `scalop`: `conda install -c bioconda scalop`
- `BioPhi`: Requires separate conda env (see BioPhi docs) + OASis database at `external/OASis_9mers_v1.db`
- `Biopython`: `conda install -c conda-forge biopython` (for pI calculation)

**Standalone usage** (for feature extraction only):
```python
from src.features import AntibodyFeatures

extractor = AntibodyFeatures(
    use_abnumber=True, use_biophi=True,
    use_scalop=True, use_sequence_features=True
)

features = extractor.extract_features(
    heavy_seq="QVKLQESGAE...",
    light_seq="DIQMTQSPSS..."
)

feature_array = extractor.features_to_array(features)  # Shape: (33,)
```

### 5. Precompute embeddings with AHO-aligned sequences

The framework supports antibody sequences aligned using the AHo numbering scheme. AHO alignment provides:
- **Fixed-length representations**: All VH and VL sequences have the same length (149 residues)
- **Position-specific analysis**: Same position corresponds to the same structural location across antibodies
- **Better attention**: Positional correspondence improves cross-attention mechanisms

```bash
# Precompute AHO-aligned embeddings
python src/utils/precompute_embeddings.py \
  --input inputs/GDPa1_complete.csv \
  --encoder esmc_300m \
  --use-aho-aligned

# Then enable in config
use_aho_aligned: true
```

**Key differences**:
- **AHO-aligned**: All sequences = 149 chars with gaps (`-`) for alignment
  - Example: `QVKLQES-GAELARPGASVKLSCKASG-YTFTN-----YWMQ...`
- **Non-aligned**: Variable lengths (VH: 111-130 chars, VL: 104-113 chars)
  - Example: `QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQ...`

**Note**: Requires separate precomputed embeddings with `_aho` suffix (e.g., `GDPa1_complete_esmc_300m_aho_*.pt`)

### 6. Precompute embeddings with full-chain sequences

The framework supports using full-chain antibody sequences (including constant regions) instead of variable regions only. This captures additional information from constant domains that can affect developability.

**Why use full-chain?**
- **Constant region information**: Includes CH domains (heavy) and CL domain (light)
- **IgG subclass differences**: Different constant regions affect properties like thermal stability
- **Example impact on Tm2**: CH2 domain sequence differs by subclass (IgG1: CPPCPAPELLGG vs IgG2: CPPCPAPPVAG), directly affecting Tm2 measurements
- **ESM model advantage**: ESM-2/ESM-C were trained on full-length proteins and can learn sequence-property relationships from constant regions

```bash
# Precompute full-chain embeddings
python src/utils/precompute_embeddings.py \
  --input inputs/GDPa1_complete.csv \
  --encoder esmc_300m \
  --full-chain

# Then enable in config
use_full_chain: true
```

**Sequence lengths** (mature proteins, signal peptide removed):
- **Variable regions only**: VH ~110-130 AA, VL ~110-130 AA
- **Full-chain**: HC ~430-460 AA, LC ~210-230 AA
  - Note: Signal peptide (17 AA) is automatically detected and removed to work with mature protein sequences

**Note**:
- Mutually exclusive with `--use-aho-aligned`
- Requires separate precomputed embeddings with `_fullchain` suffix (e.g., `GDPa1_complete_esmc_300m_fullchain_*.pt`)
- Can be used for any property, not just Tm2
- **Automatic reconstruction**: If full-chain sequences are missing (e.g., in heldout set), they are automatically reconstructed from variable regions using IgG subtype information

## Configuration

Configurations are stored in YAML files. The framework supports:

- **Default config**: `src/config/default_config.yaml` — fully documented, conservative defaults
- **Oracle-efficient config**: `src/config/oracle_efficient_config.yaml` — light, fast compute for high-throughput screening

### Example Configuration

```yaml
encoder:
  type: "esmc"
  esmc:
    model_name: "facebook/esmc_600m"
    pooling: "mean"

decoder:
  type: "attention"
  attention:
    antibody_features_dim: 33  # Enable antibody features (0 = disabled)
    antibody_features_injection_layer: "second"  # Options: first, second, third, last

# Antibody features (precomputed recommended)
antibody_features_path: "inputs/antibody_features/GDPa1_complete_antibody_features_<checksum>.pt"

training:
  finetune:
    max_epochs: 100
    batch_size: 16
    learning_rate: 1e-5
```

## Project Structure

```
CrossAbSense/
├── src/
│   ├── encoders/          # AntiBERTy, ESM-2, ESM-C, ProtT5, MultiEncoder
│   ├── decoders/          # MLP, AttentionDecoder
│   ├── features/          # Antibody feature extraction (33 descriptors)
│   ├── models/            # PyTorch Lightning modules & baselines
│   ├── data/              # Data modules, target transforms
│   ├── utils/             # Config, metrics, precompute_embeddings, sweep tools
│   ├── config/            # Default & oracle-efficient configs
│   ├── train.py           # Training script (all folds + final model)
│   ├── run_cv.py          # Cross-validation runner
│   └── predict.py         # Prediction on new sequences
├── scripts/
│   ├── extract_sweep_best.py                # Extract best sweep results
│   ├── precompute_antibody_features.py      # Precompute 33 sequence features
│   ├── precompute_prediction_cache.py       # Unified cache for prediction (embeddings + features)
│   ├── format_for_biophi_server.py          # BioPhi FASTA export
│   ├── generate_iglm_designs.py             # IgLM sequence generation
│   ├── pair_iglm_designs.py                 # Pair VH/VL designs
│   ├── run_sweep.py                         # W&B sweep runner
│   └── download_dataset.py                  # Download GDPa1 from HuggingFace
├── config/
│   └── tuning/            # Example W&B sweep configs (one per property)
├── tests/                 # Unit & integration tests
└── inputs/
    └── public_mabs_not_in_gdpa1.csv  # Example antibodies for prediction
```

## Prediction Workflow

### 1. Precompute Prediction Cache

Create a unified cache with embeddings and features for efficient prediction:

```bash
# Auto-detect all encoders from default config
python scripts/precompute_prediction_cache.py \
    results/my_designs.csv

# Output: results/my_designs_cache.pt
# Contains:
#  - Embeddings for all encoder types (esmc_300m, esmc_6b, prott5, etc.)
#  - Antibody features (abnumber, scalop, sequence features)
#  - Sequence metadata
```

**Custom configuration:**

```bash
# Use specific config
python scripts/precompute_prediction_cache.py \
    results/my_designs.csv \
    --config src/config/oracle_efficient_config.yaml

# Force recompute
python scripts/precompute_prediction_cache.py \
    results/my_designs.csv \
    --force
```

**BioPhi Integration:**

If you have BioPhi humanization results, place the summary file next to your input:
- Input: `results/my_designs.csv`
- BioPhi: `results/my_designs_biophi_summary.xlsx`

The script will automatically detect and use the BioPhi summary to skip online API calls.

**Key features:**
- ✓ Auto-detects ALL encoder types from config (including property-specific)
- ✓ Resumes from existing embeddings (no recomputation)
- ✓ BioPhi-aware (auto-loads summary if available)
- ✓ Multi-encoder support (esmc_300m, esmc_6b, prott5, antiberty)
- ✓ Output: `<input_stem>_cache.pt` in same directory as input

**⚠️ Performance Note for Live/Responsive Oracle:**

For interactive prediction workflows where speed matters:
- **Disable ESM-C 6B**: Requires API credits and is slower (~5-10x than local models)
- **Disable antibody features**: BioPhi humanness scores require external API/database

Recommended config for responsive oracle:
```yaml
encoder:
  type: "esmc"
  esmc:
    model_name: "facebook/esmc_300m"  # Use 300M, not 6B

# Or use local-only encoders
encoder:
  type: "prott5"  # Fast, runs locally

# Disable antibody features
decoder:
  attention:
    antibody_features_dim: 0  # Disable features
```

This reduces latency from ~30s to ~2s per antibody for real-time prediction.

### 2. Format Sequences for BioPhi

Generate BioPhi-compatible FASTA files for humanization analysis:

```bash
# Auto-generates output filename
python scripts/format_for_biophi_server.py \
    results/my_designs.csv

# Output: results/my_designs_biophi_formatted.fasta

# Custom output path
python scripts/format_for_biophi_server.py \
    results/my_designs.csv \
    -o custom_output.fasta

# Limit to first N sequences
python scripts/format_for_biophi_server.py \
    results/my_designs.csv \
    -n 10
```

**BioPhi server settings:**
- Humanization method: `IMGT :IMGT 1% fraction subject`
- After analysis, download summary as `<input_stem>_biophi_summary.xlsx`

**Output format:**
```
>antibody1_VH
EVQLVESGGGLVKPGGSLRLSCAASGFTF...
>antibody1_VL
DIQMTQSPSSLSASVGDRVTITCRASQD...
```

### 3. Generate IgLM Designs

Generate novel antibody sequences using IgLM:

```bash
# Generate new sequences and pair them
python scripts/generate_iglm_designs.py \
    --num_seqs 100 \
    --output_dir results/iglm_herceptin

# Just pair existing FASTA files (no generation)
python scripts/generate_iglm_designs.py \
    --pair_only \
    --vh_fasta results/iglm_herceptin/vh_designs/generated_seqs.fasta \
    --vl_fasta results/iglm_herceptin/vl_designs/generated_seqs.fasta \
    --output_csv results/iglm_herceptin/iglm_designs_for_prediction.csv

# Include Herceptin-paired designs (in addition to 1-to-1 pairs)
python scripts/generate_iglm_designs.py \
    --num_seqs 100 \
    --output_dir results/iglm_herceptin \
    --include_herceptin_pairs
```

**Pairing strategies:**
- **1-to-1 matched pairs** (default): VH seq_0 with VL seq_0, etc.
- **Herceptin-paired** (optional): VH designs with Herceptin VL, and vice versa

**Output:**
- `vh_designs/generated_seqs.fasta`: Generated VH sequences
- `vl_designs/generated_seqs.fasta`: Generated VL sequences
- `iglm_designs_for_prediction.csv`: Paired designs ready for prediction

## Properties

The framework predicts 5 developability properties across 9 assays:

| Category | Assays | Description |
|----------|--------|-------------|
| 💧 **Hydrophobicity** | HIC | Hydrophobic interaction chromatography retention time |
| 🎯 **Polyreactivity** | PR_CHO, PR_Ova | Off-target binding to CHO cell lysate and ovalbumin |
| 🧲 **Self-association** | AC-SINS_pH6.0, AC-SINS_pH7.4 | Affinity-capture self-interaction nanoparticle spectroscopy |
| 🌡️ **Thermostability** | Tm1, Tm2, Tonset | Differential scanning fluorimetry melting temperatures |
| 🧪 **Titer** | Titer | Expression yield in HEK293 cells |

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Citation

If you use CrossAbSense in your research, please cite:

```bibtex
@inproceedings{crouzet2026crossabsense,
  title={Biologically-Grounded Multi-Encoder Architectures as Developability Oracles for Antibody Design},
  author={Crouzet, Simon J.},
  booktitle={ICLR 2026 Workshop on Generative and Experimental Perspectives for Biomolecular Design (GEM)},
  year={2026},
  url={https://openreview.net/forum?id=UPUoa6mcdZ}
}
```

## Contribute & Collaborate

Contributions and collaborations are very welcome — whether it's a bug fix, a new idea, or a joint project, I'd love to hear from you.

**Code & community:**

- **Bug reports & feature requests**: Open an [issue](https://github.com/SimonCrouzet/CrossAbSense/issues)
- **Code contributions**: Fork the repo, create a branch, and submit a pull request
- **New encoders/decoders**: The modular architecture makes it straightforward to add new components (see `src/encoders/base_encoder.py` and `src/decoders/base_decoder.py`)
- **Datasets & benchmarks**: Adapting the framework to other antibody developability datasets

**Research & industry:**

- **Research collaborations**: Joint work on new PLM architectures, decoder strategies, or antibody datasets
- **Wet-lab validation**: Partnerships to experimentally test oracle predictions and close the in-silico/in-vitro loop
- **Industry applications**: Adapting the framework to proprietary antibody libraries or integrating it into existing design pipelines

I'm also available as an independent consultant — my expertise spans biomolecular design, computational antibody engineering, ML implementation and engineering for drug discovery, and predictive modeling for virtual screening. Feel free to reach out via [GitHub](https://github.com/SimonCrouzet) or [LinkedIn](https://www.linkedin.com/in/simoncrouzet/).

**Getting started:**

```bash
git clone https://github.com/SimonCrouzet/CrossAbSense
cd CrossAbSense
pip install -r requirements.txt
python -m pytest tests/ -v  # Run tests to verify setup
```
