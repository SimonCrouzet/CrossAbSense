---
license: apache-2.0
library_name: pytorch
tags:
  - antibody
  - developability
  - protein-language-model
  - regression
datasets:
  - ginkgo-datapoints/GDPa1
---

# CrossAbSense — antibody developability oracles (v0.9)

Property-specific neural oracles that predict five biophysical developability assays
for therapeutic IgGs from paired VH/VL sequences, combining frozen protein-language-model
encoders (ESM-Cambrian, ProtT5) with configurable attention decoders.

Code: https://github.com/SimonCrouzet/CrossAbSense
Dataset: [GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1) (242 IgGs, Ginkgo Bioworks)

Each property folder (`<PROPERTY>_<config-checksum>/`) contains:
`final.ckpt` (model trained on all data — used by `predict.py`), `fold0-4.ckpt`
(5-fold CV checkpoints), `config.yaml`, and `property.txt`.

## Performance (5-fold cluster-stratified CV, Spearman ρ)

| Property | This release (v0.9) | Paper (Table 1) |
|----------|--------------------:|----------------:|
| HIC (hydrophobicity)      | 0.685 | 0.644 |
| Titer (expression)        | 0.425 | 0.428 |
| PR_CHO (polyreactivity)   | 0.461 | 0.475 |
| AC-SINS (self-association)| 0.420 | 0.475 |
| Tm2 (thermostability)     | 0.442 | 0.387 |

## ⚠️ Important caveat (v0.9)

These weights were trained from the published configs but in an environment **without
BioPhi (OASis humanness) and ScaLoP** available. Those two antibody-feature sources were
substituted with sentinel values during training, so the feature inputs differ slightly
from the paper runs. This mainly affects **AC-SINS** (~0.05 below paper); the other four
properties match or exceed Table 1. A future **v1.0** will retrain the feature-using
properties with BioPhi/ScaLoP restored. Pin `revision="v0.9"` if you need exactly these weights.

## Usage

```bash
pip install huggingface_hub
python scripts/download_models.py --revision v0.9        # final.ckpt only (add --folds for CV)
python src/predict.py --input inputs/my_seqs.csv --model models/HIC_3595cc57 --output preds.csv
```

By default only `final.ckpt` (+ small metadata) is downloaded; the 5 CV fold
checkpoints are fetched only when you ask for them (`--folds`, or `predict.py
--use-cv`/`--fold`).

Or let `predict.py` fetch on demand:

```bash
python src/predict.py --input inputs/my_seqs.csv --model HIC_3595cc57 --from-hf --output preds.csv
```

## License

Apache-2.0, matching the CrossAbSense repository.
