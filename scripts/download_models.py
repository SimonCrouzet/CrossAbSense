#!/usr/bin/env python
"""Download trained CrossAbSense models from the Hugging Face Hub into models/.

Prereqs:
    pip install huggingface_hub

By default only the final checkpoint (used by predict.py) + small metadata are
fetched; pass --folds to also download the 5 CV fold checkpoints.

Usage:
    python scripts/download_models.py                       # final.ckpt for all properties
    python scripts/download_models.py --folds               # also the CV fold checkpoints
    python scripts/download_models.py --property HIC        # just the HIC model folder
    python scripts/download_models.py --repo-id me/CrossAbSense --revision v1.0
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "SimonCrouzet/CrossAbSense"
DEFAULT_REVISION = "v0.9"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    p.add_argument("--revision", default=DEFAULT_REVISION, help=f"Tag/branch/commit (default: {DEFAULT_REVISION})")
    p.add_argument("--models-dir", default="models", help="Where to place the model folders")
    p.add_argument("--property", default=None, help="Download only this property (e.g. HIC); default: all")
    p.add_argument("--folds", action="store_true", help="Also download the 5 CV fold checkpoints (large)")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.property}_*" if args.property else "*_*"
    # Default: final.ckpt + small metadata only. --folds adds the CV checkpoints.
    patterns = [f"{prefix}/final.ckpt", f"{prefix}/*.yaml", f"{prefix}/*.txt", f"{prefix}/*.json"]
    if args.folds:
        patterns.append(f"{prefix}/fold*.ckpt")

    path = snapshot_download(repo_id=args.repo_id, revision=args.revision,
                             local_dir=str(models_dir), allow_patterns=patterns)
    print(f"Downloaded to {path}")
    for d in sorted(models_dir.glob("*_*")):
        if (d / "final.ckpt").exists():
            print(f"  - {d}")


if __name__ == "__main__":
    main()
