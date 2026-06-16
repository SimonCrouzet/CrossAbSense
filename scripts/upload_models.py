#!/usr/bin/env python
"""Upload trained CrossAbSense models to the Hugging Face Hub.

Creates the repo if needed, uploads each ``models/<PROPERTY>_<checksum>/`` folder
(fold0-4.ckpt, final.ckpt, config.yaml, property.txt, ...), pushes the model card
as the repo README, and tags the release.

Prereqs:
    pip install huggingface_hub
    huggingface-cli login        # or set HF_TOKEN

Usage:
    python scripts/upload_models.py                         # all model dirs -> default repo, tag v0.9
    python scripts/upload_models.py --repo-id me/CrossAbSense --tag v0.9
    python scripts/upload_models.py --property HIC          # upload a single property
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

DEFAULT_REPO = "SimonCrouzet/CrossAbSense"
DEFAULT_TAG = "v0.9"


def find_model_dirs(models_dir: Path, only_property: str | None):
    """Return deployable model dirs: those containing a final.ckpt."""
    dirs = sorted(d for d in models_dir.iterdir() if d.is_dir() and (d / "final.ckpt").exists())
    if only_property:
        dirs = [d for d in dirs if d.name.startswith(f"{only_property}_")]
    return dirs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    p.add_argument("--models-dir", default="models", help="Local dir holding the model folders")
    p.add_argument("--property", default=None, help="Upload only this property (e.g. HIC); default: all")
    p.add_argument("--tag", default=DEFAULT_TAG, help=f"Git tag for this release (default: {DEFAULT_TAG})")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    p.add_argument("--card", default="MODEL_CARD.md", help="Model card to push as repo README.md")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    model_dirs = find_model_dirs(models_dir, args.property)
    if not model_dirs:
        raise SystemExit(f"No deployable model dirs (with final.ckpt) found under {models_dir}/")

    print(f"Repo: {args.repo_id}  |  tag: {args.tag}  |  uploading {len(model_dirs)} model(s):")
    for d in model_dirs:
        print(f"  - {d.name}")

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Model card -> README.md
    card = Path(args.card)
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=args.repo_id, repo_type="model",
                        commit_message="Add/update model card")

    # One commit per model folder (keeps history readable; allows later per-property refresh)
    for d in model_dirs:
        upload_folder(repo_id=args.repo_id, repo_type="model",
                      folder_path=str(d), path_in_repo=d.name,
                      commit_message=f"Upload {d.name}")

    api.create_tag(args.repo_id, repo_type="model", tag=args.tag, exist_ok=True)
    print(f"\nDone. https://huggingface.co/{args.repo_id}  (tag {args.tag})")


if __name__ == "__main__":
    main()
