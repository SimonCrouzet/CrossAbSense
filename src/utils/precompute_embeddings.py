#!/usr/bin/env python3
"""
Pre-compute embeddings for all encoder types.

Supports: AntiBERTy, ESM-C (300M, 600M, 6B), ProtT5
Stores raw sequence-level embeddings (seq_len, hidden_dim) for later pooling.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Encoder configurations
ENCODER_CONFIGS = {
    "antiberty": {
        "model_name": "alchemab/antiberty",
        "embedding_dim": 512,
        "requires_api": False,
    },
    "esmc_300m": {
        "model_name": "esmc_300m",
        "embedding_dim": 960,
        "requires_api": False,
    },
    "esmc_600m": {
        "model_name": "esmc_600m",
        "embedding_dim": 1152,
        "requires_api": False,
    },
    "esmc_6b": {
        "model_name": "esmc-6b-2024-12",
        "embedding_dim": 2560,
        "requires_api": True,
    },
    "prott5": {
        "model_name": "Rostlab/prot_t5_xl_uniref50",
        "embedding_dim": 1024,
        "requires_api": False,
    },
}


def compute_file_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_unique_sequences(
    csv_path: str,
    vh_col: str = "vh_protein_sequence",
    vl_col: str = "vl_protein_sequence",
    use_aho_aligned: bool = False,
    use_full_chain: bool = False
) -> Tuple[Set[str], Set[str]]:
    """Extract unique VH and VL sequences from CSV."""
    logger.info(f"Reading sequences from {csv_path}")
    df = pd.read_csv(csv_path)

    # Mutually exclusive options
    if use_aho_aligned and use_full_chain:
        raise ValueError("--use-aho-aligned and --full-chain are mutually exclusive")

    # Override columns if using AHO-aligned sequences
    if use_aho_aligned:
        vh_col = "heavy_aligned_aho"
        vl_col = "light_aligned_aho"
        logger.info("Using AHO-aligned sequences (heavy_aligned_aho, light_aligned_aho)")

    # Override columns if using full-chain sequences
    if use_full_chain:
        vh_col = "hc_protein_sequence"
        vl_col = "lc_protein_sequence"
        logger.info("Using full-chain sequences (hc_protein_sequence, lc_protein_sequence)")

        # Import helper functions
        import sys
        from pathlib import Path as PathLib
        # Add project root to path
        project_root = PathLib(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.data.gdpa1_datamodule import extract_constant_regions_from_reference, reconstruct_full_chains

        # Extract signal peptide and constant regions from reference
        reference_csv = "inputs/GDPa1_complete.csv"
        constant_regions = extract_constant_regions_from_reference(reference_csv)
        signal_peptide = constant_regions.get('signal_peptide', '')

        # Check if full-chain columns exist, if not, reconstruct them
        if vh_col not in df.columns or vl_col not in df.columns:
            logger.info("Full-chain sequences not found - reconstructing from variable regions...")
            # Reconstruct full chains (without signal peptide - mature proteins)
            df = reconstruct_full_chains(df, constant_regions, inplace=True)
        else:
            logger.info("Full-chain sequences found in CSV")
            # Strip signal peptide if present (work with mature proteins only)
            if signal_peptide:
                n_stripped_hc = 0
                n_stripped_lc = 0

                for idx in df.index:
                    if df.loc[idx, vh_col].startswith(signal_peptide):
                        df.loc[idx, vh_col] = df.loc[idx, vh_col][len(signal_peptide):]
                        n_stripped_hc += 1
                    if df.loc[idx, vl_col].startswith(signal_peptide):
                        df.loc[idx, vl_col] = df.loc[idx, vl_col][len(signal_peptide):]
                        n_stripped_lc += 1

                if n_stripped_hc > 0 or n_stripped_lc > 0:
                    logger.info(f"Stripped signal peptide ({signal_peptide}) from {n_stripped_hc} HC and {n_stripped_lc} LC sequences")
                    logger.info("Embedding mature protein sequences (signal peptide removed)")
                else:
                    logger.info("No signal peptide detected (sequences already mature)")

    vh_sequences = set()
    vl_sequences = set()

    if vh_col in df.columns:
        vh_sequences = set(df[vh_col].dropna().unique())
        logger.info(f"Found {len(vh_sequences)} unique VH sequences")

    if vl_col in df.columns:
        vl_sequences = set(df[vl_col].dropna().unique())
        logger.info(f"Found {len(vl_sequences)} unique VL sequences")

    return vh_sequences, vl_sequences


def encode_with_antiberty(sequence: str, runner) -> torch.Tensor:
    """Encode sequence with AntiBERTy (official package)."""
    # Use official AntiBERTy embed method
    # Returns tensor of shape [(seq_len + 2) x 512] (includes BOS/EOS)
    embeddings_list = runner.embed([sequence])
    emb = embeddings_list[0]  # Get first (and only) sequence

    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    return emb.cpu()




def encode_with_esmc(sequence: str, model, device) -> torch.Tensor:
    """Encode sequence with ESM-C (local models 300M/600M)."""
    # ESM-C local encoding
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)

    with torch.no_grad():
        logits_output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        embeddings = logits_output.embeddings

        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)

    return embeddings.cpu()


def encode_with_esmc_api(sequence: str, client) -> torch.Tensor:
    """Encode sequence with ESM-C 6B API."""
    from esm.sdk.api import ESMProtein, LogitsConfig, ESMProteinError

    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    if isinstance(protein_tensor, ESMProteinError):
        raise protein_tensor

    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )

    embeddings = logits_output.embeddings
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    if embeddings.dim() == 3:
        embeddings = embeddings.squeeze(0)

    return embeddings.cpu()


def encode_with_prott5(sequence: str, model, tokenizer, device) -> torch.Tensor:
    """Encode sequence with ProtT5."""
    # Add spaces between amino acids for T5
    spaced_sequence = " ".join(list(sequence))

    inputs = tokenizer(spaced_sequence, return_tensors="pt", padding=False, truncation=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Get last hidden state: (1, seq_len, 1024)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, 1024)

    return embeddings.cpu()


def load_encoder(encoder_type: str, device: str, forge_token: Optional[str] = None):
    """Load encoder model based on type."""
    config = ENCODER_CONFIGS[encoder_type]
    logger.info(f"Loading {encoder_type} encoder ({config['model_name']})...")

    if encoder_type == "antiberty":
        from antiberty import AntiBERTyRunner
        runner = AntiBERTyRunner()
        logger.info("✓ AntiBERTy loaded")
        return {"type": "antiberty", "runner": runner}

    elif encoder_type in ["esmc_300m", "esmc_600m"]:
        from esm.models.esmc import ESMC
        # Pass encoder_type directly as model name (esmc_300m or esmc_600m)
        model = ESMC.from_pretrained(encoder_type).to(device)
        model.eval()
        logger.info(f"✓ ESM-C {encoder_type} loaded")
        return {"type": "esmc_local", "model": model, "device": device}

    elif encoder_type == "esmc_6b":
        from esm.sdk.forge import ESM3ForgeInferenceClient
        token = forge_token or os.getenv("FORGE_TOKEN")
        if not token:
            raise ValueError("FORGE_TOKEN required for ESM-C 6B")

        client = ESM3ForgeInferenceClient(
            model=config["model_name"],
            url="https://forge.evolutionaryscale.ai",
            token=token,
        )
        logger.info("✓ ESM-C 6B API client initialized")
        return {"type": "esmc_api", "client": client}

    elif encoder_type == "prott5":
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained(config["model_name"], do_lower_case=False)
        model = T5EncoderModel.from_pretrained(config["model_name"]).to(device)
        model.eval()
        logger.info("✓ ProtT5 loaded")
        return {"type": "prott5", "model": model, "tokenizer": tokenizer, "device": device}

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def encode_sequence(sequence: str, encoder_dict: dict) -> torch.Tensor:
    """Encode a single sequence using the loaded encoder."""
    encoder_type = encoder_dict["type"]

    if encoder_type == "antiberty":
        return encode_with_antiberty(sequence, encoder_dict["runner"])
    elif encoder_type == "esmc_local":
        return encode_with_esmc(sequence, encoder_dict["model"], encoder_dict["device"])
    elif encoder_type == "esmc_api":
        return encode_with_esmc_api(sequence, encoder_dict["client"])
    elif encoder_type == "prott5":
        return encode_with_prott5(
            sequence, encoder_dict["model"], encoder_dict["tokenizer"], encoder_dict["device"]
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def precompute_embeddings(
    csv_path: str,
    encoder_type: str,
    output_dir: str = "inputs/embeddings",
    vh_col: str = "vh_protein_sequence",
    vl_col: str = "vl_protein_sequence",
    device: str = "cpu",
    forge_token: Optional[str] = None,
    force: bool = False,
    use_aho_aligned: bool = False,
    use_full_chain: bool = False,
    resume: bool = False,
    only_vh: bool = False,
    only_vl: bool = False
):
    """Pre-compute embeddings for all sequences using specified encoder."""

    if encoder_type not in ENCODER_CONFIGS:
        raise ValueError(f"Unknown encoder: {encoder_type}. Choose from: {list(ENCODER_CONFIGS.keys())}")

    # Mutually exclusive options
    if use_aho_aligned and use_full_chain:
        raise ValueError("--use-aho-aligned and --full-chain are mutually exclusive")

    if only_vh and only_vl:
        raise ValueError("--only-vh and --only-vl are mutually exclusive")

    config = ENCODER_CONFIGS[encoder_type]

    # Setup paths
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute checksum
    checksum = compute_file_checksum(str(csv_path))
    logger.info(f"Input file checksum: {checksum[:16]}...")

    # Output files (add suffix based on sequence type)
    suffix = ""
    if use_aho_aligned:
        suffix = "_aho"
    elif use_full_chain:
        suffix = "_fullchain"
    output_name = f"{csv_path.stem}_{encoder_type}{suffix}_{checksum[:8]}.pt"
    output_path = output_dir / output_name
    metadata_path = output_dir / f"{csv_path.stem}_{encoder_type}{suffix}_{checksum[:8]}.json"

    # Load existing embeddings if resuming
    existing_embeddings = {}
    if resume and output_path.exists():
        logger.info(f"Resume mode: loading existing embeddings from {output_path}")
        try:
            existing_embeddings = torch.load(output_path)
            logger.info(f"✓ Loaded {len(existing_embeddings)} existing embeddings")
        except Exception as e:
            logger.warning(f"Failed to load existing embeddings: {e}")
            logger.info("Starting from scratch...")
            existing_embeddings = {}

    # Check if exists and complete (not resuming)
    if output_path.exists() and not force and not resume:
        logger.info(f"Embeddings already exist at {output_path}")
        logger.info("Use --force to recompute or --resume to continue incomplete computation")

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            if metadata.get("checksum") == checksum:
                logger.info("✓ Checksum verified - embeddings are up to date")
                return
        else:
            logger.info("✓ Using existing embeddings")
            return

    # Get sequences
    vh_sequences, vl_sequences = get_unique_sequences(csv_path, vh_col, vl_col, use_aho_aligned, use_full_chain)

    # Filter by --only-vh or --only-vl flags
    if only_vh:
        vl_sequences = set()
        logger.info("Only computing VH sequences (--only-vh)")
    elif only_vl:
        vh_sequences = set()
        logger.info("Only computing VL sequences (--only-vl)")

    # Filter out already-computed sequences
    if existing_embeddings:
        vh_existing = [seq for seq in vh_sequences if f"VH:{seq}" in existing_embeddings]
        vl_existing = [seq for seq in vl_sequences if f"VL:{seq}" in existing_embeddings]

        vh_sequences = set(seq for seq in vh_sequences if f"VH:{seq}" not in existing_embeddings)
        vl_sequences = set(seq for seq in vl_sequences if f"VL:{seq}" not in existing_embeddings)

        logger.info(f"Skipping {len(vh_existing)} already-computed VH sequences")
        logger.info(f"Skipping {len(vl_existing)} already-computed VL sequences")

    total_sequences = len(vh_sequences) + len(vl_sequences)
    logger.info(f"Total unique sequences to encode: {total_sequences}")

    # Skip if no sequences to encode
    if total_sequences == 0:
        logger.info("No new sequences to encode!")
        if existing_embeddings:
            logger.info(f"All {len(existing_embeddings)} sequences already computed.")
        return

    # Load encoder
    encoder_dict = load_encoder(encoder_type, device, forge_token)

    # Start with existing embeddings (if resuming)
    embeddings = existing_embeddings.copy() if existing_embeddings else {}
    new_count = 0

    # VH sequences
    if vh_sequences:
        logger.info(f"Encoding {len(vh_sequences)} VH sequences...")
        for seq in tqdm(list(vh_sequences), desc="VH sequences"):
            try:
                emb = encode_sequence(seq, encoder_dict)
                embeddings[f"VH:{seq}"] = emb
                new_count += 1
            except Exception as e:
                logger.error(f"Failed to encode VH sequence (len={len(seq)}): {e}")
                # Continue with remaining sequences

    # VL sequences
    if vl_sequences:
        logger.info(f"Encoding {len(vl_sequences)} VL sequences...")
        for seq in tqdm(list(vl_sequences), desc="VL sequences"):
            try:
                emb = encode_sequence(seq, encoder_dict)
                embeddings[f"VL:{seq}"] = emb
                new_count += 1
            except Exception as e:
                logger.error(f"Failed to encode VL sequence (len={len(seq)}): {e}")
                # Continue with remaining sequences

    # Save embeddings (merge existing + new)
    logger.info(f"Saving {len(embeddings)} embeddings ({new_count} new) to {output_path}")
    torch.save(embeddings, output_path)

    # Save metadata
    metadata = {
        "source_file": str(csv_path),
        "checksum": checksum,
        "encoder_type": encoder_type,
        "model_name": config["model_name"],
        "embedding_dim": config["embedding_dim"],
        "num_embeddings": len(embeddings),
        "num_vh": len(vh_sequences),
        "num_vl": len(vl_sequences),
        "format": "raw",  # Raw (seq_len, hidden_dim) embeddings
        "use_aho_aligned": use_aho_aligned,  # Whether AHO-aligned sequences were used
        "use_full_chain": use_full_chain,  # Whether full-chain sequences were used
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Saved metadata to {metadata_path}")
    logger.info(f"✓ Pre-computation complete!")
    logger.info(f"   - Embeddings: {output_path}")
    logger.info(f"   - Total size: {output_path.stat().st_size / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings for antibody sequences"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with sequences"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="all",
        choices=list(ENCODER_CONFIGS.keys()) + ["all"],
        help=f"Encoder type (default: all). Options: {', '.join(ENCODER_CONFIGS.keys())}, all"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inputs/embeddings",
        help="Output directory (default: inputs/embeddings)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--forge-token",
        type=str,
        default=None,
        help="Forge API token for ESM-C 6B (or set FORGE_TOKEN env var)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation"
    )
    parser.add_argument(
        "--use-aho-aligned",
        action="store_true",
        help="Use AHO-aligned sequences (heavy_aligned_aho, light_aligned_aho columns)"
    )
    parser.add_argument(
        "--full-chain",
        action="store_true",
        help="Use full-chain sequences (hc_protein_sequence, lc_protein_sequence). Includes constant regions for better Tm2 prediction."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete computation by loading existing embeddings and computing only missing sequences"
    )
    parser.add_argument(
        "--only-vh",
        action="store_true",
        help="Only compute VH sequences (useful for splitting computation)"
    )
    parser.add_argument(
        "--only-vl",
        action="store_true",
        help="Only compute VL sequences (useful for splitting computation)"
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.use_aho_aligned and args.full_chain:
        parser.error("--use-aho-aligned and --full-chain are mutually exclusive")

    if args.only_vh and args.only_vl:
        parser.error("--only-vh and --only-vl are mutually exclusive")

    # If "all" is specified, precompute for all encoders
    if args.encoder == "all":
        encoders_to_process = list(ENCODER_CONFIGS.keys())
        logger.info(f"Precomputing embeddings for all encoders: {', '.join(encoders_to_process)}")
        logger.info("")

        for encoder_type in encoders_to_process:
            # Skip esmc_6b if no FORGE_TOKEN
            if encoder_type == "esmc_6b":
                token = args.forge_token or os.getenv("FORGE_TOKEN")
                if not token:
                    logger.warning(f"⚠️  Skipping esmc_6b (requires FORGE_TOKEN)")
                    logger.info("")
                    continue

            logger.info(f"=" * 80)
            logger.info(f"Processing encoder: {encoder_type}")
            logger.info(f"=" * 80)
            logger.info("")

            try:
                precompute_embeddings(
                    csv_path=args.input,
                    encoder_type=encoder_type,
                    output_dir=args.output_dir,
                    device=args.device,
                    forge_token=args.forge_token,
                    force=args.force,
                    use_aho_aligned=args.use_aho_aligned,
                    use_full_chain=args.full_chain,
                    resume=args.resume,
                    only_vh=args.only_vh,
                    only_vl=args.only_vl
                )
            except Exception as e:
                logger.error(f"❌ Failed to precompute {encoder_type}: {e}")
                logger.error("Continuing with next encoder...")

            logger.info("")
    else:
        # Single encoder
        precompute_embeddings(
            csv_path=args.input,
            encoder_type=args.encoder,
            output_dir=args.output_dir,
            device=args.device,
            forge_token=args.forge_token,
            force=args.force,
            use_aho_aligned=args.use_aho_aligned,
            use_full_chain=args.full_chain,
            resume=args.resume,
            only_vh=args.only_vh,
            only_vl=args.only_vl
        )


if __name__ == "__main__":
    main()
