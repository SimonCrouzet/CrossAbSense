#!/usr/bin/env python3
"""
Pre-compute antibody developability features.

Extracts sequence-based features (germline, CDR, humanness, liabilities, etc.)
and caches them for faster training. Similar to precomputed embeddings.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.antibody_features import AntibodyFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_file_checksum(filepath: str, length: int = 8) -> str:
    """Compute short SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:length]


def get_unique_sequences(
    csv_path: str,
    vh_col: str = "vh_protein_sequence",
    vl_col: str = "vl_protein_sequence",
    use_aho_aligned: bool = False,
    use_full_chain: bool = False,
) -> Tuple[Set[Tuple[str, str]], pd.DataFrame]:
    """Extract unique (VH, VL) sequence pairs from CSV."""
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
                    logger.info("Extracting features from mature protein sequences (signal peptide removed)")
                else:
                    logger.info("No signal peptide detected (sequences already mature)")

    logger.info(f"Total antibodies: {len(df)}")
    logger.info(f"Using columns: {vh_col}, {vl_col}")

    # Get unique sequence pairs
    unique_pairs = set()
    for _, row in df.iterrows():
        vh_seq = row[vh_col]
        vl_seq = row[vl_col]
        if pd.notna(vh_seq) and pd.notna(vl_seq):
            unique_pairs.add((vh_seq, vl_seq))

    logger.info(f"Unique sequence pairs: {len(unique_pairs)}")
    return unique_pairs, df


def precompute_antibody_features(
    csv_path: str,
    output_dir: str = "inputs/antibody_features",
    vh_col: str = "vh_protein_sequence",
    vl_col: str = "vl_protein_sequence",
    use_abnumber: bool = True,
    use_biophi: bool = True,
    use_scalop: bool = True,
    use_sequence_features: bool = True,
    cdr_definition: str = "north",
    force: bool = False,
    use_aho_aligned: bool = False,
    use_full_chain: bool = False,
) -> str:
    """
    Precompute antibody features for all sequences in CSV.

    Args:
        csv_path: Path to CSV file with antibody sequences
        output_dir: Directory to save precomputed features
        vh_col: Column name for heavy chain sequences (ignored if use_aho_aligned or use_full_chain=True)
        vl_col: Column name for light chain sequences (ignored if use_aho_aligned or use_full_chain=True)
        use_abnumber: Enable abnumber-based features
        use_biophi: Enable BioPhi humanness scores
        use_scalop: Enable ScaLoP canonical classes
        use_sequence_features: Enable sequence-based features
        cdr_definition: CDR definition (north, chothia, kabat, imgt)
        force: Force recomputation even if output exists
        use_aho_aligned: Use AHO-aligned sequences (heavy_aligned_aho, light_aligned_aho)
        use_full_chain: Use full-chain sequences (hc_protein_sequence, lc_protein_sequence)

    Returns:
        Path to saved features file
    """
    # Mutually exclusive options
    if use_aho_aligned and use_full_chain:
        raise ValueError("--use-aho-aligned and --full-chain are mutually exclusive")

    # Compute file checksum for version tracking
    checksum = compute_file_checksum(csv_path)
    csv_name = Path(csv_path).stem

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename (add suffix based on sequence type)
    suffix = ""
    if use_aho_aligned:
        suffix = "_aho"
    elif use_full_chain:
        suffix = "_fullchain"
    output_path = output_dir / f"{csv_name}_antibody_features{suffix}_{checksum}.pt"

    if output_path.exists() and not force:
        logger.info(f"✓ Features already exist: {output_path}")
        logger.info("Use --force to recompute")
        return str(output_path)

    logger.info("="*70)
    logger.info("Precomputing Antibody Features")
    logger.info("="*70)
    logger.info(f"Input CSV: {csv_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checksum: {checksum}")

    # Get unique sequence pairs
    unique_pairs, df = get_unique_sequences(csv_path, vh_col, vl_col, use_aho_aligned, use_full_chain)

    # Initialize feature extractor
    logger.info("\n" + "="*70)
    logger.info("Initializing AntibodyFeatures")
    logger.info("="*70)
    logger.info(f"  use_abnumber: {use_abnumber}")
    logger.info(f"  use_biophi: {use_biophi}")
    logger.info(f"  use_scalop: {use_scalop}")
    logger.info(f"  use_sequence_features: {use_sequence_features}")
    logger.info(f"  cdr_definition: {cdr_definition}")

    extractor = AntibodyFeatures(
        use_abnumber=use_abnumber,
        use_biophi=use_biophi,
        use_scalop=use_scalop,
        use_sequence_features=use_sequence_features,
        cdr_definition=cdr_definition,
        cache_abnumber=True,  # Always cache for performance
    )

    feature_dim = extractor.get_feature_dim()
    logger.info(f"\n✓ Feature dimension: {feature_dim}")
    logger.info(f"✓ Feature names: {extractor.get_feature_names()[:5]}... (first 5)")

    # Extract features for all unique pairs
    logger.info("\n" + "="*70)
    logger.info(f"Extracting features from {len(unique_pairs)} unique pairs")
    logger.info("="*70)

    features_dict = {}
    failed = 0

    for vh_seq, vl_seq in tqdm(unique_pairs, desc="Extracting features"):
        try:
            # Extract features
            features = extractor.extract_features(vh_seq, vl_seq)

            # Convert to tensor
            features_array = extractor.features_to_array(features)
            features_tensor = torch.from_numpy(features_array).float()

            # Store with key (vh_seq, vl_seq)
            features_dict[(vh_seq, vl_seq)] = features_tensor

        except Exception as e:
            logger.warning(f"Failed to extract features for sequence pair: {e}")
            failed += 1

    logger.info(f"\n✓ Extracted features for {len(features_dict)} pairs")
    if failed > 0:
        logger.warning(f"⚠️  Failed to extract features for {failed} pairs")

    # Save to disk
    logger.info(f"\nSaving features to {output_path}")
    torch.save(features_dict, output_path)

    # Verify file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved {file_size_mb:.2f} MB")

    logger.info("\n" + "="*70)
    logger.info("✅ Precomputation complete!")
    logger.info("="*70)
    logger.info(f"Features file: {output_path}")
    logger.info(f"Total pairs: {len(features_dict)}")
    logger.info(f"Feature dimension: {feature_dim}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute antibody developability features"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file with antibody sequences"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inputs/antibody_features",
        help="Directory to save precomputed features (default: inputs/antibody_features)"
    )
    parser.add_argument(
        "--vh-col",
        type=str,
        default="vh_protein_sequence",
        help="Column name for heavy chain sequences (default: vh_protein_sequence)"
    )
    parser.add_argument(
        "--vl-col",
        type=str,
        default="vl_protein_sequence",
        help="Column name for light chain sequences (default: vl_protein_sequence)"
    )
    parser.add_argument(
        "--no-abnumber",
        action="store_true",
        help="Disable abnumber-based features (germline, CDR lengths)"
    )
    parser.add_argument(
        "--no-biophi",
        action="store_true",
        help="Disable BioPhi humanness scores"
    )
    parser.add_argument(
        "--no-scalop",
        action="store_true",
        help="Disable ScaLoP canonical structure classes"
    )
    parser.add_argument(
        "--no-sequence-features",
        action="store_true",
        help="Disable sequence-based features (liabilities, charge, pI)"
    )
    parser.add_argument(
        "--cdr-definition",
        type=str,
        default="north",
        choices=["north", "chothia", "kabat", "imgt"],
        help="CDR definition scheme (default: north)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if output exists"
    )
    parser.add_argument(
        "--use-aho-aligned",
        action="store_true",
        help="Use AHO-aligned sequences (heavy_aligned_aho, light_aligned_aho)"
    )
    parser.add_argument(
        "--full-chain",
        action="store_true",
        help="Use full-chain sequences (hc_protein_sequence, lc_protein_sequence). Strips signal peptide to work with mature proteins."
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.use_aho_aligned and args.full_chain:
        parser.error("--use-aho-aligned and --full-chain are mutually exclusive")

    # Run precomputation
    output_path = precompute_antibody_features(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        vh_col=args.vh_col,
        vl_col=args.vl_col,
        use_abnumber=not args.no_abnumber,
        use_biophi=not args.no_biophi,
        use_scalop=not args.no_scalop,
        use_sequence_features=not args.no_sequence_features,
        cdr_definition=args.cdr_definition,
        force=args.force,
        use_aho_aligned=args.use_aho_aligned,
        use_full_chain=args.full_chain,
    )

    print(f"\n✅ Done! Features saved to: {output_path}")
    print(f"\nTo use in training, add to your config:")
    print(f"  antibody_features_path: \"{output_path}\"")


if __name__ == "__main__":
    main()
