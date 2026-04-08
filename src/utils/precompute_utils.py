"""Utilities for managing precomputed embeddings and antibody features."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_file_checksum(filepath: str) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        filepath: Path to file

    Returns:
        SHA256 checksum as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_precomputed_file(
    source_csv_path: str,
    precompute_dir: str,
    file_type: str,
    model_name: Optional[str] = None,
    use_aho_aligned: bool = False,
    use_full_chain: bool = False,
    log_level: str = "info"
) -> Optional[str]:
    """
    Find precomputed file for a source CSV (generic function for embeddings/features).

    Automatically detects the file by matching:
    1. Source CSV filename
    2. File type (e.g., model_name for embeddings, "antibody_features" for features)
    3. Sequence representation (normal, AHO-aligned, or full-chain) [optional]
    4. Checksum

    Args:
        source_csv_path: Path to source CSV file
        precompute_dir: Directory containing precomputed files
        file_type: Type identifier (model_name for embeddings, "antibody_features" for features)
        model_name: Deprecated, use file_type instead
        use_aho_aligned: Use AHO-aligned sequences (adds _aho suffix)
        use_full_chain: Use full-chain sequences (adds _fullchain suffix)
        log_level: Logging level ("info", "warning", "debug")

    Returns:
        Path to precomputed file, or None if not found
    """
    # Backward compatibility
    if model_name is not None and file_type == "antibody_features":
        file_type = model_name

    source_path = Path(source_csv_path)
    precompute_path = Path(precompute_dir)

    if not source_path.exists():
        logger.warning(f"Source CSV not found: {source_csv_path}")
        return None

    if not precompute_path.exists():
        if log_level == "warning":
            logger.warning(f"Precompute directory not found: {precompute_dir}")
        else:
            logger.debug(f"Precompute directory not found: {precompute_dir}")
        return None

    # Compute checksum of source file
    checksum = compute_file_checksum(source_csv_path)
    checksum_short = checksum[:8]

    # Add suffix based on sequence representation
    suffix = ""
    if use_aho_aligned:
        suffix = "_aho"
    elif use_full_chain:
        suffix = "_fullchain"

    # Expected pattern: {source_stem}_{file_type}{suffix}_{checksum_short}.pt
    # Examples: GDPa1_complete_esmc_600m_72626a47.pt (embeddings)
    #           GDPa1_complete_antibody_features_72626a47.pt (features)
    source_stem = source_path.stem
    expected_pattern = f"{source_stem}_{file_type}{suffix}_{checksum_short}.pt"

    # Look for matching file
    matching_file = precompute_path / expected_pattern

    if matching_file.exists():
        logger.info(f"✓ Found precomputed {file_type}: {matching_file}")
        return str(matching_file)
    else:
        if log_level == "warning":
            logger.warning(f"⚠️  Precomputed {file_type} not found: {expected_pattern}")
            logger.warning(f"   Expected path: {matching_file}")
        else:
            logger.debug(f"⚠️  Precomputed {file_type} not found: {expected_pattern}")
        return None


def find_precomputed_embeddings(
    source_csv_path: str,
    embeddings_dir: str = "inputs/embeddings",
    model_name: str = "esmc_6b",
    use_aho_aligned: bool = False,
    use_full_chain: bool = False
) -> Optional[str]:
    """
    Find precomputed embeddings file for a source CSV.

    Wrapper around find_precomputed_file for backward compatibility.
    """
    return find_precomputed_file(
        source_csv_path=source_csv_path,
        precompute_dir=embeddings_dir,
        file_type=model_name,
        use_aho_aligned=use_aho_aligned,
        use_full_chain=use_full_chain,
        log_level="warning"
    )


def find_precomputed_antibody_features(
    source_csv_path: str,
    features_dir: str = "inputs/antibody_features",
    use_aho_aligned: bool = False,
    use_full_chain: bool = False,
) -> Optional[str]:
    """
    Find precomputed antibody features file for a source CSV.

    Args:
        source_csv_path: Path to source CSV file
        features_dir: Directory containing precomputed antibody features
        use_aho_aligned: Use AHO-aligned sequences
        use_full_chain: Use full-chain sequences

    Returns:
        Path to precomputed antibody features file, or None if not found
    """
    return find_precomputed_file(
        source_csv_path=source_csv_path,
        precompute_dir=features_dir,
        file_type="antibody_features",
        use_aho_aligned=use_aho_aligned,
        use_full_chain=use_full_chain,
        log_level="debug"
    )


def get_embeddings_config(
    source_csv_path: str,
    encoder_type: str = "esmc",
    embeddings_dir: str = "inputs/embeddings",
    use_aho_aligned: bool = False,
    use_full_chain: bool = False
) -> dict:
    """
    Get embeddings configuration for encoder initialization.

    Automatically detects the appropriate model name based on encoder type
    and sequence representation.

    Args:
        source_csv_path: Path to source CSV file
        encoder_type: Encoder type (esmc, antiberty, prott5, esm2)
        embeddings_dir: Directory containing precomputed embeddings
        use_aho_aligned: Use AHO-aligned sequences (adds _aho suffix)
        use_full_chain: Use full-chain sequences (adds _fullchain suffix)

    Returns:
        Dict with precomputed_embeddings_path and source_csv_path,
        or empty dict if embeddings not found
    """
    # Map encoder types to model names used in embedding filenames
    # Format uses encoder_type directly (antiberty, esmc_300m, esmc_6b, etc.)
    model_name_map = {
        "esmc": "esmc_6b",       # ESM-C 6B (backward compat: esmc -> esmc_6b)
        "esmc_6b": "esmc_6b",    # ESM-C 6B
        "esmc_300m": "esmc_300m", # ESM-C 300M
        "esmc_600m": "esmc_600m", # ESM-C 600M
        "antiberty": "antiberty",
        "prott5": "prott5",
    }

    model_name = model_name_map.get(encoder_type, encoder_type)

    embeddings_path = find_precomputed_embeddings(
        source_csv_path, embeddings_dir, model_name,
        use_aho_aligned=use_aho_aligned,
        use_full_chain=use_full_chain
    )

    if embeddings_path:
        return {
            "precomputed_embeddings_path": embeddings_path,
            "source_csv_path": source_csv_path
        }
    else:
        return {}


def get_antibody_features_config(
    source_csv_path: str,
    features_dir: str = "inputs/antibody_features",
    use_aho_aligned: bool = False,
    use_full_chain: bool = False,
) -> Optional[str]:
    """
    Get antibody features configuration.

    Automatically detects precomputed antibody features file.

    Args:
        source_csv_path: Path to source CSV file
        features_dir: Directory containing precomputed antibody features
        use_aho_aligned: Use AHO-aligned sequences
        use_full_chain: Use full-chain sequences

    Returns:
        Path to precomputed features file if found, None otherwise
    """
    features_path = find_precomputed_antibody_features(
        source_csv_path, features_dir, use_aho_aligned, use_full_chain
    )

    return features_path
