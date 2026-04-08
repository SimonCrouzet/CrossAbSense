#!/usr/bin/env python3
"""
Precompute prediction cache for antibody sequences.

Creates a model-agnostic unified cache with embeddings and features
based on a config file (or default config if not specified).

Input: CSV or FASTA file + optional config
Output: Single .pt cache file with config checksum in filename

Usage:
    # Use default config (computes all common encoders + features)
    python scripts/precompute_prediction_cache.py sequences.csv

    # Use a specific config
    python scripts/precompute_prediction_cache.py sequences.csv \\
        --config config/example_esmc.yaml

    # Specify output path explicitly
    python scripts/precompute_prediction_cache.py sequences.csv \\
        --config config/esmc_300m.yaml \\
        --output cache/my_cache.pt

    # Then use in predict.py with any model
    python -m src.predict --precomputed-cache cache/sequences_a1b2c3d4.pt \\
        --models models/HIC_* --output predictions.csv
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from Bio import SeqIO

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import load_config  # noqa: E402
from src.utils.precompute_embeddings import (  # noqa: E402
    precompute_embeddings as precompute_single_encoder
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_default_config() -> dict:
    """
    Load the full default config from src/config/default_config.yaml.
    This includes all property-specific encoder settings.
    """
    default_config_path = project_root / "src" / "config" / "default_config.yaml"
    
    if default_config_path.exists():
        logger.info(f"Loading default config from {default_config_path}")
        return load_config(str(default_config_path))
    else:
        # Fallback to minimal config if default not found
        logger.warning("Default config file not found, using minimal fallback")
        return {
            'encoder': {
                'type': 'esmc_300m',
            },
            'antibody_features': {
                'enabled': True,
                'use_abnumber': True,
                'use_biophi': True,
                'use_scalop': True,
                'use_sequence_features': True,
                'cdr_definition': 'north',
                'cache_abnumber': True,
            }
        }


def compute_config_checksum(config: dict) -> str:
    """
    Compute a short checksum of the config to identify cache files.
    """
    import json
    # Only hash the relevant parts (encoder and antibody_features)
    relevant = {
        'encoder': config.get('encoder', {}),
        'antibody_features': config.get('antibody_features', {}),
    }
    config_str = json.dumps(relevant, sort_keys=True)
    checksum = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    return checksum


def parse_encoders_and_features_from_config(
    config: dict
) -> Tuple[Set[str], bool, dict]:
    """
    Parse encoder types and antibody features settings from config.
    
    Extracts ALL encoder types from:
    - encoder.type (single encoder)
    - encoder.encoder_types (list of encoders)
    - property_specific.<property>.encoder_type (per-property overrides)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (encoder_types, uses_antibody_features, ab_features_config)
    """
    encoder_types = set()
    encoder_sources = []  # Track where encoders came from
    
    # Get encoder config
    encoder_config = config.get('encoder', {})
    
    # Method 1: encoder.type (single encoder, may be multi like "esmc+antiberty")
    encoder_type = encoder_config.get('type', '')
    if encoder_type:
        for enc in encoder_type.split('+'):
            enc = enc.strip()
            encoder_types.add(enc)
            encoder_sources.append(f"{enc} (from encoder.type)")
    
    # Method 2: encoder.encoder_types (list of encoders)
    encoder_types_list = encoder_config.get('encoder_types', [])
    if encoder_types_list:
        for enc in encoder_types_list:
            enc_stripped = enc.strip()
            if enc_stripped not in encoder_types:  # Avoid duplicate logging
                encoder_sources.append(f"{enc_stripped} (from encoder.encoder_types)")
            encoder_types.add(enc_stripped)
    
    # Method 3: property_specific.<property>.encoder_type (overrides)
    property_specific = config.get('property_specific', {})
    for prop_name, property_config in property_specific.items():
        if isinstance(property_config, dict):
            prop_encoder = property_config.get('encoder_type', '')
            if prop_encoder:
                for enc in prop_encoder.split('+'):
                    enc = enc.strip()
                    if enc not in encoder_types:  # Avoid duplicate logging
                        encoder_sources.append(f"{enc} (from property_specific.{prop_name})")
                    encoder_types.add(enc)
    
    # Log sources if we found encoders
    if encoder_sources:
        logger.info("Found encoder types in config:")
        for source in encoder_sources:
            logger.info(f"  - {source}")
    
    # Normalize encoder types
    normalized_encoder_types = set()
    for enc in encoder_types:
        # Special handling for 'esmc' - try to infer variant from nested config
        if enc == 'esmc':
            esmc_config = encoder_config.get('esmc', {})
            model_name = esmc_config.get('model_name', '')
            
            # Also check encoder_configs.esmc for multi-encoder setup
            if not model_name:
                encoder_configs = encoder_config.get('encoder_configs', {})
                esmc_config = encoder_configs.get('esmc', {})
                model_name = esmc_config.get('model_name', '')
            
            # Extract variant from model name (e.g., "facebook/esmc_600m" -> "esmc_600m")
            if 'esmc_300m' in model_name or model_name == 'esmc_300m':
                enc = 'esmc_300m'
                logger.info(f"  → Normalized 'esmc' to 'esmc_300m' (from model_name: {model_name})")
            elif 'esmc_600m' in model_name or model_name == 'esmc_600m':
                enc = 'esmc_600m'
                logger.info(f"  → Normalized 'esmc' to 'esmc_600m' (from model_name: {model_name})")
            elif 'esmc_6b' in model_name or model_name == 'esmc_6b':
                enc = 'esmc_6b'
                logger.info(f"  → Normalized 'esmc' to 'esmc_6b' (from model_name: {model_name})")
            else:
                # Default to esmc_300m if no specific variant found
                enc = 'esmc_300m'
                logger.info(f"  → Normalized 'esmc' to 'esmc_300m' (default, no model_name found)")
        
        normalized_encoder_types.add(enc)
    
    # Check if antibody features are enabled
    ab_feat_config = config.get('antibody_features', {})
    uses_antibody_features = ab_feat_config.get('enabled', False)
    
    return normalized_encoder_types, uses_antibody_features, ab_feat_config


def detect_required_encoders_and_features(
    model_dirs: List[Path]
) -> Tuple[Set[str], bool]:
    """
    [DEPRECATED] Auto-detect encoder types and whether antibody features
    are needed from model configs.
    
    This function is kept for backward compatibility but is no longer
    used in the main workflow.
    
    Args:
        model_dirs: List of model directories
    
    Returns:
        Tuple of (set of encoder types, whether antibody features are used)
    """
    encoder_types = set()
    uses_antibody_features = False
    
    for model_dir in model_dirs:
        config_path = model_dir / "config.yaml"
        if not config_path.exists():
            logger.warning(f"⚠️  No config found in {model_dir}, skipping")
            continue
        
        try:
            config = load_config(str(config_path))
            
            # Get encoder types - check multiple possible locations
            # 1. New format: encoder.encoder_types (list)
            encoder_config = config.get("encoder", {})
            enc_types_list = encoder_config.get("encoder_types", [])
            if enc_types_list:
                for enc in enc_types_list:
                    encoder_types.add(enc.strip())
            
            # 2. Legacy format: encoder_type (string, may be multi-encoder)
            encoder_type = config.get("encoder_type", "")
            if encoder_type:
                # Handle multi-encoder (e.g., "esmc_300m+antiberty")
                for enc in encoder_type.split('+'):
                    encoder_types.add(enc.strip())
            
            # Check if antibody features are enabled
            ab_feat_config = config.get("antibody_features", {})
            if ab_feat_config.get("enabled", False):
                uses_antibody_features = True
        
        except Exception as e:
            logger.warning(f"⚠️  Failed to load config from {model_dir}: {e}")
    
    return encoder_types, uses_antibody_features


def load_precomputed_embeddings(
    csv_path: Path,
    encoder_types: Set[str],
    embeddings_dir: Path = Path("inputs/embeddings")
) -> Dict[Tuple[str, str], Dict[str, torch.Tensor]]:
    """
    Load precomputed embeddings from individual encoder files and combine them.
    
    Returns:
        Dict mapping (vh_seq, vl_seq) -> {encoder_type: concatenated_embedding}
    """
    # Compute checksum of input file
    sha256_hash = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    checksum = sha256_hash.hexdigest()[:8]
    
    combined_embeddings = {}
    
    for encoder_type in encoder_types:
        # Find precomputed embedding file
        expected_name = f"{csv_path.stem}_{encoder_type}_{checksum}.pt"
        embedding_path = embeddings_dir / expected_name
        
        if not embedding_path.exists():
            logger.warning(
                f"⚠️  Precomputed embeddings not found for "
                f"{encoder_type}: {embedding_path}"
            )
            logger.warning(
                f"    Run: python -m src.utils.precompute_embeddings "
                f"--input {csv_path} --encoder {encoder_type}"
            )
            continue
        
        logger.info(f"Loading {encoder_type} embeddings from {embedding_path}")
        encoder_embeddings = torch.load(
            embedding_path, map_location='cpu', weights_only=False
        )
        
        # Convert from {"VH:seq": emb} to {(vh_seq, vl_seq): {encoder: emb}}
        # Need to pair VH and VL sequences from the CSV
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            vh_seq = (row.get('vh_protein_sequence') or
                      row.get('hc_protein_sequence'))
            vl_seq = (row.get('vl_protein_sequence') or
                      row.get('lc_protein_sequence'))
            
            if pd.isna(vh_seq) or pd.isna(vl_seq):
                continue
            
            vh_key = f"VH:{vh_seq}"
            vl_key = f"VL:{vl_seq}"
            
            if vh_key in encoder_embeddings and vl_key in encoder_embeddings:
                pair_key = (vh_seq, vl_seq)
                
                if pair_key not in combined_embeddings:
                    combined_embeddings[pair_key] = {}
                
                # Concatenate VH and VL embeddings
                vh_emb = encoder_embeddings[vh_key]
                vl_emb = encoder_embeddings[vl_key]
                # Concat along sequence length
                combined = torch.cat([vh_emb, vl_emb], dim=0)
                
                combined_embeddings[pair_key][encoder_type] = combined
        
        num_pairs = len([
            k for k in combined_embeddings
            if encoder_type in combined_embeddings[k]
        ])
        logger.info(f"  ✓ Loaded embeddings for {num_pairs} pairs")
    
    return combined_embeddings


def read_fasta_to_dataframe(input_path: Path) -> pd.DataFrame:
    """
    Read sequences from FASTA file and create DataFrame.
    
    Expected FASTA headers:
        >SEQID_VH
        EVQL...
        >SEQID_VL
        DIQM...
    
    Or paired format:
        >SEQID
        EVQL...|DIQM...  (VH|VL separated by |)
    
    Returns:
        DataFrame with columns: antibody_id, vh_protein_sequence,
        vl_protein_sequence
    """
    logger.info(f"Reading FASTA file: {input_path}")
    
    sequences = {}
    for record in SeqIO.parse(input_path, "fasta"):
        seq_id = record.id
        seq_str = str(record.seq)
        
        # Check if it's paired format (VH|VL in one line)
        if '|' in seq_str:
            vh_seq, vl_seq = seq_str.split('|', 1)
            base_id = seq_id.replace('_VH', '').replace('_VL', '')
            if base_id not in sequences:
                sequences[base_id] = {}
            sequences[base_id]['vh'] = vh_seq.strip()
            sequences[base_id]['vl'] = vl_seq.strip()
        else:
            # Separate VH/VL format
            if seq_id.endswith('_VH'):
                base_id = seq_id[:-3]
                if base_id not in sequences:
                    sequences[base_id] = {}
                sequences[base_id]['vh'] = seq_str
            elif seq_id.endswith('_VL'):
                base_id = seq_id[:-3]
                if base_id not in sequences:
                    sequences[base_id] = {}
                sequences[base_id]['vl'] = seq_str
            else:
                logger.warning(
                    f"Skipping record {seq_id} - unclear if VH or VL"
                )
    
    # Convert to DataFrame
    data = []
    for antibody_id, seqs in sequences.items():
        if 'vh' in seqs and 'vl' in seqs:
            data.append({
                'antibody_id': antibody_id,
                'vh_protein_sequence': seqs['vh'],
                'vl_protein_sequence': seqs['vl']
            })
        else:
            logger.warning(f"Skipping {antibody_id} - missing VH or VL")
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} antibody pairs from FASTA")
    return df


def read_sequences_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Read sequences from CSV file.
    
    Expected columns:
        - vh_protein_sequence (or hc_protein_sequence)
        - vl_protein_sequence (or lc_protein_sequence)
        - antibody_id (optional)
    """
    logger.info(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check for columns
    vh_col = None
    vl_col = None
    
    if 'vh_protein_sequence' in df.columns:
        vh_col = 'vh_protein_sequence'
    elif 'hc_protein_sequence' in df.columns:
        vh_col = 'hc_protein_sequence'
    else:
        raise ValueError(
            "CSV must contain 'vh_protein_sequence' or "
            "'hc_protein_sequence' column"
        )
    
    if 'vl_protein_sequence' in df.columns:
        vl_col = 'vl_protein_sequence'
    elif 'lc_protein_sequence' in df.columns:
        vl_col = 'lc_protein_sequence'
    else:
        raise ValueError(
            "CSV must contain 'vl_protein_sequence' or "
            "'lc_protein_sequence' column"
        )
    
    # Standardize column names
    if vh_col != 'vh_protein_sequence':
        df['vh_protein_sequence'] = df[vh_col]
    if vl_col != 'vl_protein_sequence':
        df['vl_protein_sequence'] = df[vl_col]
    
    # Add antibody_id if missing
    if 'antibody_id' not in df.columns:
        df['antibody_id'] = [f"Ab{i:06d}" for i in range(len(df))]
    
    logger.info(f"Loaded {len(df)} antibody pairs from CSV")
    return df


def load_biophi_summary(
    input_path: Path,
    df: pd.DataFrame
) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load BioPhi summary file if it exists.
    
    Looks for file with pattern: <input_stem>_biophi_summary.xlsx
    
    Args:
        input_path: Original input file path
        df: DataFrame with antibody_id column
    
    Returns:
        Dict mapping antibody_id -> {feature_name: value}
        Returns None if file doesn't exist
    """
    # Construct expected BioPhi summary filename
    biophi_path = input_path.parent / f"{input_path.stem}_biophi_summary.xlsx"
    
    if not biophi_path.exists():
        logger.info(f"No BioPhi summary found at {biophi_path}")
        return None
    
    try:
        logger.info(f"Loading BioPhi summary from {biophi_path}")
        biophi_df = pd.read_excel(biophi_path)
        
        # Create mapping from antibody_id to BioPhi features
        biophi_dict = {}
        
        for _, row in biophi_df.iterrows():
            antibody_id = str(row['Antibody'])
            
            # Extract relevant BioPhi features
            features = {
                'oasis_percentile': row.get('OASis Percentile', 0.0),
                'oasis_identity': row.get('OASis Identity', 0.0),
                'germline_content': row.get('Germline Content', 0.0),
                'heavy_oasis_percentile': row.get('Heavy OASis Percentile', 0.0),
                'heavy_oasis_identity': row.get('Heavy OASis Identity', 0.0),
                'heavy_germline_content': row.get('Heavy Germline Content', 0.0),
                'light_oasis_percentile': row.get('Light OASis Percentile', 0.0),
                'light_oasis_identity': row.get('Light OASis Identity', 0.0),
                'light_germline_content': row.get('Light Germline Content', 0.0),
            }
            
            biophi_dict[antibody_id] = features
        
        logger.info(f"✓ Loaded BioPhi features for {len(biophi_dict)} antibodies")
        return biophi_dict
    
    except Exception as e:
        logger.warning(f"Failed to load BioPhi summary: {e}")
        return None


def precompute_antibody_features(
    df: pd.DataFrame,
    use_abnumber: bool = True,
    use_biophi: bool = True,
    use_scalop: bool = True,
    use_sequence_features: bool = True,
    cdr_definition: str = "north",
    biophi_summary: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[Tuple[str, str], torch.Tensor]:
    """
    Precompute antibody features for all unique (VH, VL) pairs.
    
    Only extracts locally-computable features (abnumber, scalop, sequence features).
    If biophi_summary is provided, BioPhi online queries are skipped but BioPhi
    features are NOT added to the cache (only used to avoid API calls).
    
    Args:
        df: DataFrame with sequences
        use_abnumber: Extract ANARCI/abnumber features
        use_biophi: Extract BioPhi features (online if not in summary)
        use_scalop: Extract SAbDab/scalop features
        use_sequence_features: Extract basic sequence features
        cdr_definition: CDR definition scheme
        biophi_summary: Pre-loaded BioPhi summary (used to skip online queries)
    
    Returns:
        Dict mapping (vh_seq, vl_seq) -> feature_tensor
    """
    from src.features.antibody_features import AntibodyFeatures
    
    # Get unique pairs with antibody IDs
    unique_pairs = []
    for _, row in df.iterrows():
        vh_seq = (row.get('vh_protein_sequence') or
                  row.get('hc_protein_sequence'))
        vl_seq = (row.get('vl_protein_sequence') or
                  row.get('lc_protein_sequence'))
        antibody_id = row.get('antibody_id', '')
        
        if pd.notna(vh_seq) and pd.notna(vl_seq):
            unique_pairs.append((vh_seq, vl_seq, antibody_id))
    
    logger.info(f"\n{'='*70}")
    logger.info(
        f"Precomputing antibody features for {len(unique_pairs)} "
        f"unique pairs"
    )
    logger.info(f"{'='*70}")
    
    # If BioPhi summary is provided, skip online BioPhi queries
    # Note: BioPhi features from summary are stored separately in cache
    if biophi_summary:
        logger.info("Using pre-computed BioPhi summary (skipping online queries)")
        use_biophi_online = False
    else:
        use_biophi_online = use_biophi
    
    # Initialize feature extractor
    extractor = AntibodyFeatures(
        use_abnumber=use_abnumber,
        use_biophi=use_biophi_online,
        use_scalop=use_scalop,
        use_sequence_features=use_sequence_features,
        cdr_definition=cdr_definition,
        cache_abnumber=True,
    )
    
    feature_dim = extractor.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")
    
    # Extract features
    features_dict = {}
    failed = 0
    
    for vh_seq, vl_seq, antibody_id in tqdm(unique_pairs, desc="Extracting features"):
        try:
            features = extractor.extract_features(vh_seq, vl_seq)
            features_array = extractor.features_to_array(features)
            features_tensor = torch.from_numpy(features_array).float()
            features_dict[(vh_seq, vl_seq)] = features_tensor
        except Exception as e:
            logger.warning(f"Failed to extract features for {antibody_id}: {e}")
            failed += 1
    
    logger.info(f"✓ Extracted features for {len(features_dict)} pairs")
    if failed > 0:
        logger.warning(f"⚠️  Failed for {failed} pairs")
    
    return features_dict


def create_prediction_cache(
    input_path: str,
    config: Optional[dict] = None,
    output_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    forge_token: Optional[str] = None,
    force: bool = False,
    embeddings_dir: str = "inputs/embeddings"
):
    """
    Create a model-agnostic prediction cache based on config.
    
    Args:
        input_path: Path to CSV or FASTA file
        config: Configuration dict (uses default if None)
        output_path: Path to save cache file (auto-generated if None)
        device: Device to use for encoding
        forge_token: Forge API token for ESMC-6B
        force: Overwrite existing cache
        embeddings_dir: Directory where embeddings are/will be stored
    """
    input_path = Path(input_path)
    embeddings_dir = Path(embeddings_dir)
    
    # Use default config if not provided
    if config is None:
        logger.info("Using default configuration")
        config = get_default_config()
    
    # Parse config to determine what to compute
    logger.info(f"\n{'='*70}")
    logger.info("Parsing configuration")
    logger.info(f"{'='*70}")
    
    encoder_types, uses_antibody_features, ab_feat_config = \
        parse_encoders_and_features_from_config(config)
    
    if not encoder_types:
        logger.warning("⚠️  No encoders specified in config!")
    else:
        logger.info(f"Encoders: {', '.join(sorted(encoder_types))}")
    
    logger.info(f"Antibody features: {'✓' if uses_antibody_features else '✗'}")
    
    # Auto-generate output path if not provided
    if output_path is None:
        # Use input file's directory and stem with _cache.pt suffix
        output_path = input_path.parent / f"{input_path.stem}_cache.pt"
        logger.info(f"Auto-generated output path: {output_path}")
    else:
        output_path = Path(output_path)
    
    # Check if output exists
    if output_path.exists() and not force:
        logger.info(f"Cache already exists: {output_path}")
        logger.info("Use --force to overwrite")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("Creating Prediction Cache")
    logger.info("="*70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device}")
    
    # Read input file (convert FASTA to CSV if needed)
    if input_path.suffix in ['.fa', '.fasta']:
        df = read_fasta_to_dataframe(input_path)
        # Save as CSV for precompute_embeddings
        csv_path = input_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Converted FASTA to CSV: {csv_path}")
    elif input_path.suffix == '.csv':
        df = read_sequences_from_csv(str(input_path))
        csv_path = input_path
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    logger.info(f"Loaded {len(df)} antibody pairs")
    
    # Initialize cache
    cache = {
        'metadata': {
            'source_file': str(input_path),
            'num_sequences': len(df),
            'encoder_types': list(encoder_types),
            'has_antibody_features': uses_antibody_features,
            'config_checksum': compute_config_checksum(config),
            'config': config,
        },
        'sequences': df.to_dict('records'),
    }
    
    # Precompute embeddings for each encoder type
    if encoder_types:
        logger.info(f"\n{'='*70}")
        logger.info("Computing embeddings in memory")
        logger.info(f"{'='*70}")
        
        embeddings_dict = {}
        
        for encoder_type in sorted(encoder_types):
            logger.info(f"\nComputing {encoder_type} embeddings...")
            
            try:
                # Load encoder
                from src.utils.precompute_embeddings import load_encoder, encode_sequence
                logger.info(f"Loading {encoder_type} encoder...")
                encoder_dict = load_encoder(encoder_type, device, forge_token)
                
                # Extract sequences
                vh_sequences = df['vh_protein_sequence'].tolist()
                vl_sequences = df['vl_protein_sequence'].tolist()
                
                # Encode VH sequences
                logger.info(f"Encoding {len(vh_sequences)} VH sequences...")
                vh_embeddings = []
                for seq in tqdm(vh_sequences, desc=f"{encoder_type} VH", ncols=80):
                    embedding = encode_sequence(seq, encoder_dict)
                    vh_embeddings.append(embedding.cpu())
                
                # Encode VL sequences
                logger.info(f"Encoding {len(vl_sequences)} VL sequences...")
                vl_embeddings = []
                for seq in tqdm(vl_sequences, desc=f"{encoder_type} VL", ncols=80):
                    embedding = encode_sequence(seq, encoder_dict)
                    vl_embeddings.append(embedding.cpu())
                
                # Store in dict (per-pair format)
                for i in range(len(df)):
                    pair_id = f"pair_{i}"
                    if pair_id not in embeddings_dict:
                        embeddings_dict[pair_id] = {}
                    embeddings_dict[pair_id][encoder_type] = {
                        'vh': vh_embeddings[i],
                        'vl': vl_embeddings[i]
                    }
                
                logger.info(f"✓ Computed {encoder_type} embeddings")
                
                # Clear GPU memory
                del encoder_dict
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to compute {encoder_type}: {e}")
                raise
        
        cache['embeddings'] = embeddings_dict
        logger.info(f"✓ Stored embeddings for {len(embeddings_dict)} pairs")
    
    # Check for BioPhi summary file
    biophi_summary = None
    if uses_antibody_features and ab_feat_config.get('use_biophi', True):
        biophi_summary = load_biophi_summary(input_path, df)
        if biophi_summary:
            cache['metadata']['has_biophi_summary'] = True
    
    # Precompute antibody features if needed
    if uses_antibody_features:
        features_dict = precompute_antibody_features(
            df,
            use_abnumber=ab_feat_config.get('use_abnumber', True),
            use_biophi=ab_feat_config.get('use_biophi', True),
            use_scalop=ab_feat_config.get('use_scalop', True),
            use_sequence_features=ab_feat_config.get(
                'use_sequence_features', True
            ),
            cdr_definition=ab_feat_config.get('cdr_definition', 'north'),
            biophi_summary=biophi_summary
        )
        cache['antibody_features'] = features_dict
    
    # Save cache
    logger.info(f"\n{'='*70}")
    logger.info("Saving unified cache")
    logger.info(f"{'='*70}")
    torch.save(cache, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved cache: {output_path} ({file_size_mb:.2f} MB)")
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ Cache creation complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Sequences: {len(df)}")
    if encoder_types:
        logger.info(f"Encoders: {', '.join(sorted(encoder_types))}")
    if uses_antibody_features:
        logger.info(f"Antibody features: ✓")
    logger.info(f"\nTo use in prediction:")
    logger.info(f"    python -m src.predict --precomputed-cache {output_path} \\")
    logger.info(f"        --models models/PROPERTY_* --output predictions.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute model-agnostic prediction cache from config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file (CSV or FASTA)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Config file (uses default if not specified)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output cache file (.pt, auto-generated if not specified)"
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="inputs/embeddings",
        help="Directory for intermediate embeddings (default: inputs/embeddings)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--forge-token",
        type=str,
        help="Forge API token for ESMC-6B (or set FORGE_TOKEN env var)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache and recompute embeddings"
    )
    
    args = parser.parse_args()
    
    # Get forge token from env if not provided
    import os
    forge_token = args.forge_token or os.environ.get("FORGE_TOKEN")
    
    # Load config if provided
    config = None
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
    
    # Create cache
    create_prediction_cache(
        input_path=args.input,
        config=config,
        output_path=args.output,
        device=args.device,
        forge_token=forge_token,
        force=args.force,
        embeddings_dir=args.embeddings_dir
    )


if __name__ == "__main__":
    main()
