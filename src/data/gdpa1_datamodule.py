"""PyTorch Lightning DataModule for GDPa1 dataset."""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.target_transforms import (
    TargetTransform,
    create_transform,
    get_recommended_transform,
)

logger = logging.getLogger(__name__)


def extract_constant_regions_from_reference(reference_csv: str = "inputs/GDPa1_complete.csv") -> Dict[str, str]:
    """
    Extract constant region and signal peptide sequences from reference dataset.

    Args:
        reference_csv: Path to CSV with full-chain sequences (hc_protein_sequence, lc_protein_sequence)

    Returns:
        Dictionary mapping:
        - 'HC_{subtype}' -> heavy chain constant region
        - 'LC_{subtype}' -> light chain constant region
        - 'signal_peptide' -> signal peptide (common for all chains)

    Raises:
        ValueError: If reference CSV doesn't have required columns or regions are inconsistent
    """
    logger.info(f"Extracting constant regions and signal peptide from {reference_csv}...")

    df = pd.read_csv(reference_csv)

    # Check required columns
    required = ['vh_protein_sequence', 'vl_protein_sequence', 'hc_protein_sequence',
                'lc_protein_sequence', 'hc_subtype', 'lc_subtype']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Reference CSV missing columns for constant region extraction: {missing}")

    constant_regions = {}

    # Extract signal peptide (common for all chains)
    # Signal peptide is what comes before the variable region in full chain
    first_ab = df.iloc[0]
    vh_start = first_ab['hc_protein_sequence'].find(first_ab['vh_protein_sequence'])
    if vh_start == -1:
        raise ValueError("Could not find VH in HC to extract signal peptide")
    signal_peptide = first_ab['hc_protein_sequence'][:vh_start]

    # Verify signal peptide is consistent across all antibodies
    for _, row in df.iterrows():
        vh_start_i = row['hc_protein_sequence'].find(row['vh_protein_sequence'])
        vl_start_i = row['lc_protein_sequence'].find(row['vl_protein_sequence'])

        if vh_start_i >= 0:
            sig_hc = row['hc_protein_sequence'][:vh_start_i]
            if sig_hc != signal_peptide:
                raise ValueError(f"Inconsistent signal peptide in HC: expected '{signal_peptide}', got '{sig_hc}'")

        if vl_start_i >= 0:
            sig_lc = row['lc_protein_sequence'][:vl_start_i]
            if sig_lc != signal_peptide:
                raise ValueError(f"Inconsistent signal peptide in LC: expected '{signal_peptide}', got '{sig_lc}'")

    constant_regions['signal_peptide'] = signal_peptide
    logger.info(f"  Detected signal peptide ({len(signal_peptide)} AA): {signal_peptide}")
    logger.info(f"  Note: Signal peptide will be removed (working with mature proteins)")

    # Extract heavy chain constant regions
    for subtype in ['IgG1', 'IgG2', 'IgG4']:
        subset = df[df['hc_subtype'] == subtype]
        if len(subset) == 0:
            continue

        # Get first antibody as reference
        first = subset.iloc[0]
        vh = first['vh_protein_sequence']
        hc = first['hc_protein_sequence']

        # Find where VH ends in HC
        vh_start = hc.find(vh)
        if vh_start == -1:
            raise ValueError(f"Could not find VH in HC for {first.get('antibody_id', 'first antibody')} ({subtype})")

        constant_region = hc[vh_start + len(vh):]

        # Verify consistency across all antibodies of this subtype
        for _, row in subset.iterrows():
            vh_i = row['vh_protein_sequence']
            hc_i = row['hc_protein_sequence']
            vh_start_i = hc_i.find(vh_i)
            if vh_start_i == -1:
                raise ValueError(f"Could not find VH in HC for {row.get('antibody_id', 'unknown')} ({subtype})")
            constant_i = hc_i[vh_start_i + len(vh_i):]
            if constant_i != constant_region:
                raise ValueError(f"Inconsistent constant regions found for {subtype}")

        constant_regions[f'HC_{subtype}'] = constant_region
        logger.info(f"  Extracted HC_{subtype} constant region ({len(constant_region)} AA)")

    # Extract light chain constant regions
    for subtype in ['Kappa', 'Lambda']:
        subset = df[df['lc_subtype'] == subtype]
        if len(subset) == 0:
            continue

        first = subset.iloc[0]
        vl = first['vl_protein_sequence']
        lc = first['lc_protein_sequence']

        vl_start = lc.find(vl)
        if vl_start == -1:
            raise ValueError(f"Could not find VL in LC for {first.get('antibody_id', 'first antibody')} ({subtype})")

        constant_region = lc[vl_start + len(vl):]

        # Verify consistency
        for _, row in subset.iterrows():
            vl_i = row['vl_protein_sequence']
            lc_i = row['lc_protein_sequence']
            vl_start_i = lc_i.find(vl_i)
            if vl_start_i == -1:
                raise ValueError(f"Could not find VL in LC for {row.get('antibody_id', 'unknown')} ({subtype})")
            constant_i = lc_i[vl_start_i + len(vl_i):]
            if constant_i != constant_region:
                raise ValueError(f"Inconsistent constant regions found for {subtype}")

        constant_regions[f'LC_{subtype}'] = constant_region
        logger.info(f"  Extracted LC_{subtype} constant region ({len(constant_region)} AA)")

    logger.info(f"✓ Extracted {len(constant_regions)} constant regions")
    return constant_regions


def reconstruct_full_chains(
    df: pd.DataFrame,
    constant_regions: Dict[str, str],
    inplace: bool = False
) -> pd.DataFrame:
    """
    Reconstruct full-chain sequences from variable regions using constant regions.

    Args:
        df: DataFrame with vh_protein_sequence, vl_protein_sequence, hc_subtype, lc_subtype
        constant_regions: Dict mapping 'HC_{subtype}' and 'LC_{subtype}' to constant regions
        inplace: Whether to modify DataFrame in place

    Returns:
        DataFrame with added hc_protein_sequence and lc_protein_sequence columns

    Raises:
        ValueError: If required columns are missing or unknown subtypes encountered
    """
    if not inplace:
        df = df.copy()

    # Check required columns
    required = ['vh_protein_sequence', 'vl_protein_sequence', 'hc_subtype', 'lc_subtype']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot reconstruct full chains: missing columns {missing}")

    logger.info(f"Reconstructing full-chain sequences for {len(df)} antibodies...")

    def reconstruct_row(row):
        # Get variable regions and subtypes
        vh = row['vh_protein_sequence']
        vl = row['vl_protein_sequence']
        hc_subtype = row['hc_subtype']
        lc_subtype = row['lc_subtype']

        # Get constant regions
        hc_const_key = f'HC_{hc_subtype}'
        lc_const_key = f'LC_{lc_subtype}'

        if hc_const_key not in constant_regions:
            raise ValueError(f"Unknown heavy chain subtype: {hc_subtype}. Available: {[k.replace('HC_', '') for k in constant_regions.keys() if k.startswith('HC_')]}")
        if lc_const_key not in constant_regions:
            raise ValueError(f"Unknown light chain subtype: {lc_subtype}. Available: {[k.replace('LC_', '') for k in constant_regions.keys() if k.startswith('LC_')]}")

        # Reconstruct full chains WITHOUT signal peptide (mature protein sequences)
        # Mature protein format: variable_region + constant_region
        # Note: Signal peptide (MRAWIFFLLCLAGRALA, 17 AA) is cleaved during maturation
        hc_full = vh + constant_regions[hc_const_key]
        lc_full = vl + constant_regions[lc_const_key]

        return pd.Series({'hc_protein_sequence': hc_full, 'lc_protein_sequence': lc_full})

    # Apply reconstruction
    reconstructed = df.apply(reconstruct_row, axis=1)
    df['hc_protein_sequence'] = reconstructed['hc_protein_sequence']
    df['lc_protein_sequence'] = reconstructed['lc_protein_sequence']

    logger.info(f"✓ Reconstructed full-chain sequences (HC ~{len(df.iloc[0]['hc_protein_sequence'])} AA, LC ~{len(df.iloc[0]['lc_protein_sequence'])} AA)")

    return df


class CustomBatchSampler(Sampler[List[int]]):
    """Custom batch sampler that merges the last batch into the previous one if it contains only 1 sample."""

    def __init__(self, data_source: Dataset, batch_size: int, shuffle: bool = False):
        """
        Args:
            data_source: Dataset to sample from
            batch_size: Size of mini-batch
            shuffle: Whether to shuffle indices
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.data_source)

        if self.shuffle:
            indices = torch.randperm(n).tolist()
        else:
            indices = list(range(n))

        # Create batches
        batches = [indices[i:i + self.batch_size] for i in range(0, n, self.batch_size)]

        # If the last batch has only 1 sample and there's more than 1 batch, merge it with the previous batch
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches[-1])
            batches = batches[:-1]

        return iter(batches)

    def __len__(self) -> int:
        n = len(self.data_source)
        num_batches = (n + self.batch_size - 1) // self.batch_size

        # If the last batch would have only 1 sample and there's more than 1 batch, we merge it
        if num_batches > 1 and n % self.batch_size == 1:
            return num_batches - 1
        return num_batches


class GDPa1Dataset(Dataset):
    """Dataset for GDPa1 antibody developability data."""

    def __init__(
        self,
        data: pd.DataFrame,
        heavy_col: str,
        light_col: str,
        target_col: str,
        precomputed_embeddings: Optional[dict] = None,
        target_transform: Optional[TargetTransform] = None,
        precomputed_antibody_features: Optional[dict] = None,
        antibody_features_config: Optional[dict] = None,
        antibody_features_mean: Optional[torch.Tensor] = None,
        antibody_features_std: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            data: DataFrame with antibody sequences and targets
            heavy_col: Column name for heavy chain sequences
            light_col: Column name for light chain sequences
            target_col: Column name for target property
            precomputed_embeddings: Dict mapping (heavy_seq, light_seq)
                -> (heavy_emb, light_emb) tensors
            target_transform: Optional transform to apply to targets
            precomputed_antibody_features: Dict mapping (heavy_seq, light_seq)
                -> antibody_features tensor (precomputed)
            antibody_features_config: Config dict for on-the-fly extraction (fallback)
        """
        self.data = data.reset_index(drop=True)
        self.heavy_col = heavy_col
        self.light_col = light_col
        self.target_col = target_col
        self.precomputed_embeddings = precomputed_embeddings
        self.target_transform = target_transform
        self.precomputed_antibody_features = precomputed_antibody_features
        self.antibody_features_mean = antibody_features_mean
        self.antibody_features_std = antibody_features_std

        # Initialize on-the-fly extractor ONLY if we don't have precomputed features
        # (avoids ScaLoP multiprocessing issues in DataLoader workers)
        self.antibody_features_extractor = None
        if antibody_features_config and antibody_features_config.get("enabled", False):
            if precomputed_antibody_features is None:
                # No precomputed features - need extractor for on-the-fly computation
                from src.features.antibody_features import AntibodyFeatures
                self.antibody_features_extractor = AntibodyFeatures(
                    use_abnumber=antibody_features_config.get("use_abnumber", True),
                    use_biophi=antibody_features_config.get("use_biophi", True),
                    use_scalop=antibody_features_config.get("use_scalop", True),
                    use_sequence_features=antibody_features_config.get("use_sequence_features", True),
                    cdr_definition=antibody_features_config.get("cdr_definition", "north"),
                    cache_abnumber=True,
                )
            # else: Have precomputed features, no need for extractor

        # Filter out rows with missing sequences or targets
        self.valid_indices = self.data[
            self.data[heavy_col].notna()
            & self.data[light_col].notna()
            & self.data[target_col].notna()
        ].index.tolist()

        n_filtered = len(self.data) - len(self.valid_indices)
        if n_filtered > 0:
            logger.debug(f"Filtered out {n_filtered} samples with missing data for {target_col}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        """Get a single antibody sample."""
        actual_idx = self.valid_indices[idx]
        row = self.data.iloc[actual_idx]

        heavy_seq = row[self.heavy_col]
        light_seq = row[self.light_col]

        # Get target value and apply transformation
        target_value = row[self.target_col]
        if self.target_transform is not None:
            # Transform target if available
            target_transformed = self.target_transform.transform_tensor(
                torch.tensor([target_value], dtype=torch.float32)
            )[0]
        else:
            target_transformed = torch.tensor(
                target_value, dtype=torch.float32
            )

        result = {
            "heavy_sequence": heavy_seq,
            "light_sequence": light_seq,
            "target": target_transformed,
            "antibody_id": row.get("id", actual_idx),
        }

        # Add cached embeddings if available
        if self.precomputed_embeddings is not None:
            key = (heavy_seq, light_seq)
            if key in self.precomputed_embeddings:
                heavy_emb, light_emb = self.precomputed_embeddings[key]
                result["heavy_embedding"] = heavy_emb
                result["light_embedding"] = light_emb

        # Add antibody features (precomputed or on-the-fly)
        key = (heavy_seq, light_seq)
        if self.precomputed_antibody_features is not None:
            # Precomputed features provided - keys MUST exist (validated at setup)
            if key not in self.precomputed_antibody_features:
                # This should never happen after setup validation, but crash explicitly if it does
                raise KeyError(
                    f"❌ FATAL: Antibody features not found for sequence pair!\n"
                    f"This should have been caught during setup validation.\n"
                    f"Heavy: {heavy_seq[:50]}...\n"
                    f"Light: {light_seq[:50]}..."
                )
            antibody_features = self.precomputed_antibody_features[key]

            # Apply z-score normalization if stats are provided
            if self.antibody_features_mean is not None and self.antibody_features_std is not None:
                antibody_features = (antibody_features - self.antibody_features_mean) / self.antibody_features_std

            result["antibody_features"] = antibody_features
        elif self.antibody_features_extractor is not None:
            # No precomputed features - compute on-the-fly
            features_dict = self.antibody_features_extractor.extract_features(heavy_seq, light_seq)
            features_array = self.antibody_features_extractor.features_to_array(features_dict)
            antibody_features = torch.from_numpy(features_array).float()

            # Apply z-score normalization if stats are provided
            if self.antibody_features_mean is not None and self.antibody_features_std is not None:
                antibody_features = (antibody_features - self.antibody_features_mean) / self.antibody_features_std

            result["antibody_features"] = antibody_features

        return result


class GDPa1DataModule(pl.LightningDataModule):
    """
    Lightning DataModule for GDPa1 dataset.

    Supports:
    - Cross-validation with stratified folds
    - Multiple target properties
    - Heavy and light chain sequences
    - Optional FLAb data augmentation
    - Precomputed embedding caching
    """

    def __init__(
        self,
        data_path: str,
        target_property: str,
        heavy_col: str = "vh_protein_sequence",
        light_col: str = "vl_protein_sequence",
        cv_fold_col: str = "hierarchical_cluster_IgG_isotype_stratified_fold",
        fold_idx: Optional[int] = 0,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        precomputed_embeddings_path: Optional[str] = None,  # Deprecated
        precomputed_embeddings_paths: Optional[dict] = None,  # NEW: Dict[encoder_type, path]
        encoder_types: Optional[List[str]] = None,  # NEW: Order for concatenating embeddings
        pooling: str = "mean",
        use_aho_aligned: bool = False,
        use_full_chain: bool = False,
        target_transform: Optional[str] = None,  # NEW: Target transformation
        target_transform_kwargs: Optional[dict] = None,  # NEW: Args for transform
        antibody_features_path: Optional[str] = None,  # NEW: Path to precomputed antibody features
        antibody_features_config: Optional[dict] = None,  # NEW: Config for on-the-fly extraction (fallback)
        normalize_antibody_features: bool = True,  # NEW: Z-score normalization from training set
    ):
        """
        Args:
            data_path: Path to GDPa1 CSV file
            target_property: Target property column name (e.g., 'HIC', 'Tm2', 'Titer')
            heavy_col: Column name for heavy chain sequences (ignored if use_aho_aligned=True)
            light_col: Column name for light chain sequences (ignored if use_aho_aligned=True)
            cv_fold_col: Column name for CV fold assignments
            fold_idx: Which fold to use as validation (0-4), or None to use all data for training
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for GPU training
            precomputed_embeddings_path: [DEPRECATED] Path to single embeddings file (for backward compatibility)
            precomputed_embeddings_paths: Dict mapping encoder_type -> embeddings file path (for MultiEncoder)
            encoder_types: List of encoder types in order (for proper concatenation in MultiEncoder)
            pooling: Pooling strategy for embeddings (mean, max, cls)
            use_aho_aligned: Whether to use AHO-aligned sequences (heavy_aligned_aho, light_aligned_aho)
                           If True, overrides heavy_col and light_col parameters.
                           IMPORTANT: Requires separate precomputed embeddings for aligned sequences!
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_path = Path(data_path)
        self.target_property = target_property
        self.cv_fold_col = cv_fold_col
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_aho_aligned = use_aho_aligned
        self.use_full_chain = use_full_chain

        # Validate mutually exclusive options
        if use_aho_aligned and use_full_chain:
            raise ValueError("use_aho_aligned and use_full_chain are mutually exclusive")

        # Select sequence columns based on use_aho_aligned or use_full_chain
        if use_aho_aligned:
            self.heavy_col = "heavy_aligned_aho"
            self.light_col = "light_aligned_aho"
            logger.info("Using AHO-aligned sequences (heavy_aligned_aho, light_aligned_aho)")
        elif use_full_chain:
            self.heavy_col = "hc_protein_sequence"
            self.light_col = "lc_protein_sequence"
            logger.info("Using full-chain sequences (hc_protein_sequence, lc_protein_sequence)")
        else:
            self.heavy_col = heavy_col
            self.light_col = light_col
            logger.info(f"Using variable region sequences ({heavy_col}, {light_col})")

        # Backward compatibility: convert single path to dict format
        if precomputed_embeddings_path is not None and precomputed_embeddings_paths is None:
            # Single encoder (legacy format)
            self.precomputed_embeddings_paths = {"default": precomputed_embeddings_path}
            self.encoder_types = ["default"]
        else:
            self.precomputed_embeddings_paths = precomputed_embeddings_paths or {}
            self.encoder_types = encoder_types or list(self.precomputed_embeddings_paths.keys())

        self.pooling = pooling

        # Target transformation
        if target_transform is None:
            # Auto-select recommended transformation
            target_transform = get_recommended_transform(target_property)
            logger.info(
                f"Auto-selected transformation for {target_property}: "
                f"{target_transform}"
            )

        self.target_transform_type = target_transform
        self.target_transform_kwargs = target_transform_kwargs or {}
        self.target_transform: TargetTransform = create_transform(
            target_transform, **self.target_transform_kwargs
        )
        logger.info(f"Created target transform: {self.target_transform.name}")

        # Antibody features
        self.antibody_features_path = antibody_features_path
        self.antibody_features_config = antibody_features_config or {}
        self.normalize_antibody_features = normalize_antibody_features

        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pooled_embeddings = None  # Cached pooled embeddings
        self.antibody_features_mean = None  # Z-score normalization stats (from training set)
        self.antibody_features_std = None
        self.cached_antibody_features = None  # Cached antibody features

    def setup(self, stage: Optional[str] = None):
        """Load and split data."""
        logger.info("="*60)
        logger.info("Setting up GDPa1 DataModule")
        logger.info("="*60)
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Target property: {self.target_property}")
        logger.info(f"CV fold: {self.fold_idx}")
        logger.info(f"Batch size: {self.batch_size}")

        # Load data
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data)} total samples")

        # Reconstruct full-chain sequences if needed
        if self.use_full_chain:
            logger.info("="*60)
            logger.info("USING FULL-CHAIN SEQUENCES (MATURE PROTEINS)")
            logger.info("Signal peptide will be automatically detected and removed")
            logger.info("="*60)

            # Extract constant regions and signal peptide from reference dataset
            reference_csv = "inputs/GDPa1_complete.csv"
            constant_regions = extract_constant_regions_from_reference(reference_csv)
            signal_peptide = constant_regions.get('signal_peptide', '')

            if 'hc_protein_sequence' not in self.data.columns or 'lc_protein_sequence' not in self.data.columns:
                logger.info("Full-chain sequences not found in dataset - reconstructing from variable regions...")

                # Reconstruct full chains (without signal peptide - mature proteins)
                self.data = reconstruct_full_chains(self.data, constant_regions, inplace=True)
            else:
                logger.info("Full-chain sequences already present in dataset")

                # Strip signal peptide if present (work with mature proteins only)
                if signal_peptide:
                    n_stripped_hc = 0
                    n_stripped_lc = 0

                    for idx in self.data.index:
                        if self.data.loc[idx, 'hc_protein_sequence'].startswith(signal_peptide):
                            self.data.loc[idx, 'hc_protein_sequence'] = self.data.loc[idx, 'hc_protein_sequence'][len(signal_peptide):]
                            n_stripped_hc += 1
                        if self.data.loc[idx, 'lc_protein_sequence'].startswith(signal_peptide):
                            self.data.loc[idx, 'lc_protein_sequence'] = self.data.loc[idx, 'lc_protein_sequence'][len(signal_peptide):]
                            n_stripped_lc += 1

                    if n_stripped_hc > 0 or n_stripped_lc > 0:
                        logger.info(f"Stripped signal peptide ({signal_peptide}) from {n_stripped_hc} HC and {n_stripped_lc} LC sequences")
                        logger.info("Working with mature protein sequences (signal peptide is cleaved during maturation)")
                    else:
                        logger.info("No signal peptide detected in sequences (already mature proteins)")

        # Load and pool precomputed embeddings if provided
        if self.precomputed_embeddings_paths:
            # MultiEncoder: load embeddings from multiple files and concatenate
            logger.info(f"Loading precomputed embeddings from {len(self.precomputed_embeddings_paths)} encoder(s)")
            logger.info(f"Encoder order: {self.encoder_types}")

            all_raw_embeddings = {}  # encoder_type -> raw_embeddings dict
            for enc_type in self.encoder_types:
                if enc_type in self.precomputed_embeddings_paths:
                    path = self.precomputed_embeddings_paths[enc_type]
                    logger.info(f"  Loading {enc_type} embeddings from {path}")
                    raw_embs = torch.load(path, weights_only=False)
                    all_raw_embeddings[enc_type] = raw_embs
                    logger.info(f"  Loaded {len(raw_embs)} raw {enc_type} embeddings")

            # Pool and concatenate embeddings from all encoders
            logger.info(f"Pooling and concatenating embeddings using strategy: {self.pooling}")
            self.pooled_embeddings = self._pool_and_concat_multi_embeddings(all_raw_embeddings, self.data)
            logger.info(f"Pooled and concatenated {len(self.pooled_embeddings)} embeddings")

        # Load precomputed antibody features (auto-detect if not provided)
        if self.antibody_features_path is None and self.antibody_features_config.get("enabled", False):
            # Auto-detect precomputed features (matching sequence representation)
            from src.utils import get_antibody_features_config
            self.antibody_features_path = get_antibody_features_config(
                self.data_path,
                use_aho_aligned=self.use_aho_aligned,
                use_full_chain=self.use_full_chain
            )

        if self.antibody_features_path:
            antibody_features_path_obj = Path(self.antibody_features_path)
            if antibody_features_path_obj.exists():
                logger.info("="*60)
                logger.info("Loading Precomputed Antibody Features")
                logger.info("="*60)
                logger.info(f"Path: {antibody_features_path_obj}")
                self.cached_antibody_features = torch.load(antibody_features_path_obj, weights_only=False)
                logger.info(f"✓ Loaded {len(self.cached_antibody_features)} antibody feature vectors")
                # Get feature dimension from first entry
                first_key = next(iter(self.cached_antibody_features))
                feature_dim = self.cached_antibody_features[first_key].shape[0]
                logger.info(f"✓ Feature dimension: {feature_dim}")
                
                # Filter features based on config (e.g., drop BioPhi if disabled)
                if self.antibody_features_config and self.antibody_features_config.get("enabled", False):
                    from src.features.antibody_features import AntibodyFeatures
                    # Get expected feature dimension from config
                    ab_cfg = self.antibody_features_config
                    valid_args = ["use_abnumber", "use_biophi", "use_scalop", "use_sequence_features", "cdr_definition", "cache_abnumber"]
                    ab_features_cfg = {k: v for k, v in ab_cfg.items() if k in valid_args}
                    ab_extractor = AntibodyFeatures(**ab_features_cfg)
                    expected_dim = ab_extractor.get_feature_dim()
                    feature_names = ab_extractor.get_feature_names()
                    
                    if expected_dim != feature_dim:
                        logger.info(f"⚠️  Feature dimension mismatch: precomputed={feature_dim}, config expects={expected_dim}")
                        logger.info(f"   Slicing features to match config...")
                        
                        # Build feature indices to keep (all features precomputed file has, in order)
                        # Compare with what config expects
                        # Assume precomputed has ALL features (33), config may want subset (31 without BioPhi)
                        # Strategy: Extract first expected_dim features that match feature_names from config
                        
                        # Create dummy full-feature extractor to get all feature names
                        full_extractor = AntibodyFeatures(
                            use_abnumber=True, use_biophi=True, use_scalop=True, 
                            use_sequence_features=True, cdr_definition=ab_cfg.get("cdr_definition", "north")
                        )
                        all_feature_names = full_extractor.get_feature_names()
                        
                        # Find indices of features we want to keep
                        keep_indices = [i for i, name in enumerate(all_feature_names) if name in feature_names]
                        logger.info(f"   Keeping {len(keep_indices)} features: {[all_feature_names[i] for i in keep_indices[:5]]}...")
                        
                        # Slice all cached features
                        filtered_features = {}
                        for key, features in self.cached_antibody_features.items():
                            filtered_features[key] = features[keep_indices]
                        
                        self.cached_antibody_features = filtered_features
                        feature_dim = expected_dim
                        logger.info(f"✓ Filtered to {feature_dim} features")

                # VALIDATE: Check that precomputed features match the sequence representation being used
                logger.info("Validating precomputed features match sequence representation...")
                sample_size = min(10, len(self.data))
                missing_keys = []
                for idx in self.data.head(sample_size).index:
                    heavy_seq = self.data.loc[idx, self.heavy_col]
                    light_seq = self.data.loc[idx, self.light_col]
                    key = (heavy_seq, light_seq)
                    if key not in self.cached_antibody_features:
                        missing_keys.append((idx, heavy_seq[:30], light_seq[:30]))

                if missing_keys:
                    seq_repr = "AHO-aligned" if self.use_aho_aligned else ("full-chain" if self.use_full_chain else "variable region")
                    error_msg = (
                        f"\n{'='*70}\n"
                        f"❌ CONFIGURATION MISMATCH: Precomputed antibody features are incompatible!\n"
                        f"{'='*70}\n"
                        f"You are using {seq_repr} sequences, but the precomputed features\n"
                        f"were generated for a different sequence representation.\n\n"
                        f"Found {len(missing_keys)} missing keys in first {sample_size} samples.\n\n"
                        f"SOLUTION:\n"
                        f"  Run: python scripts/precompute_antibody_features.py {self.data_path}"
                    )
                    if self.use_aho_aligned:
                        error_msg += " --use-aho-aligned"
                    elif self.use_full_chain:
                        error_msg += " --full-chain"
                    error_msg += f"\n{'='*70}\n"
                    raise ValueError(error_msg)

                logger.info(f"✓ Validated {sample_size} samples - all keys found")
                logger.info("="*60)
            else:
                logger.warning(f"⚠️  Antibody features path specified but not found: {antibody_features_path_obj}")
                if self.antibody_features_config.get("enabled", False):
                    logger.warning("Will compute features on-the-fly (slower)")
                else:
                    logger.warning("Antibody features disabled")

        # Fit target transformation on training data
        if self.fold_idx is None:
            # Use ALL data for training (no validation split)
            logger.info("fold_idx=None: Using ALL data for training (no CV split)")
            train_data = self.data
            val_data = pd.DataFrame()  # Empty validation set
        else:
            # Normal CV split
            train_data = self.data[self.data[self.cv_fold_col] != self.fold_idx]
            val_data = self.data[self.data[self.cv_fold_col] == self.fold_idx]

        logger.info(f"Train samples (before filtering): {len(train_data)}")
        logger.info(f"Validation samples (before filtering): {len(val_data)}")

        # Fit target transformation on training set
        train_targets = train_data[self.target_property].dropna().values
        self.target_transform.fit(train_targets)
        logger.info("="*60)
        logger.info("Target Transformation Statistics")
        logger.info("="*60)
        for key, val in self.target_transform.get_stats().items():
            logger.info(f"  {key}: {val}")
        logger.info("="*60)

        # Compute antibody features normalization stats from training set
        if self.normalize_antibody_features and self.cached_antibody_features is not None:
            logger.info("="*60)
            logger.info("Computing Antibody Features Normalization (Z-score)")
            logger.info("="*60)

            # Collect all training antibody features
            train_features_list = []
            for idx in train_data.index:
                heavy_seq = train_data.loc[idx, self.heavy_col]
                light_seq = train_data.loc[idx, self.light_col]
                key = (heavy_seq, light_seq)
                if key in self.cached_antibody_features:
                    train_features_list.append(self.cached_antibody_features[key])

            if train_features_list:
                # Stack into (n_samples, n_features) tensor
                train_features = torch.stack(train_features_list, dim=0)

                # Compute mean and std per feature (across training samples)
                self.antibody_features_mean = train_features.mean(dim=0)
                self.antibody_features_std = train_features.std(dim=0)

                # Avoid division by zero: set std=0.1 for nearly constant features
                self.antibody_features_std[self.antibody_features_std < 1e-8] = 0.1

                logger.info(f"✓ Computed normalization stats from {len(train_features_list)} training samples")
                logger.info(f"  Feature dimension: {len(self.antibody_features_mean)}")
                logger.info(f"  Mean range: [{self.antibody_features_mean.min():.4f}, {self.antibody_features_mean.max():.4f}]")
                logger.info(f"  Std range: [{self.antibody_features_std.min():.4f}, {self.antibody_features_std.max():.4f}]")
                logger.info("="*60)
            else:
                logger.warning("⚠️  No training antibody features found - normalization disabled")
                self.normalize_antibody_features = False

        # Create datasets
        if stage == "fit" or stage is None:
            self.train_dataset = GDPa1Dataset(
                train_data, self.heavy_col, self.light_col,
                self.target_property,
                precomputed_embeddings=self.pooled_embeddings,
                target_transform=self.target_transform,
                precomputed_antibody_features=self.cached_antibody_features,
                antibody_features_config=self.antibody_features_config,
                antibody_features_mean=self.antibody_features_mean,
                antibody_features_std=self.antibody_features_std,
            )
            if len(val_data) > 0:
                self.val_dataset = GDPa1Dataset(
                    val_data, self.heavy_col, self.light_col,
                    self.target_property,
                    precomputed_embeddings=self.pooled_embeddings,
                    target_transform=self.target_transform,
                    precomputed_antibody_features=self.cached_antibody_features,
                    antibody_features_config=self.antibody_features_config,
                    antibody_features_mean=self.antibody_features_mean,
                    antibody_features_std=self.antibody_features_std,
                )
            else:
                self.val_dataset = None  # No validation set
            logger.info(
                f"Train samples (after filtering): "
                f"{len(self.train_dataset)}"
            )
            if self.val_dataset:
                logger.info(
                    f"Validation samples (after filtering): "
                    f"{len(self.val_dataset)}"
                )
            else:
                logger.info(
                    "Validation samples: None (training on all data)"
                )

        if stage == "test" or stage is None:
            if len(val_data) > 0:
                self.test_dataset = GDPa1Dataset(
                    val_data, self.heavy_col, self.light_col,
                    self.target_property,
                    precomputed_embeddings=self.pooled_embeddings,
                    target_transform=self.target_transform,
                    precomputed_antibody_features=self.cached_antibody_features,
                    antibody_features_config=self.antibody_features_config,
                )
                logger.info(f"Test samples: {len(self.test_dataset)}")
            else:
                self.test_dataset = None
                logger.info("Test samples: None (no validation set)")

        logger.info("DataModule setup complete")
        logger.info("="*60)

    def _pool_all_embeddings(self, raw_embeddings: dict, data: pd.DataFrame) -> dict:
        """
        Pool all embeddings deterministically.

        Args:
            raw_embeddings: Dict with keys like "VH:sequence" and "VL:sequence"
            data: DataFrame with sequences

        Returns:
            Dict mapping (heavy_seq, light_seq) -> (heavy_emb, light_emb)
        """
        pooled = {}

        # Get unique sequence pairs from data
        unique_pairs = data[[self.heavy_col, self.light_col]].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            heavy_seq = row[self.heavy_col]
            light_seq = row[self.light_col]

            # Skip if missing
            if pd.isna(heavy_seq) or pd.isna(light_seq):
                continue

            # Get raw embeddings
            vh_key = f"VH:{heavy_seq}"
            vl_key = f"VL:{light_seq}"

            if vh_key not in raw_embeddings or vl_key not in raw_embeddings:
                logger.warning(f"Missing embeddings for sequence pair, skipping")
                continue

            vh_emb = raw_embeddings[vh_key]
            vl_emb = raw_embeddings[vl_key]

            # Ensure tensors
            if not isinstance(vh_emb, torch.Tensor):
                vh_emb = torch.tensor(vh_emb)
            if not isinstance(vl_emb, torch.Tensor):
                vl_emb = torch.tensor(vl_emb)

            # Pool using specified strategy
            vh_pooled = self._pool_single_embedding(vh_emb)
            vl_pooled = self._pool_single_embedding(vl_emb)

            # Detach and freeze (no gradients)
            vh_pooled = vh_pooled.detach()
            vl_pooled = vl_pooled.detach()
            vh_pooled.requires_grad = False
            vl_pooled.requires_grad = False

            # Store as tuple
            pooled[(heavy_seq, light_seq)] = (vh_pooled, vl_pooled)

        return pooled

    def _pool_and_concat_multi_embeddings(self, all_raw_embeddings: dict, data: pd.DataFrame) -> dict:
        """
        Pool and concatenate embeddings from multiple encoders.

        Args:
            all_raw_embeddings: Dict[encoder_type, raw_embeddings_dict]
            data: DataFrame with sequences

        Returns:
            Dict mapping (heavy_seq, light_seq) -> (concatenated_heavy_emb, concatenated_light_emb)
        """
        pooled = {}

        # Get unique sequence pairs from data
        unique_pairs = data[[self.heavy_col, self.light_col]].drop_duplicates()

        for _, row in unique_pairs.iterrows():
            heavy_seq = row[self.heavy_col]
            light_seq = row[self.light_col]

            # Skip if missing
            if pd.isna(heavy_seq) or pd.isna(light_seq):
                continue

            # Collect pooled embeddings from each encoder
            vh_embeddings_list = []
            vl_embeddings_list = []

            # Process encoders in the specified order
            for enc_type in self.encoder_types:
                if enc_type not in all_raw_embeddings:
                    logger.warning(f"Missing {enc_type} embeddings for sequence pair, skipping")
                    break

                raw_embs = all_raw_embeddings[enc_type]

                # Get raw embeddings for this encoder
                vh_key = f"VH:{heavy_seq}"
                vl_key = f"VL:{light_seq}"

                if vh_key not in raw_embs or vl_key not in raw_embs:
                    logger.warning(f"Missing {enc_type} embeddings for sequence pair, skipping")
                    break

                vh_emb = raw_embs[vh_key]
                vl_emb = raw_embs[vl_key]

                # Ensure tensors
                if not isinstance(vh_emb, torch.Tensor):
                    vh_emb = torch.tensor(vh_emb)
                if not isinstance(vl_emb, torch.Tensor):
                    vl_emb = torch.tensor(vl_emb)

                # Pool using specified strategy
                vh_pooled = self._pool_single_embedding(vh_emb)
                vl_pooled = self._pool_single_embedding(vl_emb)

                vh_embeddings_list.append(vh_pooled)
                vl_embeddings_list.append(vl_pooled)

            # Skip if we didn't get embeddings from all encoders
            if len(vh_embeddings_list) != len(self.encoder_types):
                continue

            # Concatenate embeddings from all encoders
            vh_concat = torch.cat(vh_embeddings_list, dim=0)  # (total_hidden_dim,)
            vl_concat = torch.cat(vl_embeddings_list, dim=0)  # (total_hidden_dim,)

            # Detach and freeze (no gradients)
            vh_concat = vh_concat.detach()
            vl_concat = vl_concat.detach()
            vh_concat.requires_grad = False
            vl_concat.requires_grad = False

            # Store as tuple
            pooled[(heavy_seq, light_seq)] = (vh_concat, vl_concat)

        return pooled

    def _pool_single_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Pool a single embedding tensor.

        Args:
            emb: Embedding tensor of shape (seq_len, hidden_dim)

        Returns:
            Pooled embedding of shape (hidden_dim,)
        """
        if self.pooling == "mean":
            return emb.mean(dim=0)
        elif self.pooling == "max":
            return emb.max(dim=0)[0]
        elif self.pooling == "cls":
            return emb[0]
        elif self.pooling == "sliced_wasserstein":
            # For sliced wasserstein, we need the full sequence
            # But since we're caching, just use mean for now
            # TODO: Implement batch sliced wasserstein pooling
            logger.warning("Sliced Wasserstein pooling not yet supported for caching, using mean")
            return emb.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences and cached embeddings."""
        heavy_seqs = [item["heavy_sequence"] for item in batch]
        light_seqs = [item["light_sequence"] for item in batch]
        targets = torch.stack([item["target"] for item in batch])
        antibody_ids = [item["antibody_id"] for item in batch]

        result = {
            "heavy_sequences": heavy_seqs,
            "light_sequences": light_seqs,
            "targets": targets,
            "antibody_ids": antibody_ids,
        }

        # Include cached embeddings if available
        if "heavy_embedding" in batch[0]:
            heavy_embs = torch.stack([item["heavy_embedding"] for item in batch])
            light_embs = torch.stack([item["light_embedding"] for item in batch])
            result["heavy_embeddings"] = heavy_embs
            result["light_embeddings"] = light_embs

        # Include antibody features if available
        if "antibody_features" in batch[0]:
            antibody_features = torch.stack([item["antibody_features"] for item in batch])
            result["antibody_features"] = antibody_features

        return result

    def train_dataloader(self):
        batch_sampler = CustomBatchSampler(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive between epochs
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch 4 batches per worker
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        batch_sampler = CustomBatchSampler(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive between epochs
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch 4 batches per worker
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        batch_sampler = CustomBatchSampler(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,  # Keep workers alive between epochs
            prefetch_factor=4 if self.num_workers > 0 else None,  # Prefetch 4 batches per worker
        )

    def get_num_samples(self):
        """Get number of training and validation samples."""
        return {
            "train": len(self.train_dataset) if self.train_dataset else 0,
            "val": len(self.val_dataset) if self.val_dataset else 0,
        }
