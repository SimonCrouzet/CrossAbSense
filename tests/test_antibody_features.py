"""
Test antibody feature extraction module.

Tests AntibodyFeatures on real sequences from GDPa1 dataset.
Provides detailed recap of which features work and which fail.
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import AntibodyFeatures


def test_antibody_features():
    """Test feature extraction on sample antibodies from GDPa1."""

    print("=" * 80)
    print("Testing AntibodyFeatures on GDPa1 dataset")
    print("=" * 80)

    # Load dataset
    data_path = Path(__file__).parent.parent / "inputs" / "GDPa1_complete.csv"
    if not data_path.exists():
        print(f"\n❌ Dataset not found: {data_path}")
        print("Please ensure inputs/GDPa1_complete.csv exists")
        return

    print(f"\n✓ Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Total sequences: {len(df)}")

    # Sample 5 antibodies
    n_samples = min(5, len(df))
    sample_df = df.head(n_samples)

    print(f"\n✓ Testing on {n_samples} antibodies")

    # Initialize feature extractor with all tools enabled
    print("\n" + "-" * 80)
    print("Initializing AntibodyFeatures with all tools enabled...")
    print("-" * 80)

    extractor = AntibodyFeatures(
        use_abnumber=True,
        use_biophi=True,
        use_scalop=True,
        use_sequence_features=True,
        cdr_definition="north",
        cache_abnumber=True,
    )

    # Check which tools are available
    print(f"\n  abnumber available: {'✓' if extractor._abnumber_available else '✗'}")
    print(f"  BioPhi available:   {'✓' if extractor._biophi_available else '✗'}")
    print(f"  ScaLoP available:   {'✓' if extractor._scalop_available else '✗'}")

    # Extract features for each sample
    print("\n" + "-" * 80)
    print("Extracting features from samples...")
    print("-" * 80)

    all_features = []
    for idx, row in sample_df.iterrows():
        heavy_seq = row['vh_protein_sequence']
        light_seq = row['vl_protein_sequence']

        print(f"\nSample {idx + 1}:")
        print(f"  VH length: {len(heavy_seq)}")
        print(f"  VL length: {len(light_seq)}")

        # Extract features
        features = extractor.extract_features(heavy_seq, light_seq)
        all_features.append(features)

        print(f"  ✓ Extracted {len(features)} features")

    # Analyze feature success/failure across all samples
    print("\n" + "=" * 80)
    print("Feature Extraction Summary")
    print("=" * 80)

    # Get feature names
    feature_names = list(all_features[0].keys())

    # Group features by category
    categories = {
        "abnumber (germline & CDR)": [
            "germline_identity_vh", "germline_identity_vl",
            "n_mutations_vh", "n_mutations_vl",
            "v_gene_family_vh", "v_gene_family_vl",
            "cdr_h1_length", "cdr_h2_length", "cdr_h3_length",
            "cdr_l1_length", "cdr_l2_length", "cdr_l3_length",
            "total_cdr_length"
        ],
        "BioPhi (humanness)": [
            "humanness_vh", "humanness_vl"
        ],
        "ScaLoP (canonical classes)": [
            "canonical_class_l1", "canonical_class_l2", "canonical_class_l3",
            "canonical_class_h1", "canonical_class_h2"
        ],
        "Sequence liabilities": [
            "n_deamidation_ng", "n_deamidation_ns",
            "n_glycosylation",
            "n_oxidation_m", "n_oxidation_w",
            "n_unpaired_cys"
        ],
        "CDR-H3 properties": [
            "cdr_h3_net_charge", "cdr_h3_hydrophobicity"
        ],
        "Chain properties": [
            "pi_vh", "pi_vl",
            "net_charge_vh", "net_charge_vl",
            "charge_asymmetry"
        ]
    }

    # Check each category
    for category, features_list in categories.items():
        print(f"\n{category}:")
        print("-" * 80)

        for feature_name in features_list:
            # Check if feature exists in extracted features
            if feature_name not in feature_names:
                print(f"  ✗ {feature_name:30s} - NOT FOUND")
                continue

            # Count successes (non-sentinel values)
            successes = 0
            for feat_dict in all_features:
                value = feat_dict[feature_name]

                # Check if value is sentinel (depends on feature type)
                is_sentinel = False
                if "identity" in feature_name and value == -1.0:
                    is_sentinel = True
                elif "mutations" in feature_name and value == -1:
                    is_sentinel = True
                elif "family" in feature_name and value == 0:
                    is_sentinel = True
                elif "length" in feature_name and value == -1:
                    is_sentinel = True
                elif "humanness" in feature_name and value == -1.0:
                    is_sentinel = True
                elif "canonical" in feature_name and value == 0:
                    is_sentinel = True
                elif "pi" in feature_name and value == -1.0:
                    is_sentinel = True

                if not is_sentinel:
                    successes += 1

            # Show status
            success_rate = successes / len(all_features) * 100
            status = "✓" if success_rate == 100 else "⚠" if success_rate > 0 else "✗"

            # Get sample values
            sample_values = [f"{feat_dict[feature_name]:.2f}" if isinstance(feat_dict[feature_name], float)
                           else str(feat_dict[feature_name])
                           for feat_dict in all_features[:3]]

            print(f"  {status} {feature_name:30s} {success_rate:5.1f}% ({successes}/{len(all_features)})  "
                  f"values: [{', '.join(sample_values)}...]")

    # Overall statistics
    print("\n" + "=" * 80)
    print("Overall Statistics")
    print("=" * 80)

    total_features = len(feature_names)
    successful_features = 0

    for feature_name in feature_names:
        # Count if at least one sample has non-sentinel value
        has_success = False
        for feat_dict in all_features:
            value = feat_dict[feature_name]

            # Check sentinels
            is_sentinel = (
                (value == -1.0 and ("identity" in feature_name or "humanness" in feature_name or "pi" in feature_name)) or
                (value == -1 and ("mutations" in feature_name or "length" in feature_name)) or
                (value == 0 and ("family" in feature_name or "canonical" in feature_name))
            )

            if not is_sentinel:
                has_success = True
                break

        if has_success:
            successful_features += 1

    print(f"\nTotal features:        {total_features}")
    print(f"Working features:      {successful_features}")
    print(f"Failed features:       {total_features - successful_features}")
    print(f"Success rate:          {successful_features / total_features * 100:.1f}%")

    # Convert to array
    print("\n" + "-" * 80)
    print("Testing array conversion...")
    print("-" * 80)

    import numpy as np

    feature_arrays = [extractor.features_to_array(f) for f in all_features]
    batch_array = np.stack(feature_arrays)

    print(f"\n  Batch array shape:  {batch_array.shape}")
    print(f"  Dtype:              {batch_array.dtype}")
    print(f"  Memory:             {batch_array.nbytes / 1024:.2f} KB")

    # Feature names
    print("\n" + "-" * 80)
    print("Feature names (for reference):")
    print("-" * 80)

    feature_names_ordered = extractor.get_feature_names()
    print(f"\n  Total: {len(feature_names_ordered)} features")
    print(f"  First 5:  {feature_names_ordered[:5]}")
    print(f"  Last 5:   {feature_names_ordered[-5:]}")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_antibody_features()
