#!/usr/bin/env python3
"""Run cross-validation training for all folds and properties."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import subprocess
from pathlib import Path

from src.utils import (
    aggregate_fold_metrics,
    get_dataset_column,
    is_primary_property,
    load_config,
    setup_logger,
)

logger = logging.getLogger(__name__)


def run_fold(config_path: str, property_name: str, fold: int, gpus: int = 1):
    """Run training for a single fold."""
    cmd = [
        "python",
        "-m",
        "src.train",
        "--config",
        config_path,
        "--property",
        property_name,
        "--fold",
        str(fold),
        "--gpus",
        str(gpus),
    ]

    logger.info(f"Starting training for {property_name} fold {fold}")
    logger.info(f"Command: {' '.join(cmd)}")
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.warning(f"Training failed for {property_name} fold {fold}")
        print(f"WARNING: Training failed for {property_name} fold {fold}")
        return False

    logger.info(f"Successfully completed training for {property_name} fold {fold}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run cross-validation for all folds and properties"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/default_config.yaml",
        help="Path to config file (default: src/config/default_config.yaml)",
    )
    parser.add_argument(
        "--properties",
        type=str,
        nargs="+",
        default=None,
        help="Properties to train (simplified names like 'hydrophobicity' or column names like 'HIC'. Default: all primary properties)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    args = parser.parse_args()

    # Setup logger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    logger = setup_logger(
        "run_cv",
        f"run_cv_{timestamp}.log",
        logging.INFO
    )

    logger.info("="*80)
    logger.info("Starting Cross-Validation Training")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Number of folds: {args.n_folds}")

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Compute config checksum for model directory naming (matching src/train.py)
    from src.utils import compute_file_checksum
    config_path = Path(args.config)
    config_checksum = compute_file_checksum(str(config_path))[:8]

    # Determine properties to train
    if args.properties:
        logger.info(f"Properties specified: {args.properties}")
        # Convert property names to dataset columns and validate
        properties = []
        for prop in args.properties:
            try:
                dataset_column = get_dataset_column(prop)
                if not is_primary_property(prop):
                    logger.warning(f"'{prop}' is not a primary property")
                    print(f"\n⚠️  WARNING: '{prop}' is not a primary property!")
                    print(f"    Primary properties: hydrophobicity, self-association, titer, thermostability, polyreactivity")
                    print(f"    Continuing anyway...\n")
                properties.append(dataset_column)
            except ValueError as e:
                logger.error(f"Invalid property: {e}")
                print(f"\n❌ Error: {e}\n")
                return
    else:
        # Default to primary properties only
        from src.utils.property_names import PRIMARY_PROPERTIES
        properties = list(PRIMARY_PROPERTIES.values())
        logger.info("No properties specified. Using all primary properties by default")
        print(f"No properties specified. Using all primary properties by default.")

    if not properties:
        raise ValueError(
            "No properties found. Specify --properties or add to config."
        )

    logger.info(f"Properties to train: {properties}")
    logger.info(f"Number of folds: {args.n_folds}")
    print(f"Properties to train: {properties}")
    print(f"Number of folds: {args.n_folds}")

    # Track results
    all_results = {}
    property_dfs = {}

    # Train each property across all folds
    for property_name in properties:
        print(f"\n{'=' * 80}")
        print(f"Training property: {property_name}")
        print(f"{'=' * 80}")

        fold_metrics = []
        fold_predictions = []
        
        # Determine model directory (matching src/train.py)
        if config_path == Path("src/config/default_config.yaml"):
            model_dir_name = f"{property_name}_{config_checksum}"
        else:
            config_filename = config_path.stem
            model_dir_name = f"{property_name}_{config_filename}_{config_checksum}"
        
        model_dir = Path("models") / model_dir_name

        for fold in range(args.n_folds):
            success = run_fold(args.config, property_name, fold, args.gpus)

            if success:
                # Load metrics from cv_results.json
                results_path = model_dir / "cv_results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        cv_data = json.load(f)
                        # We only want metrics for folds trained so far
                        if len(cv_data.get("folds", [])) > fold:
                            fold_metrics.append(cv_data["folds"][fold])
                
                # Load predictions from CSV
                preds_path = model_dir / f"fold{fold}_predictions.csv"
                if preds_path.exists():
                    import pandas as pd
                    df = pd.read_csv(preds_path)
                    df["fold"] = fold
                    fold_predictions.append(df)

        # Aggregate fold metrics
        if fold_metrics:
            aggregated = aggregate_fold_metrics(fold_metrics)
            all_results[property_name] = aggregated

            print(f"\n{property_name} - Aggregated Results:")
            for metric, value in aggregated.items():
                if metric.endswith("_mean"):
                    print(f"  {metric}: {value:.4f} ± {aggregated[metric.replace('_mean', '_std')]:.4f}")
        
        # Aggregate and save predictions for this property
        if fold_predictions:
            import pandas as pd
            all_preds_df = pd.concat(fold_predictions).reset_index(drop=True)
            # Rename columns to include property name
            all_preds_df = all_preds_df.rename(columns={
                "prediction": f"{property_name}_pred",
                "target": f"{property_name}_gt"
            })
            # Save property-specific aggregated predictions
            agg_preds_path = model_dir / "all_folds_predictions.csv"
            all_preds_df.to_csv(agg_preds_path, index=False)
            print(f"Saved aggregated predictions to: {agg_preds_path}")
            
            # Store for global aggregation (drop fold column)
            property_dfs[property_name] = all_preds_df.drop(columns=["fold"])
        
        # Also train the final model for this property
        print(f"\nTraining FINAL model for {property_name}...")
        run_fold(args.config, property_name, -1, args.gpus) # -1 triggers final model in src/train.py update


    # Save overall results
    results_dir = Path("results") / "cross_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "cv_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Export global predictions CSV
    if property_dfs:
        import pandas as pd
        # Merge all property dataframes on antibody_id
        global_df = None
        for prop, df in property_dfs.items():
            if global_df is None:
                global_df = df
            else:
                global_df = pd.merge(global_df, df, on="antibody_id", how="outer")
        
        global_preds_path = results_dir / "predictions.csv"
        global_df.to_csv(global_preds_path, index=False)
        print(f"Exported global predictions to: {global_preds_path}")

    print(f"\n{'=' * 80}")
    print(f"Cross-validation complete!")
    print(f"Results saved to {results_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
