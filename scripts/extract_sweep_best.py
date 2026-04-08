#!/usr/bin/env python3
"""
Extract best hyperparameters from WandB sweeps.

Usage:
    python scripts/extract_sweep_best.py --phase 3 [options]

Phases:
    1     -> phase1
    2.1   -> phase2_1_encoder
    2.2   -> phase2_2_decoder
    3     -> phase3_refinement
    4.1   -> phase4_1_seqrep
    4.2   -> phase4_2_encoder
    4.3   -> phase4_3_antibody_features

Options:
    --metric            Metric to optimize (default: cv_val_spearman)
                        Mode (min/max) is auto-determined from metric name
    --offline           Use offline mode (local files only)
    --project           WandB project (default: simon-crouzet/CrossAbSense)
    --results-dir       Local results directory (default: outputs/tuning)
    --local-wandb-dir   Local wandb cache directory (default: wandb/)
    --out-csv           Output CSV path (optional)

Modes:
    Online (default):   Uses WandB API (requires WANDB_API_KEY)
    Offline (--offline): Reads from local wandb/ directory only
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Configuration
SWEEP_IDS_FILE = "config/tuning/sweep_ids.json"
PHASE_MAPPING = {
    "1": "phase1",
    "2.1": "phase2_1_encoder",
    "2.2": "phase2_2_decoder",
    "3": "phase3_refinement",
    "4.1": "phase4_1_seqrep",
    "4.2": "phase4_2_encoder",
    "4.3": "phase4_3_antibody_features",
    "5": "phase5_bayesian",
    "5.1": "phase5_1_architecture",
    "5.2": "phase5_2_options",
    "5.3": "phase5_3_details",
    "6": "phase6_final"
}


class SweepExtractor:
    """Extract and analyze sweep results from multiple sources."""

    def __init__(self, metric: str = "cv_val_spearman"):
        self.metric = metric
        self.mode = self._determine_mode(metric)
        # For cv_val metrics, we want the mean value
        self.metric_key = f"{metric}_mean" if metric.startswith("cv_val") else metric

    @staticmethod
    def _determine_mode(metric: str) -> str:
        """Determine optimization mode based on metric name."""
        # Metrics to maximize
        maximize_metrics = ['spearman', 'pearson', 'r2', 'accuracy', 'auc']
        # Metrics to minimize
        minimize_metrics = ['loss', 'mse', 'mae', 'rmse', 'error']

        metric_lower = metric.lower()

        if any(m in metric_lower for m in maximize_metrics):
            return 'max'
        elif any(m in metric_lower for m in minimize_metrics):
            return 'min'
        else:
            raise ValueError(
                f"Cannot determine optimization mode for metric '{metric}'. "
                f"Supported metrics contain: {maximize_metrics + minimize_metrics}"
            )

    def is_better(self, value: float, best_value: Optional[float]) -> bool:
        """Check if value is better than current best."""
        import math

        # Skip invalid values
        if value is None or not isinstance(value, (int, float)):
            return False
        if isinstance(value, float) and math.isnan(value):
            return False

        if best_value is None or not isinstance(best_value, (int, float)):
            return True
        if isinstance(best_value, float) and math.isnan(best_value):
            return True

        return value < best_value if self.mode == "min" else value > best_value

    def extract_from_wandb_api(
        self,
        project: str,
        sweep_id: str
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Extract sweep results from WandB API."""
        try:
            import wandb
        except ImportError:
            raise RuntimeError("wandb required. Install: pip install wandb")

        api = wandb.Api()
        sweep_path = f"{project}/{sweep_id}"

        try:
            sweep = api.sweep(sweep_path)
            runs = sweep.runs
        except Exception as e:
            print(f"  Warning: Could not access sweep {sweep_id}: {e}")
            return [], None

        rows = []
        best = None
        best_value = None

        for run in runs:
            metric_value = run.summary_metrics.get(self.metric_key)

            row = {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config) if run.config else {},
                "metric": metric_value,
                "path": run.path,
            }
            rows.append(row)

            if metric_value is not None and self.is_better(metric_value, best_value):
                best = row
                best_value = metric_value

        return rows, best

    def extract_from_local_files(
        self,
        results_dir: str,
        sweep_id: str
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Extract sweep results from local JSON files."""
        if not os.path.isdir(results_dir):
            return [], None

        rows = []
        best = None
        best_value = None

        for root, _, files in os.walk(results_dir):
            for fname in files:
                if not fname.endswith('.json'):
                    continue

                # Check if file is related to this sweep
                path = os.path.join(root, fname)
                if sweep_id not in fname and sweep_id not in root:
                    continue

                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    continue

                if not isinstance(data, dict):
                    continue

                metric_value = data.get(self.metric_key)
                config = data.get('config') or data.get('params')

                row = {
                    'path': path,
                    'metric': metric_value,
                    'config': config,
                    'data': data
                }
                rows.append(row)

                if metric_value is not None and self.is_better(metric_value, best_value):
                    best = row
                    best_value = metric_value

        return rows, best

    def extract_from_wandb_cache(
        self,
        wandb_dir: str,
        sweep_id: str
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Extract sweep results from local wandb cache.

        Algorithm:
        1. Check wandb/sweep-{sweep_id}/ for config-*.yaml files
        2. Extract run IDs from config filenames
        3. Find corresponding wandb/run-*-{run_id}/ directories
        4. Read files/wandb-summary.json and files/config.yaml
        """
        if not os.path.isdir(wandb_dir):
            return [], None

        # Step 1: Find sweep directory
        sweep_dir = os.path.join(wandb_dir, f'sweep-{sweep_id}')
        if not os.path.isdir(sweep_dir):
            print(f"    Sweep directory not found: {sweep_dir}")
            return [], None

        # Step 2: Get all run IDs from config-*.yaml files
        run_ids = []
        for filename in os.listdir(sweep_dir):
            if filename.startswith('config-') and filename.endswith('.yaml'):
                # Extract run_id from config-{run_id}.yaml
                run_id = filename[7:-5]  # Remove 'config-' prefix and '.yaml' suffix
                run_ids.append(run_id)

        if not run_ids:
            print(f"    No config files found in {sweep_dir}")
            return [], None

        # Step 3: Find and process each run directory
        rows = []
        best = None
        best_value = None

        runs_found = 0
        runs_with_metrics = 0

        for run_id in run_ids:
            # Find ALL run directories matching: wandb/run-*-{run_id}/
            # (there can be multiple: one per fold + one aggregation run)
            matching_runs = []
            for item in os.listdir(wandb_dir):
                if item.startswith('run-') and item.endswith(f'-{run_id}'):
                    matching_runs.append(os.path.join(wandb_dir, item))

            if not matching_runs:
                continue

            # Check all matching runs to find the one with cv_val metrics
            for run_dir in matching_runs:
                if not os.path.isdir(run_dir):
                    continue

                runs_found += 1

                # Step 4: Read run data
                summary_file = os.path.join(run_dir, 'files', 'wandb-summary.json')
                config_file = os.path.join(run_dir, 'files', 'config.yaml')

                if not os.path.exists(summary_file):
                    continue

                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                except Exception:
                    continue

                # Read config if available
                config = None
                if os.path.exists(config_file):
                    try:
                        import yaml
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)
                    except Exception:
                        pass

                # Fallback to config in summary
                if not config:
                    config = summary.get('config') or summary.get('params')

                # Extract metric value
                metric_value = summary.get(self.metric_key)

                # Skip runs that don't have the metric (e.g., individual folds when looking for cv_val_mean)
                if metric_value is None:
                    continue

                runs_with_metrics += 1

                row = {
                    'run_id': run_id,
                    'run_dir': os.path.basename(run_dir),
                    'run_path': run_dir,
                    'metric': metric_value,
                    'config': config,
                    'summary': summary
                }
                rows.append(row)

                if self.is_better(metric_value, best_value):
                    best = row
                    best_value = metric_value

                # Found the run with metrics for this config, move to next config
                break

        print(f"    Found {runs_with_metrics} runs with valid '{self.metric_key}' (checked {runs_found} run directories)")

        return rows, best

    def normalize_best_entry(self, best: Optional[Dict]) -> Optional[Dict]:
        """Normalize best run entry to consistent format."""
        if best is None:
            return None

        return {
            'id': best.get('id') or best.get('run_id'),
            'metric': best.get('metric'),
            'path': best.get('path') or best.get('run_path') or best.get('run_dir'),
            'config': best.get('config'),
        }

    def extract_sweep(
        self,
        sweep_id: str,
        project: str,
        results_dir: str,
        local_wandb_dir: Optional[str] = None,
        offline: bool = False
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Extract sweep results from multiple sources.

        If offline=True:
            1. Local wandb cache
            2. Local results directory

        If offline=False:
            1. WandB API
            (falls back to local if API fails)
        """
        rows = []
        best = None

        if offline:
            # Offline mode: check local sources only
            print("  [Offline mode]")

            # Try local wandb cache first
            if local_wandb_dir:
                rows, best = self.extract_from_wandb_cache(local_wandb_dir, sweep_id)
                if rows:
                    return rows, best

            # Try local results directory
            if results_dir:
                rows, best = self.extract_from_local_files(results_dir, sweep_id)
                if rows:
                    return rows, best
        else:
            # Online mode: try WandB API
            try:
                rows, best = self.extract_from_wandb_api(project, sweep_id)
                if rows:
                    return rows, best
            except Exception as e:
                print(f"  WandB API error: {e}")
                print(f"  Falling back to local sources...")

                # Fall back to local sources
                if local_wandb_dir:
                    rows, best = self.extract_from_wandb_cache(local_wandb_dir, sweep_id)
                    if rows:
                        return rows, best

                if results_dir:
                    rows, best = self.extract_from_local_files(results_dir, sweep_id)
                    if rows:
                        return rows, best

        return rows, best


def get_swept_parameters(sweep_config_path: str) -> set:
    """Extract parameters that were swept (had multiple values)."""
    import yaml

    if not os.path.exists(sweep_config_path):
        return set()

    try:
        with open(sweep_config_path, 'r') as f:
            config = yaml.safe_load(f)

        swept_params = set()
        params = config.get('parameters', {})

        for param_name, param_config in params.items():
            # Check if parameter has multiple values
            if isinstance(param_config, dict):
                if 'values' in param_config and isinstance(param_config['values'], list):
                    if len(param_config['values']) > 1:
                        swept_params.add(param_name)
                elif 'min' in param_config or 'max' in param_config:
                    # Continuous sweep
                    swept_params.add(param_name)

        return swept_params
    except Exception as e:
        print(f"Warning: Could not parse sweep config: {e}")
        return set()


def get_default_config_values(property_name: str) -> Dict:
    """Load default config values for a property."""
    default_config_path = "src/config/default_config.yaml"

    if not os.path.exists(default_config_path):
        return {}

    try:
        import yaml
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get property-specific config if available
        property_specific = config.get('property_specific', {}).get(property_name, {})

        # Merge with general defaults (property-specific overrides general)
        defaults = {}

        # Add relevant sections
        for section in ['encoder', 'decoder', 'training']:
            if section in config:
                if isinstance(config[section], dict):
                    defaults.update(config[section])

        # Override with property-specific
        defaults.update(property_specific)

        return defaults
    except Exception as e:
        print(f"Warning: Could not load default config: {e}")
        return {}


def filter_config_diff(best_config: Dict, property_name: str, phase: str, swept_params: set) -> Dict:
    """Filter config to show only swept params + params that differ from defaults."""
    if not best_config:
        return {}

    # Get default values
    defaults = get_default_config_values(property_name)

    # Find sweep config file for this phase
    sweep_config_path = f"config/tuning/{phase}_{property_name}.yaml"
    if not os.path.exists(sweep_config_path):
        # Try alternative names
        phase_map = {
            'phase1': 'phase1_sweep',
            'phase2_1_encoder': 'phase2_1_encoder_sweep',
            'phase2_2_decoder': f'phase2_2_{property_name}',
            'phase3_refinement': f'phase3_{property_name}',
            'phase4_1_seqrep': f'phase4_1_{property_name}',
            'phase4_2_encoder': f'phase4_2_{property_name}',
            'phase4_3_antibody_features': f'phase4_3_{property_name}',
        }
        sweep_config_path = f"config/tuning/{phase_map.get(phase, phase)}.yaml"

    if os.path.exists(sweep_config_path):
        swept_params = get_swept_parameters(sweep_config_path)

    filtered = {}
    for key, value in best_config.items():
        # Extract actual value if it's wrapped in a dict with 'value' key
        actual_value = value.get('value') if isinstance(value, dict) and 'value' in value else value

        # Include if it was swept OR if it differs from default
        default_value = defaults.get(key)

        is_swept = key in swept_params
        differs_from_default = default_value is not None and actual_value != default_value

        if is_swept or differs_from_default:
            filtered[key] = {
                'value': actual_value,
                'swept': is_swept,
                'default': default_value
            }

    return filtered


def load_sweep_ids(phase: str) -> Dict[str, str]:
    """Load sweep IDs for specified phase."""
    if phase not in PHASE_MAPPING:
        raise ValueError(f"Invalid phase '{phase}'. Valid: {list(PHASE_MAPPING.keys())}")

    if not os.path.exists(SWEEP_IDS_FILE):
        raise FileNotFoundError(f"Sweep IDs file not found: {SWEEP_IDS_FILE}")

    with open(SWEEP_IDS_FILE, 'r') as f:
        all_sweeps = json.load(f)

    phase_key = PHASE_MAPPING[phase]
    if phase_key not in all_sweeps:
        raise ValueError(f"No sweeps found for phase {phase} ({phase_key})")

    return all_sweeps[phase_key]


def save_results(results: Dict, csv_path: Optional[str], json_path: str, phase_key: str):
    """Save results to JSON and optionally CSV. Merges with existing JSON."""
    # Ensure output directory exists
    json_dir = os.path.dirname(json_path)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)

    # Load existing results and merge
    all_results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                all_results = json.load(f)
            print(f"Loaded existing results from {json_path}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Update only this phase
    all_results[phase_key] = results

    # Save merged JSON
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to: {json_path} (phase: {phase_key})")

    # Save CSV if requested
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['property', 'sweep_id', 'n_runs', 'best_metric', 'best_id'])

            for prop, data in results.items():
                best = data.get('best')
                writer.writerow([
                    prop,
                    data['sweep_id'],
                    data['n_runs'],
                    best.get('metric') if best else None,
                    best.get('id') if best else None
                ])
        print(f"Saved CSV to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract best hyperparameters from WandB sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--phase',
        required=True,
        choices=['1', '2.1', '2.2', '3', '4.1', '4.2', '4.3', '5', '5.1', '5.2', '5.3', '6'],
        help='Tuning phase to extract'
    )
    parser.add_argument(
        '--metric',
        default='cv_val_spearman',
        help='Metric to optimize (default: cv_val_spearman). Mode auto-determined.'
    )
    parser.add_argument(
        '--project',
        default='simon-crouzet/CrossAbSense',
        help='WandB project path'
    )
    parser.add_argument(
        '--results-dir',
        default='outputs/tuning',
        help='Local results directory (default: outputs/tuning)'
    )
    parser.add_argument(
        '--local-wandb-dir',
        default='wandb',
        help='Local wandb cache directory (default: wandb/)'
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Use offline mode (read from local wandb cache only)'
    )
    parser.add_argument(
        '--out-csv',
        help='Output CSV path (optional)'
    )
    parser.add_argument(
        '--out-json',
        default='outputs/tuning/phase_best_summary.json',
        help='Output JSON path (default: outputs/tuning/phase_best_summary.json)'
    )
    parser.add_argument(
        '--full-config',
        action='store_true',
        help='Show full config instead of only swept parameters (default: show only swept params)'
    )

    args = parser.parse_args()

    # Check for API key if in online mode
    if not args.offline and not os.environ.get('WANDB_API_KEY'):
        print("=" * 70)
        print("WARNING: WANDB_API_KEY not found in environment!")
        print("=" * 70)
        print("\nYou have two options:")
        print("  1. Run in offline mode:  --offline")
        print("  2. Set your API key:     export WANDB_API_KEY=your_key_here")
        print(f"\nGet your API key at: https://wandb.ai/authorize")
        print("\nContinuing without API key (will only check local files)...\n")
        args.offline = True

    # Load sweep IDs for phase
    print(f"Loading sweeps for phase {args.phase}...")
    sweeps = load_sweep_ids(args.phase)
    print(f"Found {len(sweeps)} properties: {list(sweeps.keys())}\n")

    # Initialize extractor
    extractor = SweepExtractor(metric=args.metric)
    metric_info = f"Optimizing for: {args.metric}"
    if extractor.metric_key != args.metric:
        metric_info += f" (using {extractor.metric_key})"
    metric_info += f" ({extractor.mode}imize)\n"
    print(metric_info)

    # Process each sweep
    all_results = {}

    for prop, sweep_id in sweeps.items():
        print(f"\n{prop} (sweep: {sweep_id})")

        rows, best = extractor.extract_sweep(
            sweep_id=sweep_id,
            project=args.project,
            results_dir=args.results_dir,
            local_wandb_dir=args.local_wandb_dir,
            offline=args.offline
        )

        best_entry = extractor.normalize_best_entry(best)

        # Apply diff filtering by default (unless --full-config is specified)
        if not args.full_config and best_entry and best_entry.get('config'):
            phase_key = PHASE_MAPPING[args.phase]
            filtered_config = filter_config_diff(
                best_entry['config'],
                prop,
                phase_key,
                set()
            )
            best_entry['config_diff'] = filtered_config
            # Remove full config when using diff mode
            del best_entry['config']

        all_results[prop] = {
            "sweep_id": sweep_id,
            "n_runs": len(rows),
            "best": best_entry
        }

        # Display results
        print(f"  Runs: {len(rows)}")
        if best_entry and best_entry.get('metric') is not None:
            import math
            metric_val = best_entry['metric']
            if isinstance(metric_val, float) and math.isnan(metric_val):
                print(f"  ✗ No valid results (all runs returned NaN)")
            else:
                print(f"  ✓ Best {args.metric}: {metric_val:.6f} (run: {best_entry['id']})")
        else:
            print(f"  ✗ No runs found")

    # Save results
    phase_key = PHASE_MAPPING[args.phase]
    save_results(all_results, args.out_csv, args.out_json, phase_key)

    # Summary
    total_runs = sum(r['n_runs'] for r in all_results.values())
    found_best = sum(1 for r in all_results.values() if r['best'])
    print(f"\nSummary: {found_best}/{len(sweeps)} sweeps with results ({total_runs} total runs)")

    if args.full_config:
        print("\n✓ Full config mode - showing all parameters")
    else:
        print("\n✓ Diff mode (default) - showing only swept params and params that differ from defaults")


if __name__ == '__main__':
    main()
