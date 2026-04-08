#!/usr/bin/env python3
"""
Generic hyperparameter sweep runner for all tuning phases.

Usage:
    # Property-agnostic phases (1, 2.1)
    python scripts/run_sweep.py --phase 1
    python scripts/run_sweep.py --phase 2.1

    # Property-specific phases
    python scripts/run_sweep.py --phase 2.2 HIC
    python scripts/run_sweep.py --phase 4.2 Titer
    python scripts/run_sweep.py --phase 5 --all

    # Resume interrupted sweep (automatic)
    python scripts/run_sweep.py --phase 4.2 HIC

Supported phases:
    OLD PHASES:
    - Phase 1:   Learning rate + batch size (Bayesian, property-agnostic)
    - Phase 2.1: Encoder selection (Grid, property-agnostic)
    - Phase 2.2: Decoder architecture (Bayesian, per-property)

    NEW PHASES:
    - Phase 4.1: Sequence representation (Grid, per-property)
    - Phase 4.2: Encoder comparison (Grid, per-property)
    - Phase 4.3: Antibody features (Grid, per-property)
    - Phase 5:   Bayesian optimization (Bayesian, per-property)
    - Phase 5.1: Architecture tuning (Bayesian capped, per-property)
    - Phase 5.2: Options tuning (Bayesian capped, per-property)
    - Phase 6:   Final tuning (Bayesian capped, per-property)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.sweep_manager import SweepManager, check_and_resume_sweep
from src.utils.sweep_runner import create_sweep, run_sweep_agent_smart


# Property name aliases (normalize user input to canonical names)
PROPERTY_ALIASES = {
    "AC-SINS": "AC-SINS_pH7.4",
}

# Phase configurations: maps phase name to (sweep_name_prefix, config_prefix, is_property_specific)
# is_property_specific: True = per-property config, False = single shared config
PHASE_CONFIG = {
    # Old phases (1, 2.x)
    "1": ("phase1_lr", "phase1_sweep", False),  # Single config, property-agnostic
    "2.1": ("phase2_1_encoder", "phase2_1_encoder_sweep", False),  # Single config
    "2.2": ("phase2_2_decoder", "phase2_2", True),  # Per-property configs
    # New phases (4.x, 5.x)
    "4.1": ("phase4_1_seqrep", "phase4_1", True),
    "4.2": ("phase4_2_encoder", "phase4_2", True),
    "4.3": ("phase4_3_antibody_features", "phase4_3", True),
    "5": ("phase5_bayesian", "phase5", True),
    "5.1": ("phase5_1_architecture", "phase5.1", True),
    "5.2": ("phase5_2_options", "phase5.2", True),
    "6": ("phase6_final", "phase6", True),
}

# Run counts per phase
# For property-agnostic phases, use "_all" as key
# For property-specific phases, use property name as key
RUN_COUNTS = {
    # Old phases
    "1": {
        "_all": 10,
    },
    "2.1": {
        "_all": 12,  # 8 encoders × 3 pooling × 2 properties (HIC, Titer)
    },
    "2.2": {
        "HIC": 20,  # Bayesian decoder search
        "PR_CHO": 20,
        "AC-SINS_pH7.4": 20,
        "Tm2": 20,
        "Titer": 20,
    },
    # New phases
    "4.1": {
        # Grid: 3 sequence_representation values (normal, aho_aligned, full_chain)
        "HIC": 3,
        "PR_CHO": 3,
        "AC-SINS_pH7.4": 3,
        "Tm2": 3,
        "Titer": 3,
    },
    "4.2": {
        # Grid: encoders × seq_rep × antibody_features
        "HIC": 12,  # 3 × 2 × 2
        "PR_CHO": 6,  # 3 × 2 (no seq_rep variation)
        "AC-SINS_pH7.4": 6,  # 3 × 2
        "Tm2": 6,  # 3 × 2
        "Titer": 6,  # 3 × 2
    },
    "4.3": {
        # Grid: use_abfeat × normalize × projection × extra_param
        "HIC": 12,  # 2 × 2 × 3 (projection has 3 values)
        "PR_CHO": 16,  # 2 × 2 × 2 × 2 (encoder)
        "AC-SINS_pH7.4": 16,  # 2 × 2 × 2 × 2 (attention_strategy)
        "Tm2": 16,  # 2 × 2 × 2 × 2 (attention_strategy)
        "Titer": 16,  # 2 × 2 × 2 × 2 (n_layers)
    },
    "5": {
        "HIC": 40,  # Bayesian - extensive search
        "PR_CHO": 40,
        "AC-SINS_pH7.4": 40,
        "Tm2": 40,
        "Titer": 40,
    },
    "5.1": {
        # Bayesian capped at 12 (run_cap in config)
        "HIC": 12,
        "PR_CHO": 12,
        "AC-SINS_pH7.4": 12,
        "Tm2": 12,
        "Titer": 12,
    },
    "5.2": {
        # Bayesian capped at 12
        "HIC": 12,
        "PR_CHO": 12,
        "AC-SINS_pH7.4": 12,
        "Tm2": 12,
        "Titer": 12,
    },
    "6": {
        # Final tuning - Bayesian (run_cap in config)
        "HIC": 60,
        "PR_CHO": 60,
        "AC-SINS_pH7.4": 60,
        "Tm2": 60,
        "Titer": 60,
    },
}


def run_sweep_agent(sweep_id: str, count: int, name: Optional[str] = None) -> bool:
    """
    Run WandB sweep agent with smart progress tracking.

    Automatically:
    - Queries current sweep progress
    - Skips if target already reached
    - Runs only remaining runs needed

    Args:
        sweep_id: Full sweep ID (e.g., "user/project/abc123")
        count: Target number of runs to reach
        name: Optional name for progress display

    Returns:
        True if successful or sweep already complete, False otherwise
    """
    display_name = name or sweep_id

    try:
        # Use smart runner with progress tracking and resume capability
        success = run_sweep_agent_smart(
            sweep_id=sweep_id,
            target_count=count,
            label=display_name,
            verbose=True
        )
        return success

    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
        return False


def process_property(phase: str, property_name: str, project: str = "CrossAbSense") -> bool:
    """
    Process a single property: create/resume sweep and run agent.

    Args:
        phase: Phase identifier (e.g., "4.1", "4.2")
        property_name: Property to tune (e.g., HIC, Titer, AC-SINS_pH7.4)
                       Use "_all" for property-agnostic phases
        project: WandB project name

    Returns:
        True if successful, False otherwise
    """
    # Normalize property name using aliases
    if property_name != "_all":
        property_name = PROPERTY_ALIASES.get(property_name, property_name)

    if phase not in PHASE_CONFIG:
        print(f"❌ Unknown phase: {phase}")
        print(f"Supported phases: {', '.join(PHASE_CONFIG.keys())}")
        return False

    sweep_name, config_prefix, is_property_specific = PHASE_CONFIG[phase]

    # Get run count
    if phase not in RUN_COUNTS:
        print(f"❌ Phase {phase} not configured in RUN_COUNTS")
        return False

    # Determine run count key
    run_count_key = property_name if is_property_specific else "_all"
    if run_count_key not in RUN_COUNTS[phase]:
        print(f"❌ Property {property_name} not configured for phase {phase}")
        print(f"Available: {', '.join(RUN_COUNTS[phase].keys())}")
        return False

    expected_runs = RUN_COUNTS[phase][run_count_key]

    # Determine config file path
    if is_property_specific:
        # Use short property name (without _pH7.4) for config filename
        config_property_name = "AC-SINS" if property_name == "AC-SINS_pH7.4" else property_name
        config_file = f"config/tuning/{config_prefix}_{config_property_name}.yaml"
    else:
        # Property-agnostic: single config file
        config_file = f"config/tuning/{config_prefix}.yaml"

    # Check if config file exists
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return False

    # Display name for logging
    display_name = property_name if is_property_specific else "all"

    print(f"\n{'='*70}")
    print(f"Phase {phase}: {display_name}")
    print(f"Config: {config_file}")
    print(f"Expected runs: {expected_runs}")
    print(f"{'='*70}\n")

    # Initialize sweep manager
    manager = SweepManager()

    # Sweep key for storage (use "all" for property-agnostic)
    sweep_key = property_name if is_property_specific else "all"

    # Check for existing sweep
    sweep_hash = manager.get_sweep(sweep_name, sweep_key)

    if sweep_hash:
        # Reconstruct full sweep ID (entity/project/hash)
        import wandb
        entity = wandb.Api().default_entity
        sweep_id = f"{entity}/{project}/{sweep_hash}"

        print(f"✓ Found existing sweep: {sweep_id}")
        # Check status and resume if needed
        status = check_and_resume_sweep(sweep_name, sweep_key, expected_runs=expected_runs)
        if status and status["completed"] >= expected_runs and status["remaining"] == 0:
            print(f"✓ All {expected_runs} runs already completed!")
            return True
    else:
        # Create new sweep
        print("Creating new sweep...")
        sweep_id = create_sweep(config_file, project=project)
        if not sweep_id:
            print(f"❌ Failed to create sweep")
            return False

        # Extract just the hash (entity/project always same)
        sweep_hash = sweep_id.split('/')[-1]
        manager.save_sweep(sweep_name, sweep_key, sweep_hash)
        print(f"✓ Sweep created: {sweep_id}")

    # Run agent
    success = run_sweep_agent(
        sweep_id=sweep_id,
        count=expected_runs,
        name=f"{phase}_{display_name}"
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Generic hyperparameter sweep runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Property-agnostic phases (1, 2.1)
  python scripts/run_sweep.py --phase 1
  python scripts/run_sweep.py --phase 2.1

  # Property-specific phases (2.2, 4.x, 5.x)
  python scripts/run_sweep.py --phase 2.2 HIC
  python scripts/run_sweep.py --phase 4.2 Titer
  python scripts/run_sweep.py --phase 5 --all

Supported phases:
  1     - Learning rate search (property-agnostic)
  2.1   - Encoder selection (property-agnostic)
  2.2   - Decoder architecture (per-property)
  4.1   - Sequence representation (per-property)
  4.2   - Encoder comparison (per-property)
  4.3   - Antibody features (per-property)
  5     - Bayesian optimization (per-property)
  5.1   - Architecture tuning (per-property)
  5.2   - Options tuning (per-property)
  6     - Final tuning (per-property)
        """
    )

    parser.add_argument("--phase", type=str, required=True,
                       help="Phase identifier (e.g., '1', '2.1', '4.2', '5')")
    parser.add_argument("property", nargs="?", type=str,
                       help="Property to tune (e.g., HIC, Titer, AC-SINS_pH7.4)")
    parser.add_argument("--all", action="store_true",
                       help="Run all properties for this phase sequentially")
    parser.add_argument("--project", type=str, default="CrossAbSense",
                       help="WandB project name")

    args = parser.parse_args()

    # Validate phase
    if args.phase not in PHASE_CONFIG:
        print(f"❌ Unknown phase: {args.phase}")
        print(f"Supported phases: {', '.join(sorted(PHASE_CONFIG.keys(), key=lambda x: float(x)))}")
        return 1

    _, _, is_property_specific = PHASE_CONFIG[args.phase]

    # Determine properties to process
    if not is_property_specific:
        # Property-agnostic phase: ignore property argument
        if args.property:
            print(f"Note: Phase {args.phase} is property-agnostic, ignoring property argument")
        properties = ["_all"]
    elif args.all:
        # All properties for this phase
        properties = list(RUN_COUNTS[args.phase].keys())
        # Filter out special keys and ensure canonical names
        properties = [p for p in properties if p != "_all"]
    elif args.property:
        properties = [args.property]
    else:
        parser.error(f"Phase {args.phase} requires a property or --all flag")
        return 1

    # Display header
    display_props = "all properties" if args.all else (
        "property-agnostic" if properties == ["_all"] else ", ".join(properties)
    )
    print(f"\n{'='*70}")
    print(f"Phase {args.phase} Hyperparameter Sweep")
    print(f"Target: {display_props}")
    print(f"{'='*70}\n")

    # Process each property
    all_success = True
    for prop in properties:
        success = process_property(args.phase, prop, args.project)
        if not success:
            all_success = False
            display = "sweep" if prop == "_all" else prop
            print(f"\n⚠ Failed: {display}\n")
        else:
            display = "sweep" if prop == "_all" else prop
            print(f"\n✓ Completed: {display}\n")

    if all_success:
        print("\n✓ All sweeps completed successfully!")
        return 0
    else:
        print("\n⚠ Some sweeps failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
