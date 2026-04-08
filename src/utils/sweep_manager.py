#!/usr/bin/env python3
"""
Sweep ID manager for tracking WandB sweep IDs across tuning phases.

Saves sweep IDs to a JSON file for easy retrieval and reproducibility.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SweepManager:
    """Manage WandB sweep IDs for different tuning phases."""

    def __init__(self, sweep_file: str = "config/tuning/sweep_ids.json"):
        """
        Initialize sweep manager.

        Args:
            sweep_file: Path to JSON file storing sweep IDs
        """
        self.sweep_file = Path(sweep_file)
        self.sweep_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing sweep IDs
        self.sweeps = self._load_sweeps()

    def _load_sweeps(self) -> Dict:
        """Load sweep IDs from JSON file."""
        if self.sweep_file.exists():
            with open(self.sweep_file, 'r') as f:
                return json.load(f)
        else:
            logger.info(f"No existing sweep file found at {self.sweep_file}, creating new one")
            return {}

    def _save_sweeps(self):
        """Save sweep IDs to JSON file."""
        with open(self.sweep_file, 'w') as f:
            json.dump(self.sweeps, f, indent=2)
        logger.info(f"✓ Saved sweep IDs to {self.sweep_file}")

    def save_sweep(self, phase: str, property_name: str, sweep_id: str):
        """
        Save a sweep ID for a specific phase and property.

        Args:
            phase: Tuning phase (e.g., "phase1_lr", "phase2_1_encoder", "phase2_2_decoder")
            property_name: Property being optimized (e.g., "HIC", "Titer")
            sweep_id: WandB sweep ID
        """
        # Initialize phase dict if it doesn't exist
        if phase not in self.sweeps:
            self.sweeps[phase] = {}

        # Save sweep ID for this property
        self.sweeps[phase][property_name] = sweep_id

        # Save to file
        self._save_sweeps()

        logger.info(f"✓ Saved sweep ID for {phase}/{property_name}: {sweep_id}")

    def get_sweep(self, phase: str, property_name: str) -> Optional[str]:
        """
        Get sweep ID for a specific phase and property.

        Args:
            phase: Tuning phase
            property_name: Property name

        Returns:
            Sweep ID if found, None otherwise
        """
        sweep_id = self.sweeps.get(phase, {}).get(property_name)

        if sweep_id:
            logger.info(f"✓ Found sweep ID for {phase}/{property_name}: {sweep_id}")
        else:
            logger.warning(f"No sweep ID found for {phase}/{property_name}")

        return sweep_id

    def list_sweeps(self, phase: Optional[str] = None) -> Dict:
        """
        List all sweep IDs, optionally filtered by phase.

        Args:
            phase: Optional phase filter

        Returns:
            Dict of sweep IDs
        """
        if phase:
            return {phase: self.sweeps.get(phase, {})}
        else:
            return self.sweeps

    def print_sweeps(self, phase: Optional[str] = None):
        """
        Pretty-print sweep IDs.

        Args:
            phase: Optional phase filter
        """
        sweeps = self.list_sweeps(phase)

        if not sweeps:
            print("No sweep IDs saved yet.")
            return

        print("=" * 80)
        print("Saved Sweep IDs")
        print("=" * 80)

        for phase_name, properties in sweeps.items():
            if properties:
                print(f"\n{phase_name}:")
                for prop, sweep_id in properties.items():
                    print(f"  {prop}: {sweep_id}")

        print("=" * 80)

    def check_sweep_status(self, sweep_id: str, expected_runs: int = 3) -> Tuple[Dict, bool]:
        """
        Check sweep status and determine if resume is needed.

        Args:
            sweep_id: WandB sweep ID
            expected_runs: Total expected runs (default: 3 for grid search)

        Returns:
            Tuple of (status_dict, needs_resume)
            status_dict contains: completed, running, failed, crashed, total, state, remaining
            needs_resume is True if sweep is finished/stopped but has remaining runs
        """
        try:
            import wandb
            api = wandb.Api()
            sweep = api.sweep(sweep_id)

            # Count runs by state
            completed = sum(1 for run in sweep.runs if run.state == 'finished')
            running = sum(1 for run in sweep.runs if run.state == 'running')
            failed = sum(1 for run in sweep.runs if run.state == 'failed')
            crashed = sum(1 for run in sweep.runs if run.state == 'crashed')

            total = completed + running + failed + crashed
            remaining = expected_runs - total
            if remaining < 0:
                remaining = 0

            sweep_state = sweep.state if hasattr(sweep, 'state') else 'UNKNOWN'

            status = {
                'completed': completed,
                'running': running,
                'failed': failed,
                'crashed': crashed,
                'total': total,
                'state': sweep_state,
                'remaining': remaining,
                'expected': expected_runs
            }

            # Determine if resume is needed
            # Only resume if sweep is stopped/finished, not if already RUNNING
            needs_resume = (
                remaining > 0 and
                sweep_state in ['FINISHED', 'STOPPED', 'PAUSED']
            )

            return status, needs_resume

        except Exception as e:
            logger.error(f"Failed to check sweep status: {e}")
            return {
                'completed': 0,
                'running': 0,
                'failed': 0,
                'crashed': 0,
                'total': 0,
                'state': 'UNKNOWN',
                'remaining': expected_runs,
                'expected': expected_runs
            }, False

    def resume_sweep(self, sweep_id: str) -> bool:
        """
        Resume a finished/stopped sweep.

        Args:
            sweep_id: WandB sweep ID

        Returns:
            True if resume was successful, False otherwise
        """
        try:
            print(f"Resuming sweep: {sweep_id}")
            result = subprocess.run(
                ['wandb', 'sweep', '--resume', sweep_id],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 or 'Resumed sweep' in result.stdout:
                print(f"✓ Successfully resumed sweep")
                logger.info(f"✓ Successfully resumed sweep: {sweep_id}")
                return True
            else:
                print(f"⚠ Resume status unclear (exit code {result.returncode})")
                logger.warning(f"Sweep resume command executed but status unclear: {sweep_id}")
                return True  # Assume success if no error

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout while resuming sweep")
            logger.error(f"Timeout while resuming sweep: {sweep_id}")
            return False
        except Exception as e:
            print(f"✗ Failed to resume sweep: {e}")
            logger.error(f"Failed to resume sweep {sweep_id}: {e}")
            return False

    def check_and_resume(self, sweep_id: str, expected_runs: int = 3) -> Dict:
        """
        Check sweep status and auto-resume if needed.

        Args:
            sweep_id: WandB sweep ID
            expected_runs: Total expected runs

        Returns:
            Status dict with sweep information
        """
        status, needs_resume = self.check_sweep_status(sweep_id, expected_runs)

        # Debug: show current state
        print(f"Sweep state: {status['state']}, Remaining: {status['remaining']}, Needs resume: {needs_resume}")

        if needs_resume:
            print(f"⟳ Sweep is {status['state']} with {status['remaining']} remaining runs - resuming...")
            logger.info(
                f"Sweep {sweep_id} is {status['state']} but has "
                f"{status['remaining']} remaining runs - resuming..."
            )
            if self.resume_sweep(sweep_id):
                status['resumed'] = True
                print("✓ Sweep resumed successfully")
            else:
                status['resumed'] = False
                status['resume_error'] = True
                print("✗ Sweep resume failed")
        else:
            status['resumed'] = False
            if status['remaining'] > 0 and status['state'] == 'RUNNING':
                print(f"⚠ Sweep is RUNNING with {status['remaining']} remaining runs")
                print(f"  This may indicate a stale grid sweep - agent may not get scheduled work")
                print(f"  If agent hangs, manually stop and recreate the sweep")

        return status


def check_and_resume_sweep(phase: str, property_name: str, expected_runs: int = 3,
                          project: str = "CrossAbSense") -> Optional[Dict]:
    """
    Check and auto-resume a sweep if needed (phase-agnostic).

    Args:
        phase: Tuning phase (e.g., "phase3_refinement", "phase4_1_seqrep")
        property_name: Property name
        expected_runs: Expected number of runs (default: 3 for grid search)
        project: WandB project name (default: "CrossAbSense")

    Returns:
        Status dict if sweep exists, None otherwise
    """
    import os

    manager = SweepManager()
    sweep_id = manager.get_sweep(phase, property_name)

    if not sweep_id:
        return None

    # Ensure we have the full sweep ID with entity/project prefix
    if '/' not in sweep_id:
        entity = os.environ.get('WANDB_ENTITY', 'simon-crouzet')
        full_sweep_id = f"{entity}/{project}/{sweep_id}"
    else:
        full_sweep_id = sweep_id

    return manager.check_and_resume(full_sweep_id, expected_runs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage WandB sweep IDs")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Save command
    save_parser = subparsers.add_parser("save", help="Save a sweep ID")
    save_parser.add_argument("--phase", required=True, help="Tuning phase")
    save_parser.add_argument("--property", required=True, help="Property name")
    save_parser.add_argument("--sweep-id", required=True, help="WandB sweep ID")

    # Get command
    get_parser = subparsers.add_parser("get", help="Get a sweep ID")
    get_parser.add_argument("--phase", required=True, help="Tuning phase")
    get_parser.add_argument("--property", required=True, help="Property name")

    # List command
    list_parser = subparsers.add_parser("list", help="List all sweep IDs")
    list_parser.add_argument("--phase", help="Optional phase filter")

    args = parser.parse_args()

    manager = SweepManager()

    if args.command == "save":
        manager.save_sweep(args.phase, args.property, args.sweep_id)

    elif args.command == "get":
        sweep_id = manager.get_sweep(args.phase, args.property)
        if sweep_id:
            print(sweep_id)

    elif args.command == "list":
        manager.print_sweeps(args.phase)

    else:
        parser.print_help()
