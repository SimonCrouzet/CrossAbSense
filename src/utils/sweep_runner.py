"""
Utilities for running WandB sweeps with smart progress tracking and resumption.

Provides reusable functions for:
- Creating WandB sweeps
- Querying sweep progress from WandB API
- Smart sweep execution (skip completed, run only remaining)
- Multi-sweep orchestration with progress tracking
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


def create_sweep(
    config_path: Union[str, Path],
    project: str = "CrossAbSense",
    verbose: bool = True
) -> Optional[str]:
    """
    Create a WandB sweep from a YAML config file.
    
    Uses smart_create_sweep which automatically handles conditional parameters.
    
    Args:
        config_path: Path to sweep YAML config
        project: WandB project name
        verbose: Whether to print creation messages
        
    Returns:
        Sweep ID (e.g., "abc123xyz") or None on failure
        For configs with conditionals, returns first sweep ID
        
    Example:
        >>> sweep_id = create_sweep("config/tuning/phase3_HIC.yaml")
        >>> print(f"Created: {sweep_id}")
    """
    try:
        from src.utils.sweep_yaml_generator import smart_create_sweep
        
        result = smart_create_sweep(
            yaml_path=config_path,
            project=project,
            verbose=verbose
        )
        
        # If dict returned, we have multiple sweeps (conditionals)
        if isinstance(result, dict):
            if verbose:
                print("Note: Multiple sweeps created (conditional parameters)")
            # Return first sweep ID
            return list(result.values())[0]
        
        # Single sweep ID returned
        return result
        
    except Exception as e:
        print(f"ERROR: Failed to create sweep: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None


def get_sweep_progress(sweep_id: str) -> Optional[Dict[str, int]]:
    """
    Get sweep progress from WandB API.
    
    Args:
        sweep_id: WandB sweep ID (format: entity/project/sweep_id)
        
    Returns:
        dict with keys:
            - completed: Number of finished runs
            - running: Number of currently running runs
            - failed: Number of failed runs
            - crashed: Number of crashed runs
            - total: Total number of all runs
            - active_runs: completed + running (excludes failed/crashed)
        Returns None if unable to query API
        
    Example:
        >>> progress = get_sweep_progress("user/project/sweep123")
        >>> print(f"Active: {progress['active_runs']}/{target_count}")
    """
    try:
        import wandb
        api = wandb.Api()
        sweep = api.sweep(sweep_id)
        
        completed = 0
        running = 0
        failed = 0
        crashed = 0
        
        for run in sweep.runs:
            if run.state == 'finished':
                completed += 1
            elif run.state == 'running':
                running += 1
            elif run.state == 'failed':
                failed += 1
            elif run.state == 'crashed':
                crashed += 1
        
        total = completed + running + failed + crashed
        active_runs = completed + running  # Count only finished + running
        
        return {
            'completed': completed,
            'running': running,
            'failed': failed,
            'crashed': crashed,
            'total': total,
            'active_runs': active_runs
        }
    except Exception as e:
        import sys
        print(
            f"Warning: Could not query sweep progress: {e}",
            file=sys.stderr
        )
        return None


def calculate_remaining_runs(
    sweep_id: str,
    target_count: int
) -> Tuple[Optional[Dict[str, int]], int]:
    """
    Calculate how many runs are needed to reach target count.
    
    Args:
        sweep_id: WandB sweep ID
        target_count: Target number of active runs (completed + running)
        
    Returns:
        Tuple of (progress_dict, remaining_count)
        - progress_dict: Result from get_sweep_progress() or None
        - remaining_count: Number of runs needed to reach target
        
    Example:
        >>> progress, remaining = calculate_remaining_runs("sweep123", 20)
        >>> if remaining > 0:
        >>>     print(f"Need {remaining} more runs")
    """
    progress = get_sweep_progress(sweep_id)
    
    if progress is None:
        # Unable to query, assume all runs needed
        return None, target_count
    
    active_runs = progress['active_runs']
    remaining = max(0, target_count - active_runs)
    
    return progress, remaining


def print_sweep_progress(
    progress: Dict[str, int],
    target_count: int,
    label: str = ""
) -> None:
    """
    Print formatted sweep progress information.
    
    Args:
        progress: Progress dict from get_sweep_progress()
        target_count: Target number of runs
        label: Optional label to add to output (e.g., decoder type)
        
    Example:
        >>> progress = get_sweep_progress("sweep123")
        >>> print_sweep_progress(progress, 20, label="mlp")
    """
    remaining = max(0, target_count - progress['active_runs'])
    
    print("─" * 60)
    print(f"  ✓ Completed:  {progress['completed']}")
    if progress['running'] > 0:
        print(f"  ⟳ Running:    {progress['running']}")
    if progress['failed'] > 0:
        print(f"  ✗ Failed:     {progress['failed']}")
    if progress['crashed'] > 0:
        print(f"  ⚠ Crashed:    {progress['crashed']}")
    print("─" * 60)
    print(f"  Active runs:  {progress['active_runs']} / {target_count}")
    print(f"  Remaining:    {remaining}")
    print()


def run_sweep_agent_smart(
    sweep_id: str,
    target_count: int,
    label: str = "",
    verbose: bool = True
) -> bool:
    """
    Run WandB sweep agent with smart progress tracking.
    
    Automatically:
    - Queries current sweep progress
    - Skips if target already reached
    - Runs only remaining runs needed
    - Displays progress information
    
    Args:
        sweep_id: WandB sweep ID
        target_count: Target number of active runs to reach
        label: Optional label for display (e.g., decoder type, property name)
        verbose: Whether to print progress information
        
    Returns:
        bool: True if successful or sweep already complete, False on error
        
    Example:
        >>> # Run sweep until 20 active runs
        >>> success = run_sweep_agent_smart("sweep123", 20, label="HIC_mlp")
        >>> if not success:
        >>>     print("Sweep failed")
    """
    label_str = f" ({label})" if label else ""
    
    # Check current progress
    progress, remaining = calculate_remaining_runs(sweep_id, target_count)
    
    if progress and verbose:
        print(f"\nSweep{label_str}: {sweep_id}")
        print_sweep_progress(progress, target_count, label)
        
        # Skip if target already reached
        if progress['active_runs'] >= target_count:
            msg = f"✓ Sweep{label_str} already complete "
            msg += f"({progress['active_runs']}/{target_count} runs)"
            print(msg)
            print("  Skipping...")
            return True
        
        print(f"Running {remaining} remaining runs...")
    else:
        # Unable to query progress or not verbose
        if verbose:
            print(f"\nSweep{label_str}: {sweep_id}")
            if progress is None:
                print(
                    f"Unable to query progress, "
                    f"running {target_count} runs..."
                )
        remaining = target_count
    
    if verbose:
        print(f"Command: wandb agent {sweep_id} --count {remaining}")
        print("-" * 80)
    
    result = subprocess.run(
        ["wandb", "agent", sweep_id, "--count", str(remaining)],
        check=False
    )
    
    if verbose:
        print("-" * 80)
        if result.returncode == 0:
            print(f"✓ Sweep{label_str} completed successfully")
        else:
            print(
                f"✗ Sweep{label_str} failed "
                f"with exit code {result.returncode}"
            )
    
    return result.returncode == 0


def run_multiple_sweeps(
    sweep_configs: Dict[str, str],
    target_count: int,
    property_name: str = "",
    continue_on_failure: bool = True
) -> Dict[str, bool]:
    """
    Run multiple sweeps sequentially with progress tracking.
    
    Useful for running decoder-specific sweeps or multi-property batches.
    
    Args:
        sweep_configs: Dict mapping label -> sweep_id
                      e.g., {"mlp": "sweep1", "attention": "sweep2"}
        target_count: Target number of runs per sweep
        property_name: Optional property name for display
        continue_on_failure: If True, continue to next sweep on failure
        
    Returns:
        Dict mapping label -> success_status
        
    Example:
        >>> sweeps = {
        >>>     "mlp": "user/proj/sweep1",
        >>>     "attention": "user/proj/sweep2"
        >>> }
        >>> results = run_multiple_sweeps(sweeps, 20, property_name="HIC")
        >>> if all(results.values()):
        >>>     print("All sweeps completed!")
    """
    print(f"\nRunning {len(sweep_configs)} sweeps sequentially...")
    if property_name:
        print(f"Property: {property_name}")
    print(f"Target: {target_count} runs per sweep")
    print()
    
    results = {}
    
    for i, (label, sweep_id) in enumerate(sweep_configs.items(), 1):
        print()
        print(f"[{i}/{len(sweep_configs)}] {label}")
        
        success = run_sweep_agent_smart(
            sweep_id,
            target_count,
            label=label,
            verbose=True
        )
        
        results[label] = success
        
        if not success and not continue_on_failure:
            print(f"\n⚠ Sweep for {label} failed, stopping...")
            break
        elif not success:
            print(f"\n⚠ Sweep for {label} failed, continuing...")
    
    # Summary
    print()
    print("=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    for label, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {label}")
    print("=" * 60)
    
    return results
