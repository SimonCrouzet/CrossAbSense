"""
Utilities for generating W&B sweep YAML configurations with conditional
parameters.

W&B doesn't support conditional parameters natively (e.g., "only include
mlp_dropout when decoder_type=='mlp'"). This module provides automatic
detection and handling of conditional parameters in sweep YAMLs.

When you use wandb sweep with a YAML containing "condition:" keys, this
module automatically:
1. Detects conditional parameters
2. Generates separate sweep configs per condition value
3. Creates multiple sweeps (one per condition value)
4. Returns sweep IDs for each

Usage - Automatic Detection:
    from src.utils.sweep_yaml_generator import smart_create_sweep
    
    # Automatically detects and handles conditionals
    sweep_ids = smart_create_sweep("config/tuning/phase2_2_HIC.yaml")
    # Returns dict if conditionals found: {"mlp": "sweep_id", ...}
    # Returns single sweep_id string if no conditionals
    
Usage - Manual Control:
    from src.utils.sweep_yaml_generator import (
        has_conditional_params,
        create_sweeps_from_conditional_yaml
    )
    
    if has_conditional_params("config/tuning/phase2_2_HIC.yaml"):
        sweep_ids = create_sweeps_from_conditional_yaml(
            "config/tuning/phase2_2_HIC.yaml",
            condition_param="decoder_type"
        )
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def has_conditional_params(yaml_path: Union[str, Path]) -> bool:
    """
    Check if YAML contains conditional parameters.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        True if any parameter has a "condition:" key
    """
    try:
        config = parse_conditional_yaml(yaml_path)
        if "parameters" not in config:
            return False
        
        for param_def in config["parameters"].values():
            if isinstance(param_def, dict) and "condition" in param_def:
                return True
        
        return False
    except Exception:
        return False


def detect_condition_param(yaml_path: Union[str, Path]) -> Optional[str]:
    """
    Auto-detect the condition parameter from YAML.
    
    Looks for parameters used in "condition:" keys
    (e.g., "decoder_type" from "condition: decoder_type == 'mlp'")
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Name of condition parameter, or None if not found
    """
    config = parse_conditional_yaml(yaml_path)
    if "parameters" not in config:
        return None
    
    for param_def in config["parameters"].values():
        if isinstance(param_def, dict) and "condition" in param_def:
            condition_str = param_def["condition"]
            # Parse "param_name == 'value'" to extract param_name
            if "==" in condition_str:
                param_name = condition_str.split("==")[0].strip()
                return param_name
    
    return None


def smart_create_sweep(
    yaml_path: Union[str, Path],
    project: Optional[str] = None,
    entity: Optional[str] = None,
    name_template: Optional[str] = None,
    verbose: bool = True
) -> Union[str, Dict[str, str]]:
    """
    Smart sweep creation with automatic conditional parameter detection.
    
    This is the main entry point for automatic handling:
    - Detects if YAML has conditional parameters
    - If yes: Creates separate sweeps per condition value
    - If no: Creates single sweep normally
    
    Args:
        yaml_path: Path to YAML file
        project: W&B project name (overrides YAML)
        entity: W&B entity name (overrides YAML)
        name_template: Template for sweep names if conditionals found
                      (use {value} placeholder)
        verbose: Print progress messages
        
    Returns:
        - Single sweep ID (str) if no conditionals
        - Dict of sweep IDs if conditionals found: {value: sweep_id}
        
    Example:
        # Automatic detection
        result = smart_create_sweep("config/tuning/phase2_2_HIC.yaml")
        
        if isinstance(result, dict):
            # Multiple sweeps created
            for decoder_type, sweep_id in result.items():
                print(f"{decoder_type}: {sweep_id}")
        else:
            # Single sweep created
            print(f"Sweep: {result}")
    """
    yaml_path = Path(yaml_path)
    
    # Check for conditional parameters
    if not has_conditional_params(yaml_path):
        # No conditionals - create single sweep normally
        if verbose:
            print(f"No conditional parameters found in {yaml_path}")
            print("Creating single sweep...")
        
        config = parse_conditional_yaml(yaml_path)
        sweep_id = create_sweep_from_config(
            config,
            project=project,
            entity=entity,
            save_yaml=False
        )
        
        if verbose:
            print(f"✓ Created sweep: {sweep_id}")
        
        return sweep_id
    
    # Has conditionals - auto-detect condition parameter
    condition_param = detect_condition_param(yaml_path)
    
    if condition_param is None:
        raise ValueError(
            f"Found conditional parameters in {yaml_path} but could not "
            "detect condition parameter name"
        )
    
    if verbose:
        print(f"Detected conditional parameters based on: {condition_param}")
        print("Creating separate sweeps per condition value...")
    
    # Generate name template if not provided
    if name_template is None:
        base_name = yaml_path.stem
        name_template = f"{base_name}_{{value}}"
    
    # Create separate sweeps
    sweep_ids = create_sweeps_from_conditional_yaml(
        yaml_path=yaml_path,
        condition_param=condition_param,
        sweep_name_template=name_template,
        project=project,
        entity=entity,
        save_yamls=False,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n✓ Created {len(sweep_ids)} sweeps:")
        for value, sweep_id in sweep_ids.items():
            print(f"  {value}: {sweep_id}")
    
    return sweep_ids


def parse_conditional_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a YAML file that may contain conditional parameter syntax.
    
    This function reads YAML files with pseudo-conditional syntax where parameters
    have a "condition:" key. It preserves this structure for later processing.
    
    Args:
        yaml_path: Path to YAML file with potential conditional syntax
        
    Returns:
        Dictionary with parsed YAML structure, preserving condition keys
        
    Example YAML:
        parameters:
          decoder_type:
            values: ["mlp", "attention"]
          
          mlp_dropout:
            condition: decoder_type == "mlp"
            distribution: uniform
            min: 0.2
            max: 0.5
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_conditional_params(
    parameters: Dict[str, Any],
    condition_param: str
) -> Dict[str, Dict[str, Any]]:
    """
    Extract parameters that have conditions and group them by condition value.
    
    Args:
        parameters: Dictionary of parameter definitions from YAML
        condition_param: Name of the parameter used in conditions (e.g., "decoder_type")
        
    Returns:
        Dictionary mapping condition values to their associated parameters
        
    Example:
        If decoder_type has values ["mlp", "attention"] and mlp_dropout has
        condition: decoder_type == "mlp", returns:
        {
            "mlp": {
                "mlp_dropout": {distribution: "uniform", min: 0.2, max: 0.5},
                ...
            },
            "attention": {
                "attention_hidden_dim": {...},
                ...
            }
        }
    """
    # Get condition values (e.g., ["mlp", "attention", "cnn"])
    if condition_param not in parameters:
        raise ValueError(f"Condition parameter '{condition_param}' not found in parameters")
    
    condition_param_def = parameters[condition_param]
    if "values" in condition_param_def:
        condition_values = condition_param_def["values"]
    elif "value" in condition_param_def:
        condition_values = [condition_param_def["value"]]
    else:
        raise ValueError(f"Condition parameter '{condition_param}' must have 'value' or 'values' key")
    
    # Initialize result dictionary
    conditional_params: Dict[str, Dict[str, Any]] = {val: {} for val in condition_values}
    
    # Scan all parameters for conditions
    for param_name, param_def in parameters.items():
        if not isinstance(param_def, dict):
            continue
            
        if "condition" in param_def:
            condition_str = param_def["condition"]
            
            # Parse condition string (e.g., "decoder_type == 'mlp'")
            # Simple parsing: extract value after ==
            if "==" in condition_str:
                parts = condition_str.split("==")
                if len(parts) == 2:
                    condition_key = parts[0].strip()
                    condition_value = parts[1].strip().strip('"').strip("'")
                    
                    if condition_key == condition_param and condition_value in condition_values:
                        # Remove condition key from param_def
                        param_def_copy = param_def.copy()
                        param_def_copy.pop("condition", None)
                        
                        # Add to appropriate group
                        conditional_params[condition_value][param_name] = param_def_copy
    
    return conditional_params


def get_unconditional_params(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters that don't have conditions (common to all sweeps).
    
    Args:
        parameters: Dictionary of parameter definitions from YAML
        
    Returns:
        Dictionary of unconditional parameters
    """
    unconditional = {}
    
    for param_name, param_def in parameters.items():
        if not isinstance(param_def, dict):
            unconditional[param_name] = param_def
        elif "condition" not in param_def:
            unconditional[param_name] = param_def
    
    return unconditional


def generate_sweep_configs(
    base_config: Dict[str, Any],
    condition_param: str,
    sweep_name_template: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate separate sweep configurations for each condition value.
    
    Args:
        base_config: Base YAML config with conditional parameters
        condition_param: Name of parameter used in conditions (e.g., "decoder_type")
        sweep_name_template: Template for sweep names (use {value} placeholder)
                           Example: "Phase2.2_HIC_{value}"
        
    Returns:
        Dictionary mapping condition values to complete sweep configs
        
    Example:
        configs = generate_sweep_configs(
            base_config,
            condition_param="decoder_type",
            sweep_name_template="Phase2.2_HIC_{value}"
        )
        # Returns: {"mlp": {...config...}, "attention": {...config...}, ...}
    """
    if "parameters" not in base_config:
        raise ValueError("Base config must have 'parameters' key")
    
    parameters = base_config["parameters"]
    
    # Extract conditional and unconditional parameters
    conditional_params = extract_conditional_params(parameters, condition_param)
    unconditional_params = get_unconditional_params(parameters)
    
    # Generate separate configs
    configs = {}
    
    for condition_value in conditional_params.keys():
        # Create config copy
        config = base_config.copy()
        
        # Build parameters: unconditional + condition-specific
        new_parameters = unconditional_params.copy()
        
        # Set condition parameter to fixed value
        new_parameters[condition_param] = {"value": condition_value}
        
        # Add condition-specific parameters
        new_parameters.update(conditional_params[condition_value])
        
        config["parameters"] = new_parameters
        
        # Update sweep name if template provided
        if sweep_name_template and "name" in config:
            config["name"] = sweep_name_template.format(value=condition_value)
        elif sweep_name_template:
            config["name"] = sweep_name_template.format(value=condition_value)
        
        configs[condition_value] = config
    
    return configs


def config_to_yaml_string(config: Dict[str, Any]) -> str:
    """
    Convert config dictionary to YAML string.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        YAML string representation
    """
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def create_sweep_from_config(
    config: Dict[str, Any],
    name: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    save_yaml: bool = False,
    yaml_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Create a W&B sweep from configuration dictionary.
    
    Uses a temporary file approach since wandb sweep doesn't support stdin
    reliably across all versions.
    
    Args:
        config: Sweep configuration dictionary
        name: Override sweep name
        project: Override W&B project
        entity: Override W&B entity
        save_yaml: If True, save YAML permanently. If False, use temp file
        yaml_path: Path to save YAML (only if save_yaml=True)
        
    Returns:
        Sweep ID (format: entity/project/sweep_id)
        
    Raises:
        RuntimeError: If sweep creation fails
    """
    import tempfile
    
    # Override config fields if provided
    if name:
        config["name"] = name
    if project:
        config["project"] = project
    
    # Convert config to YAML string
    yaml_str = config_to_yaml_string(config)
    
    if save_yaml and yaml_path:
        # Save permanently
        yaml_path = Path(yaml_path)
        yaml_path.write_text(yaml_str)
        sweep_file = str(yaml_path)
        cleanup_file = False
    else:
        # Use temporary file
        fd, sweep_file = tempfile.mkstemp(suffix='.yaml', text=True)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(yaml_str)
        except:
            os.close(fd)
            raise
        cleanup_file = True
    
    try:
        # Create sweep from file
        cmd = ["wandb", "sweep", sweep_file]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract sweep ID from output
        output = result.stdout + result.stderr
        
        for line in output.split('\n'):
            if "wandb agent" in line:
                sweep_id = line.strip().split()[-1]
                return sweep_id
            elif "Created sweep" in line and "ID:" in line:
                parts = line.split("ID:")
                if len(parts) > 1:
                    sweep_id = parts[1].strip()
                    return sweep_id
        
        raise RuntimeError(
            f"Could not extract sweep ID from output:\n{output}"
        )
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to create sweep:\n{e.stderr}"
        raise RuntimeError(error_msg) from e
    
    finally:
        # Cleanup temporary file if needed
        if cleanup_file:
            try:
                os.unlink(sweep_file)
            except:
                pass


def create_sweeps_from_conditional_yaml(
    yaml_path: Union[str, Path],
    condition_param: str,
    sweep_name_template: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    save_yamls: bool = False,
    yaml_output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Dict[str, str]:
    """
    Complete workflow: Parse conditional YAML and create separate sweeps.
    
    This is the main high-level function that combines all steps:
    1. Parse YAML with conditional parameters
    2. Generate separate configs per condition value
    3. Create W&B sweeps for each config
    
    Args:
        yaml_path: Path to YAML with conditional parameters
        condition_param: Parameter name used in conditions (e.g., "decoder_type")
        sweep_name_template: Template for sweep names (use {value} placeholder)
        project: W&B project name
        entity: W&B entity name
        save_yamls: If True, save generated YAMLs to disk
        yaml_output_dir: Directory to save YAMLs (only if save_yamls=True)
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping condition values to sweep IDs
        
    Example:
        sweep_ids = create_sweeps_from_conditional_yaml(
            "config/tuning/phase2_2_HIC.yaml",
            condition_param="decoder_type",
            sweep_name_template="Phase2.2_HIC_{value}"
        )
        # Returns: {"mlp": "entity/project/abc123", "attention": "entity/project/def456", ...}
    """
    # Parse base config
    base_config = parse_conditional_yaml(yaml_path)
    
    # Generate separate configs
    configs = generate_sweep_configs(
        base_config,
        condition_param=condition_param,
        sweep_name_template=sweep_name_template
    )
    
    # Create sweeps
    sweep_ids = {}
    
    for condition_value, config in configs.items():
        if verbose:
            print(
                f"Creating sweep for {condition_param}={condition_value}...",
                file=sys.stderr
            )
        
        # Determine YAML save path if needed
        yaml_save_path = None
        if save_yamls:
            if yaml_output_dir:
                yaml_output_dir = Path(yaml_output_dir)
                yaml_output_dir.mkdir(parents=True, exist_ok=True)
                base_name = Path(yaml_path).stem
                yaml_save_path = yaml_output_dir / f"{base_name}_{condition_value}.yaml"
            else:
                base_name = Path(yaml_path).stem
                yaml_save_path = Path(yaml_path).parent / f"{base_name}_{condition_value}.yaml"
        
        # Create sweep
        sweep_id = create_sweep_from_config(
            config,
            project=project,
            entity=entity,
            save_yaml=save_yamls,
            yaml_path=yaml_save_path
        )
        
        sweep_ids[condition_value] = sweep_id
        if verbose:
            print(f"✓ Created sweep: {sweep_id}", file=sys.stderr)
    
    return sweep_ids


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for sweep YAML generator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate W&B sweeps from YAML with conditional parameters"
    )
    parser.add_argument(
        "yaml_path",
        type=str,
        help="Path to YAML file with conditional parameters"
    )
    parser.add_argument(
        "--condition-param",
        type=str,
        required=True,
        help="Parameter name used in conditions (e.g., 'decoder_type')"
    )
    parser.add_argument(
        "--name-template",
        type=str,
        help="Template for sweep names (use {value} placeholder)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="W&B project name (overrides YAML)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="W&B entity name (overrides YAML)"
    )
    parser.add_argument(
        "--save-yamls",
        action="store_true",
        help="Save generated YAMLs to disk (default: in-memory only)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save YAMLs (only if --save-yamls)"
    )
    
    args = parser.parse_args()
    
    # Create sweeps
    sweep_ids = create_sweeps_from_conditional_yaml(
        yaml_path=args.yaml_path,
        condition_param=args.condition_param,
        sweep_name_template=args.name_template,
        project=args.project,
        entity=args.entity,
        save_yamls=args.save_yamls,
        yaml_output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*80)
    print("Sweep IDs:")
    for condition_value, sweep_id in sweep_ids.items():
        print(f"  {condition_value}: {sweep_id}")
    print("="*80)


if __name__ == "__main__":
    main()
