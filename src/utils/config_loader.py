"""Configuration loading utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Recursively merge override config into base config.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def get_property_config(
    base_config: Dict, property_name: str
) -> Dict[str, Any]:
    """
    Get merged configuration for a specific property.
    Property-specific values override base config values.

    Args:
        base_config: Base configuration dictionary
        property_name: Name of the property (e.g., 'HIC', 'Titer')

    Returns:
        Fully merged configuration dictionary for the property
    """
    # Get property-specific overrides
    prop_overrides = base_config.get("property_specific", {}).get(property_name, {})

    # Deep merge property_specific into base config
    merged = merge_configs(base_config, prop_overrides)

    return merged


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
