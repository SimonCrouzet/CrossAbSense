"""JSON serialization utilities for numpy and torch types."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy and torch types.

    Converts:
    - numpy integers → Python int
    - numpy floats → Python float
    - numpy arrays → Python lists
    - torch tensors → Python lists
    - Path objects → strings
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy/torch types to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of obj
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def save_json(data: dict, filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file, handling numpy/torch types automatically.

    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
        indent: JSON indentation (default: 2)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)


def load_json(filepath: str) -> dict:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary loaded from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)
