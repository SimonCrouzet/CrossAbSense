#!/usr/bin/env python3
"""
Check if a model checkpoint matches expected configuration.

Usage:
    python check_model_config.py models/Titer_42eda612/fold1.ckpt wandb/sweep-jxsmgwtx/config-bxctf2wa.yaml
    python check_model_config.py models/HIC_47f5ae80/fold1.ckpt src/config/default_config.yaml --property HIC
"""

import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """Load checkpoint and extract configuration."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Extract config if available
    config = ckpt.get('config', {})
    property_name = ckpt.get('property', 'unknown')
    state_dict = ckpt.get('state_dict', {})
    
    print(f"  Property: {property_name}")
    print(f"  Has config: {bool(config)}")
    print(f"  State dict keys: {len(state_dict)}")
    
    return {
        'config': config,
        'property': property_name,
        'state_dict': state_dict,
    }


def load_yaml_config(yaml_path: str, property_name: str = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    print(f"\nLoading YAML config: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if it's a WandB sweep config (flat structure with .value)
    is_wandb_sweep = any(
        isinstance(v, dict) and 'value' in v 
        for v in config.values()
    )
    
    if is_wandb_sweep:
        print("  Format: WandB sweep config (flat)")
        # Extract values from WandB format
        flat_config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v 
                      for k, v in config.items()}
        return flat_config
    else:
        print("  Format: Standard YAML config (nested)")
        # For standard config, extract property-specific settings if provided
        if property_name and 'property_specific' in config:
            prop_config = config.get('property_specific', {}).get(property_name, {})
            print(f"  Property-specific config for: {property_name}")
            return config, prop_config
        return config, {}


def extract_decoder_params_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Extract decoder architecture parameters from state dict."""
    params = {}
    
    # Find decoder keys
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]
    
    if not decoder_keys:
        return params
    
    # Detect decoder type
    if any('attention' in k or 'attn' in k for k in decoder_keys):
        params['decoder_type'] = 'attention'
        
        # Count attention heads - look for multihead attention modules
        for key in decoder_keys:
            # Look for num_heads parameter in any attention layer
            if '.num_heads' in key and 'heavy' in key:
                try:
                    params['n_heads'] = int(state_dict[key].item())
                    break
                except:
                    pass
            # Alternative: infer from in_proj_weight shape
            if 'in_proj_weight' in key and 'heavy' in key and 'n_heads' not in params:
                weight = state_dict[key]
                # For MultiheadAttention: in_proj has shape [3*embed_dim, embed_dim]
                # We can't directly infer n_heads from this without knowing head_dim
                pass
        
        # Count attention layers
        self_attn_layers = set()
        cross_attn_layers = set()
        for key in decoder_keys:
            if 'heavy_self_attn.' in key or 'light_self_attn.' in key:
                parts = key.split('_self_attn.')[1].split('.')
                if parts[0].isdigit():
                    self_attn_layers.add(int(parts[0]))
            if 'cross_attn.' in key:
                parts = key.split('cross_attn.')[1].split('.')
                if parts[0].isdigit():
                    cross_attn_layers.add(int(parts[0]))
        
        if self_attn_layers:
            params['n_layers'] = max(self_attn_layers) + 1
        elif cross_attn_layers:
            params['n_layers'] = max(cross_attn_layers) + 1
        
        # Detect attention strategy based on layer structure
        has_self_attn = any('self_attn' in k for k in decoder_keys)
        has_cross_attn = any('cross_attn' in k for k in decoder_keys)
        has_vh_cross = any('vh_cross_attn' in k for k in decoder_keys)
        has_vl_cross = any('vl_cross_attn' in k for k in decoder_keys)
        
        if has_vh_cross and has_vl_cross and not has_self_attn:
            # Bidirectional cross-attention (VH ↔ VL)
            params['attention_strategy'] = 'bidirectional_cross'
        elif has_self_attn and has_cross_attn:
            # Has both self and cross attention
            params['attention_strategy'] = 'self_cross'
        elif has_self_attn and not has_cross_attn:
            params['attention_strategy'] = 'self_only'
        elif has_cross_attn:
            # Just cross attention (fallback)
            params['attention_strategy'] = 'bidirectional_cross'
        
        # Count output layers
        output_layers = set()
        for key in decoder_keys:
            if 'output_mlp.' in key and '.weight' in key:
                parts = key.split('output_mlp.')[1].split('.')
                if parts[0].isdigit():
                    output_layers.add(int(parts[0]))
        
        if output_layers:
            # Each Linear layer has one weight, count them
            params['n_output_layers'] = len([k for k in decoder_keys if 'output_mlp.' in k and '.weight' in k])
        
        # Detect output activation type from module names
        for key in decoder_keys:
            if 'output_activation' in key:
                # The key name might indicate the type
                if 'Softplus' in key or 'softplus' in key:
                    params['output_activation'] = 'softplus'
                elif 'Sigmoid' in key or 'sigmoid' in key:
                    params['output_activation'] = 'sigmoid'
                elif 'Exp' in key or '_exp' in key:
                    params['output_activation'] = 'exp'
                else:
                    params['output_activation'] = 'unknown'
                break
        
        # If no explicit activation, check if there's an activation module
        if 'output_activation' not in params:
            # Look for specific activation types in the state dict
            has_activation = False
            for key in decoder_keys:
                lower_key = key.lower()
                if 'final' in lower_key or 'output' in lower_key:
                    if 'softplus' in lower_key:
                        params['output_activation'] = 'softplus'
                        has_activation = True
                        break
                    elif 'exp' in lower_key and 'linear' not in lower_key:
                        params['output_activation'] = 'exp'
                        has_activation = True
                        break
                    elif 'sigmoid' in lower_key:
                        params['output_activation'] = 'sigmoid'
                        has_activation = True
                        break
            
            if not has_activation:
                params['output_activation'] = 'none'
    
    elif any('conv' in k.lower() for k in decoder_keys):
        params['decoder_type'] = 'cross_branch_cnn'
    else:
        params['decoder_type'] = 'mlp'
    
    return params


def get_nested_value(config: Dict[str, Any], path: str, default=None) -> Any:
    """Get nested dictionary value using dot notation."""
    keys = path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, {})
        else:
            return default
    return value if value != {} else default


def compare_configs(
    ckpt_info: Dict[str, Any],
    yaml_config: Any,
    property_name: str = None
) -> List[Tuple[str, Any, Any, bool]]:
    """Compare checkpoint config with YAML config.
    
    Returns:
        List of (param_name, ckpt_value, yaml_value, matches) tuples
    """
    comparisons = []
    
    # Extract relevant configs
    ckpt_config = ckpt_info['config']
    state_dict = ckpt_info['state_dict']
    ckpt_property = ckpt_info['property']
    
    # Handle different YAML formats
    if isinstance(yaml_config, tuple):
        base_config, prop_config = yaml_config
        is_standard_config = True
    else:
        # WandB sweep config (flat)
        base_config = yaml_config
        prop_config = {}
        is_standard_config = False
    
    # Get property-specific config from checkpoint if available
    ckpt_prop_config = {}
    if ckpt_property and ckpt_property != 'unknown':
        ckpt_prop_config = ckpt_config.get('property_specific', {}).get(ckpt_property, {})
    
    # Get decoder params from state dict
    state_dict_params = extract_decoder_params_from_state_dict(state_dict)
    
    if is_standard_config:
        # Standard config format
        decoder_type = prop_config.get('decoder_type') or base_config.get('decoder', {}).get('type')
        
        # Get decoder-specific config
        if decoder_type == 'attention':
            yaml_decoder = prop_config.get('attention', {})
            if not yaml_decoder:
                yaml_decoder = base_config.get('decoder', {}).get('attention', {})
        elif decoder_type == 'mlp':
            yaml_decoder = prop_config.get('mlp', {})
            if not yaml_decoder:
                yaml_decoder = base_config.get('decoder', {}).get('mlp', {})
        else:
            yaml_decoder = {}
        
        # Get training config
        yaml_training = base_config.get('training', {})
        yaml_finetune = yaml_training.get('finetune', {})
        
        # Compare decoder type (check property-specific first)
        ckpt_decoder_type = (
            ckpt_prop_config.get('decoder_type') or 
            get_nested_value(ckpt_config, 'decoder.type')
        )
        comparisons.append(('decoder_type', ckpt_decoder_type, decoder_type, ckpt_decoder_type == decoder_type))
        
        # Compare decoder parameters
        if decoder_type == 'attention':
            params_to_check = [
                ('n_heads', 'n_heads'),
                ('hidden_dim', 'hidden_dim'),
                ('n_layers', 'n_layers'),
                ('n_output_layers', 'n_output_layers'),
                ('dropout', 'dropout'),
                ('attention_strategy', 'attention_strategy'),
                ('output_activation', 'output_activation'),
            ]
            
            for yaml_key, short_key in params_to_check:
                yaml_val = yaml_decoder.get(yaml_key)
                
                # Check property-specific config first, then decoder.attention
                ckpt_val = (
                    ckpt_prop_config.get('attention', {}).get(short_key) or
                    get_nested_value(ckpt_config, f'decoder.attention.{short_key}')
                )
                
                # If not in config, try to infer from state dict
                if ckpt_val is None or ckpt_val == {}:
                    ckpt_val = state_dict_params.get(short_key)
                
                # Round floats for comparison (handle scientific notation)
                if isinstance(yaml_val, (float, int)) and isinstance(ckpt_val, (float, int)):
                    matches = abs(float(yaml_val) - float(ckpt_val)) < 1e-6
                else:
                    matches = yaml_val == ckpt_val
                
                comparisons.append((yaml_key, ckpt_val, yaml_val, matches))
        
        # Compare training parameters
        training_params = [
            ('batch_size', yaml_finetune.get('batch_size')),
            ('learning_rate', yaml_finetune.get('learning_rate')),
            ('loss', yaml_training.get('loss') or prop_config.get('loss')),
        ]
        
        for param_name, yaml_val in training_params:
            # Check property-specific first
            ckpt_val = ckpt_prop_config.get(param_name)
            if ckpt_val is None:
                ckpt_val = get_nested_value(ckpt_config, f'training.finetune.{param_name}')
            if ckpt_val is None:
                ckpt_val = get_nested_value(ckpt_config, f'training.{param_name}')
            
            if isinstance(yaml_val, float) and isinstance(ckpt_val, float):
                matches = abs(yaml_val - ckpt_val) < 0.0001
            else:
                matches = yaml_val == ckpt_val
            
            comparisons.append((param_name, ckpt_val, yaml_val, matches))
    
    else:
        # WandB sweep config (flat structure)
        params_to_check = [
            ('decoder_type', 'decoder_type'),
            ('attention_n_heads', 'n_heads'),
            ('attention_hidden_dim', 'hidden_dim'),
            ('attention_n_layers', 'n_layers'),
            ('attention_dropout', 'dropout'),
            ('attention_strategy', 'attention_strategy'),
            ('output_activation', 'output_activation'),
            ('batch_size', 'batch_size'),
            ('learning_rate', 'learning_rate'),
            ('loss', 'loss'),
            ('max_epochs', 'max_epochs'),
            ('warmup_epochs', 'warmup_epochs'),
            ('weight_decay', 'weight_decay'),
        ]
        
        for wandb_key, config_key in params_to_check:
            yaml_val = base_config.get(wandb_key)
            
            # Try to find corresponding value in checkpoint
            if wandb_key.startswith('attention_') and wandb_key != 'attention_strategy':
                # Attention decoder param (strip prefix for lookup)
                short_key = wandb_key.replace('attention_', '')
                # Check property-specific first
                ckpt_val = ckpt_prop_config.get('attention', {}).get(short_key)
                if ckpt_val is None:
                    ckpt_val = ckpt_config.get('decoder', {}).get('attention', {}).get(short_key)
                # If still not in config, try to infer from state dict
                if ckpt_val is None:
                    ckpt_val = state_dict_params.get(short_key)
            elif wandb_key == 'attention_strategy':
                # Keep full key name for attention_strategy
                ckpt_val = ckpt_prop_config.get('attention', {}).get('attention_strategy')
                if ckpt_val is None:
                    ckpt_val = ckpt_config.get('decoder', {}).get('attention', {}).get('attention_strategy')
                if ckpt_val is None:
                    ckpt_val = state_dict_params.get('attention_strategy')
            elif wandb_key == 'attention_strategy':
                # Keep full key name for attention_strategy
                ckpt_val = ckpt_prop_config.get('attention', {}).get('attention_strategy')
                if ckpt_val is None:
                    ckpt_val = ckpt_config.get('decoder', {}).get('attention', {}).get('attention_strategy')
                if ckpt_val is None:
                    ckpt_val = state_dict_params.get('attention_strategy')
            elif wandb_key in ['batch_size', 'learning_rate', 'max_epochs', 'warmup_epochs', 'weight_decay']:
                # Training param
                ckpt_val = (
                    ckpt_prop_config.get(wandb_key) or
                    ckpt_config.get('training', {}).get('finetune', {}).get(wandb_key)
                )
            elif wandb_key == 'loss':
                ckpt_val = (
                    ckpt_prop_config.get('loss') or
                    ckpt_config.get('training', {}).get('loss')
                )
            elif wandb_key == 'decoder_type':
                ckpt_val = (
                    ckpt_prop_config.get('decoder_type') or
                    ckpt_config.get('decoder', {}).get('type')
                )
                # If not in config, infer from state dict
                if ckpt_val is None:
                    ckpt_val = state_dict_params.get('decoder_type')
            elif wandb_key == 'output_activation':
                # Check property-specific first
                ckpt_val = (
                    ckpt_prop_config.get('attention', {}).get('output_activation') or
                    ckpt_config.get('decoder', {}).get('attention', {}).get('output_activation')
                )
                # If not in config, infer from state dict
                if ckpt_val is None:
                    ckpt_val = state_dict_params.get('output_activation')
            else:
                # Decoder level param
                ckpt_val = (
                    ckpt_prop_config.get(wandb_key) or
                    ckpt_config.get('decoder', {}).get(wandb_key)
                )
            
            # Compare (handle scientific notation for floats)
            # Convert string scientific notation to float if needed
            if isinstance(yaml_val, str):
                try:
                    yaml_val_float = float(yaml_val)
                    if isinstance(ckpt_val, (float, int)):
                        matches = abs(yaml_val_float - float(ckpt_val)) < 1e-6
                    else:
                        matches = yaml_val == ckpt_val
                except ValueError:
                    matches = yaml_val == ckpt_val
            elif isinstance(yaml_val, (float, int)) and isinstance(ckpt_val, (float, int)):
                matches = abs(float(yaml_val) - float(ckpt_val)) < 1e-6
            else:
                matches = yaml_val == ckpt_val
            
            comparisons.append((config_key, ckpt_val, yaml_val, matches))
    
    return comparisons


def print_comparison(comparisons: List[Tuple[str, Any, Any, bool]]) -> None:
    """Print comparison results in a nice format."""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    # Calculate column widths
    max_param_len = max(len(c[0]) for c in comparisons)
    max_ckpt_len = max(len(str(c[1])) for c in comparisons)
    max_yaml_len = max(len(str(c[2])) for c in comparisons)
    
    # Print header
    print(f"{'Parameter':<{max_param_len}}  {'Checkpoint':<{max_ckpt_len}}  {'YAML Config':<{max_yaml_len}}  {'Match'}")
    print("-" * 80)
    
    # Print each comparison
    all_match = True
    for param_name, ckpt_val, yaml_val, matches in comparisons:
        match_symbol = "✓" if matches else "✗"
        if not matches:
            all_match = False
        
        # Color code if in terminal
        if matches:
            status = f"  {match_symbol}"
        else:
            status = f"  {match_symbol} MISMATCH"
        
        print(f"{param_name:<{max_param_len}}  {str(ckpt_val):<{max_ckpt_len}}  {str(yaml_val):<{max_yaml_len}}{status}")
    
    print("="*80)
    if all_match:
        print("✓ ALL PARAMETERS MATCH")
    else:
        print("✗ SOME PARAMETERS DO NOT MATCH")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Check if model checkpoint matches expected configuration"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--property",
        type=str,
        help="Property name (for standard config files with property-specific settings)",
    )
    args = parser.parse_args()
    
    # Validate paths
    ckpt_path = Path(args.checkpoint)
    config_path = Path(args.config)
    
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return 1
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    # Load checkpoint
    ckpt_info = load_checkpoint(str(ckpt_path))
    
    # Determine property name
    property_name = args.property or ckpt_info['property']
    
    # Load YAML config
    yaml_config = load_yaml_config(str(config_path), property_name)
    
    # Compare
    comparisons = compare_configs(ckpt_info, yaml_config, property_name)
    
    # Print results
    print_comparison(comparisons)
    
    return 0


if __name__ == "__main__":
    exit(main())
