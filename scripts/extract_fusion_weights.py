#!/usr/bin/env python3
"""
Extract learnable chain fusion weights from a trained model checkpoint.
Supports Exp 5: Learned Fusion Weights analysis.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def extract_fusion_weights(checkpoint_path: str):
    """Extract VH/VL fusion weights from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Identify chain fusion weight key
    # It should be something like 'decoder.chain_weight'
    weight_key = None
    for key in state_dict.keys():
        if 'chain_weight' in key:
            weight_key = key
            break
            
    if weight_key is None:
        print("❌ No learnable chain fusion weights found in this checkpoint.")
        print("Ensure the model was trained with use_learnable_chain_fusion='per_chain' or 'per_dim'.")
        return

    weight = state_dict[weight_key]
    
    # Apply sigmoid to get w_VH in [0, 1]
    w_vh = torch.sigmoid(weight)
    w_vl = 1 - w_vh
    
    print(f"Found weights: {weight_key}")
    print(f"Shape: {w_vh.shape}")
    
    if w_vh.dim() == 0 or (w_vh.dim() == 1 and w_vh.shape[0] == 1):
        # per_chain (scalar)
        val_vh = w_vh.item()
        val_vl = w_vl.item()
        print(f"Results (per_chain):")
        print(f"  w_VH (Heavy Chain weight): {val_vh:.4f}")
        print(f"  w_VL (Light Chain weight): {val_vl:.4f}")
        
        if val_vh > 0.6:
            print("  Conclusion: Heavy chain dominance detected.")
        elif val_vh < 0.4:
            print("  Conclusion: Light chain dominance detected.")
        else:
            print("  Conclusion: Balanced contribution from both chains.")
            
    else:
        # per_dim (vector)
        vh_np = w_vh.numpy()
        print(f"Results (per_dim):")
        print(f"  Mean w_VH: {vh_np.mean():.4f} ± {vh_np.std():.4f}")
        print(f"  Min w_VH:  {vh_np.min():.4f}")
        print(f"  Max w_VH:  {vh_np.max():.4f}")
        
        # Save to CSV for detailed analysis
        output_csv = Path(checkpoint_path).with_suffix('.fusion_weights.csv')
        df = pd.DataFrame({
            'dim': range(len(vh_np)),
            'w_vh': vh_np,
            'w_vl': 1 - vh_np
        })
        df.to_csv(output_csv, index=False)
        print(f"Detailed per-dimension weights saved to: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract fusion weights from checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to .ckpt file")
    args = parser.parse_args()
    
    extract_fusion_weights(args.checkpoint)

if __name__ == "__main__":
    main()
