#!/usr/bin/env python3
"""
Download the GDPa1 antibody developability benchmark dataset (Ginkgo Bioworks).

Source: https://huggingface.co/datasets/ginkgo-datapoints/GDPa1
"""

import pandas as pd
from datasets import load_dataset

def download_gdpa1_dataset():
    """Download the complete GDPa1 dataset from HuggingFace"""
    print("Downloading GDPa1 dataset...")
    
    # Load the dataset from HuggingFace
    dataset = load_dataset("ginkgo-datapoints/GDPa1")
    
    # Convert to pandas DataFrame
    df = dataset['train'].to_pandas()
    
    # Save to CSV
    output_path = "inputs/GDPa1_complete.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    df = download_gdpa1_dataset()