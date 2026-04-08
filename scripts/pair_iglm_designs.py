#!/usr/bin/env python3
"""
Pair IgLM VH and VL designs from FASTA files to create matched antibodies.
Creates a CSV with 1-to-1 pairing of generated sequences.
"""

import argparse
from pathlib import Path
import pandas as pd
from Bio import SeqIO


def pair_designs(vh_fasta_path, vl_fasta_path, output_csv_path):
    """
    Pair VH and VL sequences from FASTA files.
    
    Args:
        vh_fasta_path: Path to VH sequences FASTA file
        vl_fasta_path: Path to VL sequences FASTA file
        output_csv_path: Path to output CSV file
    """
    # Load VH sequences
    vh_sequences = {}
    for record in SeqIO.parse(vh_fasta_path, "fasta"):
        vh_sequences[record.id] = str(record.seq)
    
    # Load VL sequences
    vl_sequences = {}
    for record in SeqIO.parse(vl_fasta_path, "fasta"):
        vl_sequences[record.id] = str(record.seq)
    
    print(f"Loaded {len(vh_sequences)} VH sequences")
    print(f"Loaded {len(vl_sequences)} VL sequences")
    
    # Pair sequences by ID
    paired_designs = []
    
    # Get all sequence IDs (assuming same IDs in both files)
    all_ids = sorted(set(vh_sequences.keys()) & set(vl_sequences.keys()))
    
    for seq_id in all_ids:
        paired_designs.append({
            "antibody_id": f"IgLM_VH_{seq_id}",
            "vh_protein_sequence": vh_sequences[seq_id],
            "vl_protein_sequence": vl_sequences[seq_id],
            "design_source": "IgLM_VH"
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(paired_designs)
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nCreated {len(df)} paired designs at {output_csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 3 entries:")
    print(df.head(3))


def main():
    parser = argparse.ArgumentParser(
        description="Pair IgLM VH and VL designs into matched antibodies"
    )
    parser.add_argument(
        "--vh_fasta",
        type=str,
        required=True,
        help="Path to VH sequences FASTA file"
    )
    parser.add_argument(
        "--vl_fasta",
        type=str,
        required=True,
        help="Path to VL sequences FASTA file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    vh_fasta_path = Path(args.vh_fasta)
    vl_fasta_path = Path(args.vl_fasta)
    output_csv_path = Path(args.output)
    
    # Validate inputs
    if not vh_fasta_path.exists():
        raise FileNotFoundError(f"VH FASTA not found: {vh_fasta_path}")
    if not vl_fasta_path.exists():
        raise FileNotFoundError(f"VL FASTA not found: {vl_fasta_path}")
    
    # Create output directory if needed
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pair sequences
    pair_designs(vh_fasta_path, vl_fasta_path, output_csv_path)


if __name__ == "__main__":
    main()
