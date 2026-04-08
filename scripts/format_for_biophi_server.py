#!/usr/bin/env python3
"""
Format antibody sequences for BioPhi server.

BioPhi requires:
- Multiple antibody sequences in FASTA format
- Both chains of an antibody should have the same ID
- IDs should have _HC/_LC or _VH/_VL suffix
"""

import argparse
import pandas as pd
from pathlib import Path


def format_sequences_for_biophi(input_csv, output_fasta, vh_col='vh_protein_sequence', 
                                vl_col='vl_protein_sequence', id_col=None, suffix_type='VH_VL'):
    """
    Convert antibody sequences from CSV to BioPhi-compatible FASTA format.
    
    Args:
        input_csv: Path to input CSV file with antibody sequences
        output_fasta: Path to output FASTA file
        vh_col: Column name for heavy chain variable region sequence
        vl_col: Column name for light chain variable region sequence  
        id_col: Column name for antibody ID (if None, uses 'antibody_id' or 'antibody_name')
        suffix_type: Type of suffix to use ('VH_VL' or 'HC_LC')
    """
    
    # Read CSV file
    df = pd.read_csv(input_csv)
    
    # Determine ID column
    if id_col is None:
        if 'antibody_id' in df.columns:
            id_col = 'antibody_id'
        elif 'antibody_name' in df.columns:
            id_col = 'antibody_name'
        else:
            raise ValueError("Could not find antibody_id or antibody_name column. Please specify id_col.")
    
    # Check for required columns
    if vh_col not in df.columns or vl_col not in df.columns:
        raise ValueError(f"Required columns not found. Available columns: {df.columns.tolist()}")
    
    # Set suffix based on type
    if suffix_type == 'VH_VL':
        heavy_suffix = '_VH'
        light_suffix = '_VL'
    elif suffix_type == 'HC_LC':
        heavy_suffix = '_HC'
        light_suffix = '_LC'
    else:
        raise ValueError("suffix_type must be 'VH_VL' or 'HC_LC'")
    
    # Write FASTA file
    with open(output_fasta, 'w') as f:
        for idx, row in df.iterrows():
            antibody_id = str(row[id_col])
            vh_seq = str(row[vh_col])
            vl_seq = str(row[vl_col])
            
            # Skip if sequences are missing
            if pd.isna(vh_seq) or pd.isna(vl_seq) or vh_seq == 'nan' or vl_seq == 'nan':
                print(f"Warning: Skipping {antibody_id} - missing sequence")
                continue
            
            # Write heavy chain
            f.write(f">{antibody_id}{heavy_suffix}\n")
            f.write(f"{vh_seq}\n")
            
            # Write light chain
            f.write(f">{antibody_id}{light_suffix}\n")
            f.write(f"{vl_seq}\n")
    
    print(f"✓ Successfully formatted {len(df)} antibodies to {output_fasta}")
    print(f"  Format: IDs with {heavy_suffix}/{light_suffix} suffixes")


def main():
    parser = argparse.ArgumentParser(
        description='Format antibody sequences for BioPhi server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Format GDPa1 sequences
  python format_for_biophi_server.py inputs/GDPa1_v1.2_sequences.csv -o biophi_sequences.fasta
  
  # Format heldout set with HC/LC suffixes
  python format_for_biophi_server.py inputs/heldout-set-sequences.csv -o biophi_heldout.fasta --suffix HC_LC
  
  # Limit to first 10 sequences
  python format_for_biophi_server.py inputs/GDPa1_v1.2_sequences.csv -o biophi_sample.fasta -n 10
        """
    )
    
    parser.add_argument('input_csv', type=str, help='Input CSV file with antibody sequences')
    parser.add_argument('-o', '--output', type=str, help='Output FASTA file (default: auto-generated from input name)',
                       default=None)
    parser.add_argument('--vh-col', type=str, default='vh_protein_sequence',
                       help='Column name for heavy chain variable region (default: vh_protein_sequence)')
    parser.add_argument('--vl-col', type=str, default='vl_protein_sequence',
                       help='Column name for light chain variable region (default: vl_protein_sequence)')
    parser.add_argument('--id-col', type=str, default=None,
                       help='Column name for antibody ID (default: auto-detect antibody_id or antibody_name)')
    parser.add_argument('--suffix', type=str, choices=['VH_VL', 'HC_LC'], default='VH_VL',
                       help='Suffix type to use (default: VH_VL)')
    parser.add_argument('-n', '--num-sequences', type=int, default=None,
                       help='Limit to first N sequences (default: all)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        return
    
    # Auto-generate output filename if not specified
    if args.output is None:
        args.output = str(input_path.parent / f"{input_path.stem}_biophi_formatted.fasta")
    
    # Read and optionally limit sequences
    df = pd.read_csv(args.input_csv)
    if args.num_sequences:
        df = df.head(args.num_sequences)
        temp_file = str(input_path.parent / f"{input_path.stem}_temp.csv")
        df.to_csv(temp_file, index=False)
        input_file = temp_file
    else:
        input_file = args.input_csv
    
    # Format sequences
    format_sequences_for_biophi(
        input_file,
        args.output,
        vh_col=args.vh_col,
        vl_col=args.vl_col,
        id_col=args.id_col,
        suffix_type=args.suffix
    )
    
    # Clean up temp file
    if args.num_sequences:
        temp_path = Path(temp_file)
        if temp_path.exists():
            temp_path.unlink()


if __name__ == '__main__':
    main()
