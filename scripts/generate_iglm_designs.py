#!/usr/bin/env python3
"""
Script to generate antibody designs using IgLM.
Generates 1000 VH and 1000 VL sequences and pairs them 1-to-1.
"""

import argparse
import os
import subprocess
from pathlib import Path
import pandas as pd
from Bio import SeqIO

def run_command(cmd):
    """Run a command in the iglm conda environment."""
    conda_cmd = f"conda run -n iglm {cmd}"
    print(f"Running: {conda_cmd}")
    result = subprocess.run(conda_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result


def pair_designs_from_fastas(vh_fasta_path, vl_fasta_path, output_csv_path, 
                              herceptin_vh=None, herceptin_vl=None):
    """
    Pair VH and VL sequences from FASTA files into matched antibodies.
    
    Creates three types of designs:
    1. VH designs paired 1-to-1 with VL designs (matched pairs)
    2. VH designs paired with Herceptin VL (if provided)
    3. VL designs paired with Herceptin VH (if provided)
    
    Args:
        vh_fasta_path: Path to VH sequences FASTA file
        vl_fasta_path: Path to VL sequences FASTA file
        output_csv_path: Path to output CSV file
        herceptin_vh: Optional Herceptin VH sequence for pairing
        herceptin_vl: Optional Herceptin VL sequence for pairing
    """
    print(f"\nPairing designs from FASTA files...")
    
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
    
    all_designs = []
    
    # 1. Create 1-to-1 paired designs (VH seq_0 with VL seq_0, etc.)
    all_ids = sorted(set(vh_sequences.keys()) & set(vl_sequences.keys()))
    print(f"Creating {len(all_ids)} matched VH-VL pairs...")
    
    for seq_id in all_ids:
        all_designs.append({
            "antibody_id": f"IgLM_VH_{seq_id}",
            "vh_protein_sequence": vh_sequences[seq_id],
            "vl_protein_sequence": vl_sequences[seq_id],
            "design_source": "IgLM_VH"
        })
    
    # 2. Optionally pair VH designs with Herceptin VL
    if herceptin_vl:
        print(f"Creating {len(vh_sequences)} VH designs with Herceptin VL...")
        for seq_id, vh_seq in vh_sequences.items():
            all_designs.append({
                "antibody_id": f"IgLM_VH_HercVL_{seq_id}",
                "vh_protein_sequence": vh_seq,
                "vl_protein_sequence": herceptin_vl,
                "design_source": "IgLM_VH_HercVL"
            })
    
    # 3. Optionally pair VL designs with Herceptin VH
    if herceptin_vh:
        print(f"Creating {len(vl_sequences)} VL designs with Herceptin VH...")
        for seq_id, vl_seq in vl_sequences.items():
            all_designs.append({
                "antibody_id": f"IgLM_VL_HercVH_{seq_id}",
                "vh_protein_sequence": herceptin_vh,
                "vl_protein_sequence": vl_seq,
                "design_source": "IgLM_VL_HercVH"
            })
    
    # 4. Add Herceptin baseline if both chains provided
    if herceptin_vh and herceptin_vl:
        all_designs.append({
            "antibody_id": "Herceptin_Baseline",
            "vh_protein_sequence": herceptin_vh,
            "vl_protein_sequence": herceptin_vl,
            "design_source": "Baseline"
        })
    
    # Create DataFrame and save
    if all_designs:
        df = pd.DataFrame(all_designs)
        df.to_csv(output_csv_path, index=False)
        
        print(f"\n✓ Created {len(df)} total designs at {output_csv_path}")
        print("\nDesign breakdown:")
        print(df["design_source"].value_counts())
        print(f"\nFirst 3 entries:")
        print(df.head(3)[["antibody_id", "design_source"]].to_string(index=False))
        
        return df
    else:
        print("No designs created!")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Generate antibody designs with IgLM and pair them",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate new sequences and pair them
  python scripts/generate_iglm_designs.py --num_seqs 100 --output_dir results/iglm_herceptin
  
  # Just pair existing FASTA files without generating
  python scripts/generate_iglm_designs.py --pair_only \\
      --vh_fasta results/iglm_herceptin/vh_designs/generated_seqs.fasta \\
      --vl_fasta results/iglm_herceptin/vl_designs/generated_seqs.fasta \\
      --output_csv results/iglm_herceptin/iglm_designs_for_prediction.csv
        """
    )
    parser.add_argument("--num_seqs", type=int, default=1000, 
                       help="Number of sequences to generate per chain")
    parser.add_argument("--output_dir", type=str, default="external/iglm", 
                       help="Output directory")
    parser.add_argument("--pair_only", action="store_true",
                       help="Only pair existing FASTA files, don't generate new sequences")
    parser.add_argument("--vh_fasta", type=str,
                       help="Path to existing VH FASTA (for --pair_only mode)")
    parser.add_argument("--vl_fasta", type=str,
                       help="Path to existing VL FASTA (for --pair_only mode)")
    parser.add_argument("--output_csv", type=str,
                       help="Output CSV path (for --pair_only mode)")
    parser.add_argument("--include_herceptin_pairs", action="store_true",
                       help="Also create designs with Herceptin VH/VL (in addition to 1-to-1 pairs)")
    args = parser.parse_args()
    
    # Herceptin Reference (from PDB)
    herceptin_vh = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGYAMDYWGQGTLVTVSS"
    herceptin_vl = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
    
    if args.pair_only:
        # Just pair existing FASTA files
        if not args.vh_fasta or not args.vl_fasta or not args.output_csv:
            parser.error("--pair_only requires --vh_fasta, --vl_fasta, and --output_csv")
        
        vh_fasta = Path(args.vh_fasta)
        vl_fasta = Path(args.vl_fasta)
        output_csv = Path(args.output_csv)
        
        if not vh_fasta.exists():
            raise FileNotFoundError(f"VH FASTA not found: {vh_fasta}")
        if not vl_fasta.exists():
            raise FileNotFoundError(f"VL FASTA not found: {vl_fasta}")
        
        # Pair designs
        pair_designs_from_fastas(
            vh_fasta, 
            vl_fasta, 
            output_csv,
            herceptin_vh=herceptin_vh if args.include_herceptin_pairs else None,
            herceptin_vl=herceptin_vl if args.include_herceptin_pairs else None
        )
        
    else:
        # Generate new sequences and pair them
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Generate VH designs
        print(f"Generating {args.num_seqs} VH sequences...")
        vh_dir = output_path / "vh_designs"
        vh_dir.mkdir(exist_ok=True)
        run_command(f"iglm_generate --prompt_sequence EVQ --chain_token \"[HEAVY]\" --species_token \"[HUMAN]\" --num_seqs {args.num_seqs} --output_dir {vh_dir}")
        
        # 2. Generate VL designs
        print(f"Generating {args.num_seqs} VL sequences...")
        vl_dir = output_path / "vl_designs"
        vl_dir.mkdir(exist_ok=True)
        run_command(f"iglm_generate --prompt_sequence DIQ --chain_token \"[LIGHT]\" --species_token \"[HUMAN]\" --num_seqs {args.num_seqs} --output_dir {vl_dir}")
        
        # 3. Pair the generated designs
        vh_fasta = vh_dir / "generated_seqs.fasta"
        vl_fasta = vl_dir / "generated_seqs.fasta"
        csv_path = output_path / "iglm_designs_for_prediction.csv"
        
        if vh_fasta.exists() and vl_fasta.exists():
            pair_designs_from_fastas(
                vh_fasta,
                vl_fasta,
                csv_path,
                herceptin_vh=herceptin_vh if args.include_herceptin_pairs else None,
                herceptin_vl=herceptin_vl if args.include_herceptin_pairs else None
            )

if __name__ == "__main__":
    main()
