#!/usr/bin/env python3
"""
Simple script to convert CSV to FASTA format with just >name headers
"""
import pandas as pd
import sys

if len(sys.argv) != 3:
    print("Usage: python csv_to_fasta_simple.py <input_csv> <output_fasta>")
    sys.exit(1)

input_csv = sys.argv[1]
output_fasta = sys.argv[2]

# Read CSV
df = pd.read_csv(input_csv)

# Check for required columns
if 'name' not in df.columns or 'cds_sequence' not in df.columns:
    print("Error: CSV file must have 'name' and 'cds_sequence' columns")
    sys.exit(1)

# Filter valid sequences
df = df[df['cds_sequence'].notna() & (df['cds_sequence'] != '')]

# Write FASTA file
with open(output_fasta, 'w') as f:
    for _, row in df.iterrows():
        f.write(f">{row['name']}\n")
        f.write(f"{row['cds_sequence']}\n")

print(f"Successfully created {output_fasta} with {len(df)} sequences")