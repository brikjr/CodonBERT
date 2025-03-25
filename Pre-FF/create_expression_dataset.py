#!/usr/bin/env python3
"""
Create a simulated expression dataset from RNA elements for model training.
This script generates expression values for RNA elements based on sequence features
and saves the enriched data to a new CSV file.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from Bio.Seq import Seq
import random
from pathlib import Path

# Define function to calculate GC content since import is failing
def calculate_gc_content(sequence):
    """Calculate the GC content of a DNA/RNA sequence as a percentage."""
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    if total == 0:
        return 0.0
    return (gc_count / total) * 100.0

# Parse arguments
parser = argparse.ArgumentParser(description='Create simulated expression dataset from RNA elements')
parser.add_argument('--input', type=str, default='rna_elements.csv', 
                    help='Path to input RNA elements CSV file')
parser.add_argument('--output', type=str, default='rna_elements_expression.csv', 
                    help='Path to output CSV file with expression data')
parser.add_argument('--element_type', type=str, default=None, 
                    help='Filter by specific RNA element type (RBP, UTR, miRNA)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)
random.seed(args.seed)

# Define motifs that might influence expression (for simulation purposes)
# These are simplified examples - real motifs would be more complex
EXPRESSION_ENHANCING_MOTIFS = ['AUGG', 'CCCA', 'GGGA', 'UACG', 'AGCG', 'UGCG']
EXPRESSION_REDUCING_MOTIFS = ['AUAU', 'UAAU', 'CCCC', 'AAAA', 'UUUU', 'GAGA']

def count_kozak_like_sequences(seq):
    """Count Kozak-like sequences in a given RNA sequence."""
    # Basic Kozak consensus: GCCACCAUGG
    kozak_pattern = 'GCCACCAUGG'
    count = 0
    for i in range(len(seq) - len(kozak_pattern) + 1):
        window = seq[i:i+len(kozak_pattern)]
        matches = sum(1 for a, b in zip(window, kozak_pattern) if a == b)
        if matches >= 7:  # Allow for some mismatches
            count += 1
    return count

def count_motifs(seq, motif_list):
    """Count occurrences of motifs in sequence."""
    count = 0
    for motif in motif_list:
        # Count overlapping motifs
        for i in range(len(seq) - len(motif) + 1):
            if seq[i:i+len(motif)] == motif:
                count += 1
    return count

def compute_minimum_free_energy_proxy(seq):
    """
    Compute a simple proxy for minimum free energy based on sequence features.
    This is a very simplified model - in practice you would use a proper RNA folding algorithm.
    """
    gc_content = calculate_gc_content(seq) / 100.0
    # A rough proxy using GC content and sequence length
    return -1 * (gc_content * 0.8 + 0.2) * np.log(len(seq)) * 2

def generate_expression_level(seq, seq_type=None):
    """
    Generate a simulated expression level based on sequence features.
    
    In a real-world scenario, this would be replaced by actual experimental measurements,
    but for demonstration purposes, we're creating a model to simulate expression levels.
    """
    if not seq or pd.isna(seq) or seq == '':
        return np.nan
    
    # Convert to uppercase and standardize to RNA (replace T with U)
    seq = seq.upper().replace('T', 'U')
    
    # Calculate basic sequence features
    length = len(seq)
    if length < 10:  # Skip very short sequences
        return np.nan

    gc_content = calculate_gc_content(seq) / 100.0
    utr5_proxy = seq[:min(100, length//3)]  # First third (proxy for 5' UTR)
    utr3_proxy = seq[max(0, length-100):]   # Last part (proxy for 3' UTR)
    
    # Count motifs
    enhancing_motifs = count_motifs(seq, EXPRESSION_ENHANCING_MOTIFS)
    reducing_motifs = count_motifs(seq, EXPRESSION_REDUCING_MOTIFS)
    kozak_like = count_kozak_like_sequences(seq)
    
    # Calculate MFE proxy
    mfe_proxy = compute_minimum_free_energy_proxy(seq)
    
    # Base expression level from length and GC content
    # We assume longer sequences with moderate GC content express better
    base_expression = (np.log(length) * 0.5) * (1 - abs(gc_content - 0.5) * 2)
    
    # Adjust for motifs
    motif_effect = (enhancing_motifs * 0.2) - (reducing_motifs * 0.15)
    
    # Adjust for Kozak-like sequences
    kozak_effect = kozak_like * 0.3
    
    # Adjust for MFE proxy
    mfe_effect = -mfe_proxy * 0.1
    
    # Type-specific adjustments
    type_effect = 0
    if seq_type:
        if seq_type.lower() == 'rbp':
            type_effect = 0.5  # Assume RBPs have higher baseline expression
        elif seq_type.lower() == 'utr':
            type_effect = -0.3  # UTRs might repress expression
        elif seq_type.lower() == 'mirna':
            type_effect = -0.5  # miRNAs typically repress expression
    
    # Combine all effects
    expression = base_expression + motif_effect + kozak_effect + mfe_effect + type_effect
    
    # Add random noise (normal distribution)
    noise = np.random.normal(0, 0.5)
    expression += noise
    
    # Scale to a reasonable range (0-10) and ensure non-negative
    expression = max(0, min(10, expression))
    
    return expression

def main():
    # Load RNA elements data
    print(f"Loading RNA elements from {args.input}")
    df = pd.read_csv(args.input)
    
    # Filter by element type if specified
    if args.element_type:
        df = df[df['type'] == args.element_type]
        print(f"Filtered to {len(df)} elements of type {args.element_type}")
    
    # Check for CDS sequence, use it if available, otherwise use the full sequence
    sequence_col = 'cds_sequence' if 'cds_sequence' in df.columns else 'sequence'
    
    print(f"Using '{sequence_col}' column for sequence data")
    print(f"Processing {len(df)} RNA elements")
    
    # Generate expression levels for each sequence
    expressions = []
    for idx, row in df.iterrows():
        seq = row[sequence_col]
        seq_type = row.get('type')
        
        # Generate expression level
        expr = generate_expression_level(seq, seq_type)
        expressions.append(expr)
        
        # Show progress every 100 entries
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} elements")
    
    # Add expression column to dataframe
    df['expression'] = expressions
    
    # Add a few additional simulated columns for more features
    df['expression_variance'] = df['expression'] * np.random.uniform(0.1, 0.3, size=len(df))
    df['translation_efficiency'] = df['expression'] * np.random.uniform(0.5, 1.5, size=len(df))
    
    # Filter out rows with NaN expression
    df_filtered = df.dropna(subset=['expression'])
    print(f"Final dataset contains {len(df_filtered)} elements with expression values")
    
    # Save to output CSV
    output_path = args.output
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved expression dataset to {output_path}")
    
    # Print statistics
    print("\nExpression statistics:")
    print(f"Mean: {df_filtered['expression'].mean():.2f}")
    print(f"Median: {df_filtered['expression'].median():.2f}")
    print(f"Min: {df_filtered['expression'].min():.2f}")
    print(f"Max: {df_filtered['expression'].max():.2f}")
    print(f"Std Dev: {df_filtered['expression'].std():.2f}")

if __name__ == "__main__":
    main()