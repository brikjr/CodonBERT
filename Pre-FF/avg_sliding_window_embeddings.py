import numpy as np
import os
import glob
import pandas as pd
from collections import defaultdict

# Define paths
embeddings_dir = "/Users/brik/Projects/ML/CodonBERT/output/ires_sliding_window_npy"
csv_path = "/Users/brik/Projects/ML/CodonBERT/data/car_t_mrna_dataset.csv"
mapping_path = "/Users/brik/Projects/ML/CodonBERT/output/ires_sliding_window_npy/ires_file_mapping.csv"
output_dir = "/Users/brik/Projects/ML/CodonBERT/output/processed"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the existing file mapping
mapping_df = pd.read_csv(mapping_path)
file_mapping = dict(zip(mapping_df['numbered_file'], mapping_df['original_name']))
print(f"Loaded mapping for {len(file_mapping)} files")

# Load the CSV file with gene metadata
df = pd.read_csv(csv_path)
print(f"Loaded CSV with {len(df)} entries")

# Create a mapping from gene names to metadata
gene_to_metadata = {}
for idx, row in df.iterrows():
    gene_symbol = row.get('gene_symbol', None)
    ires = row.get('ires', None)
    protein_family = row.get('protein_family', None)
    genus = row.get('genus', None)
    species = row.get('species', None)
    
    # Use the gene name as a key (matching with your filenames)
    if ires:
        gene_to_metadata[ires] = {
            'gene_symbol': gene_symbol,
            'protein_family': protein_family,
            'genus': genus,
            'species': species
        }

print(f"Created metadata mapping for {len(gene_to_metadata)} genes")

# Find all npy files
npy_files = glob.glob(os.path.join(embeddings_dir, "*.npy"))
print(f"Found {len(npy_files)} npy files")

# Process each npy file and create the averaged embeddings
all_embeddings = []
all_labels = []
gene_names = []
expected_shape = (100, 768)  # Based on your output
skipped_files = []
included_files = []

for npy_file in sorted(npy_files):
    # Get the numbered file name
    numbered_name = os.path.basename(npy_file)
    
    # Look up the original gene name
    original_name = file_mapping.get(numbered_name)
    if not original_name:
        print(f"Warning: Could not find original name for {numbered_name}")
        skipped_files.append((numbered_name, "No mapping found"))
        continue
    
    print(f"Processing {numbered_name} -> {original_name}")
    
    # Load the embedding
    embedding = np.load(npy_file, allow_pickle=True)
    
    # Convert to numpy array if it's a list
    if isinstance(embedding, list):
        print(f"  Converting list to numpy array")
        embedding = np.array(embedding)
    
    # Print the shape to debug
    print(f"  Shape of embedding: {embedding.shape}")
    
    # Calculate the average embedding
    if len(embedding.shape) > 1:
        # Average across the first dimension
        avg_embedding = np.mean(embedding, axis=0)
    else:
        # It's already a 1D array or empty
        if embedding.shape == (0,):
            print(f"  WARNING: Empty embedding array for {numbered_name}, skipping")
            skipped_files.append((numbered_name, original_name, "Empty array"))
            continue
        avg_embedding = embedding
    
    # Verify the embedding has the expected shape
    if avg_embedding.shape != expected_shape:
        print(f"  WARNING: Unexpected shape {avg_embedding.shape} for {numbered_name}, skipping")
        skipped_files.append((numbered_name, original_name, f"Unexpected shape: {avg_embedding.shape}"))
        continue
    
    print(f"  Shape of averaged embedding: {avg_embedding.shape}")
    
    # Add to our lists
    all_embeddings.append(avg_embedding)
    gene_names.append(original_name)
    included_files.append((numbered_name, original_name))
    
    # Get metadata for this gene
    metadata = gene_to_metadata.get(original_name, {})
    all_labels.append(metadata)

# Convert to numpy arrays
all_embeddings = np.array(all_embeddings)
print(f"Final shape of all embeddings: {all_embeddings.shape}")

# Report on files that were skipped
print("\nSkipped files:")
for file_info in skipped_files:
    print(f"  {file_info}")

# Report on files that were included (in order)
print("\nIncluded files (in order of stacking):")
for i, (file_name, gene_name) in enumerate(included_files):
    print(f"  {i}: {file_name} -> {gene_name}")

# Save the processed data
np.save(os.path.join(output_dir, "ires_embeddings.npy"), all_embeddings)

# Save the labels as a Python file
with open(os.path.join(output_dir, "labels.py"), "w") as f:
    f.write("# Generated gene labels\n\n")
    f.write("gene_names = " + repr(gene_names) + "\n\n")
    f.write("gene_metadata = " + repr(all_labels) + "\n")

# Also save as CSV for easier inspection
labels_df = pd.DataFrame({
    'gene_name': gene_names,
    'gene_symbol': [label.get('gene_symbol', '') for label in all_labels],
    'protein_family': [label.get('protein_family', '') for label in all_labels],
    'genus': [label.get('genus', '') for label in all_labels],
    'species': [label.get('species', '') for label in all_labels]
})
labels_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

# Save the skipped/included info for reference
with open(os.path.join(output_dir, "processing_report.txt"), "w") as f:
    f.write("Skipped files:\n")
    for file_info in skipped_files:
        f.write(f"  {file_info}\n")
    
    f.write("\nIncluded files (in order of stacking):\n")
    for i, (file_name, gene_name) in enumerate(included_files):
        f.write(f"  {i}: {file_name} -> {gene_name}\n")

print(f"Processed {len(all_embeddings)} gene embeddings")
print(f"Saved embeddings to {os.path.join(output_dir, 'ires_embeddings.npy')}")
print(f"Saved labels to {os.path.join(output_dir, 'ires_labels.py')} and {os.path.join(output_dir, 'ires_labels.csv')}")
print(f"Saved processing report to {os.path.join(output_dir, 'ires_processing_report.txt')}")