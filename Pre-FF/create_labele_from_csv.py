import pandas as pd
import numpy as np
import os
import re

# Paths
csv_path = "/Users/brik/Projects/ML/CodonBERT/data/car_t_mrna_dataset.csv"
processed_dir = "/Users/brik/Projects/ML/CodonBERT/output/processed"
output_dir = "/Users/brik/Projects/ML/CodonBERT/output/processed"

# Load the processed gene names (from the processing report)
with open(os.path.join(processed_dir, "processing_report.txt"), "r") as f:
    content = f.read()

# Extract gene names from the "Included files" section
included_section = content.split("Included files (in order of stacking):")[1]
lines = included_section.strip().split("\n")
processed_genes = []

for line in lines:
    if line.strip():
        # Extract gene name from format like "0: sliding-window00001.npy -> 4-1BB_windows"
        parts = line.split("->")
        if len(parts) > 1:
            gene_name = parts[1].strip().replace("_windows", "")
            processed_genes.append(gene_name)

print(f"Found {len(processed_genes)} processed genes")

# Load the CSV data
df = pd.read_csv(csv_path)
print(f"Loaded CSV with {len(df)} entries")

# Let's examine which columns we have in the CSV
print("\nCSV columns:")
print(df.columns.tolist())

# Try to find which column contains gene names that match our processed genes
matching_columns = []
for column in df.columns:
    if column in ['gene_symbol', 'ires', 'name']:  # Common columns for gene names
        values = df[column].dropna().astype(str).tolist()
        matches = [gene for gene in processed_genes if gene in values]
        if matches:
            matching_columns.append((column, len(matches)))
            print(f"Found {len(matches)} matching genes in column '{column}'")

# Create mappings for both protein family and functional category
gene_to_family = {}
gene_to_category = {}

if matching_columns:
    # Use the column with the most matches
    best_col = max(matching_columns, key=lambda x: x[1])[0]
    print(f"Using column '{best_col}' for gene matching")
    
    # Create mappings from gene names to classifications
    for idx, row in df.iterrows():
        gene_name = str(row[best_col])
        
        # Map to protein family
        protein_family = row.get('protein_family', None)
        if gene_name and protein_family and not pd.isna(protein_family):
            gene_to_family[gene_name] = protein_family
            
        # Map to functional category
        functional_category = row.get('functional_category', None)
        if gene_name and functional_category and not pd.isna(functional_category):
            gene_to_category[gene_name] = functional_category
else:
    print("No good matching column found. Trying a more flexible matching approach.")
    
    # Create a list of all gene names from the processed list
    processed_gene_patterns = [re.compile(re.escape(gene), re.IGNORECASE) for gene in processed_genes]
    
    # Check each row for matches
    for idx, row in df.iterrows():
        # Process protein family
        protein_family = row.get('protein_family', None)
        functional_category = row.get('functional_category', None)
        
        if (not protein_family or pd.isna(protein_family)) and (not functional_category or pd.isna(functional_category)):
            continue
        
        # Convert all string columns to string and check for matches
        for col in df.columns:
            val = row.get(col, None)
            if not isinstance(val, str):
                continue
                
            # Check if any of our patterns match this value
            for i, pattern in enumerate(processed_gene_patterns):
                if pattern.search(val):
                    gene = processed_genes[i]
                    if protein_family and not pd.isna(protein_family):
                        gene_to_family[gene] = protein_family
                    if functional_category and not pd.isna(functional_category):
                        gene_to_category[gene] = functional_category
                    break

# Assign classifications to the processed genes
protein_families = []
functional_categories = []
family_matched_genes = []
category_matched_genes = []
family_unknown_genes = []
category_unknown_genes = []

for gene in processed_genes:
    # Process protein family
    family = gene_to_family.get(gene, "unknown")
    protein_families.append(family)
    if family != "unknown":
        family_matched_genes.append(gene)
    else:
        family_unknown_genes.append(gene)
        
    # Process functional category
    category = gene_to_category.get(gene, "unknown")
    functional_categories.append(category)
    if category != "unknown":
        category_matched_genes.append(gene)
    else:
        category_unknown_genes.append(gene)

print(f"\nProtein Family stats:")
print(f"Assigned protein families to {len(family_matched_genes)} genes out of {len(processed_genes)}")
print(f"Found {len(family_unknown_genes)} genes without protein family info")
if family_matched_genes:
    print(f"Matched genes examples: {family_matched_genes[:5]}")
if family_unknown_genes:
    print(f"Unmatched genes examples: {family_unknown_genes[:5]}")

print(f"\nFunctional Category stats:")
print(f"Assigned functional categories to {len(category_matched_genes)} genes out of {len(processed_genes)}")
print(f"Found {len(category_unknown_genes)} genes without functional category info")
if category_matched_genes:
    print(f"Matched genes examples: {category_matched_genes[:5]}")
if category_unknown_genes:
    print(f"Unmatched genes examples: {category_unknown_genes[:5]}")

# Save the protein families as a NumPy array
np.save(os.path.join(output_dir, "protein_families.npy"), np.array(protein_families))

# Save the functional categories as a NumPy array
np.save(os.path.join(output_dir, "functional_categories.npy"), np.array(functional_categories))

# Save protein families as a Python file
with open(os.path.join(output_dir, "protein_families.py"), "w") as f:
    f.write("# Protein family labels for processed genes\n\n")
    f.write("gene_names = " + repr(processed_genes) + "\n\n")
    f.write("protein_families = " + repr(protein_families) + "\n")
    
    # Add a count of protein family categories
    f.write("\n# Protein family distribution:\n")
    family_counts = {}
    for family in protein_families:
        family_counts[family] = family_counts.get(family, 0) + 1
    
    for family, count in sorted(family_counts.items()):
        f.write(f"# {family}: {count}\n")

# Save functional categories as a Python file
with open(os.path.join(output_dir, "functional_categories.py"), "w") as f:
    f.write("# Functional category labels for processed genes\n\n")
    f.write("gene_names = " + repr(processed_genes) + "\n\n")
    f.write("functional_categories = " + repr(functional_categories) + "\n")
    
    # Add a count of functional category distribution
    f.write("\n# Functional category distribution:\n")
    category_counts = {}
    for category in functional_categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    for category, count in sorted(category_counts.items()):
        f.write(f"# {category}: {count}\n")

print(f"Saved protein families to {os.path.join(output_dir, 'protein_families.npy')} and {os.path.join(output_dir, 'protein_families.py')}")
print(f"Saved functional categories to {os.path.join(output_dir, 'functional_categories.npy')} and {os.path.join(output_dir, 'functional_categories.py')}")