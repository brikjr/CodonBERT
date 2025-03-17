#!/bin/bash
FASTA_DIR="/Users/brik/Projects/ML/CodonBERT/data/sliding_window"
CONFIG_TEMPLATE="config.yaml"
OUT_DIR="/Users/brik/Projects/ML/CodonBERT/output/sliding_window_npy"
DIGITS=5
MODEL_DIR="/Users/brik/Projects/ML/CodonBERT/model"
MAPPING_FILE="${OUT_DIR}/file_mapping.csv"

# Ensure OUT_DIR exists
mkdir -p "${OUT_DIR}"

# Create mapping file header
echo "numbered_file,original_name" > "${MAPPING_FILE}"

# Generate an array of fasta files sorted numerically
readarray -t fasta_files < <(find "${FASTA_DIR}" -name "*.fasta" | sort -V)

# Use a counter for consistent numbering
counter=1

for fasta_file in "${fasta_files[@]}"; do
    echo "Processing ${fasta_file}..."
    
    # Get original base name
    original_name=$(basename "${fasta_file}" .fasta)
    
    # Generate the output file path
    output_file="${OUT_DIR}/${original_name}.npy"
    
    # Run the Python script with the corrected model_dir
    python extract_embed.py \
        hydra.run.dir=. \
        embed.data_path="${fasta_file}" \
        embed.output_path="${output_file}" \
        embed.model_dir="${MODEL_DIR}"
    
    if [ -f "${output_file}" ]; then
        # Create a consistent numbered output name
        printf -v new_name "sliding-window%0${DIGITS}d.npy" $counter
        new_output_file="${OUT_DIR}/${new_name}"
        
        mv "${output_file}" "${new_output_file}"
        echo "Created ${new_output_file}"
        
        # Add to mapping file
        echo "${new_name},${original_name}" >> "${MAPPING_FILE}"
        
        # Increment the counter
        ((counter++))
    else
        echo "Failed to generate output for ${fasta_file}."
    fi
done

echo "Mapping saved to ${MAPPING_FILE}"