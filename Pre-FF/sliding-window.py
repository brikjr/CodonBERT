from pathlib import Path
from Bio import SeqIO
import re

def get_next_file_number(directory_path):
    # List all files in the directory with the matching pattern
    files = list(Path(directory_path).glob('sliding-window*.fasta'))
    max_number = 0
    for file in files:
        # Extract the number from the filename using a regular expression
        match = re.search(r'sliding-window(\d+).fasta', file.name)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    return max_number + 1

def generate_windows(input_fasta_path, output_directory_path, window_size=300):
    # Check if input is from an S3 bucket or local filesystem
    if input_fasta_path.startswith('s3'):
        input_seq_list = [sequence for sequence in SeqIO.parse(S3Path(input_fasta_path).open(),'fasta')]
    else:
        input_seq_list = [sequence for sequence in SeqIO.parse(Path(input_fasta_path).open(),'fasta')]

    # Loop over each sequence in the fasta file
    for sequence_rec in input_seq_list:
        window_counter = 0
        full_seq = str(sequence_rec.seq)
        fasta_text = ""
        
        # Generate all windows for the sequence
        for start_pos in range(0, len(full_seq) - window_size + 1, 1):
            window = full_seq[start_pos:start_pos + window_size]
            fasta_text += f">{sequence_rec.id}_window{window_counter}\n{window}\n"
            window_counter += 1
        
        # Save all windows of the sequence to one file
        output_file_path = f'{output_directory_path}/{sequence_rec.id}_windows.fasta'
        with open(output_file_path, 'w') as output_file:
            output_file.write(fasta_text)

# Example usage
output_directory_path = 'sliding_window/'
generate_windows("car_t_cds.fasta", output_directory_path)
