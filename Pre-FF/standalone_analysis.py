#!/usr/bin/env python3
"""
Standalone analysis script for fine-tuned CodonBERT model.
This script is intentionally self-contained to avoid import conflicts.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Define custom tokenizer function
def get_tokenizer():
    """Create tokenizer."""
    lst_ele = list("AUGC")
    lst_voc = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for a1 in lst_ele:
        for a2 in lst_ele:
            for a3 in lst_ele:
                lst_voc.extend([f"{a1}{a2}{a3}"])
    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
    tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = BertProcessing(
        ("[SEP]", dic_voc["[SEP]"]),
        ("[CLS]", dic_voc["[CLS]"]),
    )
    return tokenizer

# Function to tokenize sequence into kmers
def mytok(seq, kmer_len=3, s=1):
    """Tokenize a sequence into kmers."""
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, s):
        kmer_list.append(seq[j : j + kmer_len])
    return kmer_list

# Process sequence for the model
def process_sequence(sequence, tokenizer, max_length=512):
    """Process a sequence for model input."""
    # Tokenize sequence into kmers
    tokens = mytok(sequence, kmer_len=3, s=1)
    
    # Convert to space-separated string for the tokenizer
    tokens_str = " ".join(tokens)
    
    # Encode with the tokenizer
    encoding = tokenizer.encode(tokens_str)
    
    # Get IDs as a list
    input_ids = encoding.ids
    
    # Truncate if too long
    if len(input_ids) > max_length - 2:  # -2 for [CLS] and [SEP]
        input_ids = input_ids[:max_length - 2]
    
    # Add special tokens
    input_ids = [tokenizer.token_to_id("[CLS]")] + input_ids + [tokenizer.token_to_id("[SEP]")]
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)
    
    # Pad sequences
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([tokenizer.token_to_id("[PAD]")] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
    
    # Convert to tensors
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    
    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor
    }

# Define argument parser
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Standalone model analysis tool')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the fine-tuned model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data CSV file')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    parser.add_argument('--window_size', type=int, default=300, help='Size of sliding window')
    parser.add_argument('--stride', type=int, default=100, help='Stride for sliding window')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print received arguments for debugging
    print(f"Received arguments:")
    print(f"  model_dir: {args.model_dir}")
    print(f"  test_data: {args.test_data}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  window_size: {args.window_size}")
    print(f"  stride: {args.stride}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    # Define sliding window function
    def sliding_windows(sequence, window_size=300, stride=100):
        """Generate sliding windows from a sequence."""
        windows = []
        for i in range(0, len(sequence) - window_size + 1, stride):
            windows.append(sequence[i:i + window_size])
        if not windows and len(sequence) > 0:  # If sequence is shorter than window_size
            windows = [sequence]
        return windows
    
    # Load tokenizer and model
    try:
        print(f"Loading model from {args.model_dir}")
        
        # Create custom tokenizer
        print("Initializing custom RNA tokenizer...")
        tokenizer = get_tokenizer()
        print("Custom tokenizer initialized successfully")
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load test data
    try:
        print(f"Loading test data from {args.test_data}")
        test_df = pd.read_csv(args.test_data)
        print(f"Loaded {len(test_df)} samples from test data")
        
        # Check available columns for debugging
        print(f"Available columns in test data: {test_df.columns.tolist()}")
        
        # Find sequence column
        sequence_col = None
        for col in ['sequence', 'cds_sequence', 'rna_sequence', 'nucleotide_sequence']:
            if col in test_df.columns:
                sequence_col = col
                break
        
        if sequence_col is None:
            print("No sequence column found in the data!")
            print("Available columns:", test_df.columns.tolist())
            sys.exit(1)
        
        print(f"Using '{sequence_col}' as sequence column")
        
        # Find target column - default to 'expression'
        target_col = 'expression'
        if target_col not in test_df.columns:
            print(f"Target column '{target_col}' not found in data!")
            print("Available columns:", test_df.columns.tolist())
            sys.exit(1)
        
        # Extract sequences and labels
        sequences = []
        labels = []
        
        for _, row in test_df.iterrows():
            if pd.isna(row[sequence_col]) or row[sequence_col] == '':
                continue
                
            sequences.append(str(row[sequence_col]))
            labels.append(float(row[target_col]))
        
        print(f"Prepared {len(sequences)} valid sequences with expression values")
        
    except Exception as e:
        print(f"Error processing test data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process sequences using sliding windows
    print("Processing sequences...")
    all_windows = []
    sequence_indices = []
    
    for i, seq in enumerate(sequences):
        windows = sliding_windows(seq, window_size=args.window_size, stride=args.stride)
        all_windows.extend(windows)
        sequence_indices.extend([i] * len(windows))
    
    print(f"Created {len(all_windows)} windows from {len(sequences)} sequences")
    
    # Process windows for input to model
    processed_windows = []
    for window in all_windows:
        processed_windows.append(process_sequence(window, tokenizer, max_length=512))
    
    # Run inference in batches
    print("Running inference...")
    window_predictions = []
    
    batch_size = 16
    num_batches = (len(processed_windows) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(processed_windows))
            
            batch_inputs = processed_windows[start_idx:end_idx]
            
            # Prepare batch tensors
            input_ids = torch.stack([item['input_ids'] for item in batch_inputs]).to(device)
            attention_mask = torch.stack([item['attention_mask'] for item in batch_inputs]).to(device)
            
            # Run model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.cpu().numpy().flatten()
            
            window_predictions.extend(predictions)
            
            # Print progress for large datasets
            if (i+1) % 10 == 0 or i == num_batches - 1:
                print(f"  Processed {end_idx}/{len(processed_windows)} windows ({(end_idx/len(processed_windows))*100:.1f}%)")
    
    # Aggregate window predictions by sequence
    print("Aggregating predictions...")
    unique_indices = sorted(set(sequence_indices))
    sequence_predictions = []
    sequence_labels = labels
    
    for i in unique_indices:
        indices = [j for j, idx in enumerate(sequence_indices) if idx == i]
        if indices:
            # Average predictions for this sequence
            seq_pred = np.mean([window_predictions[j] for j in indices])
            sequence_predictions.append(seq_pred)
    
    # Calculate metrics
    print("Calculating metrics...")
    rmse = np.sqrt(mean_squared_error(sequence_labels, sequence_predictions))
    r2 = r2_score(sequence_labels, sequence_predictions)
    pearson_corr, _ = pearsonr(sequence_labels, sequence_predictions)
    spearman_corr, _ = spearmanr(sequence_labels, sequence_predictions)
    
    # Print metrics
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    
    # Save metrics to file
    metrics = {
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr),
        'num_samples': len(sequence_labels)
    }
    
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create a simple scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(sequence_labels, sequence_predictions, alpha=0.6)
    plt.plot([min(sequence_labels), max(sequence_labels)], 
             [min(sequence_labels), max(sequence_labels)], 'r--')
    plt.title(f'Predicted vs. Actual Values\nRMSE: {rmse:.3f}, R²: {r2:.3f}')
    plt.xlabel('Actual Expression Value')
    plt.ylabel('Predicted Expression Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "predictions_vs_actual.png"))
    plt.close()
    
    # Save predictions
    results_df = pd.DataFrame({
        'actual': sequence_labels,
        'predicted': sequence_predictions
    })
    results_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
    
    print(f"Analysis complete. Results saved to {args.output_dir}") 