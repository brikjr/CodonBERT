#!/usr/bin/env python3
"""
Fine-tune CodonBERT for mRNA expression prediction using RNA binding proteins, UTRs, and miRNAs.
Uses a sliding window approach to process sequence data.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../benchmarks'))

# Import tokenizer from benchmarks/utils
try:
    from utils.tokenizer import get_tokenizer
except ImportError:
    # Fallback to direct import
    from benchmarks.utils.tokenizer import get_tokenizer

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune CodonBERT for mRNA expression prediction')
parser.add_argument('--data', type=str, default='rna_elements.csv', help='Path to RNA elements CSV file')
parser.add_argument('--model', type=str, default='../model', help='Path to CodonBERT model directory')
parser.add_argument('--output_dir', type=str, default='output/fine_tuned', help='Output directory for fine-tuned model')
parser.add_argument('--window_size', type=int, default=300, help='Size of sliding window for sequence processing')
parser.add_argument('--stride', type=int, default=100, help='Stride for sliding window')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
parser.add_argument('--use_lora', action='store_true', help='Use LoRA for parameter-efficient fine-tuning')
parser.add_argument('--lora_rank', type=int, default=32, help='LoRA rank parameter')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha parameter')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
parser.add_argument('--element_type', type=str, default=None, help='Specific RNA element type to focus on (RBP, UTR, miRNA)')
parser.add_argument('--target_metric', type=str, default='expression', help='Target metric for prediction')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def sliding_windows(sequence, window_size=300, stride=100):
    """Generate sliding windows from a sequence."""
    windows = []
    seq_len = len(sequence)
    
    if seq_len <= window_size:
        # If sequence is shorter than window, return the whole sequence
        return [sequence]
    
    for i in range(0, seq_len - window_size + 1, stride):
        windows.append(sequence[i:i + window_size])
    
    return windows

def process_sequence(seq, tokenizer, max_length):
    """Process sequence for input to model using codon tokenization."""
    # Convert to uppercase and replace T with U for RNA
    seq = seq.upper().replace("T", "U")
    
    # Split into codons (3 nucleotides)
    codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
    
    # Join codons with spaces for tokenizer
    seq_str = " ".join(codons)
    
    # Check tokenizer type and tokenize sequence appropriately
    if hasattr(tokenizer, 'encode') and callable(tokenizer.encode):
        # This is a tokenizers.Tokenizer object
        encoding = tokenizer.encode(seq_str)
        input_ids = encoding.ids
        attention_mask = [1] * len(input_ids)
        
        # Pad or truncate to max_length
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        elif len(input_ids) < max_length:
            padding = [0] * (max_length - len(input_ids))
            input_ids = input_ids + padding
            attention_mask = attention_mask + padding
        
        encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    else:
        # Try using it as a transformers tokenizer
        encoded = tokenizer(seq_str, padding="max_length", truncation=True, max_length=max_length)
    
    return encoded

class RNAElementsDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=1024, window_size=300, stride=100):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        
        # Process all sequences into windows
        self.all_windows = []
        self.all_window_labels = []
        self.sequence_indices = []  # Track which sequence each window belongs to
        
        for i, seq in enumerate(sequences):
            windows = sliding_windows(seq, window_size, stride)
            for window in windows:
                self.all_windows.append(window)
                self.all_window_labels.append(labels[i])
                self.sequence_indices.append(i)
    
    def __len__(self):
        return len(self.all_windows)
    
    def __getitem__(self, idx):
        window = self.all_windows[idx]
        label = self.all_window_labels[idx]
        
        # Process window
        encoded = process_sequence(window, self.tokenizer, self.max_length)
        
        # Return with label
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(eval_pred):
    """Compute metrics for regression task."""
    predictions, labels = eval_pred
    predictions = predictions.reshape(-1)
    
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions)
    pearson_corr, _ = pearsonr(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }

def prepare_datasets(data_path, tokenizer, max_length, window_size, stride, element_type=None, target_metric='expression'):
    """Prepare datasets from RNA elements CSV."""
    # Load data
    df = pd.read_csv(data_path)
    
    # Filter by element type if specified
    if element_type:
        df = df[df['type'] == element_type]
    
    # Check if we have enough data
    if len(df) < 10:
        raise ValueError(f"Not enough data after filtering: {len(df)} samples")
    
    # Extract sequences and labels
    sequences = []
    labels = []
    
    # Check for CDS sequence, use it if available, otherwise use the full sequence
    sequence_col = 'cds_sequence' if 'cds_sequence' in df.columns else 'sequence'
    
    for _, row in df.iterrows():
        if pd.isna(row[sequence_col]) or row[sequence_col] == '':
            continue
            
        # Define expression or other target metric if available
        # For now, we'll use a simulated expression value based on sequence length
        # In a real scenario, you would have actual expression measurements
        if target_metric in df.columns:
            label = row[target_metric]
        else:
            # Simulate an expression metric if not available
            # (This is just for demonstration - real data would have actual measurements)
            label = len(row[sequence_col]) / 1000.0  # Normalized by 1000
            
        sequences.append(row[sequence_col])
        labels.append(label)
    
    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.3, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed
    )
    
    # Create datasets
    train_dataset = RNAElementsDataset(X_train, y_train, tokenizer, max_length, window_size, stride)
    val_dataset = RNAElementsDataset(X_val, y_val, tokenizer, max_length, window_size, stride)
    test_dataset = RNAElementsDataset(X_test, y_test, tokenizer, max_length, window_size, stride)
    
    print(f"Train set: {len(train_dataset)} windows from {len(X_train)} sequences")
    print(f"Validation set: {len(val_dataset)} windows from {len(X_val)} sequences")
    print(f"Test set: {len(test_dataset)} windows from {len(X_test)} sequences")
    
    return train_dataset, val_dataset, test_dataset

def main():
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    # Load base model
    model = BertForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,  # Regression task
        problem_type="regression"
    )
    
    # Apply LoRA for parameter-efficient fine-tuning if requested
    if args.use_lora:
        print("Using LoRA for parameter-efficient fine-tuning")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "key", "value"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        args.data, 
        tokenizer, 
        args.max_length, 
        args.window_size, 
        args.stride,
        args.element_type,
        args.target_metric
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        report_to="none",
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test results: {test_results}")
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
    
    # Save model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print(f"Fine-tuning complete. Model saved to {os.path.join(args.output_dir, 'final_model')}")

if __name__ == "__main__":
    main() 