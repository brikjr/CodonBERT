#!/usr/bin/env python3
"""
Analyze the performance of the fine-tuned CodonBERT model for mRNA expression prediction.
This script evaluates the model on a test set, visualizes predictions vs. actual values,
and analyzes feature importance through attention patterns.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import BertForSequenceClassification, AutoModel
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../benchmarks'))

# Import tokenizer and dataset from fine-tuning script
try:
    from utils.tokenizer import get_tokenizer
    from fine_tune_rna_elements import process_sequence, RNAElementsDataset, sliding_windows
except ImportError:
    print("Could not import from fine_tune_rna_elements.py, make sure it exists in the same directory")
    sys.exit(1)

# Parse arguments
parser = argparse.ArgumentParser(description='Analyze fine-tuned CodonBERT model performance')
parser.add_argument('--model_dir', type=str, default='output/fine_tuned/final_model', 
                   help='Directory containing the fine-tuned model')
parser.add_argument('--test_data', type=str, default='rna_elements_expression.csv', 
                   help='Path to test data CSV with expression values')
parser.add_argument('--output_dir', type=str, default='output/analysis', 
                   help='Directory to save analysis results')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
parser.add_argument('--window_size', type=int, default=300, help='Size of sliding window for sequences')
parser.add_argument('--stride', type=int, default=100, help='Stride for sliding window')
parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
parser.add_argument('--element_type', type=str, default=None, 
                   help='Specific RNA element type to analyze (RBP, UTR, miRNA)')
parser.add_argument('--target_metric', type=str, default='expression', 
                   help='Target metric for prediction')
parser.add_argument('--num_samples', type=int, default=100, 
                   help='Number of samples to use for detailed attention analysis')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def save_prediction_plot(y_true, y_pred, output_path):
    """Create a scatter plot of predicted vs. actual expression values."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # Add metrics to plot
    plt.title(f'Predicted vs. Actual Expression Values\nRMSE: {rmse:.3f}, R²: {r2:.3f}, Pearson: {pearson_corr:.3f}')
    plt.xlabel('Actual Expression')
    plt.ylabel('Predicted Expression')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add confidence ellipse
    sns.kdeplot(x=y_true, y=y_pred, levels=5, color='blue', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved prediction plot to {output_path}")

def save_error_distribution(y_true, y_pred, output_path):
    """Create a histogram of prediction errors."""
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved error distribution plot to {output_path}")

def extract_attention_weights(model, dataset, tokenizer, num_samples=10):
    """Extract attention weights from model for analysis."""
    # Switch model to base BERT model to access attention
    if not hasattr(model, 'encoder'):
        print("Model doesn't have an accessible encoder with attention. Using alternative approach...")
        return None
    
    device = next(model.parameters()).device
    attention_weights = []
    sample_sequences = []
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in sample_indices:
            sample = dataset[idx]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            
            # Forward pass with output_attentions=True
            outputs = model(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           output_attentions=True)
            
            # Get the attention weights
            if outputs.attentions:
                # Average over all layers and heads
                att_weights = torch.stack(outputs.attentions).mean(dim=[0, 1])  # Average across layers and heads
                attention_weights.append(att_weights.cpu().numpy())
                
                # Get the sequence for this sample
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                sequence = " ".join(tokens)
                sample_sequences.append(sequence)
    
    return attention_weights, sample_sequences

def analyze_attention_patterns(attention_weights, sequences, output_dir):
    """Analyze attention patterns and visualize them."""
    if attention_weights is None or not attention_weights:
        print("No attention weights available for analysis.")
        return
    
    os.makedirs(os.path.join(output_dir, "attention"), exist_ok=True)
    
    for i, (att_matrix, seq) in enumerate(zip(attention_weights, sequences)):
        plt.figure(figsize=(12, 10))
        sns.heatmap(att_matrix[0], cmap="viridis")
        plt.title(f"Attention Pattern for Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"attention/sample_{i+1}_attention.png"))
        plt.close()
        
        # Save sequence for reference
        with open(os.path.join(output_dir, f"attention/sample_{i+1}_sequence.txt"), "w") as f:
            f.write(seq)
    
    # Calculate and visualize average attention
    if len(attention_weights) > 0:
        avg_attention = np.mean([att[0] for att in attention_weights], axis=0)
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attention, cmap="viridis")
        plt.title("Average Attention Pattern")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attention/average_attention.png"))
        plt.close()

def analyze_by_element_type(df, y_true, y_pred, output_dir):
    """Analyze model performance by RNA element type."""
    if 'type' not in df.columns:
        print("No 'type' column found in the data, skipping element type analysis.")
        return
    
    # Create a DataFrame with predictions
    results_df = pd.DataFrame({
        'true_value': y_true,
        'predicted_value': y_pred,
        'absolute_error': np.abs(y_true - y_pred),
        'type': df['type'].values
    })
    
    # Save results to CSV
    results_df.to_csv(os.path.join(output_dir, "prediction_results_by_type.csv"), index=False)
    
    # Group by type and calculate metrics
    type_metrics = []
    for element_type, group in results_df.groupby('type'):
        metrics = {
            'type': element_type,
            'count': len(group),
            'rmse': np.sqrt(mean_squared_error(group['true_value'], group['predicted_value'])),
            'r2': r2_score(group['true_value'], group['predicted_value']),
            'mean_error': group['absolute_error'].mean(),
            'median_error': group['absolute_error'].median()
        }
        type_metrics.append(metrics)
    
    # Convert to DataFrame and save
    type_metrics_df = pd.DataFrame(type_metrics)
    type_metrics_df.to_csv(os.path.join(output_dir, "metrics_by_type.csv"), index=False)
    
    # Create bar plot of RMSE by type
    plt.figure(figsize=(12, 6))
    sns.barplot(x='type', y='rmse', data=type_metrics_df)
    plt.title('RMSE by RNA Element Type')
    plt.xlabel('Element Type')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rmse_by_type.png"))
    plt.close()
    
    # Create box plot of errors by type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='type', y='absolute_error', data=results_df)
    plt.title('Prediction Error by RNA Element Type')
    plt.xlabel('Element Type')
    plt.ylabel('Absolute Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_type.png"))
    plt.close()

def main():
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    
    # Load model
    print(f"Loading fine-tuned model from {args.model_dir}")
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_df = pd.read_csv(args.test_data)
    
    # Filter by element type if specified
    if args.element_type:
        test_df = test_df[test_df['type'] == args.element_type]
        print(f"Filtered to {len(test_df)} elements of type {args.element_type}")
    
    # Check if target metric exists
    if args.target_metric not in test_df.columns:
        print(f"Target metric '{args.target_metric}' not found in data. Available columns: {test_df.columns.tolist()}")
        sys.exit(1)
    
    # Check for CDS sequence, use it if available, otherwise use the full sequence
    sequence_col = 'cds_sequence' if 'cds_sequence' in test_df.columns else 'sequence'
    
    # Extract sequences and labels
    sequences = []
    labels = []
    
    for _, row in test_df.iterrows():
        if pd.isna(row[sequence_col]) or row[sequence_col] == '':
            continue
            
        sequences.append(row[sequence_col])
        labels.append(row[args.target_metric])
    
    print(f"Prepared {len(sequences)} test sequences with {args.target_metric} values")
    
    # Create dataset
    test_dataset = RNAElementsDataset(
        sequences, labels, tokenizer, args.max_length, args.window_size, args.stride
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Run inference
    print("Running inference on test data...")
    all_predictions = []
    all_labels = []
    window_predictions = []  # Store predictions for each window
    window_indices = []      # Store which sequence each window belongs to
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.cpu().numpy().flatten()
            
            window_predictions.extend(predictions)
            window_indices.extend(test_dataset.sequence_indices[len(all_predictions):len(all_predictions) + len(predictions)])
            all_labels.extend(labels)
    
    # Aggregate window predictions by sequence
    sequence_predictions = []
    for i in range(len(sequences)):
        # Find all windows for this sequence
        seq_windows = [window_predictions[j] for j in range(len(window_predictions)) if window_indices[j] == i]
        if seq_windows:
            # Average predictions for all windows of this sequence
            sequence_predictions.append(np.mean(seq_windows))
    
    # Extract true labels for sequences (not windows)
    sequence_labels = labels
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(sequence_labels, sequence_predictions))
    r2 = r2_score(sequence_labels, sequence_predictions)
    pearson_corr, _ = pearsonr(sequence_labels, sequence_predictions)
    spearman_corr, _ = spearmanr(sequence_labels, sequence_predictions)
    
    # Print metrics
    print(f"\nModel Performance on Test Set:")
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
    
    # Create visualizations
    save_prediction_plot(
        sequence_labels, sequence_predictions, 
        os.path.join(args.output_dir, "predictions_vs_actual.png")
    )
    
    save_error_distribution(
        sequence_labels, sequence_predictions,
        os.path.join(args.output_dir, "error_distribution.png")
    )
    
    # Analyze by RNA element type
    analyze_by_element_type(test_df.iloc[[i for i in range(len(test_df)) if i < len(sequence_labels)]], 
                           sequence_labels, sequence_predictions, args.output_dir)
    
    # Extract and analyze attention weights
    print("\nExtracting attention patterns for analysis...")
    attention_weights, sample_sequences = extract_attention_weights(
        model, test_dataset, tokenizer, args.num_samples
    )
    
    analyze_attention_patterns(attention_weights, sample_sequences, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 