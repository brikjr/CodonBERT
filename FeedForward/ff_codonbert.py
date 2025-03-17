import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import yaml
import json
import time
from datetime import datetime
import argparse

# Check if MPS is available (Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# ===== 1. Configuration Management =====
def load_config(config_path):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config, save_path):
    """
    Save configuration to a JSON file for later reference.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    with open(save_path, 'w') as file:
        json.dump(config, file, indent=2)

# ===== 2. Model Definition =====
class CodonBERTPredictor(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dims=[512, 256], dropout_rate=0.3, activation="relu"):
        """
        Feedforward neural network for predicting mRNA properties from CodonBERT embeddings.
        
        Args:
            embedding_dim: Dimension of the CodonBERT embeddings (768 from config)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
            activation: Activation function to use (relu, leaky_relu, selu)
        """
        super().__init__()
        
        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        elif activation == "selu":
            act_fn = nn.SELU()
        else:
            act_fn = nn.ReLU()
            print(f"Warning: Unknown activation '{activation}', using ReLU instead")
        
        layers = []
        input_dim = embedding_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output layer (single value for regression)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ===== 3. Data Loading and Preparation =====
def load_and_prepare_data(embeddings_path, labels_path, test_size=0.2, batch_size=32, random_seed=42):
    """
    Load embeddings and labels, split into train/val sets, and create DataLoaders.
    
    Args:
        embeddings_path: Path to .npy file containing CodonBERT embeddings
        labels_path: Path to .npy file containing target labels
        test_size: Fraction of data to use for validation
        batch_size: Batch size for training
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, embedding_dim
    """
    # Load pre-computed embeddings and labels
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Process embeddings based on shape
    if len(embeddings.shape) == 3:
        print(f"Found sequence-level embeddings with shape {embeddings.shape}")
        # If embeddings have shape [n_samples, seq_length, embedding_dim]
        # We'll pool across sequence length to get a fixed-size representation
        embeddings = np.mean(embeddings, axis=1)
        print(f"Pooled to shape: {embeddings.shape}")
    
    embedding_dim = embeddings.shape[1]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_seed
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    # Create DataLoader objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, embedding_dim

# ===== 4. Early Stopping Implementation =====
class EarlyStopper:
    """
    Early stopping implementation to prevent overfitting.
    """
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
        
    def should_stop(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

# ===== 5. Training and Evaluation Functions =====
def create_optimizer(model, optimizer_name, initial_lr, weight_decay, momentum=0.9):
    """
    Create optimizer based on configuration.
    
    Args:
        model: The neural network model
        optimizer_name: Name of the optimizer (sgd, adam, adamw)
        initial_lr: Initial learning rate
        weight_decay: L2 regularization strength
        momentum: Momentum factor for SGD
        
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}', using SGD instead")
        return optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

def create_lr_scheduler(optimizer, scheduler_config):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: The optimizer
        scheduler_config: Configuration for the scheduler
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = scheduler_config.get("type", "plateau")
    
    if scheduler_type.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
            min_lr=scheduler_config.get("min_lr", 1e-6),
            verbose=True
        )
    elif scheduler_type.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("factor", 0.5)
        )
    elif scheduler_type.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("t_max", 50),
            eta_min=scheduler_config.get("min_lr", 1e-6)
        )
    else:
        print(f"Warning: Unknown scheduler '{scheduler_type}', using ReduceLROnPlateau instead")
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

def train_model(model, train_loader, val_loader, config, run_name):
    """
    Train the model with early stopping, learning rate scheduling, and TensorBoard logging.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        run_name: Name for the current training run
        
    Returns:
        Trained model and training history
    """
    # Create output directories
    checkpoint_dir = os.path.join(config["output"]["checkpoint_dir"], run_name)
    logs_dir = os.path.join(config["output"]["logs_dir"], run_name)
    plots_dir = os.path.join(config["output"]["plot_dir"], run_name)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save run configuration
    save_config(config, os.path.join(checkpoint_dir, "config.json"))
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(logs_dir)
    
    # Add model graph to TensorBoard
    sample_input = next(iter(train_loader))[0][0:1].to(device)
    writer.add_graph(model, sample_input)
    
    # Move model to device
    model = model.to(device)
    
    # Create criterion, optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = create_optimizer(
        model,
        config["training"]["optimizer"],
        config["training"]["initial_lr"],
        config["training"]["weight_decay"],
        config["training"].get("momentum", 0.9)
    )
    
    scheduler = create_lr_scheduler(optimizer, config["callbacks"]["lr_scheduler"])
    
    # Initialize early stopper
    early_stopper = EarlyStopper(
        patience=config["callbacks"]["early_stopping"]["patience"],
        min_delta=config["callbacks"]["early_stopping"]["min_delta"]
    )
    
    # Initialize training history
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_r2': [], 
        'learning_rate': []
    }
    
    # Initialize best validation loss for model checkpointing
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    # Training loop
    num_epochs = config["training"]["num_epochs"]
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Log batch loss to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Batch/Train Loss', loss.item(), global_step)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate R-squared for validation set
        val_r2 = r2_score(all_targets, all_preds)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['learning_rate'].append(current_lr)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val R²: {val_r2:.4f}, '
              f'LR: {current_lr:.6f}, '
              f'Time: {epoch_time:.2f}s')
        
        # Log to TensorBoard
        writer.add_scalar('Epoch/Train Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Validation Loss', val_loss, epoch)
        writer.add_scalar('Epoch/R-squared', val_r2, epoch)
        writer.add_scalar('Epoch/Learning Rate', current_lr, epoch)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
            }, checkpoint_path)
        
        # Check early stopping
        if early_stopper.should_stop(val_loss):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
        
        # Add prediction vs actual plot to TensorBoard
        if (epoch + 1) % 5 == 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(all_targets, all_preds, alpha=0.5)
            ax.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Epoch {epoch+1}: Prediction vs Actual (R² = {val_r2:.4f})')
            ax.grid(True)
            writer.add_figure('Prediction vs Actual', fig, epoch)
            plt.close(fig)
    
    # Close TensorBoard writer
    writer.close()
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save final training curves
    plot_training_history(history, os.path.join(plots_dir, 'training_history.png'))
    
    # Save history as JSON
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key in history:
            if isinstance(history[key], np.ndarray):
                history[key] = history[key].tolist()
        json.dump(history, f, indent=2)
    
    return model, history

# ===== 6. Plot Training History =====
def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation loss, R², and learning rate history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation R²
    plt.subplot(2, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('Validation R²')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rate'], label='Learning Rate', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    
    # Plot correlation between predicted and actual values if available
    if 'predictions' in history and 'targets' in history:
        plt.subplot(2, 2, 4)
        plt.scatter(history['targets'], history['predictions'], alpha=0.5)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction Correlation')
        
        # Add diagonal line for perfect predictions
        min_val = min(min(history['targets']), min(history['predictions']))
        max_val = max(max(history['targets']), max(history['predictions']))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ===== 7. Model Evaluation =====
def evaluate_model(model, data_loader):
    """
    Evaluate the model on the given data.
    
    Args:
        model: Trained model
        data_loader: DataLoader containing evaluation data
        
    Returns:
        Predictions, targets, MSE, and R²
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return all_preds, all_targets, mse, r2

# ===== 8. Prediction Function =====
def predict(model, embeddings):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        embeddings: Input embeddings (numpy array or torch tensor)
        
    Returns:
        Predictions as numpy array
    """
    model.eval()
    model = model.to(device)
    
    # Convert to tensor if numpy array
    if isinstance(embeddings, np.ndarray):
        if len(embeddings.shape) == 3:
            # Pool if sequence-level embeddings
            embeddings = np.mean(embeddings, axis=1)
        embeddings = torch.FloatTensor(embeddings).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(embeddings).cpu().numpy()
    
    return predictions

# ===== 9. Main Function =====
def main():
    """
    Main function to run the training pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train mRNA prediction model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create a unique run name using timestamp
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting run: {run_name}")
    
    # Create output directories
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["logs_dir"], exist_ok=True)
    os.makedirs(config["output"]["plot_dir"], exist_ok=True)
    
    # Load and prepare data
    train_loader, val_loader, embedding_dim = load_and_prepare_data(
        config["data"]["embeddings_path"],
        config["data"]["labels_path"],
        test_size=config["data"]["test_size"],
        batch_size=config["training"]["batch_size"],
        random_seed=config["data"]["random_seed"]
    )
    
    # Update embedding_dim in config if needed
    if config["model"]["embedding_dim"] != embedding_dim:
        print(f"Warning: Updating embedding_dim in config from {config['model']['embedding_dim']} to {embedding_dim}")
        config["model"]["embedding_dim"] = embedding_dim
    
    # Create model
    model = CodonBERTPredictor(
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        dropout_rate=config["model"]["dropout_rate"],
        activation=config["model"].get("activation", "relu")
    )
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    trained_model, history = train_model(
        model, train_loader, val_loader, config, run_name
    )
    
    # Evaluate final model on validation set
    val_preds, val_targets, val_mse, val_r2 = evaluate_model(trained_model, val_loader)
    print(f"Final Validation Results - MSE: {val_mse:.6f}, R²: {val_r2:.6f}")
    
    # Save final evaluation metrics
    eval_results = {
        "val_mse": val_mse,
        "val_r2": val_r2,
        "run_name": run_name
    }
    
    with open(os.path.join(config["output"]["checkpoint_dir"], run_name, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Training complete! Run name: {run_name}")
    print(f"TensorBoard logs saved to: {os.path.join(config['output']['logs_dir'], run_name)}")
    print("To view TensorBoard logs, run:")
    print(f"tensorboard --logdir={config['output']['logs_dir']}")

if __name__ == "__main__":
    main()