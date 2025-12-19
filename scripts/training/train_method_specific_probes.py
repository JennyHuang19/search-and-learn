#!/usr/bin/env python3
"""
Script to train separate MLP probes for each method type.

This script:
1. Loads the processed data with method information
2. Creates separate datasets for each method type
3. Trains individual MLP probes for each method
4. Evaluates and compares performance across methods
5. Saves results and visualizations

Methods supported:
- majority
- naive  
- weighted
- beam_search
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# =========================
# Method-Specific Dataset
# =========================
class MethodSpecificDataset(Dataset):
    def __init__(self, embedding_features, auxiliary_features, labels, method_info, target_method):
        """
        Create dataset for a specific method type.
        
        Args:
            embedding_features: Question embeddings
            auxiliary_features: Auxiliary features
            labels: Soft labels (sl values)
            method_info: List of method information dictionaries
            target_method: The specific method to filter for ('majority', 'naive', 'weighted', 'beam_search')
        """
        # Filter data for the specific method
        method_indices = []
        for i, info in enumerate(method_info):
            if info['method'] == target_method:
                method_indices.append(i)
        
        if len(method_indices) == 0:
            raise ValueError(f"No samples found for method: {target_method}")
        
        self.embedding_features = torch.tensor(embedding_features[method_indices], dtype=torch.float32)
        self.auxiliary_features = torch.tensor(auxiliary_features[method_indices], dtype=torch.float32)
        self.labels = torch.tensor(labels[method_indices], dtype=torch.float32).unsqueeze(1)
        self.method_info = [method_info[i] for i in method_indices]
        
        print(f"Created dataset for method '{target_method}' with {len(method_indices)} samples")

    def __len__(self):
        return len(self.embedding_features)

    def __getitem__(self, idx):
        return (self.embedding_features[idx], self.auxiliary_features[idx]), self.labels[idx]

# =========================
# Method-Specific MLP Model
# =========================
class MethodSpecificMLP(nn.Module):
    def __init__(self, embedding_dim, auxiliary_dim, hidden_dims=[32, 32], dropout_rate=0.2, method_name="unknown"):
        super(MethodSpecificMLP, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.auxiliary_dim = auxiliary_dim
        self.hidden_dims = hidden_dims
        self.method_name = method_name
        
        # First layer: embedding + auxiliary features
        self.layer1 = nn.Linear(embedding_dim + auxiliary_dim, hidden_dims[0])
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0])
        
        # Second layer: hidden + auxiliary features (repeated injection)
        self.layer2 = nn.Linear(hidden_dims[0] + auxiliary_dim, hidden_dims[1])
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[1], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, embedding_features, auxiliary_features):
        # First layer: concatenate embedding and auxiliary features
        x = torch.cat([embedding_features, auxiliary_features], dim=1)
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)
        
        # Second layer: concatenate hidden features with auxiliary features again
        x = torch.cat([x, auxiliary_features], dim=1)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

# =========================
# Training Functions
# =========================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for (emb_features, aux_features), labels in train_loader:
        emb_features = emb_features.to(device)
        aux_features = aux_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(emb_features, aux_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for (emb_features, aux_features), labels in val_loader:
            emb_features = emb_features.to(device)
            aux_features = aux_features.to(device)
            labels = labels.to(device)
            
            outputs = model(emb_features, aux_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(all_predictions).flatten()
    true_labels = np.array(all_labels).flatten()
    
    # For soft labels, use MSE
    mse = mean_squared_error(true_labels, predictions)
    
    # For binary classification (if needed), convert to binary
    binary_predictions = (predictions >= 0.5).astype(int)
    binary_labels = (true_labels >= 0.5).astype(int)
    accuracy = accuracy_score(binary_labels, binary_predictions)
    
    return total_loss / num_batches, mse, accuracy

# =========================
# Method-Specific Training
# =========================
def train_method_specific_probe(data_dir, output_dir, method_name, config=None):
    """
    Train a probe specifically for one method type.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training probe for method: {method_name}")
    print(f"Using device: {device}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    X_train_emb = np.load(os.path.join(data_dir, "X_train_embeddings.npy"))
    X_train_aux = np.load(os.path.join(data_dir, "X_train_auxiliary.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    
    X_test_emb = np.load(os.path.join(data_dir, "X_test_embeddings.npy"))
    X_test_aux = np.load(os.path.join(data_dir, "X_test_auxiliary.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Load method information
    try:
        with open(os.path.join(data_dir, "method_info_train.pkl"), 'rb') as f:
            method_info_train = pickle.load(f)
        with open(os.path.join(data_dir, "method_info_test.pkl"), 'rb') as f:
            method_info_test = pickle.load(f)
        print("Method information loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError("Method information not found. Please run extract_features_improved.py first.")
    
    # Load model info
    with open(os.path.join(data_dir, "model_info.json"), 'r') as f:
        model_info = json.load(f)
    
    # Create method-specific datasets
    try:
        train_dataset = MethodSpecificDataset(X_train_emb, X_train_aux, y_train, method_info_train, method_name)
        test_dataset = MethodSpecificDataset(X_test_emb, X_test_aux, y_test, method_info_test, method_name)
    except ValueError as e:
        print(f"Error creating dataset for method '{method_name}': {e}")
        return None
    
    # Training configuration
    if config is None:
        config = {
            'batch_size': 64,  # Smaller batch size for method-specific training
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 50,  # Fewer epochs for method-specific training
            'patience': 10,
            'hidden_dims': [200, 200],
            'dropout_rate': 0.2
        }
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = MethodSpecificMLP(
        embedding_dim=model_info['embedding_dim'],
        auxiliary_dim=model_info['auxiliary_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        method_name=method_name
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"Starting training for method '{method_name}'...")
    train_losses = []
    val_losses = []
    val_mses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val MSE':<12} {'Val Acc':<8} {'LR':<10}")
    print("-" * 70)
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_mse, val_accuracy = validate_epoch(model, test_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_accuracies.append(val_accuracy)
        
        # Print formatted output
        print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {val_mse:<12.6f} {val_accuracy:<8.4f} {optimizer.param_groups[0]['lr']:<10.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            method_output_dir = os.path.join(output_dir, method_name)
            os.makedirs(method_output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'method_name': method_name
            }, os.path.join(method_output_dir, 'best_model.pth'))
            
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("-" * 70)
    print(f"Training completed for method '{method_name}'. Best validation loss: {best_val_loss:.6f}")
    
    # Load best model for final evaluation
    method_output_dir = os.path.join(output_dir, method_name)
    checkpoint = torch.load(os.path.join(method_output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    final_val_loss, final_val_mse, final_val_accuracy = validate_epoch(model, test_loader, criterion, device)
    
    print(f"\nFinal Results for method '{method_name}':")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Final Val MSE: {final_val_mse:.6f}")
    print(f"  Final Val Accuracy: {final_val_accuracy:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mses': val_mses,
        'val_accuracies': val_accuracies,
        'best_epoch': checkpoint['epoch'],
        'config': config,
        'method_name': method_name,
        'final_metrics': {
            'best_val_loss': best_val_loss,
            'final_val_mse': final_val_mse,
            'final_val_accuracy': final_val_accuracy
        }
    }
    
    with open(os.path.join(method_output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate predictions
    test_predictions = generate_test_predictions(model, test_loader, device)
    np.save(os.path.join(method_output_dir, 'test_predictions.npy'), test_predictions)
    
    return {
        'method_name': method_name,
        'final_metrics': history['final_metrics'],
        'predictions': test_predictions,
        'history': history
    }

def generate_test_predictions(model, test_loader, device):
    """Generate predictions on the test set."""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for (emb_features, aux_features), _ in test_loader:
            emb_features = emb_features.to(device)
            aux_features = aux_features.to(device)
            
            outputs = model(emb_features, aux_features)
            predictions = outputs.squeeze().cpu().numpy()
            all_predictions.extend(predictions)
    
    return np.array(all_predictions)

# =========================
# Main Training Script
# =========================
def main(data_dir, output_dir, methods=None):
    """
    Train separate probes for each method type.
    
    Args:
        data_dir: Directory containing processed data
        output_dir: Directory to save results
        methods: List of methods to train. If None, trains all methods.
    """
    if methods is None:
        methods = ['majority', 'naive', 'weighted', 'beam_search']
    
    print(f"Training method-specific probes for: {methods}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train probes for each method
    results = {}
    for method in methods:
        try:
            result = train_method_specific_probe(data_dir, output_dir, method)
            if result is not None:
                results[method] = result
        except Exception as e:
            print(f"Error training probe for method '{method}': {e}")
            continue
    
    # Generate comparison report
    if results:
        generate_comparison_report(results, output_dir)
    
    print(f"\n{'='*60}")
    print("METHOD-SPECIFIC PROBE TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"Methods trained: {list(results.keys())}")

def generate_comparison_report(results, output_dir):
    """Generate a comparison report of all method-specific probes."""
    
    # Create comparison table
    comparison_data = []
    for method, result in results.items():
        metrics = result['final_metrics']
        comparison_data.append({
            'Method': method,
            'Best Val Loss': metrics['best_val_loss'],
            'Final Val MSE': metrics['final_val_mse'],
            'Final Val Accuracy': metrics['final_val_accuracy']
        })
    
    # Create DataFrame for easy comparison
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save comparison table
    comparison_file = os.path.join(output_dir, 'method_comparison.csv')
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\nMethod comparison saved to: {comparison_file}")
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("METHOD COMPARISON RESULTS")
    print(f"{'='*60}")
    print(df_comparison.to_string(index=False, float_format='%.6f'))
    
    # Create visualization
    create_comparison_visualization(results, output_dir)
    
    # Save detailed results
    detailed_results = {
        'comparison_table': df_comparison.to_dict('records'),
        'individual_results': results
    }
    
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)

def create_comparison_visualization(results, output_dir):
    """Create visualizations comparing method-specific probe performance."""
    
    # Prepare data for plotting
    methods = list(results.keys())
    val_losses = [results[m]['final_metrics']['best_val_loss'] for m in methods]
    val_mses = [results[m]['final_metrics']['final_val_mse'] for m in methods]
    val_accuracies = [results[m]['final_metrics']['final_val_accuracy'] for m in methods]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Method-Specific Probe Performance Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Validation Loss
    axes[0, 0].bar(methods, val_losses, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    axes[0, 0].set_title('Best Validation Loss (Lower is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Validation MSE
    axes[0, 1].bar(methods, val_mses, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    axes[0, 1].set_title('Final Validation MSE (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Validation Accuracy
    axes[1, 0].bar(methods, val_accuracies, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    axes[1, 0].set_title('Final Validation Accuracy (Higher is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Combined metrics heatmap
    metrics_data = np.array([val_losses, val_mses, val_accuracies])
    im = axes[1, 1].imshow(metrics_data, cmap='RdYlBu_r', aspect='auto')
    axes[1, 1].set_xticks(range(len(methods)))
    axes[1, 1].set_xticklabels(methods, rotation=45)
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(['Val Loss', 'Val MSE', 'Val Acc'])
    axes[1, 1].set_title('Metrics Heatmap', fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 1])
    
    # Add value annotations on bars
    for i, (ax, values, title) in enumerate([(axes[0, 0], val_losses, 'Loss'), 
                                           (axes[0, 1], val_mses, 'MSE'),
                                           (axes[1, 0], val_accuracies, 'Accuracy')]):
        for j, v in enumerate(values):
            ax.text(j, v + max(values) * 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Comparison visualization saved to: {os.path.join(output_dir, 'method_comparison.png')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train method-specific MLP probes")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save method-specific probe results")
    parser.add_argument("--methods", type=str, nargs='+', 
                       default=['majority', 'naive', 'weighted', 'beam_search'],
                       help="Methods to train probes for")
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir, args.methods)
