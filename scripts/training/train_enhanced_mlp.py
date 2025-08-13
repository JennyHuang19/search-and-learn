import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import os
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

# =========================
# Enhanced Dataset for Multi-Feature Input
# =========================
class EnhancedDataset(Dataset):
    def __init__(self, embedding_features, auxiliary_features, labels, method_info=None):
        self.embedding_features = torch.tensor(embedding_features, dtype=torch.float32)
        self.auxiliary_features = torch.tensor(auxiliary_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.method_info = method_info  # Store method information for analysis

    def __len__(self):
        return len(self.embedding_features)

    def __getitem__(self, idx):
        if self.method_info is not None:
            return (self.embedding_features[idx], self.auxiliary_features[idx]), self.labels[idx], self.method_info[idx]
        else:
            return (self.embedding_features[idx], self.auxiliary_features[idx]), self.labels[idx]

# Small helper to get method name from a batched method_info entry
# Handles both list-of-dicts (default) and dict-of-lists (PyTorch collate over dicts)
def _get_method_name(method_info_batched, index):
    # list of dicts: [ {'method': 'weighted', ...}, ... ]
    if isinstance(method_info_batched, list):
        entry = method_info_batched[index]
        if isinstance(entry, dict) and 'method' in entry:
            return entry['method']
        return None
    # dict of lists: { 'method': ['weighted', ...], 'method_naive': [...], ... }
    if isinstance(method_info_batched, dict):
        methods_list = method_info_batched.get('method')
        if isinstance(methods_list, (list, tuple)) and index < len(methods_list):
            return methods_list[index]
        return None
    return None

# =========================
# Enhanced MLP Model with Multi-Layer Feature Injection
# =========================
class EnhancedMLP(nn.Module):
    def __init__(self, embedding_dim, auxiliary_dim, hidden_dims=[32, 32], dropout_rate=0.2):
        super(EnhancedMLP, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.auxiliary_dim = auxiliary_dim
        self.hidden_dims = hidden_dims
        
        # First layer: embedding + auxiliary features
        self.layer1 = nn.Linear(embedding_dim + auxiliary_dim, hidden_dims[0])
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0]) # layer norm, group norm.
        
        # Second layer: hidden + auxiliary features
        self.layer2 = nn.Linear(hidden_dims[0] + auxiliary_dim, hidden_dims[1]) #
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
        
        # Second layer: concatenate hidden features with auxiliary features
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

    # Track method-specific training losses
    method_train_losses = {
        'majority': {'losses': [], 'count': 0},
        'naive': {'losses': [], 'count': 0},
        'weighted': {'losses': [], 'count': 0},
        'beam_search': {'losses': [], 'count': 0}
    }

    for batch_data in train_loader:
        if len(batch_data) == 3:  # Has method info
            (emb_features, aux_features), labels, method_info = batch_data
        else:  # No method info (backward compatibility)
            (emb_features, aux_features), labels = batch_data
            method_info = None

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

        # Track method-specific losses if method info is available
        if method_info is not None:
            batch_size = labels.shape[0]
            for i in range(batch_size):
                sample_loss = criterion(outputs[i:i+1], labels[i:i+1]).item()
                method_name = _get_method_name(method_info, i)
                if method_name in method_train_losses:
                    method_train_losses[method_name]['losses'].append(sample_loss)
                    method_train_losses[method_name]['count'] += 1

    # Calculate method-specific average losses and counts
    method_avg_losses = {}
    method_counts = {}
    for method, data in method_train_losses.items():
        method_counts[method] = data['count']
        if data['count'] > 0:
            method_avg_losses[method] = float(np.mean(data['losses']))
        else:
            method_avg_losses[method] = 0.0

    return total_loss / max(num_batches, 1), method_avg_losses, method_counts

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) == 3:  # Has method info
                (emb_features, aux_features), labels, _ = batch_data
            else:  # No method info (backward compatibility)
                (emb_features, aux_features), labels = batch_data

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
    
    return total_loss / max(num_batches, 1), mse, accuracy

def validate_epoch_by_method(model, val_loader, criterion, device):
    """
    Validate the model and return statistics broken down by method type.
    """
    model.eval()
    
    # Dictionary to store results by method
    method_results = {
        'majority': {'predictions': [], 'labels': [], 'losses': []},
        'naive': {'predictions': [], 'labels': [], 'losses': []},
        'weighted': {'predictions': [], 'labels': [], 'losses': []},
        'beam_search': {'predictions': [], 'labels': [], 'losses': []}
    }
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) != 3:  # No method info available
                return validate_epoch(model, val_loader, criterion, device), None
                
            (emb_features, aux_features), labels, method_info_batch = batch_data
            
            emb_features = emb_features.to(device)
            aux_features = aux_features.to(device)
            labels = labels.to(device)
            
            outputs = model(emb_features, aux_features)
            
            # Calculate loss for each sample
            batch_size = labels.shape[0]
            for i in range(batch_size):
                sample_loss = criterion(outputs[i:i+1], labels[i:i+1]).item()
                prediction = outputs[i].cpu().numpy().item()
                label = labels[i].cpu().numpy().item()
                method_name = _get_method_name(method_info_batch, i)
                
                if method_name in method_results:
                    method_results[method_name]['predictions'].append(prediction)
                    method_results[method_name]['labels'].append(label)
                    method_results[method_name]['losses'].append(sample_loss)
            
            total_loss += criterion(outputs, labels).item()
            num_batches += 1
    
    # Calculate overall metrics
    overall_loss = total_loss / max(num_batches, 1)
    
    # Calculate method-specific metrics
    method_stats = {}
    for method, data in method_results.items():
        if len(data['predictions']) > 0:
            predictions = np.array(data['predictions'])
            true_labels = np.array(data['labels'])
            losses = np.array(data['losses'])
            
            # MSE
            mse = mean_squared_error(true_labels, predictions)
            
            # Accuracy (binary)
            binary_predictions = (predictions >= 0.5).astype(int)
            binary_labels = (true_labels >= 0.5).astype(int)
            accuracy = accuracy_score(binary_labels, binary_predictions)
            
            # Average loss
            avg_loss = float(np.mean(losses))
            
            method_stats[method] = {
                'count': len(predictions),
                'loss': avg_loss,
                'mse': float(mse),
                'accuracy': float(accuracy),
                'mean_prediction': float(np.mean(predictions)),
                'mean_label': float(np.mean(true_labels))
            }
        else:
            method_stats[method] = {
                'count': 0,
                'loss': 0.0,
                'mse': 0.0,
                'accuracy': 0.0,
                'mean_prediction': 0.0,
                'mean_label': 0.0
            }
    
    # Calculate overall metrics for comparison
    all_predictions = []
    all_labels = []
    for data in method_results.values():
        all_predictions.extend(data['predictions'])
        all_labels.extend(data['labels'])
    
    if len(all_predictions) > 0:
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        overall_mse = mean_squared_error(all_labels, all_predictions)
        binary_predictions = (all_predictions >= 0.5).astype(int)
        binary_labels = (all_labels >= 0.5).astype(int)
        overall_accuracy = accuracy_score(binary_labels, binary_predictions)
    else:
        overall_mse = 0.0
        overall_accuracy = 0.0
    
    return (overall_loss, overall_mse, overall_accuracy), method_stats

# =========================
# Main Training Script
# =========================
def main(data_dir, output_dir, config=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    X_train_emb = np.load(os.path.join(data_dir, "X_train_embeddings.npy"))
    X_train_aux = np.load(os.path.join(data_dir, "X_train_auxiliary.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    
    X_test_emb = np.load(os.path.join(data_dir, "X_test_embeddings.npy"))
    X_test_aux = np.load(os.path.join(data_dir, "X_test_auxiliary.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Load method information if available
    method_info_train = None
    method_info_test = None
    try:
        with open(os.path.join(data_dir, "method_info_train.pkl"), 'rb') as f:
            method_info_train = pickle.load(f)
        with open(os.path.join(data_dir, "method_info_test.pkl"), 'rb') as f:
            method_info_test = pickle.load(f)
        print("Method information loaded successfully.")
    except FileNotFoundError:
        print("Method information not found. Using standard validation.")
        pass
    
    # Load model info
    with open(os.path.join(data_dir, "model_info.json"), 'r') as f:
        model_info = json.load(f)
    
    print(f"Data shapes:")
    print(f"  Train embeddings: {X_train_emb.shape}")
    print(f"  Train auxiliary: {X_train_aux.shape}")
    print(f"  Test embeddings: {X_test_emb.shape}")
    print(f"  Test auxiliary: {X_test_aux.shape}")
    
    # Create datasets
    train_dataset = EnhancedDataset(X_train_emb, X_train_aux, y_train, method_info_train)
    test_dataset = EnhancedDataset(X_test_emb, X_test_aux, y_test, method_info_test)
    
    # Training configuration
    if config is None:
        config = {
            'batch_size': 128,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': 100,
            'patience': 20,
            'hidden_dims': [32, 32],
            'dropout_rate': 0.2
        }
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = EnhancedMLP(
        embedding_dim=model_info['embedding_dim'],
        auxiliary_dim=model_info['auxiliary_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
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
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    val_mses = []
    val_accuracies = []

    # Per-method loss histories
    method_names = ['majority', 'naive', 'weighted', 'beam_search']
    method_train_loss_history = {m: [] for m in method_names}
    method_val_loss_history = {m: [] for m in method_names}
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val MSE':<12} {'Val Acc':<8} {'LR':<10}")
    print("-" * 70)
    
    # Print method distribution if available
    if method_info_test is not None:
        method_counts = {}
        for info in method_info_test:
            method = info['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        print(f"Method distribution in test set: {method_counts}")
        print("-" * 70)
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss, method_train_losses, method_train_counts = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation (try method-specific validation first)
        if method_info_test is not None:
            (val_loss, val_mse, val_accuracy), method_stats = validate_epoch_by_method(model, test_loader, criterion, device)
        else:
            val_loss, val_mse, val_accuracy = validate_epoch(model, test_loader, criterion, device)
            method_stats = None
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mses.append(val_mse)
        val_accuracies.append(val_accuracy)

        # Record per-method histories if available
        if method_stats is not None:
            for m in method_names:
                # Train loss per method: only if we saw samples in this epoch
                if method_train_counts.get(m, 0) > 0:
                    method_train_loss_history[m].append(method_train_losses.get(m, np.nan))
                else:
                    method_train_loss_history[m].append(np.nan)
                # Val loss per method: only if there were val samples for this method
                if method_stats[m]['count'] > 0:
                    method_val_loss_history[m].append(method_stats[m]['loss'])
                else:
                    method_val_loss_history[m].append(np.nan)
        
        # Print formatted output
        print(f"{epoch+1:<6} {train_loss:<12.6f} {val_loss:<12.6f} {val_mse:<12.6f} {val_accuracy:<8.4f} {optimizer.param_groups[0]['lr']:<10.2e}")
        
        # Print method-specific statistics every 5 epochs or on the last epoch
        if method_stats is not None and (epoch % 5 == 4 or epoch == config['num_epochs'] - 1):
            print(f"    Method-specific stats (Epoch {epoch+1}):")
            print(f"    {'Method':<12} {'Train_Loss':<12} {'Val_Loss':<12} {'Val_MSE':<10} {'Val_Acc':<8} {'Count':<6}")
            print(f"    {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*6}")
            for method, stats in method_stats.items():
                if stats['count'] > 0:
                    train_loss_method = method_train_losses.get(method, np.nan)
                    print(f"    {method:<12} {train_loss_method:<12.6f} {stats['loss']:<12.6f} {stats['mse']:<10.6f} {stats['accuracy']:<8.4f} {stats['count']:<6}")
            print()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(output_dir, 'best_model.pth'))
            
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("-" * 70)
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    # Plot per-method loss curves if available
    if method_info_test is not None:
        # Individual plots per method
        for m in method_names:
            plt.figure(figsize=(8, 5))
            plt.plot(method_train_loss_history[m], label=f'{m} Train Loss', linewidth=2)
            plt.plot(method_val_loss_history[m], label=f'{m} Val Loss', linewidth=2)
            plt.title(f'Train/Val Loss by Epoch - {m}', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'method_loss_{m}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Combined grid plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Per-Method Train/Val Loss', fontsize=16, fontweight='bold')
        positions = [(0,0), (0,1), (1,0), (1,1)]
        for (m, pos) in zip(method_names, positions):
            ax = axes[pos]
            ax.plot(method_train_loss_history[m], label='Train Loss', linewidth=2)
            ax.plot(method_val_loss_history[m], label='Val Loss', linewidth=2)
            ax.set_title(m)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'method_loss_all.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    if method_info_test is not None:
        (final_val_loss, final_val_mse, final_val_accuracy), final_method_stats = validate_epoch_by_method(model, test_loader, criterion, device)
    else:
        final_val_loss, final_val_mse, final_val_accuracy = validate_epoch(model, test_loader, criterion, device)
        final_method_stats = None

    print(f"\nFinal Results:")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Final Val MSE: {final_val_mse:.6f}")
    print(f"  Final Val Accuracy: {final_val_accuracy:.4f}")

    # Print final method-specific statistics
    if final_method_stats is not None:
        print(f"\nFinal Method-Specific Results:")
        print(f"{'Method':<12} {'Count':<6} {'Val_Loss':<10} {'Val_MSE':<10} {'Val_Acc':<8} {'Mean_Pred':<10} {'Mean_Label':<10}")
        print(f"{'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
        for method, stats in final_method_stats.items():
            if stats['count'] > 0:
                print(f"{method:<12} {stats['count']:<6} {stats['loss']:<10.6f} {stats['mse']:<10.6f} {stats['accuracy']:<8.4f} {stats['mean_prediction']:<10.4f} {stats['mean_label']:<10.4f}")
        print()

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mses': val_mses,
        'val_accuracies': val_accuracies,
        'best_epoch': checkpoint['epoch'],
        'config': config
    }
    if method_info_test is not None:
        history['method_train_loss_history'] = method_train_loss_history
        history['method_val_loss_history'] = method_val_loss_history
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(20, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='orange', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Plot 2: Validation MSE
    plt.subplot(2, 3, 2)
    plt.plot(val_mses, label='Val MSE', color='red', linewidth=2)
    plt.title('Validation MSE', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Validation Accuracy
    plt.subplot(2, 3, 3)
    plt.plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
    plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Loss comparison (linear scale)
    plt.subplot(2, 3, 4)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='orange', linewidth=2)
    plt.title('Training and Validation Loss (Linear Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Training progress summary
    plt.subplot(2, 3, 5)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='gray')
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Final metrics comparison
    plt.subplot(2, 3, 6)
    metrics = ['Train Loss', 'Val Loss', 'Val MSE', 'Val Acc']
    final_values = [train_losses[-1], val_losses[-1], val_mses[-1], val_accuracies[-1]]
    colors = ['blue', 'orange', 'red', 'green']
    
    bars = plt.bar(metrics, final_values, color=colors, alpha=0.7)
    plt.title('Final Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    
    # Print final summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Final validation MSE: {val_mses[-1]:.6f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Best validation accuracy: {max(val_accuracies):.4f}")
    print(f"Best validation MSE: {min(val_mses):.6f}")
    
    # Check for overfitting
    train_val_diff = train_losses[-1] - val_losses[-1]
    if abs(train_val_diff) > 0.1:
        print(f"Warning: Large gap between train ({train_losses[-1]:.6f}) and val ({val_losses[-1]:.6f}) loss")
        print("This might indicate overfitting or underfitting.")
    
    print(f"{'='*60}")
    
    plt.show()
    
    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_predictions = generate_test_predictions(model, test_loader, device)
    
    # Save predictions
    np.save(os.path.join(output_dir, 'test_predictions.npy'), test_predictions)
    print(f"Test predictions saved to: {os.path.join(output_dir, 'test_predictions.npy')}")
    
    print(f"\nTraining completed. Results saved to {output_dir}")
    print(f"Final test predictions shape: {test_predictions.shape}")
    print(f"Test predictions range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    
    return test_predictions

def generate_test_predictions(model, test_loader, device):
    """
    Generate predictions on the test set and return as numpy array.
    Similar to the notebook functionality.
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:  # Has method info
                (emb_features, aux_features), _, method_info = batch_data
            else:  # No method info (backward compatibility)
                (emb_features, aux_features), _ = batch_data
                
            emb_features = emb_features.to(device)
            aux_features = aux_features.to(device)
            
            outputs = model(emb_features, aux_features)
            predictions = outputs.squeeze().cpu().numpy()
            all_predictions.extend(predictions)
    
    return np.array(all_predictions)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save training results")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    args = parser.parse_args()
    
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'hidden_dims': [32, 32],
        'dropout_rate': 0.2
    }
    
    main(args.data_dir, args.output_dir, config)
