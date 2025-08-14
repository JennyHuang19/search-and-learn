#!/usr/bin/env python3
"""
Training script for Transformer + MLP model that:
1. Takes question strings and numerical features as input
2. Passes questions through BERT or Qwen to get embeddings (with backprop)
3. Optionally applies PCA dimension reduction to embeddings
4. Concatenates embeddings with numerical features
5. Passes combined features through MLP classifier
6. Uses step-by-step loss tracking and validation every 200 steps

Usage examples:
    # Train with BERT (default)
    python train_mlp.py --train-csv train.csv --val-csv val.csv
    
    # Train with Qwen
    python train_mlp.py --train-csv train.csv --val-csv val.csv --model-type qwen
    
    # Train with Qwen and custom model path
    python train_mlp.py --train-csv train.csv --val-csv val.csv --model-type qwen --model-path /path/to/qwen/model
    
    # Train with PCA reduction
    python train_mlp.py --train-csv train.csv --val-csv val.csv --pca-components 32
    
    # Train with frozen transformer
    python train_mlp.py --train-csv train.csv --val-csv val.csv --freeze-transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import pandas as pd
import numpy as np
import argparse
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class TransformerMLPDataset(Dataset):
    """Dataset class for Transformer + MLP training"""
    
    def __init__(self, questions, numerical_features, labels, tokenizer, max_length=512):
        self.questions = questions
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        numerical_feat = torch.tensor(self.numerical_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Tokenize the question
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numerical_features': numerical_feat,
            'labels': label
        }


class TransformerMLPModel(nn.Module):
    """Transformer + MLP model that allows backprop through BERT/Qwen with optional PCA reduction"""
    
    def __init__(self, model_name, model_type, numerical_feature_dim, hidden_dims, freeze_transformer=False, pca_components=None): 
        super(TransformerMLPModel, self).__init__()
        
        # Load transformer model
        self.model_type = model_type
        if self.model_type == "bert":
            self.transformer = AutoModel.from_pretrained(model_name)
        elif self.model_type == "qwen":
            self.transformer = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Use 'bert' or 'qwen'")
        
        # Get embedding dimension based on model type
        if model_type == "bert":
            self.embedding_dim = self.transformer.config.hidden_size
        elif model_type == "qwen":
            self.embedding_dim = self.transformer.config.hidden_size
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'bert' or 'qwen'")
        
        # PCA configuration
        self.pca_components = pca_components
        self.pca = None
        if pca_components is not None:
            self.pca = PCA(n_components=pca_components, svd_solver="randomized")
            self.effective_embedding_dim = pca_components
        else:
            self.effective_embedding_dim = self.embedding_dim

        
        # Freeze transformer if specified
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Build MLP dynamically based on hidden_dims
        layers = []
        input_dim = self.effective_embedding_dim + numerical_feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU()
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
    
    def fit_pca(self, embeddings):
        """Fit PCA on transformer embeddings"""
        if self.pca_components is not None:
            print(f"Fitting PCA with {self.pca_components} components on {embeddings.shape[0]} samples")
            self.pca.fit(embeddings.detach().cpu().numpy())
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
    
    def forward(self, input_ids, attention_mask, numerical_features):
        # Forward pass through transformer (with gradients if not frozen)
        if self.model_type == "bert":
            transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        elif self.model_type == "qwen":
            transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        
        # Get embeddings with configurable pooling
        if self.pooling_method == "cls":
            # Use CLS token embedding (first token) - traditional BERT approach
            cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, embedding_dim]
        elif self.pooling_method == "max":
            # Max pooling over all tokens
            cls_embedding = transformer_outputs.hidden_states[-1].max(dim=1)[0]
        else:  # mean
            # Mean pooling over all tokens
            cls_embedding = transformer_outputs.hidden_states[-1].mean(dim=1)
        
        # Apply PCA reduction if configured
        if self.pca is not None and self.pca_components is not None:
            # Convert to numpy for PCA, then back to tensor
            cls_embedding_np = cls_embedding.detach().cpu().numpy()
            cls_embedding_reduced = self.pca.transform(cls_embedding_np)
            cls_embedding = torch.tensor(cls_embedding_reduced, dtype=torch.float32, device=cls_embedding.device)
        
        # Concatenate embeddings with numerical features
        combined_features = torch.cat([cls_embedding, numerical_features], dim=1)
        
        # Forward pass through MLP
        output = self.mlp(combined_features)
        
        return output


def load_data(csv_path, question_col, numerical_cols, label_col):
    """Load and preprocess data from CSV"""
    df = pd.read_csv(csv_path)
    
    questions = df[question_col].values
    numerical_features = df[numerical_cols].values
    labels = df[label_col].values
    
    return questions, numerical_features, labels


def collect_transformer_embeddings(model, data_loader, device, pooling_method):
    """Collect transformer embeddings from data for PCA fitting"""
    model.eval()
    embeddings = []
    
    print("Collecting transformer embeddings for PCA fitting...")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i % 100 == 0:
                print(f"Processing batch {i+1}/{len(data_loader)}")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get transformer outputs
            if model.model_type == "bert":
                transformer_outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
            elif model.model_type == "qwen":
                transformer_outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            else:
                raise ValueError(f"Unsupported model type: {model.model_type}. Use 'bert' or 'qwen'")
            
            # cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]  # CLS token
            # Get embeddings with configurable pooling
            if pooling_method == "cls":
                # Use CLS token embedding (first token) - traditional BERT approach
                cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, embedding_dim]
            elif pooling_method == "max":
                # Max pooling over all tokens
                cls_embedding = transformer_outputs.hidden_states[-1].max(dim=1)[0]
            else:  # mean
                # Mean pooling over all tokens
                cls_embedding = transformer_outputs.hidden_states[-1].mean(dim=1)
            
            
            embeddings.append(cls_embedding.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    print(f"Collected {all_embeddings.shape[0]} embeddings with dimension {all_embeddings.shape[1]}")
    
    return all_embeddings


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_path):
    """Training loop with step-by-step loss tracking and validation every 10 steps"""
    
    # Loss tracking arrays
    agg_train_loss = []
    agg_val_loss = []
    step_train_loss = []
    step_val_loss = []
    best_val_loss = float('inf')
    counter = 0
    train_steps, val_steps = [], []

    
    for epoch in range(num_epochs):
        model.train()
        train_loss = []

        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1} of {len(train_loader)}")
            counter += 1
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            train_steps.append(counter)
            train_loss.append(loss.item())
            step_train_loss.append(loss.item())
            
            # Validation every 10 steps
            if (i + 1) % 200 == 0:
                model.eval()
                val_loss_step = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                        val_numerical_features = val_batch['numerical_features'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        
                        val_preds = model(val_input_ids, val_attention_mask, val_numerical_features)
                        val_loss_step.append(criterion(val_preds.squeeze(), val_labels).item())
                
                mean_val_loss_step = np.mean(val_loss_step)
                step_val_loss.append(mean_val_loss_step)

                val_steps.append(counter)
                
                print(f"Epoch {epoch+1}, Step {i+1}, Train Loss: {loss.item():.4f}, Val Loss: {mean_val_loss_step:.4f}")
                
                # Save best model based on validation loss
                if mean_val_loss_step < best_val_loss:
                    best_val_loss = mean_val_loss_step
                    torch.save(model.state_dict(), save_path)
                    print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
                
                model.train()
        
        # End of epoch
        epoch_train_loss = np.mean(train_loss)
        epoch_val_loss = np.mean(step_val_loss[-len(train_loader)//10:]) if len(step_val_loss) > 0 else 0
        
        agg_train_loss.append(epoch_train_loss)
        agg_val_loss.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1} completed:")
        print(f"  Average Train Loss: {epoch_train_loss:.4f}")
        print(f"  Average Val Loss: {epoch_val_loss:.4f}")
        print("-" * 50)
    
    return {
        'agg_train_loss': agg_train_loss,
        'agg_val_loss': agg_val_loss,
        'step_train_loss': step_train_loss,
        'step_val_loss': step_val_loss,
        'train_steps': train_steps,
        'val_steps': val_steps,
        'best_val_loss': best_val_loss
    } # track time if needed.


def plot_training_curves(training_history, save_path):
    """Plot training and validation loss curves with moving average smoothing"""
    
    # Plot epoch-level losses
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_history['agg_train_loss'], label='Training Loss', color='blue', marker='o')
    plt.plot(training_history['agg_val_loss'], label='Validation Loss', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch-level Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot step-level losses with moving average smoothing
    plt.subplot(1, 2, 2)
    
    # Moving average smoothing for training loss
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    window = 20  # tweak as needed
    smoothed_train_loss = moving_average(training_history['step_train_loss'], window)
    smoothed_train_steps = training_history['train_steps'][window-1:] if len(training_history['train_steps']) == len(training_history['step_train_loss']) else np.arange(len(smoothed_train_loss))
    
    plt.plot(smoothed_train_steps, smoothed_train_loss, 
             label=f'Train Loss (smoothed, window={window})', color='blue')
    plt.plot(training_history['val_steps'], training_history['step_val_loss'], 
             label='Val Loss (per step)', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Train/Val Loss per Step')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a separate detailed step-level plot
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_train_steps, smoothed_train_loss, 
             label=f'Train Loss (smoothed, window={window})', color='blue')
    plt.plot(training_history['val_steps'], training_history['step_val_loss'], 
             label='Val Loss (per step)', color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Train/Val Loss per Step')
    plt.legend()
    plt.grid(True)
    
    # Save the detailed step plot
    step_plot_path = save_path.replace('.png', '_step_detail.png')
    plt.savefig(step_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Step-level plot saved to: {step_plot_path}')


def plot_soft_label_calibration(model, val_loader, device, output_dir):
    """Plot predicted vs true soft labels for calibration analysis"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, numerical_features)
            preds = outputs.squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    softLabel_preds_numpy = np.array(all_preds)
    y_val = np.array(all_labels)
    
    # Create calibration plot
    plt.figure(figsize=(6, 6))
    plt.scatter(softLabel_preds_numpy, y_val, alpha=0.1)
    plt.xlabel("Predicted probability")
    plt.ylabel("True label (soft)")
    plt.title("Predicted vs True (Soft Label)")
    plt.grid(True)
    
    # Add diagonal line for perfect calibration
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect calibration')
    plt.legend()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'soft_label_calibration.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Soft label calibration plot saved to: {plot_path}')
    
    return softLabel_preds_numpy, y_val


def save_transformer_last_layer_weights(model, output_dir, stage="before"):
    """Save only the mean and std of transformer's last layer weights"""
    transformer_layers = list(model.transformer.encoder.layer)
    if transformer_layers:
        last_layer = transformer_layers[-1]
        stats = {}
        total_params = 0
        for name, param in last_layer.named_parameters():
            mean = param.mean().item()
            std = param.std().item()
            stats[name] = {'mean': mean, 'std': std, 'shape': tuple(param.shape)}
            total_params += param.numel()
            print(f'  {name}: {param.shape}, mean: {mean:.6f}, std: {std:.6f}')
        print(f'  Total parameters in last layer: {total_params:,}')
        stats_path = os.path.join(output_dir, f'transformer_last_layer_stats_{stage}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f'Transformer last layer stats ({stage}) saved to: {stats_path}')
        return stats_path
    else:
        print(f'Warning: Could not find transformer layers to save stats for {stage} stage')
        return None


def main():
    parser = argparse.ArgumentParser(description='Train BERT + MLP model')
    parser.add_argument('--train-csv', required=True, help='Path to training CSV file')
    parser.add_argument('--val-csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--question-col', default='question', help='Column name for questions')
    parser.add_argument('--numerical-cols', nargs='+', default=['N', 'question_length', 'method_beam_search', 'method_majority', 'method_naive', 'method_weighted'], help='Column names for numerical features')
    parser.add_argument('--label-col', default='sl', help='Column name for labels')
    parser.add_argument('--model-type', choices=['bert', 'qwen'], default='bert', help='Type of transformer model to use (bert or qwen)')
    parser.add_argument('--model-path', default=None, help='Path to transformer model (local directory or HF model id). If not specified, will use default paths for the selected model type.')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[16, 4], help='Hidden layer dimensions for MLP (e.g., 16 4 for two hidden layers)')
    parser.add_argument('--freeze-transformer', action='store_true', help='Freeze transformer parameters')
    parser.add_argument('--pca-components', type=int, default=None, help='Number of PCA components for embeddings (default: None, no PCA)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--pooling', choices=['cls', 'mean', 'max'], default='cls', 
                    help='Pooling method: cls (CLS token), mean (mean pooling), or max (max pooling)')
    parser.add_argument('--output-dir', default='/dccstor/gma2/jhjenny9/search-and-learn/training-res/numinaMath', help='Output directory')
    
    
    args = parser.parse_args()
    
    # Set default model paths if not specified
    if args.model_path is None:
        if args.model_type == 'bert':
            args.model_path = '/u/jhjenny9/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
        elif args.model_type == 'qwen':
            args.model_path = 'Qwen/Qwen2.5-1.5B-Instruct'
        print(f'Using default {args.model_type} model path: {args.model_path}')
    
    # Validate model type and path
    if args.model_type not in ['bert', 'qwen']:
        raise ValueError(f"Unsupported model type: {args.model_type}. Use 'bert' or 'qwen'")
    
    print(f'Selected model type: {args.model_type}')
    print(f'Model path: {args.model_path}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading training data...')
    train_questions, train_numerical, train_labels = load_data(
        args.train_csv, args.question_col, args.numerical_cols, args.label_col
    )
    
    print('Loading validation data...')
    val_questions, val_numerical, val_labels = load_data(
        args.val_csv, args.question_col, args.numerical_cols, args.label_col
    )
    
    # Normalize numerical features
    scaler = StandardScaler()
    train_numerical = scaler.fit_transform(train_numerical)
    val_numerical = scaler.transform(val_numerical)
    
    # Save scaler for inference
    scaler_path = os.path.join(args.output_dir, 'numerical_scaler.pkl')
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Load tokenizer
    if args.model_type == "bert":
        print(f'Loading tokenizer for {args.model_path}...')
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.model_type == "qwen":
        print(f'Loading tokenizer for {args.model_path}...')
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}. Use 'bert' or 'qwen'")
    
    # Create datasets
    train_dataset = TransformerMLPDataset(train_questions, train_numerical, train_labels, tokenizer, args.max_length)
    val_dataset = TransformerMLPDataset(val_questions, val_numerical, val_labels, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) # can use larger batch size for validation.
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Numerical feature dimension: {len(args.numerical_cols)}')
    print(f'Transformer type: {args.model_type}')
    print(f'Batch size: {args.batch_size}')
    print(f'Steps per epoch: {len(train_loader)}')
    print(f'Validation every 200 steps')
    
    # Create model
    model = TransformerMLPModel(args.model_path, args.model_type, len(args.numerical_cols), args.hidden_dims, args.freeze_transformer, args.pca_components)
    model.to(device)
    
    print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    if args.freeze_transformer:
        print('Transformer parameters are frozen')
    else:
        print('Transformer parameters are trainable')
    
    # Fit PCA if specified
    if args.pca_components is not None:
        print(f'\n=== Fitting PCA with {args.pca_components} components ===')
        # Collect transformer embeddings from training data
        transformer_embeddings = collect_transformer_embeddings(model, train_loader, device, args.pooling)
        # Fit PCA on the collected embeddings
        model.fit_pca(transformer_embeddings)
        print(f'Transformer embeddings reduced from {transformer_embeddings.shape[1]} to {args.pca_components} dimensions')
    else:
        print('\n=== No PCA reduction applied ===')
        print(f'Using full transformer embedding dimension: {model.embedding_dim}')
    
    # Save transformer last layer weights BEFORE training
    print('\n=== Transformer Last Layer Weights BEFORE Training ===')
    before_weights_path = save_transformer_last_layer_weights(model, args.output_dir, "before")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training
    print('\nStarting training...')
    model_save_path = os.path.join(args.output_dir, 'best_model.pth')
    training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.num_epochs, device, model_save_path
    )
    
    # Save transformer last layer weights AFTER training
    print('\n=== Transformer Last Layer Weights AFTER Training ===')
    after_weights_path = save_transformer_last_layer_weights(model, args.output_dir, "after")

    # Create soft label predictions and calibration plot
    print('Creating soft label calibration plot...')
    softLabel_preds_numpy, y_val = plot_soft_label_calibration(
        model, val_loader, device, args.output_dir
    )
    # Explicitly save softLabel_preds_numpy to args.output_dir
    np.save(os.path.join(args.output_dir, 'softLabel_preds.npy'), softLabel_preds_numpy)

    # Compute and save BCE loss lower bound
    BCE_loss_lower_bound = float(np.mean(
        -val_loader.dataset.labels * np.log(val_loader.dataset.labels + 1e-8)
        - (1 - val_loader.dataset.labels) * np.log(1 - val_loader.dataset.labels + 1e-8)
    ))
    training_history["BCE_loss_lower_bound"] = BCE_loss_lower_bound
    training_history["transformer_weights_before"] = before_weights_path
    training_history["transformer_weights_after"] = after_weights_path
    training_history["model_type"] = args.model_type
    if args.pca_components is not None:
        training_history["pca_components"] = args.pca_components
        training_history["pca_explained_variance"] = float(model.pca.explained_variance_ratio_.sum()) if model.pca else None
    print(f"BCE loss lower bound: {BCE_loss_lower_bound}")
    if args.pca_components is not None:
        print(f"PCA explained variance ratio: {training_history['pca_explained_variance']:.4f}")
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(training_history, plot_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f'Training completed!')
    print(f'Best validation loss: {training_history["best_val_loss"]:.4f}')
    print(f'Model type: {args.model_type}')
    if args.pca_components is not None:
        print(f'PCA reduction: {model.embedding_dim} → {args.pca_components} dimensions')
        print(f'PCA explained variance: {training_history["pca_explained_variance"]:.4f}')
    print(f'Model saved to: {model_save_path}')
    print(f'Training curves saved to: {plot_path}')
    print(f'Training history saved to: {history_path}')
    print(f'Scaler saved to: {scaler_path}')


if __name__ == '__main__':
    main()

