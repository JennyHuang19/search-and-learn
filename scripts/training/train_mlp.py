#!/usr/bin/env python3
"""
Training script for MLP model that:
1. Takes question strings and numerical features as input
2. Passes questions through either BERT or Qwen to get embeddings (with backprop)
3. Concatenates embeddings with numerical features
4. Passes combined features through MLP classifier
5. Uses step-by-step loss tracking and validation every 10 steps
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
import matplotlib.pyplot as plt


class MLPDataset(Dataset):
    """Dataset class for MLP training with either BERT or Qwen"""
    
    def __init__(self, questions, numerical_features, labels, tokenizer, max_length=512, model_type='bert'):
        self.questions = questions
        self.numerical_features = numerical_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        numerical_feat = torch.tensor(self.numerical_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.model_type == 'bert':
            # Tokenize the question for BERT
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
        else:
            # For Qwen, just store the text and tokenize later
            return {
                'text': question,
                'numerical_features': numerical_feat,
                'labels': label
            }


class MLPModel(nn.Module):
    """MLP model that can use either BERT or Qwen embeddings"""
    
    def __init__(self, model_name, model_type, numerical_feature_dim, hidden_dims=[16, 4], freeze_embedding=False):
        super(MLPModel, self).__init__()
        
        self.model_type = model_type
        
        # Load embedding model
        if model_type == 'bert':
            self.embedding_model = AutoModel.from_pretrained(model_name)
            self.embedding_dim = self.embedding_model.config.hidden_size
        else:  # qwen
            self.embedding_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.embedding_dim = self.embedding_model.config.hidden_size
        
        # Freeze embedding model if specified
        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
        
        # Build MLP dynamically based on hidden_dims
        layers = []
        input_dim = self.embedding_dim + numerical_feature_dim
        
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
        
    def forward(self, input_ids=None, attention_mask=None, text=None, numerical_features=None):
        if self.model_type == 'bert':
            # Forward pass through BERT
            bert_outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            # Get CLS token embedding (first token)
            embedding = bert_outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, embedding_dim]
        else:
            # Forward pass through Qwen
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(next(self.parameters()).device)
            
            outputs = self.embedding_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            # Get mean pooling over all tokens from last hidden layer
            last_hidden_state = outputs.hidden_states[-1]  # shape: [batch_size, seq_len, hidden_dim]
            embedding = last_hidden_state.mean(dim=1)  # shape: [batch_size, hidden_dim]
        
        # Concatenate embeddings with numerical features
        combined_features = torch.cat([embedding, numerical_features], dim=1)
        
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


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_path, model_type):
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
            
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if model_type == 'bert':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
            else:
                text = batch['text']
                outputs = model(text=text, numerical_features=numerical_features)
            
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            train_steps.append(counter)
            train_loss.append(loss.item())
            step_train_loss.append(loss.item())
            
            # Validation every 10 steps
            if (i + 1) % 10 == 0:
                model.eval()
                val_loss_step = []
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_numerical_features = val_batch['numerical_features'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        
                        if model_type == 'bert':
                            val_input_ids = val_batch['input_ids'].to(device)
                            val_attention_mask = val_batch['attention_mask'].to(device)
                            val_preds = model(input_ids=val_input_ids, attention_mask=val_attention_mask, numerical_features=val_numerical_features)
                        else:
                            val_text = val_batch['text']
                            val_preds = model(text=val_text, numerical_features=val_numerical_features)
                        
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
    }


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


def plot_soft_label_calibration(model, val_loader, device, output_dir, model_type):
    """Plot predicted vs true soft labels for calibration analysis"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)
            
            if model_type == 'bert':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, numerical_features=numerical_features)
            else:
                text = batch['text']
                outputs = model(text=text, numerical_features=numerical_features)
            
            preds = outputs.squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    softLabel_preds_numpy = np.array(all_preds)
    y_val = np.array(all_labels)
    
    # Save predictions to .npy file
    preds_path = os.path.join(output_dir, 'softLabel_preds.npy')
    np.save(preds_path, softLabel_preds_numpy)
    print(f'Soft label predictions saved to: {preds_path}')
    
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


def main():
    parser = argparse.ArgumentParser(description='Train MLP model with either BERT or Qwen embeddings')
    parser.add_argument('--embedding-model', choices=['bert', 'qwen'], required=True, help='Type of embedding model to use')
    parser.add_argument('--model-path', required=True, help='Path to the embedding model')
    parser.add_argument('--train-csv', required=True, help='Path to training CSV file')
    parser.add_argument('--val-csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--question-col', default='question', help='Column name for questions')
    parser.add_argument('--numerical-cols', nargs='+', default=['N', 'question_length', 'method_beam_search', 'method_maj', 'method_naive', 'method_weighted'], help='Column names for numerical features')
    parser.add_argument('--label-col', default='sl', help='Column name for labels')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[16, 4], help='Hidden layer dimensions for MLP (e.g., 16 4 for two hidden layers)')
    parser.add_argument('--freeze-embedding', action='store_true', help='Freeze embedding model parameters')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--output-dir', default='./mlp_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Using {args.embedding_model.upper()} as embedding model')
    
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
    print(f'Loading tokenizer for {args.model_path}...')
    if args.embedding_model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:  # qwen
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Create datasets
    train_dataset = MLPDataset(train_questions, train_numerical, train_labels, tokenizer, args.max_length, args.embedding_model)
    val_dataset = MLPDataset(val_questions, val_numerical, val_labels, tokenizer, args.max_length, args.embedding_model)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Numerical feature dimension: {len(args.numerical_cols)}')
    print(f'Batch size: {args.batch_size}')
    print(f'Steps per epoch: {len(train_loader)}')
    print(f'Validation every 10 steps')
    
    # Create model
    model = MLPModel(args.model_path, args.embedding_model, len(args.numerical_cols), args.hidden_dims, args.freeze_embedding)
    model.to(device)
    
    print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'MLP architecture: {args.embedding_model.upper()} + {len(args.numerical_cols)} features -> {args.hidden_dims} -> 1')
    if args.freeze_embedding:
        print('Embedding model parameters are frozen')
    else:
        print('Embedding model parameters are trainable')
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training
    print('Starting training...')
    model_save_path = os.path.join(args.output_dir, 'best_model.pth')
    training_history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.num_epochs, device, model_save_path, args.embedding_model
    )
    
    # Create soft label predictions and calibration plot
    print('Creating soft label calibration plot...')
    softLabel_preds_numpy, y_val = plot_soft_label_calibration(
        model, val_loader, device, args.output_dir, args.embedding_model
    )
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(training_history, plot_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    print(f'Training completed!')
    print(f'Best validation loss: {training_history["best_val_loss"]:.4f}')
    print(f'Model saved to: {model_save_path}')
    print(f'Training curves saved to: {plot_path}')
    print(f'Training history saved to: {history_path}')
    print(f'Scaler saved to: {scaler_path}')


if __name__ == '__main__':
    main()

