#!/usr/bin/env python3
"""
Script to generate predictions using a trained enhanced MLP model.

This script loads a trained model and generates predictions on test data,
returning the results as numpy arrays (softLabel_preds_numpy).
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from torch.utils.data import DataLoader

# Import the model and dataset classes
from train_enhanced_mlp import EnhancedMLP, EnhancedDataset

def load_model_and_data(model_dir, data_dir):
    """
    Load the trained model and test data.
    """
    # Load model configuration
    with open(os.path.join(data_dir, "model_info.json"), 'r') as f:
        model_info = json.load(f)
    
    # Load training configuration
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pth'))
    config = checkpoint['config']
    
    # Load test data
    X_test_emb = np.load(os.path.join(data_dir, "X_test_embeddings.npy"))
    X_test_aux = np.load(os.path.join(data_dir, "X_test_auxiliary.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedMLP(
        embedding_dim=model_info['embedding_dim'],
        auxiliary_dim=model_info['auxiliary_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset and dataloader
    test_dataset = EnhancedDataset(X_test_emb, X_test_aux, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return model, test_loader, device, y_test

def generate_predictions(model, test_loader, device):
    """
    Generate predictions on the test set.
    Returns numpy array of predictions (softLabel_preds_numpy).
    """
    model.eval()
    all_predictions = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for i, ((emb_features, aux_features), _) in enumerate(test_loader):
            emb_features = emb_features.to(device)
            aux_features = aux_features.to(device)
            
            outputs = model(emb_features, aux_features)
            predictions = outputs.squeeze().cpu().numpy()
            all_predictions.extend(predictions)
            
            if (i + 1) % 10 == 0:
                print(f"Processed batch {i + 1}")
    
    softLabel_preds_numpy = np.array(all_predictions)
    return softLabel_preds_numpy

def main():
    parser = argparse.ArgumentParser(description="Generate predictions using trained model")
    parser.add_argument("--model_dir", type=str, required=True, 
                       help="Directory containing trained model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed data")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save predictions (optional)")
    
    args = parser.parse_args()
    
    # Load model and data
    print("Loading model and data...")
    model, test_loader, device, y_test = load_model_and_data(args.model_dir, args.data_dir)
    
    # Generate predictions
    softLabel_preds_numpy = generate_predictions(model, test_loader, device)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"  Predictions shape: {softLabel_preds_numpy.shape}")
    print(f"  Predictions range: [{softLabel_preds_numpy.min():.4f}, {softLabel_preds_numpy.max():.4f}]")
    print(f"  Predictions mean: {softLabel_preds_numpy.mean():.4f}")
    print(f"  Predictions std: {softLabel_preds_numpy.std():.4f}")
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, accuracy_score
    
    mse = mean_squared_error(y_test, softLabel_preds_numpy)
    binary_preds = (softLabel_preds_numpy >= 0.5).astype(int)
    binary_labels = (y_test >= 0.5).astype(int)
    accuracy = accuracy_score(binary_labels, binary_preds)
    
    print(f"  MSE: {mse:.6f}")
    print(f"  Binary Accuracy: {accuracy:.4f}")
    
    # Save predictions if output file specified
    if args.output_file:
        np.save(args.output_file, softLabel_preds_numpy)
        print(f"\nPredictions saved to: {args.output_file}")
    
    # Return predictions (for interactive use)
    return softLabel_preds_numpy

if __name__ == "__main__":
    predictions = main()
    print(f"\nReturned predictions array with shape: {predictions.shape}")
    print("You can access this as 'softLabel_preds_numpy' in your environment.")
