#!/usr/bin/env python3
"""
Script to run the improved MLP training pipeline with enhanced feature engineering.

This script demonstrates the improved data processing that includes:
1. Feature normalization using StandardScaler
2. Repeated auxiliary features to increase signal strength
3. Multi-layer auxiliary feature injection in the MLP architecture
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run improved MLP training pipeline")
    parser.add_argument("--train_csv", type=str, required=True, 
                       help="Path to training CSV file")
    parser.add_argument("--test_csv", type=str, required=True, 
                       help="Path to test CSV file")
    parser.add_argument("--data_output_dir", type=str, required=True,
                       help="Directory to save processed data")
    parser.add_argument("--model_output_dir", type=str, required=True,
                       help="Directory to save trained model")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=20,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Step 1: Process data with improved feature engineering
    print("Step 1: Processing data with improved feature engineering...")
    data_cmd = [
        "python", "scripts/extract_features_improved.py",
        "--train_csv", args.train_csv,
        "--test_csv", args.test_csv,
        "--output_dir", args.data_output_dir
    ]
    
    if not run_command(data_cmd, "Data Processing"):
        print("Data processing failed. Exiting.")
        sys.exit(1)
    
    # Step 2: Train the enhanced MLP model
    print("Step 2: Training enhanced MLP model...")
    train_cmd = [
        "python", "scripts/train_enhanced_mlp.py",
        "--data_dir", args.data_output_dir,
        "--output_dir", args.model_output_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_epochs", str(args.num_epochs),
        "--patience", str(args.patience)
    ]
    
    if not run_command(train_cmd, "Model Training"):
        print("Model training failed. Exiting.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Training pipeline completed successfully!")
    print(f"Processed data saved to: {args.data_output_dir}")
    print(f"Trained model saved to: {args.model_output_dir}")
    print(f"{'='*60}")
    
    # Print summary of improvements
    print("\nImprovements implemented:")
    print("1. ✅ Feature normalization using StandardScaler")
    print("2. ✅ Repeated auxiliary features (10x repetition) for signal strength")
    print("3. ✅ Multi-layer auxiliary feature injection in MLP architecture")
    print("4. ✅ Enhanced MLP with BatchNorm, Dropout, and GELU activations")
    print("5. ✅ Learning rate scheduling and early stopping")
    print("6. ✅ Comprehensive training monitoring and visualization")

if __name__ == "__main__":
    main()
