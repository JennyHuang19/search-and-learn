#!/usr/bin/env python3
"""
Script to concatenate all completion files matching the pattern:
/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32_numina_{N}_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl

Outputs a concatenated DataFrame with columns (problem, completion_tokens).
"""

import os
import json
import pandas as pd
import glob
from pathlib import Path
import argparse


def find_completion_files(base_dir):
    """
    Find all completion files matching the pattern.
    
    Args:
        base_dir (str): Base directory to search in
        
    Returns:
        list: List of file paths matching the pattern
    """
    # Pattern to match: best_of_32_numina_{N}_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl
    pattern = os.path.join(base_dir, "best_of_32_numina_*_chunk", "best_of_n_completions.jsonl", "best_of_n_completions.jsonl")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} completion files:")
    for f in files:
        print(f"  {f}")
    
    return files


def load_completion_file(file_path):
    """
    Load a single completion file and return DataFrame.
    
    Args:
        file_path (str): Path to the completion file
        
    Returns:
        pd.DataFrame: DataFrame with columns (problem, completion_tokens)
    """
    try:
        with open(file_path, "r") as f:
            completions_data = [json.loads(line) for line in f]
        
        df = pd.DataFrame(completions_data)
        
        # Extract only the required columns
        if "problem" in df.columns and "completion_tokens" in df.columns:
            return df[["problem", "completion_tokens"]]
        else:
            print(f"Warning: Missing required columns in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def concatenate_completions(base_dir, output_file=None):
    """
    Main function to concatenate all completion files.
    
    Args:
        base_dir (str): Base directory to search in
        output_file (str, optional): Path to save the concatenated DataFrame
        
    Returns:
        pd.DataFrame: Concatenated DataFrame
    """
    # Find all completion files
    files = find_completion_files(base_dir)
    
    if not files:
        print("No completion files found!")
        return pd.DataFrame()
    
    # Load each file
    dataframes = []
    total_rows = 0
    
    for file_path in files:
        print(f"Loading {file_path}...")
        df = load_completion_file(file_path)
        
        if not df.empty:
            dataframes.append(df)
            total_rows += len(df)
            print(f"  Loaded {len(df)} rows")
        else:
            print(f"  Skipped (empty or error)")
    
    if not dataframes:
        print("No valid data loaded!")
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    print(f"\nConcatenating {len(dataframes)} DataFrames...")
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Final DataFrame shape: {concatenated_df.shape}")
    print(f"Total rows: {total_rows}")
    
    # Save if output file specified
    if output_file:
        print(f"Saving to {output_file}...")
        concatenated_df.to_csv(output_file, index=False)
        print(f"Saved successfully!")
    
    return concatenated_df


def main():
    parser = argparse.ArgumentParser(description="Concatenate completion files")
    parser.add_argument(
        "--base-dir", 
        default="/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct",
        help="Base directory to search for completion files"
    )
    parser.add_argument(
        "--output", 
        default="/dccstor/gma2/jhjenny9/search-and-learn/training-res/numinaMath/bon_token_counts_16000.csv",
        help="Output CSV file path"
    )
    
    args = parser.parse_args()
    
    print(f"Searching for completion files in: {args.base_dir}")
    print(f"Output will be saved to: {args.output}")
    print("-" * 50)
    
    # Run the concatenation
    df = concatenate_completions(args.base_dir, args.output)
    
    if not df.empty:
        print("\nFirst few rows of concatenated DataFrame:")
        print(df.head())
        print(f"\nDataFrame info:")
        print(df.info())


if __name__ == "__main__":
    main()
