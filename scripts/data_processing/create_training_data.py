#!/usr/bin/env python3
"""
Script to join all softlabels_beam CSV files into one unified training dataset.
This creates a single CSV file with columns: question, N, method, SL.
"""

import pandas as pd
import os
import glob
import argparse
from pathlib import Path

def find_csv_files(input_dir):
    """
    Find all relevant CSV files in the input directory.
    Includes both softlabels_beam and parallel_df files, 
    which are expected to be in 'beam' and 'parallel' subdirectories, respectively.

    Args:
        input_dir (str): Directory containing 'beam' and 'parallel' subdirectories

    Returns:
        list: List of CSV file paths
    """
    # Find beam search files in the 'beam' subdirectory
    beam_dir = os.path.join(input_dir, "beam")
    beam_pattern = os.path.join(beam_dir, "softlabels_beam*.csv")
    beam_files = glob.glob(beam_pattern)

    # Find parallel files in the 'parallel' subdirectory
    parallel_dir = os.path.join(input_dir, "parallel")
    parallel_pattern = os.path.join(parallel_dir, "parallel_df_*_chunk*.csv")
    parallel_files = glob.glob(parallel_pattern)

    # Combine and sort all files
    all_files = beam_files + parallel_files
    all_files = sorted(all_files)

    if not all_files:
        print(f"Warning: No relevant CSV files found in {input_dir}/beam or {input_dir}/parallel")
        return []

    print(f"Found {len(all_files)} CSV files:")
    print(f"  Beam search files ({len(beam_files)}):")
    for file in beam_files:
        print(f"    - {os.path.basename(file)}")
    print(f"  Parallel files ({len(parallel_files)}):")
    for file in parallel_files:
        print(f"    - {os.path.basename(file)}")

    return all_files

def load_and_clean_csv(file_path):
    """
    Load a CSV file and ensure it has the expected columns.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check if the DataFrame has the expected columns
        expected_columns = ['problem', 'method', 'N', 'sl']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: {os.path.basename(file_path)} is missing columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Rename columns to match the desired output format
        df = df.rename(columns={
            'problem': 'question',
            'sl': 'SL'
        })
        
        # Ensure the DataFrame has the correct column order
        df = df[['question', 'N', 'method', 'SL']]
        
        print(f"  Loaded {len(df)} rows from {os.path.basename(file_path)}")
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Join all softlabels_beam and parallel_df CSV files into one unified training dataset")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/dccstor/gma2/jhjenny9/search-and-learn/data/sl",
        help="Directory containing 'beam' and 'parallel' subdirectories with CSV files (default: /dccstor/gma2/jhjenny9/search-and-learn/data/sl)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/dccstor/gma2/jhjenny9/search-and-learn/data/sl/combined_training_data.csv",
        help="Output CSV file path (default: /dccstor/gma2/jhjenny9/search-and-learn/data/sl/combined_training_data.csv)"
    )
    parser.add_argument(
        "--remove_duplicates",
        action="store_true",
        help="Remove duplicate question-N-method combinations"
    )
    
    args = parser.parse_args()
    
    print(f"Searching for softlabels_beam CSV files in: {args.input_dir}")
    
    # Find all CSV files
    csv_files = find_csv_files(args.input_dir)
    
    if not csv_files:
        print("No CSV files found. Exiting.")
        return
    
    # Load and combine all CSV files
    all_dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        df = load_and_clean_csv(csv_file)
        if df is not None:
            all_dataframes.append(df)
            total_rows += len(df)
    
    if not all_dataframes:
        print("No valid CSV files could be loaded. Exiting.")
        return
    
    # Combine all DataFrames
    print(f"\nCombining {len(all_dataframes)} DataFrames...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Total rows loaded: {total_rows}")
    
    # Check for any data quality issues
    print(f"\nData quality check:")
    print(f"  - Questions: {combined_df['question'].nunique()} unique")
    print(f"  - N values: {sorted(combined_df['N'].unique())}")
    print(f"  - Methods: {combined_df['method'].unique()}")
    print(f"  - Missing values: {combined_df.isnull().sum().sum()}")
    
    # Remove duplicates if requested
    if args.remove_duplicates:
        initial_rows = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['question', 'N', 'method'], keep='first')
        final_rows = len(combined_df)
        print(f"  - Duplicates removed: {initial_rows - final_rows} rows")
        print(f"  - Final shape: {combined_df.shape}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the combined dataset
    print(f"\nSaving combined dataset to: {args.output_file}")
    combined_df.to_csv(args.output_file, index=False)
    
    print(f"\nSuccessfully created training dataset!")
    print(f"Output file: {args.output_file}")
    print(f"Final shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Show sample of the data
    print(f"\nSample data:")
    print(combined_df.head(10))
    
    # Summary statistics
    print(f"\nSummary by N value:")
    for n in sorted(combined_df['N'].unique()):
        n_data = combined_df[combined_df['N'] == n]
        print(f"  N={n}: {len(n_data)} rows")
    
    print(f"\nSummary by method:")
    for method in combined_df['method'].unique():
        method_data = combined_df[combined_df['method'] == method]
        print(f"  {method}: {len(method_data)} rows")

if __name__ == "__main__":
    main()
