### This script take in a series of completions.jsonl files and generates a unified dataframe with columns "problem", "method", "N", "sl". This generated dataframe is used for probe training.
import pandas as pd
import subprocess
import os
import sys
import tempfile
import shutil
import json
import re

def extract_n_from_path(file_path):
    """
    Extract the n value from file paths with different naming conventions.
    
    Examples:
    - beam_search_N_4_40_chunk_...
    - Numina_beam_search_N_4_40_...
    
    Args:
        file_path (str): Path to extract n from
        
    Returns:
        int: The extracted n value, or None if not found
    """
    # Pattern 1: beam_search_N_4_40_chunk_...
    pattern1 = r'beam_search_(\d+)_4_40'
    
    # Pattern 2: Numina_beam_search_N_4_40_...
    pattern2 = r'Numina_beam_search_(\d+)_4_40'
    
    # Try pattern 1 first
    match = re.search(pattern1, file_path)
    if match:
        return int(match.group(1))
    
    # Try pattern 2
    match = re.search(pattern2, file_path)
    if match:
        return int(match.group(1))
    
    # If no pattern matches, return None
    print(f"Warning: Could not extract n value from path: {file_path}")
    return None


def run_generate_softlabels(input_jsonl, output_file, n_bootstrap, sample_size): # for each input, we will generate softlabel_N.jsonl for each N, which will create keys ['problem', 'sl_weighted_N', 'sl_naive_N', 'sl_maj_N']. we will then run this for each input file, which results in softlabel_N_i.jsonl.
    """Run the generate_softlabels.py script as a subprocess"""
    cmd = [
        "python", "scripts/data_processing/generate_softlabels.py",
        "--input_jsonl", input_jsonl, # next file.
        "--output_file", output_file,
        "--n_bootstrap", str(n_bootstrap),
        "--sample_size", str(sample_size) # the 8 in beam_search_8_4_40.
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running generate_softlabels.py:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"generate_softlabels.py failed with return code {result.returncode}")
    
    print(f"Successfully generated: {output_file}")
    return result.stdout

def extract_sl_N_beam(file_path, n):
    """
    Reads a softlabel JSONL file and extracts the "sl_maj_N" field from each line.
    (For the beam search method, the final aggregation method used is always "maj," or majority voting.)
    
    Args:
        file_path (str): Path to the JSONL file
        n (int): The N value to extract (e.g., 2 for "sl_maj_2", 4 for "sl_maj_4")
        
    Returns:
        pd.DataFrame: DataFrame with columns "problem", "method", "N", "sl"
        
    Raises:
        FileNotFoundError: If the file_path doesn't exist
        ValueError: If n is not a positive integer
    """
    data = []
    line_count = 0
    error_count = 0
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                # Extract sl, method, n.
                sl = entry.get(f"sl_maj_{n}")
                data.append({
                    "problem": entry.get("problem"),
                    "method": "beam_search",
                    "N": n,
                    "sl": sl
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    return pd.DataFrame(data)

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create unified dataframe from beam search completion files")
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap samples to generate per problem (default: 30)"
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        default=[
            '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_fifth_chunk/beam_search_completions.jsonl',
            '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_sixth_chunk/beam_search_completions.jsonl',
            "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_4_4_40_second_chunk/beam_search_completions.jsonl"
        ],
        help="List of input JSONL files to process (default: three specific Numina beam search files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/dccstor/gma2/jhjenny9/search-and-learn/data/sl/beam",
        help="Output directory for the combined dataset (default: /dccstor/gma2/jhjenny9/search-and-learn/data/sl/beam)"
    )
    args = parser.parse_args()
    
    # Input file paths from command line arguments
    input_files = args.input_files
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    softlabels_beam_df = pd.DataFrame()
    
    try:
        for i, input_file_path in enumerate(input_files):
            print(f"\nProcessing input file {i+1}/{len(input_files)}: {os.path.basename(input_file_path)}")
            
            # Extract n from the file name.
            n = extract_n_from_path(input_file_path)
        
            print(f"  Processing N={n}")
            
            # Create temporary output file for softlabels
            temp_softlabels_file = os.path.join(temp_dir, f"softlabels_{n}_{i}.jsonl")
            
            try:
                # Run generate_softlabels.py for each file.
                run_generate_softlabels(
                    input_file_path, 
                    temp_softlabels_file, 
                    n_bootstrap=args.n_bootstrap, 
                    sample_size=n # ensures that the correct n is used for the current beam search file.
                )
                
                # Extract appropriate columns from the soft label file to create a dataframe with columns "problem", "method", "N", "sl".
                softlabels_n = extract_sl_N_beam(temp_softlabels_file, n)
                # print N for bookkeeping.
                print(f"processing N: {n}, the {i}th file.") # 4,4,4,4, 8,8,8 etc.
                print(f"the shape of the softlabels dataframe is {softlabels_n.shape}")
                
                # Concatenate to main dataframe
                softlabels_beam_df = pd.concat([softlabels_beam_df, softlabels_n], ignore_index=True)
                
                print(f"    Successfully processed N={n}")
                
            except Exception as e:
                print(f"    Error processing N={n}: {e}")
                continue
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_softlabels_file):
                    os.remove(temp_softlabels_file)
        
        # Save the combined dataset
        # Use the chunk directory name from each input file to generate the output file name
        def extract_chunk_dir_name(path):
            # Get the parent directory name containing the file
            return os.path.basename(os.path.dirname(path))
        chunk_dir_names = [extract_chunk_dir_name(f) for f in input_files] # job name isall of the chunk directory names combined.
        chunk_str = "-".join(chunk_dir_names)
        output_file = os.path.join(output_dir, f"softlabels_beam_df_{chunk_str}.csv")
        softlabels_beam_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully created combined dataset:")
        print(f"Output file: {output_file}")
        print(f"Dataset shape: {softlabels_beam_df.shape}")
        print(f"Columns: {list(softlabels_beam_df.columns)}")
        print(f"Methods: {softlabels_beam_df['method'].unique()}")
        print(f"N values: {sorted(softlabels_beam_df['N'].unique())}")
        
        # Show sample of the data
        print(f"\nSample data:")
        print(softlabels_beam_df.head())
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()