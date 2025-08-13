### This script take in a series of completions.jsonl files and generates a unified dataframe with columns "problem", "sl", "method", "N", for probe training.


import pandas as pd
import subprocess
import os
import sys
import tempfile
import shutil
import json



def extract_sl_method_N(file_path):
    """
    Reads a JSONL file and for each line, creates 3 rows for each of the keys:
    'sl_weighted_N', 'sl_naive_N', 'sl_maj_N', extracting the method and N from the key.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        pd.DataFrame: DataFrame with columns "problem", "sl", "method", "N"
    """
    data = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                problem = entry.get("problem")
                for key in entry:
                    if key.startswith("sl_"):
                        parts = key.split("_")
                        if len(parts) == 3:
                            _, method, n = parts
                            sl_value = entry[key]
                            data.append({
                                "problem": problem,
                                "sl": sl_value,
                                "method": method,
                                "N": int(n)
                            })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    return pd.DataFrame(data)

def run_generate_softlabels(input_jsonl, output_file, n_bootstrap, sample_size): # for each input, we will generate softlabel_N.jsonl for each N, which will create keys ['problem', 'sl_weighted_N', 'sl_naive_N', 'sl_maj_N']. we will then run this for each input file, which results in softlabel_N_i.jsonl.
    """Run the generate_softlabels.py script as a subprocess"""
    cmd = [
        "python", "scripts/data_processing/generate_softlabels.py",
        "--input_jsonl", input_jsonl,
        "--output_file", output_file,
        "--n_bootstrap", str(n_bootstrap),
        "--sample_size", str(sample_size)
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

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Create parallel dataset from multiple input files")
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap samples to generate per problem (default: 30)"
    )
    args = parser.parse_args()
    
    # Input file paths
    input_files = [
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32_numina_second_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl", # contains sl_maj_2, sl_weighted_2, sl_naive_2, sl_maj_4, sl_weighted_4, sl_naive_4, sl_maj_8, sl_weighted_8, sl_naive_8, sl_maj_16, sl_weighted_16, sl_naive_16, sl_maj_32, sl_weighted_32, sl_naive_32
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32_numina_third_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl" # contains sl_maj_2, sl_weighted_2, sl_naive_2, sl_maj_4, sl_weighted_4, sl_naive_4, sl_maj_8, sl_weighted_8, sl_naive_8, sl_maj_16, sl_weighted_16, sl_naive_16, sl_maj_32, sl_weighted_32, sl_naive_32
        # "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32_numina_fourth_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl",
        # "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32_numina_fifth_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl",
        # "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32_numina_sixth_chunk/best_of_n_completions.jsonl/best_of_n_completions.jsonl"
    ]
    
    # N values to process
    n_values = [2, 4] # 8, 16, 32
    
    # Create output directory
    output_dir = "/dccstor/gma2/jhjenny9/search-and-learn/data/sl/parallel"
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    softlabels_parallel_df = pd.DataFrame()
    
    try:
        for i, input_file_path in enumerate(input_files):
            print(f"\nProcessing input file {i+1}/{len(input_files)}: {os.path.basename(input_file_path)}")
            
            # Extract chunk name for method identification
            chunk_name = os.path.basename(os.path.dirname(input_file_path))
            
            for n in n_values:
                print(f"  Processing N={n}")
                
                # Create temporary output file for softlabels
                temp_softlabels_file = os.path.join(temp_dir, f"softlabels_{n}_{i}.jsonl")
                
                try:
                    # Run generate_softlabels.py for each N
                    run_generate_softlabels(
                        input_file_path, 
                        temp_softlabels_file, 
                        n_bootstrap=args.n_bootstrap, 
                        sample_size=n
                    )
                    
                    # Extract soft labels using extract_sl_maj_N
                    softlabels_n = extract_sl_method_N(temp_softlabels_file)
                    
                    
                    # Concatenate to main dataframe
                    softlabels_parallel_df = pd.concat([softlabels_parallel_df, softlabels_n], ignore_index=True)
                    
                    print(f"    Successfully processed N={n}")
                    
                except Exception as e:
                    print(f"    Error processing N={n}: {e}")
                    continue
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_softlabels_file):
                        os.remove(temp_softlabels_file)
        
        # Save the combined dataset
        output_file = os.path.join(output_dir, "softlabels_parallel_df.csv")
        softlabels_parallel_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully created combined dataset:")
        print(f"Output file: {output_file}")
        print(f"Dataset shape: {softlabels_parallel_df.shape}")
        print(f"Columns: {list(softlabels_parallel_df.columns)}")
        print(f"Methods: {softlabels_parallel_df['method'].unique()}")
        print(f"N values: {sorted(softlabels_parallel_df['N'].unique())}")
        
        # Show sample of the data
        print(f"\nSample data:")
        print(softlabels_parallel_df.head())
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()