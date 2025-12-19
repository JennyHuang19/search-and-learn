#!/usr/bin/env python3
"""
Script to submit create_df_beam.py jobs in batches of 2 files.
This helps avoid memory issues and makes the jobs more manageable.
"""

import subprocess
import os
import time
from typing import List

def submit_beam_job(input_files: List[str], job_name: str, n_bootstrap: int = 30, memory: str = "100G"):
    """
    Submit a single LSF job for processing a batch of input files.
    
    Args:
        input_files: List of 2 input file paths
        job_name: Name for the LSF job
        n_bootstrap: Number of bootstrap samples (default: 30)
        memory: Memory request for LSF (default: "100G")
    """
    
    # Build the LSF command
    cmd = [
        "bsub",
        "-gpu", "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB",
        "-M", memory,
        "-J", job_name,
        "-oo", f"{job_name}.out",
        "-eo", f"{job_name}.err",
        "python", "scripts/data_processing/create_df_beam.py",
        "--n_bootstrap", str(n_bootstrap),
        "--input_files"
    ] + input_files
    
    print(f"Submitting job: {job_name}")
    print(f"Files: {len(input_files)} files")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Job submitted successfully: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {job_name}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    # Clean input files list (fixed formatting issues)
    input_files = [
        '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_second_chunk/beam_search_completions.jsonl',
        '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_third_chunk/beam_search_completions.jsonl',
        '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_fourth_chunk/beam_search_completions.jsonl',
        '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_fifth_chunk/beam_search_completions.jsonl',
        '/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_2_4_40_sixth_chunk/beam_search_completions.jsonl',
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_4_4_40_second_chunk/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_4_4_40_third_chunk/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_4_4_40_fourth_chunk/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_4_4_40_fifth_chunk/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/Numina_beam_search_4_4_40_sixth_chunk/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_22000_23000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_23000_24000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_24000_25000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_25000_26000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_26000_27000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_27000_28000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_28000_29000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_8_4_40_chunk_29000_30000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_11000_12000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_12000_13000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_13000_14000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_14000_15000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_15000_16000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_20000_21000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_21000_22000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_22000_23000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_23000_24000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_24000_25000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_25000_26000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_26000_27000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_27000_28000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_28000_29000/beam_search_completions.jsonl",
        "/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam/beam_search_16_4_40_chunk_29000_30000/beam_search_completions.jsonl"
    ]
    
    print(f"Total input files: {len(input_files)}")
    print(f"Will submit jobs in batches of 2")

    def extract_chunk_range(dirname):
            # Looks for chunk_{start}_{end} in the directory name
            import re
            match = re.search(r'chunk_(\d+)_(\d+)', dirname)
            if match:
                return f"chunk_{match.group(1)}_{match.group(2)}"
            else:
                return dirname  # fallback
    
    # Group files into batches of 2
    batches = []
    for i in range(0, len(input_files), 2):
        batch = input_files[i:i+2]
        batches.append(batch)
    
    print(f"Number of batches: {len(batches)}")
    
    # Submit jobs for each batch
    successful_submissions = 0
    for i, batch in enumerate(batches):
        # Create job name based on the files in the batch
        first_file = os.path.basename(os.path.dirname(batch[0]))
        second_file = os.path.basename(os.path.dirname(batch[1])) if len(batch) > 1 else "single"
        
        # Extract the chunk range from the directory name for the job name.
        chunk1 = extract_chunk_range(os.path.basename(os.path.dirname(batch[0])))
        chunk2 = extract_chunk_range(os.path.basename(os.path.dirname(batch[1]))) if len(batch) > 1 else "single"
        job_name = f"beam-batch-{i+1:02d}-{chunk1}-{chunk2}"
        
        # Submit the job
        if submit_beam_job(batch, job_name):
            successful_submissions += 1
        
        # Small delay between submissions to avoid overwhelming the scheduler
        if i < len(batches) - 1:  # Don't sleep after the last job
            print(f"Waiting 2 seconds before next submission...")
            time.sleep(2)
    
    print(f"\nJob submission complete!")
    print(f"Successfully submitted: {successful_submissions}/{len(batches)} batches")
    
    if successful_submissions == len(batches):
        print("All jobs submitted successfully!")
    else:
        print(f"Some jobs failed to submit. Check the error messages above.")

if __name__ == "__main__":
    main()
