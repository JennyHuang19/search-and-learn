#!/usr/bin/env python3
"""
Script to submit beam search jobs in chunks of 1000 questions.
Updates YAML configuration for each chunk and submits LSF jobs.
"""

import os
import yaml
import subprocess
import argparse
from pathlib import Path
import time


def update_yaml_config(yaml_path, yaml_template, dataset_start, dataset_end, num_samples, output_dir):
    """Update YAML configuration with new dataset range and output directory"""
    # Copy the template YAML file
    subprocess.run(['cp', yaml_template, yaml_path], check=True)
    
    # Read the copied YAML file
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the configuration
    config['dataset_start'] = dataset_start
    config['dataset_end'] = dataset_end
    config['output_dir'] = output_dir
    
    # Remove num_samples if it exists to avoid conflicts
    if 'num_samples' in config:
        del config['num_samples']
    
    # Write the updated configuration back
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated {yaml_path} with dataset_start={dataset_start}, dataset_end={dataset_end}, output_dir={output_dir}")


def submit_job(job_name, yaml_path):
    """
    Submit a beam search job using LSF.
    
    Args:
        job_name (str): Name for the LSF job
        yaml_path (str): Path to the YAML configuration file
    """
    # Construct the bsub command
    cmd = [
        'bsub',
        '-gpu', 'num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB',
        '-M', '100G',
        '-J', job_name,
        '-oo', f'{job_name}.out',
        '-eo', f'{job_name}.err',
        'python', 'scripts/test_time_compute.py', yaml_path
    ]
    
    print(f"Submitting job: {' '.join(cmd)}")
    
    # Submit the job
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Job submitted successfully: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Submit beam search jobs in chunks')
    parser.add_argument('--yaml-template', 
                       default='/dccstor/gma2/jhjenny9/search-and-learn/recipes/Qwen2.5-1.5B-Instruct/beam_search_16_4_40.yaml',
                       help='Path to the YAML template file')
    parser.add_argument('--start-chunk', type=int, default=30000,
                       help='Starting chunk index (default: 5000)')
    parser.add_argument('--end-chunk', type=int, default=50000,
                       help='Ending chunk index (default: 10000)')
    parser.add_argument('--chunk-size', type=int, default=2000,
                       help='Size of each chunk (default: 1000)')
    parser.add_argument('--base-output-dir',
                       default='/dccstor/gma2/jhjenny9/search-and-learn/data/Numina_Beam',
                       help='Base output directory path')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would be done without actually submitting jobs')
    
    args = parser.parse_args()
    
    # Check if YAML template exists
    if not os.path.exists(args.yaml_template):
        print(f"Error: YAML template file not found: {args.yaml_template}")
        return
    
    # Loop through chunks
    for chunk_start in range(args.start_chunk, args.end_chunk, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, args.end_chunk)
        
        # Create unique YAML file for this chunk
        chunk_yaml = f"beam_search_16_4_40_{chunk_start}_{chunk_end}.yaml"
        chunk_yaml_path = os.path.abspath(chunk_yaml)
        
        # Create output directory name - match the YAML naming convention
        chunk_name = f"beam_search_16_4_40_chunk_{chunk_start}_{chunk_end}"
        output_dir = os.path.join(args.base_output_dir, chunk_name)
        
        # Create job name
        job_name = f"Beam-16-{chunk_start}-{chunk_end}"
        
        print(f"\n{'='*60}")
        print(f"Processing chunk: {chunk_start} to {chunk_end}")
        print(f"YAML file: {chunk_yaml}")
        print(f"Output directory: {output_dir}")
        print(f"Job name: {job_name}")
        print(f"{'='*60}")
        
        if args.dry_run:
            print("DRY RUN - Would update YAML and submit job")
            continue
        
        # Update the YAML configuration for this chunk
        update_yaml_config(chunk_yaml_path, args.yaml_template, chunk_start, chunk_end, args.chunk_size, output_dir)
        
        # Submit the job with the chunk-specific YAML
        submit_job(job_name, chunk_yaml_path)
        
        # Wait a bit before submitting the next job
        if chunk_end < args.end_chunk:  # Don't wait after the last job
            print(f"Waiting 5 seconds before next job...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("All jobs submitted successfully!")
    print(f"Total chunks processed: {(args.end_chunk - args.start_chunk) // args.chunk_size}")
    print(f"{'='*60}")
    
    # Clean up chunk YAML files
    print("\nCleaning up chunk YAML files...")
    for chunk_start in range(args.start_chunk, args.end_chunk, args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, args.end_chunk)
        chunk_yaml = f"beam_search_16_4_40_{chunk_start}_{chunk_end}.yaml"
        if os.path.exists(chunk_yaml):
            os.remove(chunk_yaml)
            print(f"Removed {chunk_yaml}")
    
    print("Cleanup complete!")


if __name__ == "__main__":
    main()
