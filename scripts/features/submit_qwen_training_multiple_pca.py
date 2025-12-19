#!/usr/bin/env python3
"""
Submit Qwen training and validation jobs for multiple PCA dimensions.
This script submits LSF jobs to train Qwen models with different PCA component counts
and then validates them using the fitted scalers and PCA transformations.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path


def submit_qwen_training_job(
    csv_path: str,
    output_dir: str,
    model_name: str,
    pca_components: int,
    aux_weight_multiplier: float = 1.0,
    pooling_method: str = "mean",
    memory: str = "100G",
    gpu: str = "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
):
    """Submit a Qwen training job for a specific PCA dimension."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file paths
    output_npy = os.path.join(output_dir, f"qwen_train_pca{pca_components}.npy")
    output_meta = os.path.join(output_dir, f"qwen_train_pca{pca_components}_meta.json")
    
    # Create job name
    job_name = f"QWEN-train-pca-{pca_components}"
    
    # Create output file names
    out_file = os.path.join(output_dir, f"{job_name}.out")
    err_file = os.path.join(output_dir, f"{job_name}.err")
    
    # Build the command
    cmd = [
        "bsub",
        "-gpu", gpu,
        "-M", memory,
        "-J", job_name,
        "-oo", out_file,
        "-eo", err_file,
        "python", "scripts/features/get_reduced_qwen.py",
        "--csv", csv_path,
        "--out", output_npy,
        "--meta", output_meta,
        "--model", model_name,
        "--pca-components", str(pca_components),
        "--aux-weight-multiplier", str(aux_weight_multiplier),
        "--pooling", pooling_method,
        "--fit-scaler"
    ]
    
    print(f"Submitting training job for PCA {pca_components}:")
    print(f"  Job name: {job_name}")
    print(f"  Output dir: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    
    # Submit the job
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✅ Job submitted successfully")
        print(f"  Output: {result.stdout.strip()}")
        return True
    else:
        print(f"  ❌ Job submission failed")
        print(f"  Error: {result.stderr.strip()}")
        return False


def submit_qwen_validation_job(
    csv_path: str,
    output_dir: str,
    model_name: str,
    pca_components: int,
    train_meta_path: str,
    aux_weight_multiplier: float = 1.0,
    pooling_method: str = "mean",
    memory: str = "100G",
    gpu: str = "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB"
):
    """Submit a Qwen validation job for a specific PCA dimension."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file paths
    output_npy = os.path.join(output_dir, f"qwen_val_pca{pca_components}.npy")
    output_meta = os.path.join(output_dir, f"qwen_val_pca{pca_components}_meta.json")
    
    # Create job name
    job_name = f"QWEN-val-pca-{pca_components}"
    
    # Create output file names
    out_file = os.path.join(output_dir, f"{job_name}.out")
    err_file = os.path.join(output_dir, f"{job_name}.err")
    
    # Build the command
    cmd = [
        "bsub",
        "-gpu", gpu,
        "-M", memory,
        "-J", job_name,
        "-oo", out_file,
        "-eo", err_file,
        "python", "scripts/features/get_reduced_qwen.py",
        "--csv", csv_path,
        "--out", output_npy,
        "--meta", output_meta,
        "--model", model_name,
        "--pca-components", str(pca_components),
        "--aux-weight-multiplier", str(aux_weight_multiplier),
        "--pooling", pooling_method,
        "--scaler", train_meta_path
    ]
    
    print(f"Submitting validation job for PCA {pca_components}:")
    print(f"  Job name: {job_name}")
    print(f"  Output dir: {output_dir}")
    print(f"  Train meta: {train_meta_path}")
    print(f"  Command: {' '.join(cmd)}")
    
    # Submit the job
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✅ Job submitted successfully")
        print(f"  Output: {result.stdout.strip()}")
        return True
    else:
        print(f"  ❌ Job submission failed")
        print(f"  Error: {result.stderr.strip()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit Qwen training and validation jobs for multiple PCA dimensions"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--val-csv", 
        type=str,
        required=True,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for all jobs"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Qwen model name (default: Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--pca-dimensions",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="PCA dimensions to test (default: 32 64 128 256)"
    )
    parser.add_argument(
        "--aux-weight-multiplier",
        type=float,
        default=1.0,
        help="Auxiliary weight multiplier for group reweighting (default: 1.0)"
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Pooling method for token embeddings (default: mean)"
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="100G",
        help="Memory request for LSF jobs (default: 100G)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB",
        help="GPU specification for LSF jobs"
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Only submit training jobs, skip validation"
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Only submit validation jobs, skip training"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay between job submissions in seconds (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.training_only and args.validation_only:
        print("Error: Cannot specify both --training-only and --validation-only")
        exit(1)
    
    # Create base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Qwen Multi-PCA Job Submission ===")
    print(f"Training CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Output directory: {base_output_dir}")
    print(f"Model: {args.model_name}")
    print(f"PCA dimensions: {args.pca_dimensions}")
    print(f"Auxiliary weight multiplier: {args.aux_weight_multiplier}")
    print(f"Pooling method: {args.pooling_method}")
    print(f"Memory request: {args.memory}")
    print(f"GPU spec: {args.gpu}")
    print()
    
    # Submit training jobs
    if not args.validation_only:
        print("=== Submitting Training Jobs ===")
        for pca_dim in args.pca_dimensions:
            # Create output directory for this PCA dimension
            output_dir = base_output_dir / f"pca_{pca_dim}"
            
            success = submit_qwen_training_job(
                csv_path=args.train_csv,
                output_dir=str(output_dir),
                model_name=args.model_name,
                pca_components=pca_dim,
                aux_weight_multiplier=args.aux_weight_multiplier,
                pooling_method=args.pooling_method,
                memory=args.memory,
                gpu=args.gpu
            )
            
            if success:
                print(f"  Training job for PCA {pca_dim} submitted successfully")
            else:
                print(f"  Failed to submit training job for PCA {pca_dim}")
                continue
            
            # Add delay between submissions
            if args.delay > 0:
                time.sleep(args.delay)
        
        print()
    
    # Submit validation jobs
    if not args.training_only:
        print("=== Submitting Validation Jobs ===")
        for pca_dim in args.pca_dimensions:
            # Create output directory for this PCA dimension
            output_dir = base_output_dir / f"pca_{pca_dim}"
            
            # Check if training metadata exists
            train_meta_path = output_dir / f"qwen_train_pca{pca_dim}_meta.json"
            
            if not train_meta_path.exists():
                print(f"  ⚠️  Training metadata not found for PCA {pca_dim}: {train_meta_path}")
                print(f"  Skipping validation job for PCA {pca_dim}")
                continue
            
            success = submit_qwen_validation_job(
                csv_path=args.val_csv,
                output_dir=str(output_dir),
                model_name=args.model_name,
                pca_components=pca_dim,
                train_meta_path=str(train_meta_path),
                aux_weight_multiplier=args.aux_weight_multiplier,
                pooling_method=args.pooling_method,
                memory=args.memory,
                gpu=args.gpu
            )
            
            if success:
                print(f"  Validation job for PCA {pca_dim} submitted successfully")
            else:
                print(f"  Failed to submit validation job for PCA {pca_dim}")
            
            # Add delay between submissions
            if args.delay > 0:
                time.sleep(args.delay)
        
        print()
    
    print("=== Job Submission Complete ===")
    print(f"All jobs submitted to output directory: {base_output_dir}")
    print()
    print("Job status can be checked with:")
    print(f"  bjobs -a | grep 'QWEN-'")
    print()
    print("Output files will be in:")
    for pca_dim in args.pca_dimensions:
        output_dir = base_output_dir / f"pca_{pca_dim}"
        print(f"  PCA {pca_dim}: {output_dir}")
    
    if not args.validation_only:
        print()
        print("Note: Validation jobs will wait for training metadata files to be created.")
        print("You may need to wait for training jobs to complete before validation jobs can start.")


if __name__ == "__main__":
    main()
