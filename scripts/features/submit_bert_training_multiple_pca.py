#!/usr/bin/env python3
"""
Script to submit BERT training jobs for multiple PCA dimensions.
This submits jobs for PCA dimensions: 32, 64, 128, 256, and no PCA.
Also includes validation jobs that use the fitted scalers and PCA from training.
"""

import subprocess
import os
import time
import argparse
from typing import List, Optional

def submit_bert_training_job(
    csv_path: str,
    output_dir: str,
    pca_components: Optional[int],
    model_path: str,
    job_name: str,
    memory: str = "250G",
    aux_weight_multiplier: float = 1.0
) -> bool:
    """
    Submit a single LSF job for BERT training with specified PCA dimensions.
    
    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save output files
        pca_components: Number of PCA components (None for no PCA)
        model_path: Path to BERT model
        job_name: Name for the LSF job
        memory: Memory request for LSF
        aux_weight_multiplier: Multiplier for auxiliary feature weights
        
    Returns:
        bool: True if job submitted successfully, False otherwise
    """
    
    # Create output file names
    if pca_components is None:
        # No PCA case
        output_npy = os.path.join(output_dir, "train_features_BERT_no_pca.npy")
        meta_json = os.path.join(output_dir, "train_meta_no_pca.json")
        pca_arg = ["--pca-components", "none"]  # Use 'none' string for CLI
    else:
        # PCA case
        output_npy = os.path.join(output_dir, f"train_features_BERT_pca_{pca_components}.npy")
        meta_json = os.path.join(output_dir, f"train_meta_pca_{pca_components}.json")
        pca_arg = ["--pca-components", str(pca_components)]
    
    # Build the LSF command
    cmd = [
        "bsub",
        "-gpu", "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB",
        "-M", memory,
        "-J", job_name,
        "-oo", f"{job_name}.out",
        "-eo", f"{job_name}.err",
        "python", "scripts/features/get_reduced_bert.py",
        "--csv", csv_path,
        "--out", output_npy,
        "--meta", meta_json,
        "--model", model_path,
        "--enable-gradients",
        "--fit-scaler",
        "--aux-weight-multiplier", str(aux_weight_multiplier)
    ] + pca_arg
    
    print(f"Submitting training job: {job_name}")
    print(f"PCA components: {pca_components if pca_components else 'No PCA'}")
    print(f"Output file: {output_npy}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Training job submitted successfully: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error submitting training job {job_name}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def submit_bert_validation_job(
    csv_path: str,
    output_dir: str,
    pca_components: Optional[int],
    model_path: str,
    train_meta_path: str,
    job_name: str,
    memory: str = "100G",
    aux_weight_multiplier: float = 1.0
) -> bool:
    """
    Submit a single LSF job for BERT validation using fitted scaler and PCA from training.
    
    Args:
        csv_path: Path to validation CSV file
        output_dir: Directory to save output files
        pca_components: Number of PCA components (None for no PCA)
        model_path: Path to BERT model
        train_meta_path: Path to training metadata (contains fitted scaler and PCA)
        job_name: Name for the LSF job
        memory: Memory request for LSF
        aux_weight_multiplier: Multiplier for auxiliary feature weights
        
    Returns:
        bool: True if job submitted successfully, False otherwise
    """
    
    # Create output file names
    if pca_components is None:
        # No PCA case
        output_npy = os.path.join(output_dir, "val_features_BERT_no_pca.npy")
        pca_arg = ["--pca-components", "none"]  # Use 'none' string for CLI
    else:
        # PCA case
        output_npy = os.path.join(output_dir, f"val_features_BERT_pca_{pca_components}.npy")
        pca_arg = ["--pca-components", str(pca_components)]
    
    # Build the LSF command for validation (no --fit-scaler, uses --scaler instead)
    cmd = [
        "bsub",
        "-gpu", "num=1/task:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB",
        "-M", memory,
        "-J", job_name,
        "-oo", f"{job_name}.out",
        "-eo", f"{job_name}.err",
        "python", "scripts/features/get_reduced_bert.py",
        "--csv", csv_path,
        "--out", output_npy,
        "--model", model_path,
        "--scaler", train_meta_path,
        "--aux-weight-multiplier", str(aux_weight_multiplier)
    ] + pca_arg
    
    print(f"Submitting validation job: {job_name}")
    print(f"PCA components: {pca_components if pca_components else 'No PCA'}")
    print(f"Output file: {output_npy}")
    print(f"Using scaler from: {train_meta_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Validation job submitted successfully: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error submitting validation job {job_name}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def submit_validation_only(
    val_csv_path: str,
    output_dir: str,
    model_path: str,
    pca_dimensions: List[Optional[int]],
    memory: str = "200G",
    aux_weight_multiplier: float = 1.0
) -> None:
    """
    Submit only validation jobs, assuming training jobs have already completed.
    
    Args:
        val_csv_path: Path to validation CSV file
        output_dir: Directory containing training metadata and where to save validation outputs
        model_path: Path to BERT model
        pca_dimensions: List of PCA dimensions to process
        memory: Memory request for LSF
        aux_weight_multiplier: Multiplier for auxiliary feature weights
    """
    
    print(f"Submitting BERT validation jobs only")
    print(f"Validation CSV: {val_csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    print(f"PCA dimensions: {[d if d is not None else 'No PCA' for d in pca_dimensions]}")
    print(f"Total validation jobs to submit: {len(pca_dimensions)}")
    print("-" * 80)
    
    # Check if training metadata files exist
    missing_metadata = []
    for pca_dim in pca_dimensions:
        if pca_dim is None:
            meta_path = os.path.join(output_dir, "train_meta_no_pca.json")
        else:
            meta_path = os.path.join(output_dir, f"train_meta_pca_{pca_dim}.json")
        
        if not os.path.exists(meta_path):
            missing_metadata.append(f"PCA {pca_dim if pca_dim is not None else 'No PCA'}: {meta_path}")
    
    if missing_metadata:
        print("WARNING: Some training metadata files are missing:")
        for missing in missing_metadata:
            print(f"  - {missing}")
        print("\nMake sure training jobs have completed before running validation.")
        print("You can check job status with: bjobs | grep 'BERT-train'")
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborting validation job submission.")
            return
    
    # Submit validation jobs
    successful_validation_submissions = 0
    
    for i, pca_dim in enumerate(pca_dimensions):
        # Create job name
        if pca_dim is None:
            job_name = f"BERT-val-no-pca"
            train_meta_path = os.path.join(output_dir, "train_meta_no_pca.json")
        else:
            job_name = f"BERT-val-pca-{pca_dim}"
            train_meta_path = os.path.join(output_dir, f"train_meta_pca_{pca_dim}.json")
        
        # Submit the validation job
        if submit_bert_validation_job(
            csv_path=val_csv_path,
            output_dir=output_dir,
            pca_components=pca_dim,
            model_path=model_path,
            train_meta_path=train_meta_path,
            job_name=job_name,
            memory=memory,
            aux_weight_multiplier=aux_weight_multiplier
        ):
            successful_validation_submissions += 1
        
        # Small delay between submissions
        if i < len(pca_dimensions) - 1:
            print(f"Waiting 3 seconds before next validation submission...")
            time.sleep(3)
    
    # Summary
    print("-" * 80)
    print(f"Validation job submission complete!")
    print(f"Successfully submitted: {successful_validation_submissions}/{len(pca_dimensions)} validation jobs")
    
    if successful_validation_submissions == len(pca_dimensions):
        print("All validation jobs submitted successfully!")
        print("\nValidation job names:")
        for pca_dim in pca_dimensions:
            if pca_dim is None:
                print(f"  - BERT-val-no-pca")
            else:
                print(f"  - BERT-val-pca-{pca_dim}")
        
        print(f"\nOutput files will be saved to: {output_dir}")
        print("Monitor job progress with: bjobs | grep 'BERT-val'")
        
    else:
        print(f"Some validation jobs failed to submit. Check the error messages above.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Submit BERT training OR validation jobs (not both)")
    parser.add_argument("--training-only", action="store_true", 
                       help="Submit only training jobs")
    parser.add_argument("--validation-only", action="store_true", 
                       help="Submit only validation jobs (assumes training is complete)")
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
        "--model-path",
        type=str,
        default="/u/jhjenny9/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        help="BERT model path (default: local bert-base-uncased)"
    )
    parser.add_argument(
        "--memory", 
        default="100G", 
        help="Memory request for jobs (default: 100G)"
    )
    args = parser.parse_args()
    
    # Configuration from command line arguments
    train_csv_path = args.train_csv
    val_csv_path = args.val_csv
    output_dir = args.output_dir
    model_path = args.model_path
    
    # PCA dimensions to test (including no PCA)
    pca_dimensions = [32, 64, 128, 256, None]  # None means no PCA
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check that exactly one option is selected
    if args.training_only and args.validation_only:
        print("Error: Cannot specify both --training-only and --validation-only")
        print("Use --training-only to submit training jobs first")
        print("Use --validation-only to submit validation jobs after training completes")
        return
    elif not args.training_only and not args.validation_only:
        print("Error: Must specify either --training-only or --validation-only")
        print("Use --training-only to submit training jobs first")
        print("Use --validation-only to submit validation jobs after training completes")
        return
    
    if args.training_only:
        # Submit only training jobs
        print(f"Submitting BERT training jobs for multiple PCA dimensions")
        print(f"Training CSV: {train_csv_path}")
        print(f"Output directory: {output_dir}")
        print(f"Model path: {model_path}")
        print(f"PCA dimensions: {[d if d is not None else 'No PCA' for d in pca_dimensions]}")
        print(f"Total training jobs to submit: {len(pca_dimensions)}")
        print("-" * 80)
        
        successful_training_submissions = 0
        
        for i, pca_dim in enumerate(pca_dimensions):
            # Create job name
            if pca_dim is None:
                job_name = f"BERT-train-no-pca"
            else:
                job_name = f"BERT-train-pca-{pca_dim}"
            
            # Submit the training job
            if submit_bert_training_job(
                csv_path=train_csv_path,
                output_dir=output_dir,
                pca_components=pca_dim,
                model_path=model_path,
                job_name=job_name,
                memory="250G",
                aux_weight_multiplier=1.0
            ):
                successful_training_submissions += 1
            
            # Small delay between submissions to avoid overwhelming the scheduler
            if i < len(pca_dimensions) - 1:  # Don't sleep after the last job
                print(f"Waiting 3 seconds before next training submission...")
                time.sleep(3)
        
        # Summary
        print("-" * 80)
        print(f"Training job submission complete!")
        print(f"Successfully submitted: {successful_training_submissions}/{len(pca_dimensions)} training jobs")
        
        if successful_training_submissions == len(pca_dimensions):
            print("All training jobs submitted successfully!")
            print("\nTraining job names:")
            for pca_dim in pca_dimensions:
                if pca_dim is None:
                    print(f"  - BERT-train-no-pca")
                else:
                    print(f"  - BERT-train-pca-{pca_dim}")
            
            print(f"\nOutput files will be saved to: {output_dir}")
            print("Monitor job progress with: bjobs | grep 'BERT-train'")
            print("\nAfter training jobs complete, run validation with:")
            print(f"python {__file__} --train-csv {train_csv_path} --val-csv {val_csv_path} --output-dir {output_dir} --validation-only")
            
        else:
            print(f"Some training jobs failed to submit. Check the error messages above.")
    
    elif args.validation_only:
        # Submit only validation jobs
        submit_validation_only(
            val_csv_path=val_csv_path,
            output_dir=output_dir,
            model_path=model_path,
            pca_dimensions=pca_dimensions,
            memory=args.memory,
            aux_weight_multiplier=1.0
        )

if __name__ == "__main__":
    main()
