import argparse
import numpy as np
import json
from collections import defaultdict
from datasets import Dataset, load_dataset
from sal.utils.score import bootstrap_completions


def process_bootstrap_results(bootstrap_dataset, sample_size):
    """
    Process bootstrap results to compute averages (soft labels).
    This function replicates the logic from process_bootstrap.py
    """
    # Initialize a dictionary to store sums and counts for each question
    results = defaultdict(lambda: {
        f"bs_indicator_weighted@{sample_size}": 0, 
        f"bs_indicator_naive@{sample_size}": 0, 
        f"bs_indicator_maj@{sample_size}": 0, 
        "count": 0
    })

    # Process each line in the bootstrap dataset
    for data in bootstrap_dataset:
        question_id = data["problem"]  # "problem" uniquely identifies each question

        # Update sums of indicator columns.
        results[question_id][f"bs_indicator_weighted@{sample_size}"] += data.get(f"bs_indicator_weighted", 0)
        results[question_id][f"bs_indicator_naive@{sample_size}"] += data.get(f"bs_indicator_naive", 0)
        results[question_id][f"bs_indicator_maj@{sample_size}"] += data.get(f"bs_indicator_maj", 0)
        results[question_id]["count"] += 1

    # Compute averages for each question and prepare rows for the dataset
    rows = []
    for question_id, values in results.items():
        count = values["count"]
        rows.append({
            "problem": question_id,
            f"sl_weighted_{sample_size}": values[f"bs_indicator_weighted@{sample_size}"] / count if count > 0 else 0,
            f"sl_naive_{sample_size}": values[f"bs_indicator_naive@{sample_size}"] / count if count > 0 else 0,
            f"sl_maj_{sample_size}": values[f"bs_indicator_maj@{sample_size}"] / count if count > 0 else 0,
        })

    return rows


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run bootstrap sampling on completions and process to generate soft labels."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file containing completions.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file where soft labels will be saved.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=30,
        help="Number of bootstrap samples to generate per problem (default: 30).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=8,
        help="Number of completions to sample per bootstrap (default: 8).",
    )
    args = parser.parse_args()

    print(f"Starting combined bootstrap processing...")
    print(f"Input file: {args.input_jsonl}")
    print(f"Output file: {args.output_file}")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Sample size: {args.sample_size}")

    # Step 1: Load the completions file and run bootstrap sampling
    print("Step 1: Loading completions and running bootstrap sampling...")
    dataset = load_dataset("json", data_files=args.input_jsonl, split="train")
    
    bootstrapped_dataset = bootstrap_completions(
        dataset, n_bootstrap=args.n_bootstrap, sample_size=args.sample_size
    )
    print(f"Bootstrap sampling completed. Generated {len(bootstrapped_dataset)} samples.")

    # Step 2: Process bootstrap results to compute soft labels
    print("Step 2: Processing bootstrap results to compute soft labels...")
    soft_label_rows = process_bootstrap_results(bootstrapped_dataset, args.sample_size)
    print(f"Soft label computation completed. Generated {len(soft_label_rows)} soft labels.")

    # Step 3: Create final dataset and save
    print("Step 3: Creating final dataset and saving...")
    final_dataset = Dataset.from_list(soft_label_rows)
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    
    # Save the final soft labels
    final_dataset.to_json(args.output_file, orient="records", lines=True)
    print(f"Successfully saved soft labels to: {args.output_file}")
    print(f"Final dataset shape: {len(final_dataset)} rows")


if __name__ == "__main__":
    main()
