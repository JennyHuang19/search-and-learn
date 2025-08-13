import argparse
import numpy as np
import json
from datasets import Dataset, load_dataset
from sal.utils.score import bootstrap_completions

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run bootstrap sampling on completions.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Path to the input JSONL file containing completions.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file where bootstrapped results will be saved.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10,
        help="Number of bootstrap samples to generate per problem.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2,
        help="Number of completions to sample per bootstrap.",
    )
    args = parser.parse_args()


# Path to your input and output files
# input_jsonl = "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32/best_of_n_completions.jsonl"
# output_jsonl = "/dccstor/gma2/jhjenny9/search-and-learn/data/Qwen/Qwen2.5-1.5B-Instruct/best_of_32/best_of_2_bootstrap_out.jsonl"

    # Load the completions file (a .jsonl file) as a HuggingFace Dataset
    dataset = load_dataset("json", data_files=args.input_jsonl, split="train")

    # Run the bootstrap sampling
    bootstrapped_dataset = bootstrap_completions(
        dataset, n_bootstrap=args.n_bootstrap, sample_size=args.sample_size
    )

    # Save the result to a new .jsonl file
    bootstrapped_dataset.to_json(args.output_jsonl, orient="records", lines=True)
    print(f"Saved bootstrapped dataset to {args.output_jsonl}")

if __name__ == "__main__":
    main()