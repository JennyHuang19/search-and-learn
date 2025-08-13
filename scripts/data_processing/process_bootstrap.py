import argparse
import json
from collections import defaultdict
from datasets import Dataset

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute averages (aka, soft labels).")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing bootstrap results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSON file where averages will be saved.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2,
        help="Number of completions to sample per bootstrap.",
    )
    args = parser.parse_args()

    # Initialize a dictionary to store sums and counts for each question
    results = defaultdict(lambda: {f"bs_indicator_weighted@{args.sample_size}": 0, f"bs_indicator_naive@{args.sample_size}": 0, f"bs_indicator_maj@{args.sample_size}": 0, "count": 0})

    # Read the input JSONL file
    with open(args.input_file, "r") as infile:
        for line in infile:
            data = json.loads(line)
            question_id = data["problem"]  # "problem" uniquely identifies each question

            # Update sums and counts
            results[question_id][f"bs_indicator_weighted@{args.sample_size}"] += data.get(f"bs_indicator_weighted@2", 0) # JH: notice that 2 is hardcoded. this should be correct, because we also hardcoded 2 in bootstrap_completions in sal.utils.score.
            results[question_id][f"bs_indicator_naive@{args.sample_size}"] += data.get(f"bs_indicator_naive@2", 0)
            results[question_id][f"bs_indicator_maj@{args.sample_size}"] += data.get(f"bs_indicator_maj@2", 0)
            results[question_id]["count"] += 1

    # Compute averages for each question and prepare rows for the dataset
    rows = []
    for question_id, values in results.items():
        count = values["count"]
        rows.append({
            "problem": question_id,
            "sl_weighted_2": values[f"bs_indicator_weighted@{args.sample_size}"] / count if count > 0 else 0,
            "sl_naive_2": values[f"bs_indicator_naive@{args.sample_size}"] / count if count > 0 else 0,
            "sl_maj_2": values[f"bs_indicator_maj@{args.sample_size}"] / count if count > 0 else 0,
        })

    # Create a HuggingFace Dataset
    dataset = Dataset.from_list(rows)

    # Save the dataset
    dataset.to_json(args.output_file, orient="records", lines=True)
    print(f"Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()