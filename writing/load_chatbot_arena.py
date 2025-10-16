#!/usr/bin/env python3
"""
Script to load the Chatbot Arena dataset from Hugging Face.
"""

import os
from datasets import load_dataset

def load_chatbot_arena_data():
    """
    Load the Chatbot Arena dataset from Hugging Face.

    Returns:
        Dataset: The loaded Chatbot Arena dataset
    """
    print("Loading Chatbot Arena dataset from Hugging Face...")

    # Load the dataset (lmsys/chatbot_arena_conversations)
    dataset = load_dataset("lmarena-ai/arena-human-preference-55k")

    print(f"Dataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")

    # Display basic information about the dataset
    for split_name, split_data in dataset.items():
        print(f"\n{split_name} split:")
        print(f"  Number of examples: {len(split_data)}")
        print(f"  Features: {split_data.features}")

    return dataset


if __name__ == "__main__":
    # Load the dataset
    dataset = load_chatbot_arena_data()

    # Display a sample from the dataset
    if len(dataset) > 0:
        split_name = list(dataset.keys())[0]
        print(f"\n\nSample from {split_name} split:")
        print(dataset[split_name][0])

    # Save each split as a CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for split_name, split_data in dataset.items():
        csv_filename = os.path.join(script_dir, f"chatbot_arena_{split_name}.csv")
        print(f"\nSaving {split_name} split to {csv_filename}...")

        # Convert to pandas DataFrame and save as CSV
        df = split_data.to_pandas()
        df.to_csv(csv_filename, index=False)

        print(f"Saved {len(df)} rows to {csv_filename}")
