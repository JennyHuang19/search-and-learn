# score_rm.py
import argparse
import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer
import torch

def build_rm_pipeline(model_name):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        device=device,
        model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
        top_k=None,
        function_to_apply="none",
    )
    return rm_pipe

def score_completions(row, rm_pipe, batch_size=2):
    completions = row["completions"]
    prompt = row["prompt"]

    chat_template = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": completions},
    # {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]


    inputs = [prompt + "\n" + c for c in completions]
    outputs = rm_pipe(inputs, batch_size=min(batch_size, len(inputs)))
    scores = [float(max(d["score"] for d in out)) for out in outputs]
    best_idx = int(np.argmax(scores))
    return pd.Series({
        "rm_scores": scores,
        "best_response": completions[best_idx]
    })

def main():
    parser = argparse.ArgumentParser(description="Score completions with a reward model and select the best response.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file with generations.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file with scores.")
    parser.add_argument("--rm_model_name", type=str, default="NCSOFT/Llama-3-OffsetBias-RM-8B", help="Reward model name.") # replace with skyworks.
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for reward model scoring.") # k in best of k.
    args = parser.parse_args() 

    print(f"Loading generations from {args.input_path}")
    df = pd.read_json(args.input_path, lines=True)

    print(f"Loading reward model: {args.rm_model_name}")
    rm_pipe = build_rm_pipeline(args.rm_model_name)

    print("Scoring completions...")
    scored = df.apply(lambda row: score_completions(row, rm_pipe, args.batch_size), axis=1)
    df[["rm_scores", "best_response"]] = scored

    print(f"Saving results to {args.output_path}")
    df.to_json(args.output_path, orient="records", lines=True)
    print("Done.")

if __name__ == "__main__":
    main() 