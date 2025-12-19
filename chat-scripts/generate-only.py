#!/usr/bin/env python3
import argparse, os, time, json
from typing import List

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
import yaml

# -------------------- tiny utils --------------------
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

# -------------------- main generation pipeline --------------------
def main():
    ap = argparse.ArgumentParser(description="Generate completions only with vLLM.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # minimal config fields
    repo_id       = cfg["dataset_repo"]                # HF dataset repo id
    split         = cfg.get("split", "train")          # which split
    prompt_col    = cfg.get("prompt_column", "prompt")
    out_path      = cfg["out_path"]                    # local JSONL output
    gen_model     = cfg["generator_model"]
    begin_row     = cfg["begin_row"]
    end_row       = cfg["end_row"]
    n             = int(cfg.get("n", 4))               # number of completions per prompt
    
    # sampling parameters follow from the sal general config file.
    max_tokens    = int(cfg.get("max_tokens", 512))
    temperature   = float(cfg.get("temperature", 0.8))
    top_p         = float(cfg.get("top_p", 1.0))

    # 1) load prompts from HF hub
    ds = load_dataset(repo_id, split=split)
    ds_filtered = ds.select(range(begin_row, end_row))

    if prompt_col not in ds_filtered.column_names:
        raise ValueError(f"Column '{prompt_col}' not in dataset columns {ds_filtered.column_names}")
    prompts = ds_filtered[prompt_col]
    if len(prompts) == 0:
        raise ValueError("No rows in dataset split.")

    print(f"Loaded {len(prompts)} prompts from {repo_id}")
    print(f"Will generate {n} completions per prompt (from config)")

    # 2) build generator (vLLM)
    t_load0 = time.time()
    llm = LLM(model=gen_model, trust_remote_code=True, dtype="auto", tensor_parallel_size=1)
    print(f"[vLLM] model load took {time.time()-t_load0:.2f}s")
    
    # Get tokenizer for counting tokens
    tokenizer = llm.get_tokenizer()
    
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=1.05,           # gentle anti-loop
        stop=["</s>", "<|im_end|>"],       # Qwen3 chat EOT markers
        n=1,  # we duplicate prompts to get N
    )


    # 3) generate
    expanded = [p for p in prompts for _ in range(n)]
    all_responses = []
    generation_times = []
    t0 = time.time()
    
    print(f"Generating {len(expanded)} total completions...")
    for i in range(0, len(expanded), n):   # simple batching
        batch = expanded[i:i+n]
        batch_start = time.time()
        out = llm.generate(batch, sampling_params=sampling, use_tqdm=False)
        batch_end = time.time()
        all_responses.extend(out)
        # Record time for each prompt (batch contains n generations for one prompt)
        generation_times.extend([batch_end - batch_start] * n)
        
        # Print progress every 10 batches
        if (i // n + 1) % 10 == 0:
            elapsed = time.time() - t0
            completed = i + n
            rate = completed / elapsed
            print(f"  Completed {completed}/{len(expanded)} completions ({rate:.1f} completions/sec)")
    
    print(f"[vLLM] generation took {time.time()-t0:.2f}s")

    expected = len(prompts) * n
    if len(all_responses) != expected:
        raise RuntimeError(f"Expected {expected} generations, got {len(all_responses)}")

    # Process completions
    completions = [[] for _ in range(len(prompts))]
    token_counts = [[] for _ in range(len(prompts))]
    for i in range(len(prompts)):
        chunk = all_responses[i*n:(i+1)*n]
        texts = [o.text for r in chunk for o in r.outputs]
        completions[i] = texts
        # Count tokens for each completion
        token_counts[i] = [len(tokenizer.encode(text)) for text in texts]

    # 4) save results locally
    ensure_dir(out_path)
    with open(out_path, "w") as f:
        for i, (p, cand_list, token_list) in enumerate(zip(prompts, completions, token_counts)):
            rec = {
                "prompt": p,
                "completions": cand_list,
                "token_counts": token_list,
                "n_completions": n,
                "generation_time": generation_times[i*n] if i*n < len(generation_times) else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"[done] wrote {len(prompts)} prompts with {n} completions each to {out_path}")
    print(f"Total completions generated: {len(prompts) * n}")

if __name__ == "__main__":
    main()
