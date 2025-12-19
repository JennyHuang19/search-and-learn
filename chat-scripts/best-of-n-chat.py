#!/usr/bin/env python3
import argparse, os, time, json
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
import yaml

# request 2 GPUs. place generator on device 0, reward model on device 1.

# -------------------- tiny utils --------------------
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


import time
import torch
from tqdm import tqdm

def score_pairs(rm_pipe, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
    inputs = [p + "\n" + r for (p, r) in pairs]
    scores = []
    
    start_time = time.time()
    total_batches = (len(inputs) + batch_size - 1) // batch_size
    
    print(f"Starting to score {len(inputs)} items in {total_batches} batches...")
    
    for i in tqdm(range(0, len(inputs), batch_size), desc="Scoring batches"):
        batch_start = time.time()
        
        # Get batch inputs
        batch_inputs = inputs[i:i+batch_size]
        
        # Use the pipeline with proper device handling
        with torch.cuda.amp.autocast():  # Use mixed precision for efficiency
            out = rm_pipe(batch_inputs) # this is the reward model doing its scoring, {"label": "...", "score": ...}.
        
        batch_time = time.time() - batch_start
        batch_size_actual = min(batch_size, len(inputs) - i)
        
        for item in out:
            scores.append(float(max(d["score"] for d in item)))
        
        # Print batch timing info
        if (i // batch_size + 1) % 10 == 0:  # Print every 10 batches
            elapsed = time.time() - start_time
            avg_time_per_item = elapsed / (i + batch_size_actual)
            items_per_sec = (i + batch_size_actual) / elapsed
            print(f"  Batch {i//batch_size + 1}/{total_batches}: {batch_time:.2f}s for {batch_size_actual} items "
                  f"({items_per_sec:.1f} items/sec, {avg_time_per_item:.3f}s per item)")
    
    total_time = time.time() - start_time
    avg_time_per_item = total_time / len(inputs)
    items_per_sec = len(inputs) / total_time
    
    print(f"\nScoring completed in {total_time:.2f}s")
    print(f"Average: {avg_time_per_item:.3f}s per item ({items_per_sec:.1f} items/sec)")
    
    return scores

# -------------------- main pipeline --------------------
def main():
    ap = argparse.ArgumentParser(description="chat best-of-N with HuggingFace Hub dataset.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    # minimal config fields
    repo_id       = cfg["dataset_repo"]                # HF dataset repo id
    split         = cfg.get("split", "train")          # which split
    prompt_col    = cfg.get("prompt_column", "prompt")
    out_path      = cfg["out_path"]                    # local JSONL output
    gen_model     = cfg["generator_model"]
    rm_model      = cfg["reward_model"]
    begin_row     = cfg["begin_row"]
    end_row       = cfg["end_row"]
    n             = int(cfg.get("n", 4))
    # sampling parameters follow from the sal general config file.
    max_tokens    = int(cfg.get("max_tokens", 2048))
    temperature   = float(cfg.get("temperature", 0.8))
    top_p         = float(cfg.get("top_p", 1.0))

    # 1) load prompts from HF hub
    ds = load_dataset(repo_id, split=split)

    ds_filtered = ds.select(range(begin_row, end_row))   # rows 100 through 199

    if prompt_col not in ds_filtered.column_names:
        raise ValueError(f"Column '{prompt_col}' not in dataset columns {ds_filtered.column_names}")
    prompts = ds_filtered[prompt_col]
    if len(prompts) == 0:
        raise ValueError("No rows in dataset split.")

    # 2) build generator (vLLM)
    t_load0 = time.time()
    llm = LLM(model=gen_model, trust_remote_code=True, dtype="auto", tensor_parallel_size=1) # Generator model loading.
    print(f"[vLLM] model load took {time.time()-t_load0:.2f}s")
    
    # Get tokenizer for counting tokens
    tokenizer = llm.get_tokenizer()
    
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,  # we duplicate prompts to get N
    )

    # 3) generate
    expanded = [p for p in prompts for _ in range(n)]
    all_responses = []
    generation_times = []
    t0 = time.time()
    for i in range(0, len(expanded), n):   # simple batching
        batch = expanded[i:i+n]
        batch_start = time.time()
        out = llm.generate(batch, sampling_params=sampling, use_tqdm=False)
        batch_end = time.time()
        all_responses.extend(out)
        # Record time for each prompt (batch contains n generations for one prompt)
        generation_times.extend([batch_end - batch_start] * n) # this is a list of n times for each prompt.
    print(f"[vLLM] generation took {time.time()-t0:.2f}s")


    import gc
    import torch
    del llm
    gc.collect()
    breakpoint() # gpu memory already freed up.

    # free up the GPU memory before loading in the reward model. the reward model is currently loaded on CPU.
    from transformers import AutoTokenizer, pipeline

    # -------------------- reward model --------------------
    def build_rm_pipeline(model_name: str):
        # Force GPU usage - try multiple approaches
        device = "cuda:0"  # Explicitly specify GPU
        
        tok = AutoTokenizer.from_pretrained(model_name)
        rm = pipeline(
            "sentiment-analysis",       # sequence classification head
            model=model_name,
            tokenizer=tok,
            device=device,  # Use explicit device string
            model_kwargs={"torch_dtype": torch.float16},
            top_k=None,
            # return_all_scores=True,
            function_to_apply="none",   # raw logits
        )
        
        # Double-check and force GPU if needed
        if hasattr(rm, 'model') and rm.model is not None:
            rm.model = rm.model.to(device)
            print(f"RM model loaded on device: {rm.model.device}")
            
            # Also ensure the tokenizer's model is on GPU
            if hasattr(rm, 'tokenizer') and hasattr(rm.tokenizer, 'model'):
                rm.tokenizer.model = rm.tokenizer.model.to(device)
        
        return rm

    expected = len(prompts) * n
    if len(all_responses) != expected:
        raise RuntimeError(f"Expected {expected} generations, got {len(all_responses)}")

    completions = [[] for _ in range(len(prompts))]
    token_counts = [[] for _ in range(len(prompts))]
    for i in range(len(prompts)):
        chunk = all_responses[i*n:(i+1)*n]
        texts = [o.text for r in chunk for o in r.outputs]
        completions[i] = texts
        # Count tokens for each completion
        token_counts[i] = [len(tokenizer.encode(text)) for text in texts]


    # 4) reward model scoring
    # Debug GPU availability
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[DEBUG] CUDA device name: {torch.cuda.get_device_name()}")
    ### 
    t_load1 = time.time()
    rm_pipe = build_rm_pipeline(rm_model) # Reward model loading.
    
    print(f"[RM] model load took {time.time()-t_load1:.2f}s")

    pairs = [(p, c) for p, cand_list in zip(prompts, completions) for c in cand_list]
    t1 = time.time()
    flat_scores = score_pairs(rm_pipe, pairs, batch_size=2) # this takes a really long time with batch_size=32. this takes in a reward model and the pairs of prompts and completions.
    print(f"[RM] scoring took {time.time()-t1:.2f}s")

    if len(flat_scores) != expected:
        raise RuntimeError(f"Expected {expected} scores, got {len(flat_scores)}")

    scores = []
    k = 0
    for _ in range(len(prompts)):
        scores.append(flat_scores[k:k+n])
        k += n

    chosen = [cand_list[int(np.argmax(s_list))] for cand_list, s_list in zip(completions, scores)]

    # 5) save results locally
    ensure_dir(out_path)
    with open(out_path, "w") as f:
        for i, (p, cand_list, s_list, best) in enumerate(zip(prompts, completions, scores, chosen)):
            rec = {
                "prompt": p,
                "completions": cand_list,
                "token_counts": token_counts[i],
                "scores": s_list,
                "chosen": best,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] wrote results to {out_path}")

if __name__ == "__main__":
    main()
