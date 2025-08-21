#!/usr/bin/env python3
import argparse, os, time, json
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from vllm import LLM, SamplingParams
import yaml

# -------------------- tiny utils --------------------
def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

# -------------------- reward model --------------------
def build_rm_pipeline(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    rm = pipeline(
        "sentiment-analysis",       # sequence classification head
        model=model_name,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,              # GPU if available
        model_kwargs={"torch_dtype": torch.float16},
        return_all_scores=True,
        function_to_apply="none",   # raw logits
    )
    return rm

def score_pairs(rm_pipe, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[float]:
    inputs = [p + "\n" + r for (p, r) in pairs]
    scores = []
    for i in range(0, len(inputs), batch_size):
        out = rm_pipe(inputs[i:i+batch_size], batch_size=min(batch_size, len(inputs)))
        for item in out:
            scores.append(float(max(d["score"] for d in item)))
    return scores

# -------------------- main pipeline --------------------
def main():
    ap = argparse.ArgumentParser(description="Minimal best-of-N with Hugging Face Hub dataset.")
    ap.add_argument("--config", required=True, help="Path to simple YAML config.")
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
    max_tokens    = int(cfg.get("max_tokens", 256))

    # 1) load prompts from HF hub
    ds = load_dataset(repo_id, split=split)

    ds_filtered = ds.select(range(begin_row, end_row))   # rows 100 through 199

    if prompt_col not in ds_filtered.column_names:
        raise ValueError(f"Column '{prompt_col}' not in dataset columns {ds_filtered.column_names}")
    prompts = ds_filtered[prompt_col]
    if len(prompts) == 0:
        raise ValueError("No rows in dataset split.")

    # 2) build generator (vLLM)
    llm = LLM(model=gen_model, trust_remote_code=True, dtype="auto", tensor_parallel_size=1)
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=max_tokens,
        n=1,  # we duplicate prompts to get N
    )

    # 3) generate
    expanded = [p for p in prompts for _ in range(n)]
    all_responses = []
    t0 = time.time()
    for i in range(0, len(expanded), n):   # simple batching
        batch = expanded[i:i+n]
        out = llm.generate(batch, sampling_params=sampling, use_tqdm=False)
        all_responses.extend(out)
    print(f"[vLLM] generation took {time.time()-t0:.2f}s")

    expected = len(prompts) * n
    if len(all_responses) != expected:
        raise RuntimeError(f"Expected {expected} generations, got {len(all_responses)}")

    completions = [[] for _ in range(len(prompts))]
    for i in range(len(prompts)):
        chunk = all_responses[i*n:(i+1)*n]
        texts = [o.text for r in chunk for o in r.outputs]
        completions[i] = texts

    # 4) reward model scoring
    rm_pipe = build_rm_pipeline(rm_model)
    pairs = [(p, c) for p, cand_list in zip(prompts, completions) for c in cand_list]
    t1 = time.time()
    flat_scores = score_pairs(rm_pipe, pairs, batch_size=32)
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
        for p, cand_list, s_list, best in zip(prompts, completions, scores, chosen):
            rec = {
                "prompt": p,
                "completions": cand_list,
                "scores": s_list,
                "chosen": best,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[done] wrote results to {out_path}")

if __name__ == "__main__":
    main()
