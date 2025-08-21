#!/usr/bin/env python3
import copy
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline

from sal.config import Config
from sal.utils.score import aggregate_scores

logger = logging.getLogger(__name__)


# -----------------------------
# Minimal Beam container
# -----------------------------
@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str = ""
    pruned: bool = False
    completed: bool = False
    stop_reasons: Optional[List[str]] = None
    history: Optional[List[str]] = None
    all_scores: Optional[List[float]] = None
    completion_tokens: int = 0

    def __post_init__(self):
        if self.history is None:
            self.history = []


# -----------------------------
# VersaPRM (reward model) scorer
# -----------------------------
class VersaPRMScorer:
    """
    Wraps the UW-Madison-Lee-Lab/VersaPRM model as a sequence-classification pipeline.
    Returns a single scalar per (prompt, response) pair (max logit).
    """
    def __init__(self, model_name: str = "UW-Madison-Lee-Lab/VersaPRM", torch_dtype="float16"):
        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.float16
        tok = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "sentiment-analysis",      # uses sequence classification head
            model=model_name,
            tokenizer=tok,
            device="auto",
            model_kwargs={"torch_dtype": dtype},
            return_all_scores=True,    # list of {label, score}
            function_to_apply="none",  # raw logits (no softmax)
        )

    def score(self, prompts: List[str], completions: List[List[str]]) -> List[List[float]]:
        """
        Inputs:
          prompts:     [P]
          completions: [ [c1, c2, ...], ... ] aligned to prompts
        Returns:
          scores:      [ [s(c1), s(c2), ...], ... ] aligned to completions
        """
        flat_pairs: List[Tuple[str, str]] = []
        for p, cand_list in zip(prompts, completions):
            for c in cand_list:
                flat_pairs.append((p, c))

        inputs = [p + "\n" + r for (p, r) in flat_pairs]
        flat_scores: List[float] = []
        # conservative batch size to avoid OOM; tune if needed
        batch_size = 32
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i:i + batch_size]
            outs = self.pipe(chunk, batch_size=min(batch_size, len(chunk)))
            # each 'outs[j]' is a list of dicts; we take max raw logit
            for item in outs:
                flat_scores.append(float(max(d["score"] for d in item)))

        # unflatten
        scores: List[List[float]] = []
        k = 0
        for cand_list in completions:
            n = len(cand_list)
            scores.append(flat_scores[k:k + n])
            k += n
        return scores


# -----------------------------
# Helpers for generation
# -----------------------------
def _make_inputs_for_generation(beams: List[Beam]) -> List[str]:
    # no chat template; just prompt + what we already have
    return [b.prompt + b.current_text for b in beams]


def _generate_step(
    llm: LLM,
    beams: List[Beam],
    lookahead: int,
    sampling_params: SamplingParams,
    stop: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Generate up to `lookahead` tokens for each beam’s current prefix.
    Returns:
      next_texts:     the newly generated slices (not cumulative)
      stop_reasons:   list of "EOS" | "length" | "" per item
      token_counts:   tokens added this call, per item
    """
    # customize sampling per step
    step_params = SamplingParams(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=lookahead,
        n=1,
        stop=stop or [],
        include_stop_str_in_output=True if stop else False,
    )

    inputs = _make_inputs_for_generation(beams)
    outputs = llm.generate(inputs, sampling_params=step_params, use_tqdm=False)

    next_texts, reasons, toks = [], [], []
    for out in outputs:
        # vLLM returns one Output per input when n=1
        o = out.outputs[0]
        text = o.text
        token_ids = o.token_ids or []
        # crude stop detection (vLLM doesn't expose reason directly)
        reason = ""
        if stop:
            if any(s in text for s in stop):
                reason = "EOS"
        # if we hit max_tokens exactly, treat as "length"
        if len(token_ids) >= lookahead and lookahead > 0 and reason == "":
            reason = "length"

        next_texts.append(text)
        reasons.append(reason)
        toks.append(len(token_ids))

    return next_texts, reasons, toks


# -----------------------------
# Beam search function
# -----------------------------
def _beam_search(batch_of_prompts: List[str], config: Config, llm: LLM, prm: VersaPRMScorer) -> List[Beam]:
    """
    Chat-data-friendly beam search:
      - Input is raw prompts (strings) from a HF dataset 'prompt' column.
      - No system prompt or tokenizer.chat_template.
      - PRM scoring = VersaPRM.
    """
    base_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,  # used only on last iteration
        n=1,
    )

    stop = getattr(config, "stop", None) or ["\n\n"]  # default simple stop if not provided
    beams: List[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(Beam(prompt=prompt, index=i))

    completed_beams: List[Beam] = []
    total_start_time = time.time()

    # iterations
    for it in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        # active set
        if it == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # ensure exactly config.n active beams (duplicate if needed)
        if len(active_beams) != config.n:
            repeats = (config.n // max(len(active_beams), 1)) + 1
            extended = [copy.deepcopy(b) for b in (active_beams * repeats)[:config.n]]
            active_beams = extended
            if len(active_beams) != config.n:
                raise ValueError(f"Expected {config.n} active beams, got {len(active_beams)}")

        # last iteration: allow full decode to max_tokens
        lookahead = 0 if it == config.num_iterations - 1 else config.lookahead
        step_stop = [] if it == config.num_iterations - 1 else stop

        # one generation step
        next_texts, stop_reasons, token_counts = _generate_step(
            llm=llm,
            beams=active_beams,
            lookahead=(config.max_tokens if lookahead == 0 else lookahead),
            sampling_params=base_params,
            stop=step_stop,
        )

        # update beams
        prompts_for_score, comps_for_score = [], []
        for b, new_text, reason, tok_add in zip(active_beams, next_texts, stop_reasons, token_counts, strict=True):
            b.current_text += new_text
            b.history.append(new_text)
            b.stop_reasons = [reason]
            b.completion_tokens += tok_add

            # mark completion
            if reason in ("EOS", "length") or new_text == "":
                b.completed = True
                completed_beams.append(b)

            # PRM scoring is over the *current prefix* as a single candidate
            prompts_for_score.append(b.prompt)
            comps_for_score.append([b.current_text])

        # score all active beams with VersaPRM
        scores = prm.score(prompts_for_score, comps_for_score)  # shape [B][1]
        agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores]  # one number per active beam

        # record raw scores on the beam (keep last vector for compatibility)
        for b, s in zip(active_beams, scores, strict=True):
            b.all_scores = s[0]

        # remove completed from further consideration
        keep_mask = [not b.completed for b in active_beams]
        active_beams = [b for b, keep in zip(active_beams, keep_mask, strict=True) if keep]
        agg_scores = [s for s, keep in zip(agg_scores, keep_mask, strict=True) if keep]

        # early stop if all done
        if len(active_beams) == 0:
            break

        # optional duplicate filtering
        if getattr(config, "filter_duplicates", False):
            uniq = {}
            for idx, b in enumerate(active_beams):
                if b.current_text not in uniq:
                    uniq[b.current_text] = idx
            keep_indices = list(uniq.values())
            active_beams = [active_beams[i] for i in keep_indices]
            agg_scores = [agg_scores[i] for i in keep_indices]

        # prune to top (n / beam_width)
        width = max(1, config.n // max(1, getattr(config, "beam_width", 1)))
        top_indices = np.argsort(np.array(agg_scores))[-width:]
        for idx, b in enumerate(active_beams):
            if idx not in top_indices:
                b.pruned = True

    # finalize: choose top-n among completed
    if getattr(config, "sort_completed", True):
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    # pad if not enough
    if len(completed_beams) != config.n and len(completed_beams) > 0:
        repeats = (config.n // len(completed_beams)) + 1
        extended = [copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]]
        completed_beams = extended

    total_end_time = time.time()
    print(f"Total beam search time: {total_end_time - total_start_time:.2f} seconds.")
    return completed_beams


# -----------------------------
# Public API (HF chat data)
# -----------------------------
def beam_search(examples, config: Config, llm: LLM):
    """
    `examples` is a HF Dataset batch with a 'prompt' column (chat prompts).
    PRM is fixed to VersaPRM.
    """
    problems = examples["prompt"]  # <-- chat prompts column
    prm = VersaPRMScorer(model_name="UW-Madison-Lee-Lab/VersaPRM", torch_dtype="float16")

    beam_results = _beam_search(problems, config, llm, prm)

    # group per prompt
    grouped = defaultdict(list)
    for b in beam_results:
        grouped[b.prompt].append(b)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}
    for p in problems:
        beams = grouped[p]
        completions = [b.current_text for b in beams]
        agg_scores = [aggregate_scores(b.all_scores, config.agg_strategy) for b in beams]
        pred = completions[int(np.argmax(agg_scores))]

        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
