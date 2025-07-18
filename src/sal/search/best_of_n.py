#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores


def best_of_n(x, config: Config, llm: LLM, prm: PRM):
    tokenizer = llm.get_tokenizer()

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    tokenizer = llm.get_tokenizer()
    # TODO: set the augmented template from a file
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )

    # Process conversations in batches of 8
    batch_size = 8
    all_responses = []
    
    for i in range(0, len(templated_convs), batch_size):
        batch_convs = templated_convs[i:i + batch_size]
        
        # Print GPU memory usage before processing batch
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"Batch {i//batch_size + 1}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_memory:.2f}GB")
            else:
                print(f"Batch {i//batch_size + 1}: Processing batch of size {len(batch_convs)}")
        except Exception as e:
            print(f"Batch {i//batch_size + 1}: Processing batch of size {len(batch_convs)} (GPU monitoring unavailable: {e})")
        
        batch_responses = llm.generate(
            batch_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        all_responses.extend(batch_responses)
    
    responses = all_responses
    
    if len(responses) != len(x["problem"]) * config.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem'] * config.n)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    # scores = prm.score(x["problem"], completions) # (to-do: batch this). print statements to figure out where oom occurs.

    ### Batched scoring
    batch_size = 8
    all_scores = []
    print(f"Starting scoring: {len(x['problem'])} problems, batch size {batch_size}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Before scoring] GPU Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    for i in range(0, len(x["problem"]), batch_size):
        batch_problems = x["problem"][i:i + batch_size]
        batch_completions = completions[i:i + batch_size]
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[Before batch {i//batch_size + 1}] GPU Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            batch_scores = prm.score(batch_problems, batch_completions)
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[After batch {i//batch_size + 1}] GPU Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            print(f"Batch {i//batch_size + 1}: Scored {len(batch_scores)} problems.")
        except Exception as e:
            print(f"OOM or error during scoring batch {i//batch_size + 1}: {e}")
            raise
        all_scores.extend(batch_scores)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[After scoring] GPU Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    scores = all_scores

    ###

    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens

    return x
