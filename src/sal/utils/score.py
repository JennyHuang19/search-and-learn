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

import pdb
import math
import numpy as np

from typing import Literal

from datasets import Dataset
from tqdm import tqdm


from sal.config import Config
from sal.utils.math import (
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    subsample_completions,
    add_indicator_columns,
    add_correctness_list,
    compute_weighted_pred_bs,
    compute_maj_pred_bs,
    compute_naive_pred_bs,
    add_indicator_columns_bs,
)


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score(dataset: Dataset, config: Config) -> Dataset:
    dataset = dataset.map(
        lambda x: {"agg_scores": [aggregate_scores(s, "last") for s in x["scores"]]}
    )
    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for n in tqdm(subsets, desc="Computing majority & weighted predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Extract answers {n}",
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute weighted pred {n}",
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute majority pred {n}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute naive pred {n}",
        )
        # Add indicator columns
        dataset = dataset.map(
            add_indicator_columns,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Add indicator columns {n}",
        )
        # Add correctness list
        dataset = dataset.map(
            add_correctness_list,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Add correctness list {n}",
        )
        # Nuke unused columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset

def bootstrap_completions(dataset, n_bootstrap=10, sample_size=2, random_seed=42):
    """
    For each problem, sample `sample_size` completions (with replacement) from the list of completions,
    do this `n_bootstrap` times, and for each sample compute the predictions for each method
    and their indicators or correctness. Each bootstrap sample becomes a new entry with a unique sample id.
    """
    rng = np.random.default_rng(random_seed)
    new_rows = []

    for idx, row in enumerate(dataset):
        # Add a breakpoint to inspect the row object
        print(f"Processing row {idx}...")
        # pdb.set_trace() 

        problem = row["problem"] # problem.
        answer = row["answer"]  # answer.
        completions = row["completions"] # completions.
        scores = row["agg_scores"] # prm scores.
        preds = extract_completion_answers(row)["preds"] # extracted answers from completions.

        for b in range(n_bootstrap):
            # Sample indices with replacement
            sample_indices = rng.choice(len(completions), size=sample_size, replace=True)
            sampled_completions = [completions[i] for i in sample_indices]
            sampled_scores = [scores[i] for i in sample_indices]
            sampled_preds = [preds[i] for i in sample_indices]

            # one realization of bootstrap sample.
            x = {
                "completions": sampled_completions,
                "sampled_scores": sampled_scores,  # shape: [1, sample_size]
                "sampled_preds": sampled_preds,
                "answer": answer,  # answer.
            }

            # Add a breakpoint to inspect x.
            # pdb.set_trace() 

            # Compute predictions using math utilities
            pred_weighted = compute_weighted_pred_bs(x, n=sample_size)[f"pred_weighted@{sample_size}"]
            pred_naive = compute_naive_pred_bs(x, n=sample_size)[f"pred_naive@{sample_size}"]
            pred_maj = compute_maj_pred_bs(x, n=sample_size)[f"pred_maj@{sample_size}"]

            weighted_ind = add_indicator_columns_bs(x, n=sample_size)[f"indicator_weighted@{sample_size}"]
            naive_ind = add_indicator_columns_bs(x, n=sample_size)[f"indicator_naive@{sample_size}"]
            maj_ind = add_indicator_columns_bs(x, n=sample_size)[f"indicator_maj@{sample_size}"]
            
            # Compose new row
            new_row = dict(row)  # copy original row
            new_row.update({
                "problem": problem,
                "answer": answer,
                "bootstrap_sample_id": f"{idx}_{b}",
                "bootstrap_sample_indices": list(sample_indices),
                # "bootstrap_completions": sampled_completions,
                "bootstrap_scores": sampled_scores,
                "bs_pred_weighted": pred_weighted,
                "bs_pred_naive": pred_naive,
                "bs_pred_maj": pred_maj,
                "bs_indicator_weighted": weighted_ind,
                "bs_indicator_naive": naive_ind,
                "bs_indicator_maj": maj_ind,
            })

            # Add a breakpoint to inspect answers after adding bootstrap.
            # pdb.set_trace() 

            new_rows.append(new_row)

    # Return as a new HuggingFace Dataset
    return Dataset.from_list(new_rows)
