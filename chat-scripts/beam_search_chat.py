#!/usr/bin/env python3
import argparse
import copy
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from sal.config import Config
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import aggregate_scores
from utils_chat import ChatBeam, build_chat_conv, generate_k_steps_chat, last
import pdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tokenizer(model_id): # mostly taken from score-versaprm.py.
    """Get tokenizer with proper configuration for chat models."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer

def score_completion_chat(model, tokenizer, device, prompt: str, completion: str, 
                         candidate_tokens=[12, 10]): # mostly taken from score-versaprm.py.
    """Score a single prompt-completion pair using reward model."""
    try:
        # Format input as per reward model requirements
        input_text = prompt + ' \n\n' + completion + ' \n\n\n\n'
        input_id = torch.tensor([tokenizer.encode(input_text)]).to(device)
        
        with torch.no_grad():
            logits = model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,1] 
            final_score = scores[0, -1].item()
            
        return final_score
    except Exception as e:
        logger.warning(f"Error scoring completion: {e}")
        return 0.0

def score_completions_chat(model, tokenizer, device, prompts: List[str], 
                          completions: List[List[str]], candidate_tokens=[12, 10]):
    """Score multiple completions using reward model.""" # mostly taken from score-versaprm.py.
    all_scores = []
    for prompt, completion_list in zip(prompts, completions): # completions is a list of lists.
        prompt_scores = []
        for completion in completion_list: # completion (str), each element is a string up to that point in the response (aka, up to iteration i).
            score = score_completion_chat(model, tokenizer, device, prompt, completion, candidate_tokens)
            # pdb.set_trace() # jyh: prompt_scores, score.
            prompt_scores.append(score)
        all_scores.append(prompt_scores)
    return all_scores

def _beam_search_chat(prompt: str, config: Config, llm: LLM, 
                     reward_model, reward_tokenizer, device) -> tuple[List[ChatBeam], int]: # jyh: devices should correctly be the prm device.
    """Main beam search function for chat.
    
    Returns:
        tuple: (List of ChatBeam objects, total tokens generated for this prompt)
    """
    
    # Chat-specific sampling parameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens, # limit max tokens to 256.
        top_p=config.top_p,
        stop=["\n\n", "Human:", "Assistant", "User", "Bot"],
        include_stop_str_in_output=True,
        n=1,
    )

    # Initialize beams
    beams: List[ChatBeam] = []
    for i in range(config.n):
        beams.append(
            ChatBeam(
                prompt=prompt,
                index=i,
                current_text="",
                next_texts=None,
                lookahead_texts=None,
                pruned=False,
                completed=False,
                stop_reasons=None,
                history=[],
                best_scores=[],
                all_scores=[],
                previous_text=None,
                completion_tokens=0,
            )
        )

    completed_beams: List[ChatBeam] = []
    total_start_time = time.time()
    total_prompt_tokens = 0  # Track total tokens for this prompt
    
    # BREAKPOINT 1: Check initialization
    # pdb.set_trace()  # Inspect: len(beams), len(batch_of_prompts), config.n, config.num_iterations

    # Main beam search loop
    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        # BREAKPOINT 2: Check iteration start
        # pdb.set_trace()  # Inspect: i, config.num_iterations, len(beams), len(completed_beams)
        
        if i == 0:
            active_beams = [b for b in beams if not b.pruned] # len(beams) (114954), len(active_beams) (2). b.pruned is False for all beams. so why is the length of active_beams only 2?
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Ensure we have exactly config.n active beams
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(f"Extending active_beams with {repeats} repetitions to reach size {config.n}")
            extended_active_beams = [copy.deepcopy(b) for b in (active_beams * repeats)[:config.n]]
            active_beams = extended_active_beams
        
        # BREAKPOINT 3: Check active beams before generation
        # pdb.set_trace()  # Inspect: len(active_beams), config.n, [b.current_text[:50] for b in active_beams[:3]]

        # Last iteration: generate to completion
        if i == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        # Build conversations for each beam
        convs = [build_chat_conv(b.prompt, b.current_text, config.system_prompt) for b in active_beams] # jyh: b.prompt is a list of strings? but it should be a string. it is dependent on the input to beam_search_chat, which i have (as of 11:29am) changed from a list of strings to a string.
        
        # Apply chat template
        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            print(f"Using custom chat template set in config!")
            tokenizer.chat_template = config.custom_chat_template
        # jyh: do i have to change config.custom_chat_template for chat? config.custom_chat_template seems to be a lot of info on tool usage and formatting.
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=(i == 0),
            continue_final_message=(i > 0),
            tokenize=False,
        ) # jyh: templated_convs seems correct.
        
        # BREAKPOINT 4: Check conversations before generation
        # pdb.set_trace()  # Inspect: len(convs), len(templated_convs), templated_convs[0][:200] if templated_convs else None

        # Generate next tokens
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        gen_results, batch_tokens = generate_k_steps_chat(templated_convs, lookahead, llm, sampling_params, config.beam_width)
        total_prompt_tokens += batch_tokens
        
        # BREAKPOINT 5: Check generation results
        # pdb.set_trace()  # Inspect: len(gen_results), lookahead, gen_results[0].next_texts if gen_results else None, gen_results[0].stop_reasons if gen_results else None

        # Update beam states
        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            # Check completion
            if (
                beam.stop_reasons[0] == "length" # jyh: stop reason is not reaching length.
                or beam.stop_reasons[0] == "EOS" 
                or beam.next_texts[0] == ""
            ):
                beam.completed = True # 11:06pm never being reached.
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text]) # a list of lists of length 1.

        # BREAKPOINT 6: Check before scoring
        # pdb.set_trace()  # Inspect: len(prompts), len(completions), completions[0][0][:100] if completions else None
        
        # prompts is a list of the same prompt repeated beam_width times.
        # completions is a beam_width length list of the completions.
        # Score completions
        scores = score_completions_chat(reward_model, reward_tokenizer, device, prompts, completions) # jyh: have we made sure the device is GPU 2?        
        # pdb.set_trace() # jyh: scores. [[1.0, 1.0]], [[1.0, 1.0]]
        # Aggregate scores
        # agg_scores = [[aggregate_scores(s, config.agg_strategy) for s in score] for score in scores] # jyh: expects list[list[list[float]]].
        agg_scores = [[aggregate_scores(s, config.agg_strategy)] for s in scores] # jyh: input list[list[float]], output list[list[float]].


        # Update beam scores
        for beam, score in zip(active_beams, scores):
            # beam.all_scores = [score[0]] # jyh: changed from a float to a list of floats to handle output of prm.score and accommodate agg_scores of line 230.
            beam.all_scores.append(score[0]) # jyh: shouldn't this be beam.all_scores.append(score[0])?

        # BREAKPOINT 7: Check scoring results
        # pdb.set_trace()  # Inspect: len(scores), len(agg_scores), agg_scores[0] if agg_scores else None, config.agg_strategy

        # Remove completed beams
        agg_scores = [agg_scores[i] for i, b in enumerate(active_beams) if not b.completed]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping
        if len(active_beams) == 0:
            break

        # Filter duplicates
        if config.filter_duplicates:
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = i
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # Prune low-scoring beams
        top_indices = np.argsort(np.array(agg_scores).flatten())[-(config.n // config.beam_width):]
        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True
        
        # BREAKPOINT 8: Check pruning results
        # pdb.set_trace()  # Inspect: len(active_beams), len(top_indices), top_indices, [b.pruned for b in active_beams]

    # Final selection
    if config.sort_completed:
        # # pdb.set_trace()
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy) if isinstance(b.all_scores, list) else b.all_scores, # If b.all_scores is a list: Call aggregate_scores() to reduce it to a single float. If b.all_scores is already a float: Use it directly without aggregation.
            reverse=True
        )[:config.n] # b.all_scores are stepwise scores from the reward model. we perform mv by taking the last score.
    else:
        completed_beams = completed_beams[:config.n]

    # Ensure we have exactly config.n beams
    if len(completed_beams) != config.n:
        repeats = config.n // (len(completed_beams) + 1)
        extended_completed_beams = [copy.deepcopy(b) for b in (completed_beams * repeats)[:config.n]]
        completed_beams = extended_completed_beams

    # BREAKPOINT 9: Check final results
    # # pdb.set_trace()  # Inspect: len(completed_beams), config.n, [b.current_text[:100] for b in completed_beams[:3]]

    total_end_time = time.time()
    logger.info(f"Total beam search time: {total_end_time - total_start_time:.2f} seconds.")
    logger.info(f"Total tokens generated for this prompt: {total_prompt_tokens}")
    
    return completed_beams, total_prompt_tokens

def beam_search_chat(example: Dict[str, Any], config: Config, llm: LLM, 
                    reward_model, reward_tokenizer, device):
    """Main function to run beam search for chat."""
    problem = example["prompt"]
    beam_results, prompt_token_count = _beam_search_chat(problem, config, llm, reward_model, reward_tokenizer, device)

    # Group results by prompt
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "scores": [], "pred": [], "completion_tokens": [], "total_tokens": prompt_token_count}

    beams = grouped_results[problem]

    ### The issue is: beams is empty, which means no beams were found for the given problem. why are some beams empty?
    if len(beams) == 0:
        print("ERROR: No beams found for this problem!")
        return {"completions": [[]], "scores": [[]], "pred": [""], "completion_tokens": [[]], "total_tokens": prompt_token_count}
    ### 

    completions = [b.current_text for b in beams]
    agg_scores = [aggregate_scores(b.all_scores, config.agg_strategy) if isinstance(b.all_scores, list) else b.all_scores for b in beams] # jyh: 11:16pm why are some elements scalars while others lists, agg_scores = [0.85, [0.8, 0.9, 0.7]]  # Mixed types!
    pred = completions[np.argmax(agg_scores)]
    results["completions"].append(completions)
    results["scores"].append([b.all_scores for b in beams])
    results["pred"].append(pred)
    results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    
    # Load chat prompts
    df = pd.read_csv("/dccstor/gma2/mehuldamani/search-and-learn/chat-results/preferred_responses.csv")
    examples = {"prompt": df["prompt"].tolist()}
    print("number of prompts:", len(examples["prompt"]))
    # restrict to examples in dataset_start and dataset_end
    examples["prompt"] = examples["prompt"][config.dataset_start:config.dataset_end]
    print("num prompts between dataset_start and dataset_end:", len(examples["prompt"]))
    
    # Initialize models
    # Ensure both GPUs 0 and 1 are visible (if not already set by the environment)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print("Loading language model...")
    # vLLM will use the first visible GPU with tensor_parallel_size=1
    llm = LLM(model=config.model_path, trust_remote_code=True, dtype="auto", tensor_parallel_size=1)
    
    print("Loading reward model...")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("This script requires at least 2 CUDA GPUs for separate generator and PRM placement.")
    gen_device = 'cuda:0'
    prm_device = 'cuda:1'
    reward_tokenizer = get_tokenizer(config.prm_path)
    # Load PRM directly on GPU 1 to avoid intermediate moves
    reward_model = AutoModelForCausalLM.from_pretrained(
        config.prm_path, device_map={"": 1}
    )
    device = prm_device
    print(f"PRM loaded on device {prm_device}")
    
    # Set chat-specific system prompt if not already set
    if not hasattr(config, 'system_prompt') or config.system_prompt is None:
        config.system_prompt = "You are a helpful AI chat assistant. Provide clear, helpful responses to user questions."
    print(f"System prompt: {config.system_prompt}")

    # Run beam search one prompt at a time and build rows incrementally
    print("Running beam search...")
    rows = []
    for idx, prompt in enumerate(examples['prompt']):
        start_time = time.time()
        single_example = {"prompt": prompt}  # jyh changed to get rid of brackets in user prompts.
        single_result = beam_search_chat(single_example, config, llm, reward_model, reward_tokenizer, device)
        elapsed_time = time.time() - start_time
        print(f"Question {idx}: Time taken = {elapsed_time:.2f} seconds")
        row = {
            'prompt': prompt,
            'completions': single_result['completions'][0],
            'best_completion': single_result['pred'][0],
            'scores': single_result['scores'][0],
            # 'completion_tokens': single_result['completion_tokens'][0],
            'token_count': single_result['total_tokens'],
            'time': elapsed_time
        }
        rows.append(row)

    output_df = pd.DataFrame(rows)

    # Save results
    if config.output_dir:
        output_df.to_json(config.output_dir, orient='records', lines=True)
        print(f"Results saved to {config.output_dir}")
    
    # Print summary
    print(f"\nBeam search completed!")
    print(f"Processed {len(examples['prompt'])} prompts")
    print(f"Generated {len(output_df)} completion sets")
    
    return

if __name__ == "__main__":
    main()
