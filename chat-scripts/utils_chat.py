import copy
import logging
from dataclasses import dataclass
import pdb

import numpy as np
from vllm import LLM, SamplingParams

logger = logging.getLogger()


def build_chat_conv(
    prompt: str, response: str | None, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """Build conversation format for chat prompts."""
    if system_prompt:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        conversation = [
            {"role": "user", "content": prompt},
        ]

    if response and response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    """Get the last element of a list."""
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    """Calculate the mean of a list."""
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class ChatBeam:
    """Beam class for chat beam search."""
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    previous_text: str | None
    pruned: bool = False
    history: list[str] = None
    completed: bool = False
    completion_tokens: int = 0

    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class ChatGenResult:
    """Generation result for chat beam search."""
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    completion_tokens: int = 0


def generate_k_steps_chat(
    templated_convs,
    lookahead_steps: int,
    llm: LLM,
    sampling_params: SamplingParams,
    beam_width: int,
) -> tuple[list[ChatBeam], int]:
    """Generate k steps for chat conversations.
    
    Returns:
        tuple: (list of ChatBeam objects, total tokens generated for this batch)
    """

    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = ChatGenResult(
                index=i,
                initial_prompt=text, # jyh: this should be a string, but currently, the user portion is a list of strings.
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
                completion_tokens=0,
            )
            gen_results.append(gen_result) # jyh: list of length templated_convs * beam_width.
    
    # BREAKPOINT 1: Check initialization
    # pdb.set_trace()  # Inspect: len(gen_results), len(templated_convs), beam_width, lookahead_steps

    gen_sampling_params = copy.deepcopy(sampling_params)
    tokenizer = llm.get_tokenizer()  # tokenizer for token counting
    total_batch_tokens = 0  # Track total tokens for this batch

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps

        # BREAKPOINT 2: Check iteration start
        # pdb.set_trace()  # Inspect: i, lookahead_steps, len(gen_results), gen_sampling_params.temperature

        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        
        if not current_gen:
            break
        
        # BREAKPOINT 3: Check active generations
        # pdb.set_trace()  # Inspect: len(current_gen), [g.stop_reason for g in current_gen]
            
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        
        # BREAKPOINT 4: Check before generation
        # pdb.set_trace()  # Inspect: len(gen_prompts), gen_prompts[0][:100] if gen_prompts else None
        
        llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        step_tokens = sum([len(k.outputs[0].token_ids) for k in llm_outputs])
        total_batch_tokens += step_tokens

        curr_step_tokens = []
        
        # BREAKPOINT 5: Check after generation
        # pdb.set_trace()  # Inspect: len(llm_outputs), llm_outputs[0].outputs[0].text if llm_outputs else None, llm_outputs[0].outputs[0].finish_reason if llm_outputs else None
        
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text

            # count the number of generated tokens for each beam
            gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
            gen_result.completion_tokens += len(gen_tokens)
            curr_step_tokens.append(len(gen_tokens))

            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.outputs[0].finish_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.outputs[0].finish_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"
        
        # Print the total number of tokens generated in the current step
        print(f"Number of tokens generated in current step: {step_tokens}")

    # BREAKPOINT 6: Check after all iterations
    # pdb.set_trace()  # Inspect: len(gen_results), gen_results[0].first_step_text, gen_results[0].lookahead_text, gen_results[0].stop_reason

    outputs: list[ChatBeam] = []

    counter = 0
    for i, text in enumerate(templated_convs): # jyh: why do we only end up with len(templated_convs) outputs?
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        completion_tokens = 0  # track total completion tokens per beam

        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            completion_tokens += gen_result.completion_tokens
            counter += 1

        beam_result = ChatBeam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            completion_tokens=completion_tokens,
        )
        outputs.append(beam_result)

    # BREAKPOINT 7: Check final output
    # pdb.set_trace()  # Inspect: len(outputs), outputs[0].next_texts, outputs[0].lookahead_texts, outputs[0].stop_reasons
    # each output (beam_result) has a length of beam_width.
    print(f"Total number of tokens generated for this batch: {total_batch_tokens}")
    
    return outputs, total_batch_tokens

