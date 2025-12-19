# Search and Learn - Workflow Documentation

## Overview

This repository implements **test-time compute scaling** techniques for LLMs, inspired by the idea that giving models more "time to think" can improve performance on complex problems. The project provides infrastructure for search-based inference methods that use **Process Reward Models (PRMs)** to guide LLMs toward better solutions.

## Repository Structure

```
search-and-learn/
├── src/sal/                    # Core library
│   ├── config.py               # Configuration dataclass
│   ├── search/                 # Search algorithms
│   │   ├── best_of_n.py        # Best-of-N sampling
│   │   ├── beam_search.py      # Beam search
│   │   ├── diverse_verifier_tree_search.py  # DVTS
│   │   └── utils.py            # Beam utilities
│   ├── models/                 # Reward model implementations
│   │   ├── reward_models.py    # PRM classes (MathShepherd, RLHFFlow, SkyworkO1, etc.)
│   │   └── skywork_o1_prm/     # Skywork PRM implementation
│   └── utils/                  # Utilities (data, parser, scoring)
├── scripts/                    # Execution scripts
│   ├── test_time_compute.py    # Main entry point for running search
│   ├── merge_chunks.py         # Merge parallel job results
│   ├── data_generation/        # Dataset generation scripts
│   ├── data_processing/        # Data processing utilities
│   ├── features/               # Feature extraction scripts
│   └── training/               # Training scripts for probes
├── recipes/                    # YAML configs for different models
│   ├── Llama-3.2-1B-Instruct/  # Llama 1B configs
│   ├── Llama-3.2-3B-Instruct/  # Llama 3B configs
│   ├── Qwen2.5-1.5B-Instruct/  # Qwen configs
│   └── AceMath-7B-Instruct/    # AceMath configs
├── chat-scripts/               # Chat-specific implementations
│   ├── beam_search_chat.py     # Beam search adapted for chat
│   ├── best-of-n-chat.py       # Best-of-N for chat
│   ├── utils_chat.py           # Chat utilities
│   ├── score_rm.py             # Reward model scoring
│   └── eval_*.py               # Evaluation scripts
├── chat-recipes/               # YAML configs for chat experiments
└── chat-results/               # Chat experiment results
```

## Core Workflow

### 1. Search Algorithms

The repository supports three main search approaches:

#### Best-of-N (`src/sal/search/best_of_n.py`)
- Generates `n` complete solutions for each problem
- Scores all solutions using a PRM
- Selects the highest-scoring solution
- Simplest approach, good baseline

#### Beam Search (`src/sal/search/beam_search.py`)
- Generates solutions step-by-step (steps separated by `\n\n`)
- Maintains `n` parallel beams
- At each iteration:
  1. Generates next step for all active beams
  2. Scores partial completions with PRM
  3. Prunes low-scoring beams, keeps top `n/beam_width` beams
  4. Repeats for `num_iterations` steps
- Produces `n` complete solutions at the end

#### DVTS - Diverse Verifier Tree Search (`src/sal/search/diverse_verifier_tree_search.py`)
- Maintains `n/beam_width` independent search trees
- Each tree explores `beam_width` branches per step
- Selects best branch at each step based on PRM scores
- Promotes diversity through independent trees

### 2. Process Reward Models (PRMs)

PRMs score partial solutions step-by-step (`src/sal/models/reward_models.py`):

- **RLHFFlow** (default): `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data`
- **MathShepherd**: `peiyi9979/math-shepherd-mistral-7b-prm`
- **SkyworkO1**: `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-{1.5B,7B}`
- **Qwen2.5-Math**: `Qwen/Qwen2.5-Math-PRM-7B`

PRMs provide step-level scores that are aggregated using strategies:
- `last`: Use the last step's score
- `min`: Use the minimum score across steps
- `prod`: Use the product of all scores

### 3. Running Experiments

**Basic Usage:**
```bash
# Run with a recipe config
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

# Override parameters
python scripts/test_time_compute.py $CONFIG \
    --n=256 \
    --seed=0 \
    --push_to_hub=true
```

**Key Configuration Parameters** (`src/sal/config.py`):
- `approach`: `best_of_n`, `beam_search`, or `dvts`
- `n`: Number of completions/beams
- `beam_width`: Beams per tree (for beam_search/dvts)
- `num_iterations`: Max generation steps
- `temperature`, `top_p`: Sampling parameters
- `agg_strategy`: Score aggregation method
- `prm_path`: Path to PRM model

### 4. Chat-Specific Workflow

The `chat-scripts/` directory contains adaptations for open-ended chat:

**`beam_search_chat.py`** - Main chat beam search:
1. Loads chat prompts from CSV
2. Uses vLLM for generation (GPU 0)
3. Uses reward model for scoring (GPU 1)
4. Runs beam search with chat-specific:
   - Stop tokens: `\n\n`, `Human:`, `Assistant`, etc.
   - System prompts for chat
   - Custom scoring via `score_completions_chat()`

**Utilities (`utils_chat.py`)**:
- `ChatBeam`: Dataclass for chat beam state
- `build_chat_conv()`: Build conversation format
- `generate_k_steps_chat()`: Step-by-step generation

### 5. Data Processing Pipeline

1. **Generate completions**: `scripts/test_time_compute.py`
2. **Merge chunks**: `scripts/merge_chunks.py` (for parallel runs)
3. **Process results**: Scripts in `scripts/data_processing/`
4. **Extract features**: Scripts in `scripts/features/`
5. **Train probes**: Scripts in `scripts/training/`

## Key Concepts

### Beams
A `Beam` tracks:
- `prompt`: Original problem
- `current_text`: Generated text so far
- `history`: List of generated steps
- `all_scores`: PRM scores at each step
- `pruned`/`completed`: Status flags

### Step Detection
Steps are identified by double newlines (`\n\n`), matching the math problem format:
```
## Step 1: [description]
[calculation]

## Step 2: [description]
[calculation]
```

### Scoring Flow
1. Generate partial completion
2. Format for PRM (model-specific)
3. Get step-level scores
4. Aggregate scores for ranking
5. Prune low-scoring beams

## Quick Start

```bash
# Install
conda create -n sal python=3.11 && conda activate sal
pip install -e '.[dev]'

# Run test (first 10 problems)
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

# Full run with 256 completions
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --n=256 --num_samples=500 --push_to_hub=true
```

## Hardware Requirements

- Experiments run on H100s (80GB VRAM)
- vLLM uses ~50% GPU memory by default (`gpu_memory_utilization=0.5`)
- PRM uses remaining GPU memory
- For chat: GPU 0 = generator, GPU 1 = PRM

## References

- [HuggingFace Blog Post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)
- [Original Paper: Snell et al. 2024](https://arxiv.org/abs/2408.03314)
- [Rich Sutton's Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)
