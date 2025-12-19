# Search and Learn (clean-sal branch) - Workflow Documentation

## Overview

This is the **clean-sal** branch - a streamlined version of the HuggingFace Search and Learn repository focused on **test-time compute scaling** for math problem solving. It implements search-based inference methods that use **Process Reward Models (PRMs)** to guide LLMs toward better solutions by giving them more "time to think."

Based on the paper: [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (Snell et al., 2024)

## Repository Structure

```
search-and-learn/
├── src/sal/                    # Core library
│   ├── config.py               # Configuration dataclass (all parameters)
│   ├── search/                 # Search algorithm implementations
│   │   ├── best_of_n.py        # Best-of-N sampling
│   │   ├── beam_search.py      # Beam search with PRM guidance
│   │   ├── diverse_verifier_tree_search.py  # DVTS algorithm
│   │   └── utils.py            # Beam dataclass, generation utilities
│   ├── models/                 # Reward model implementations
│   │   ├── reward_models.py    # PRM classes (RLHFFlow, MathShepherd, etc.)
│   │   └── skywork_o1_prm/     # Skywork PRM model code
│   └── utils/                  # Utilities
│       ├── data.py             # Dataset loading and saving
│       ├── score.py            # Score aggregation and evaluation
│       ├── math.py             # Math answer extraction and voting
│       ├── parser.py           # Argument parsing
│       └── hub.py              # HuggingFace Hub utilities
├── scripts/
│   ├── test_time_compute.py    # Main entry point
│   └── merge_chunks.py         # Merge parallel job results
├── recipes/                    # YAML configs per model
│   ├── Llama-3.2-1B-Instruct/  # Llama 1B configs
│   ├── Llama-3.2-3B-Instruct/  # Llama 3B configs
│   ├── Qwen2.5-1.5B-Instruct/  # Qwen 1.5B configs
│   └── AceMath-7B-Instruct/    # AceMath configs
└── tests/                      # Unit tests
```

## Core Workflow

### Entry Point: `scripts/test_time_compute.py`

The main script orchestrates the entire pipeline:

```python
def main():
    # 1. Parse config from YAML + CLI args
    config = parser.parse()

    # 2. Initialize LLM (vLLM for fast inference)
    llm = LLM(model=config.model_path, ...)

    # 3. Load Process Reward Model
    prm = load_prm(config)

    # 4. Load dataset (default: MATH-500)
    dataset = get_dataset(config)

    # 5. Run search algorithm over dataset
    dataset = dataset.map(approach_fn, ...)

    # 6. Score and compute predictions
    dataset = score(dataset, config)

    # 7. Save results (local or Hub)
    save_dataset(dataset, config)
```

### Search Algorithms

#### 1. Best-of-N (`src/sal/search/best_of_n.py`)

The simplest approach - generate N complete solutions and pick the best:

```
For each problem:
  1. Generate N complete solutions using vLLM
  2. Score all solutions with PRM
  3. Select solution with highest aggregated score
```

**Key characteristics:**
- Generates full solutions in one pass
- Can subsample results for different compute budgets
- Supports continuous batching for efficiency

#### 2. Beam Search (`src/sal/search/beam_search.py`)

Step-by-step generation with pruning at each step:

```
For each problem:
  1. Initialize N beams
  2. For each iteration (up to num_iterations):
     a. Generate next step for all active beams (stop at "\n\n")
     b. Score partial completions with PRM
     c. Keep top N/beam_width beams, prune the rest
     d. Mark completed beams (hit EOS or max length)
  3. Return top N completed solutions
```

**Key characteristics:**
- Steps are delimited by double newlines (`\n\n`)
- Requires `search_batch_size=1` (non-batched)
- More compute-intensive but can find better solutions
- Supports lookahead for better scoring

#### 3. DVTS - Diverse Verifier Tree Search (`src/sal/search/diverse_verifier_tree_search.py`)

Maintains multiple independent search trees for diversity:

```
For each problem:
  1. Initialize n_beams = N/beam_width independent trees
  2. For each iteration:
     a. Generate beam_width candidates per tree
     b. Score all candidates with PRM
     c. Select best candidate per tree (local selection)
     d. Prune trees that hit EOS or "\boxed{"
  3. Expand final results to N solutions
```

**Key characteristics:**
- Promotes diversity through independent trees
- Local selection (within tree) vs global (beam search)
- Automatically prunes when answer is found (`\boxed{`)

### Process Reward Models (PRMs)

PRMs score partial solutions step-by-step (`src/sal/models/reward_models.py`):

| Model | Path | Notes |
|-------|------|-------|
| **RLHFFlow** (default) | `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` | Predicts +/- for each step |
| **MathShepherd** | `peiyi9979/math-shepherd-mistral-7b-prm` | Uses special `ки` token |
| **SkyworkO1** | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-{1.5B,7B}` | Custom PRM architecture |
| **Qwen2.5-Math** | `Qwen/Qwen2.5-Math-PRM-7B` | Uses `<extra_0>` step separator |

### Score Aggregation Strategies

Step-level scores are aggregated into a single score (`src/sal/utils/score.py`):

- **`last`** (default): Use the final step's score
- **`min`**: Use the minimum score across all steps
- **`prod`**: Multiply all step scores together

## Configuration (`src/sal/config.py`)

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `approach` | `best_of_n` | Search algorithm: `best_of_n`, `beam_search`, `dvts` |
| `model_path` | `meta-llama/Llama-3.2-1B-Instruct` | Generator LLM |
| `prm_path` | `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` | Process Reward Model |
| `n` | 4 | Number of completions/beams |
| `beam_width` | 4 | Beams per tree (for DVTS) |
| `num_iterations` | 40 | Max generation steps |
| `temperature` | 0.8 | Sampling temperature |
| `agg_strategy` | `last` | Score aggregation method |
| `dataset_name` | `HuggingFaceH4/MATH-500` | Evaluation dataset |

## Running Experiments

### Basic Usage

```bash
# Install
conda create -n sal python=3.11 && conda activate sal
pip install -e '.[dev]'

# Run Best-of-N (fast test - 10 problems)
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml

# Run Beam Search
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/beam_search.yaml

# Run DVTS
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/dvts.yaml
```

### Full Experiments

```bash
# Best-of-N with 256 completions
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
    --n=256 --num_samples=500 --seed=0 --push_to_hub=true

# Beam Search (run separately for each n)
for n in 4 16 64 256; do
    python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/beam_search.yaml \
        --n=$n --num_samples=500 --seed=0
done

# DVTS with 256 completions
python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/dvts.yaml \
    --n=256 --num_samples=500 --seed=0
```

### Overriding Parameters

```bash
# Different model
python scripts/test_time_compute.py $CONFIG --model_path=meta-llama/Llama-3.2-3B-Instruct

# Different PRM
python scripts/test_time_compute.py $CONFIG --prm_path=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B

# Different dataset
python scripts/test_time_compute.py $CONFIG --dataset_name=AI-MO/aimo-validation-aime
```

## Output Format

Results are saved as JSONL with these fields:

```json
{
  "problem": "Original math problem",
  "answer": "Ground truth answer",
  "completions": ["solution1", "solution2", ...],
  "scores": [[step_scores], [step_scores], ...],
  "pred": "Selected best solution",
  "completion_tokens": [token_counts]
}
```

## Hardware Requirements

- Experiments run on H100s (80GB VRAM)
- GPU memory split: 50% vLLM, 50% PRM (`gpu_memory_utilization=0.5`)
- Beam search/DVTS are much slower than Best-of-N (~60+ hrs vs ~3 hrs for 500 problems at n=256)

## Key Differences from Main Branch

The `clean-sal` branch is a focused, minimal version:
- No chat-specific scripts (`chat-scripts/`, `chat-recipes/`)
- No chat results data (`chat-results/`)
- No clustering scripts (`cluster-questions/`)
- No additional data processing pipelines
- Core search algorithms only

## References

- [HuggingFace Blog Post](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)
- [Snell et al. 2024 - Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314)
- [Rich Sutton's Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)
