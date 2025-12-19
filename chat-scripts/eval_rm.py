#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import time

def build_eval_rm_pipeline(rm_model_name: str):
    """Build evaluation reward model pipeline."""
    print(f"Loading evaluation reward model: {rm_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(rm_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create a pipeline for reward scoring
    reward_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False,
        truncation=True,
        max_length=2048,
    )
    
    return reward_pipe

def get_reward_score(reward_pipe, text):
    """Get reward score for a single text."""
    # The reward model expects a string input
    try:
        result = reward_pipe(text)
        # result is a list of dicts, e.g. [{'label': 'LABEL_1', 'score': 0.87}]
        return result[0]['score']
    except Exception as e:
        print(f"Error scoring: {e}")
        return None

def evaluate_with_rm(input_path: str, eval_rm_model: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                    output_path: str = None, save_results: bool = True):
    """
    Load scored generations and evaluate best responses with a reward model.
    
    Args:
        input_path: Path to the JSONL file containing scored generations
        eval_rm_model: Name of the evaluation reward model to use
        output_path: Path to save the evaluation results (optional)
        save_results: Whether to save results to file
    
    Returns:
        DataFrame with original data plus evaluation scores
    """
    print(f"Loading scored generations from: {input_path}")
    
    # Load the scored generations
    df_scored = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df_scored)} prompts with scored completions")
    
    # Check if best_response column exists, fallback to best_completion
    if 'best_response' not in df_scored.columns:
        if 'best_completion' not in df_scored.columns:
            raise ValueError("Input file must contain either 'best_response' or 'best_completion' column. Please run score-generations.py first.")
        response_column = 'best_completion'
        print("Using 'best_completion' column for evaluation")
    else:
        response_column = 'best_response'
        print("Using 'best_response' column for evaluation")
    
    # Build evaluation reward model pipeline
    reward_pipe = build_eval_rm_pipeline(eval_rm_model)
    
    # Score each response
    print(f"Evaluating {response_column} with reward model...")
    t0 = time.time()
    
    # Use tqdm for progress tracking
    eval_scores = []
    for i, response in enumerate(tqdm(df_scored[response_column], desc="Evaluating responses")):
        score = get_reward_score(reward_pipe, response)
        eval_scores.append(score)
    
    total_time = time.time() - t0
    
    # Add evaluation scores to dataframe
    df_scored['eval_rm'] = eval_scores
    
    # Calculate summary statistics
    valid_scores = [s for s in eval_scores if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    print(f"\nEvaluation completed!")
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Average time per response: {total_time/len(df_scored):.3f}s")
    print(f"Total responses evaluated: {len(df_scored)}")
    print(f"Valid scores: {len(valid_scores)}/{len(eval_scores)}")
    print(f"Average evaluation score: {avg_score:.4f}")
    
    # Save results if requested
    if save_results and output_path:
        print(f"Saving evaluation results to: {output_path}")
        df_scored.to_json(output_path, orient='records', lines=True)
    
    return df_scored

def main():
    parser = argparse.ArgumentParser(description="Evaluate best responses with a reward model")
    parser.add_argument("--input_path", required=True, 
                       help="Path to the JSONL file containing scored generations")
    parser.add_argument("--eval_rm_model", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                       help="Name of the evaluation reward model to use")
    parser.add_argument("--output_path", 
                       help="Path to save the evaluation results (optional)")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Determine output path if not provided
    if not args.output_path and not args.no_save:
        # Create output path based on input path
        base_path = args.input_path.replace('.jsonl', '')
        if base_path.endswith('_scored'):
            base_path = base_path[:-7]  # Remove '_scored' suffix
        args.output_path = f"{base_path}_eval_rm.jsonl"
    
    # Evaluate the generations
    df_evaluated = evaluate_with_rm(
        input_path=args.input_path,
        eval_rm_model=args.eval_rm_model,
        output_path=args.output_path,
        save_results=not args.no_save
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Input file: {args.input_path}")
    print(f"Evaluation reward model: {args.eval_rm_model}")
    if args.output_path and not args.no_save:
        print(f"Output file: {args.output_path}")
    print(f"Total responses: {len(df_evaluated)}")
    
    # Calculate and display score statistics
    valid_scores = df_evaluated['eval_rm'].dropna()
    if len(valid_scores) > 0:
        print(f"Valid scores: {len(valid_scores)}/{len(df_evaluated)}")
        print(f"Average score: {valid_scores.mean():.4f}")
        print(f"Min score: {valid_scores.min():.4f}")
        print(f"Max score: {valid_scores.max():.4f}")
        print(f"Score std: {valid_scores.std():.4f}")
    
    return df_evaluated

if __name__ == "__main__":
    df_result = main()
