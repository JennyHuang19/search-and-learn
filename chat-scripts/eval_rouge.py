#!/usr/bin/env python3
import argparse
import pandas as pd
from rouge_score import rouge_scorer
import time

def build_rouge_scorer(rouge_types=None):
    """Build ROUGE scorer with specified metrics."""
    if rouge_types is None:
        rouge_types = ["rougeL"]
    
    print(f"Initializing ROUGE scorer with metrics: {rouge_types}")
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    return scorer

def rouge_similarity(scorer, hypothesis, reference):
    """Compute ROUGE similarity between hypothesis and reference."""
    try:
        result = scorer.score(str(hypothesis), str(reference))
        # Return the f-measure for the first metric (usually rougeL)
        return result[list(result.keys())[0]].fmeasure
    except Exception as e:
        print(f"Error computing ROUGE similarity: {e}")
        return None

def evaluate_with_rouge(input_path: str, preferred_responses_path: str,
                       output_path: str = None, save_results: bool = True,
                       rouge_types: list = None):
    """
    Load scored generations and evaluate best responses against preferred responses using ROUGE.
    
    Args:
        input_path: Path to the JSONL file containing scored generations
        preferred_responses_path: Path to the CSV file containing preferred responses
        output_path: Path to save the evaluation results (optional)
        save_results: Whether to save results to file
        rouge_types: List of ROUGE metrics to use (default: ["rougeL"])
    
    Returns:
        DataFrame with original data plus ROUGE scores
    """
    print(f"Loading scored generations from: {input_path}")
    
    # Load the scored generations
    df_scored = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df_scored)} prompts with scored completions")
    
    # Check if best_response column exists
    if 'best_response' not in df_scored.columns:
        raise ValueError("Input file must contain 'best_response' column. Please run score-generations.py first.")
    
    print(f"Loading preferred responses from: {preferred_responses_path}")
    
    # Load the preferred responses
    preferred_responses = pd.read_csv(preferred_responses_path)
    print(f"Loaded {len(preferred_responses)} preferred responses")
    
    # Check if required columns exist
    required_cols = ["prompt", "preferred_response"]
    missing_cols = [col for col in required_cols if col not in preferred_responses.columns]
    if missing_cols:
        raise ValueError(f"Preferred responses file missing required columns: {missing_cols}")
    
    # Build ROUGE scorer
    scorer = build_rouge_scorer(rouge_types)
    
    # Merge on prompt to align preferred_response with best_response
    print("Merging data on prompts...")
    merged = pd.merge(
        df_scored,
        preferred_responses[["prompt", "preferred_response"]],
        on="prompt",
        how="left"
    )
    
    print(f"df_scored has {len(df_scored)} rows    ")
    print(f"Merged data has {len(merged)} rows     ")
    print(f"Rows with preferred responses: {merged['preferred_response'].notna().sum()}")
    
    # Compute ROUGE similarity for each merged row (handles possible duplicates)
    print("Computing ROUGE similarities...")
    t0 = time.time()
    
    rouge_scores_per_row = []
    for idx, row in merged.iterrows():
        if pd.notnull(row["preferred_response"]) and pd.notnull(row["best_response"]):
            score = rouge_similarity(scorer, row["best_response"], row["preferred_response"])
            rouge_scores_per_row.append(score)
        else:
            rouge_scores_per_row.append(None)
    
    merged["_score_rouge_row"] = rouge_scores_per_row
    total_time = time.time() - t0
    
    # Aggregate per prompt to get a single score per prompt (e.g., max over duplicates)
    per_prompt_scores = (
        merged.groupby("prompt")["_score_rouge_row"]
        .max()
    )
    
    # Map back to df_scored 1:1 length
    df_scored["score_rouge"] = df_scored["prompt"].map(per_prompt_scores)
    
    # Calculate summary statistics
    valid_scores = [s for s in df_scored["score_rouge"].tolist() if s is not None]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    print(f"\nROUGE evaluation completed!") # currently in a state where i don't know if the scores are correct due to df merging / indexing.
    print(f"Total evaluation time: {total_time:.2f}s")
    print(f"Average time per response: {total_time/len(df_scored):.3f}s")
    print(f"Total responses evaluated: {len(df_scored)}")
    print(f"Valid scores: {len(valid_scores)}/{len(df_scored)}")
    print(f"Average ROUGE score: {avg_score:.4f}")
    
    # Save results if requested
    if save_results and output_path:
        print(f"Saving evaluation results to: {output_path}")
        df_scored.to_json(output_path, orient='records', lines=True)
    
    return df_scored

def main():
    parser = argparse.ArgumentParser(description="Evaluate best responses against preferred responses using ROUGE")
    parser.add_argument("--input_path", required=True, 
                       help="Path to the JSONL file containing scored generations")
    parser.add_argument("--preferred_responses_path", required=True,
                       help="Path to the CSV file containing preferred responses")
    parser.add_argument("--output_path", 
                       help="Path to save the evaluation results (optional)")
    parser.add_argument("--rouge_types", nargs="+", default=["rougeL"],
                       help="ROUGE metrics to use (default: rougeL)")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Determine output path if not provided
    if not args.output_path and not args.no_save:
        # Create output path based on input path
        base_path = args.input_path.replace('.jsonl', '')
        if base_path.endswith('_scored'):
            base_path = base_path[:-7]  # Remove '_scored' suffix
        elif base_path.endswith('_eval_rm'):
            base_path = base_path[:-8]  # Remove '_eval_rm' suffix
        args.output_path = f"{base_path}_eval_rouge.jsonl"
    
    # Evaluate the generations
    df_evaluated = evaluate_with_rouge(
        input_path=args.input_path,
        preferred_responses_path=args.preferred_responses_path,
        output_path=args.output_path,
        save_results=not args.no_save,
        rouge_types=args.rouge_types
    )
    
    # Print summary
    print("\n" + "="*50)
    print("ROUGE EVALUATION SUMMARY")
    print("="*50)
    print(f"Input file: {args.input_path}")
    print(f"Preferred responses file: {args.preferred_responses_path}")
    print(f"ROUGE metrics: {args.rouge_types}")
    if args.output_path and not args.no_save:
        print(f"Output file: {args.output_path}")
    print(f"Total responses: {len(df_evaluated)}")
    
    # Calculate and display score statistics
    valid_scores = df_evaluated['score_rouge'].dropna()
    if len(valid_scores) > 0:
        print(f"Valid scores: {len(valid_scores)}/{len(df_evaluated)}")
        print(f"Average ROUGE score: {valid_scores.mean():.4f}")
        print(f"Min ROUGE score: {valid_scores.min():.4f}")
        print(f"Max ROUGE score: {valid_scores.max():.4f}")
        print(f"ROUGE score std: {valid_scores.std():.4f}")
    
    return df_evaluated

if __name__ == "__main__":
    df_result = main()
