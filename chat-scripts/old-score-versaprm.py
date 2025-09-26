#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def get_tokenizer(model_id):
    """Get tokenizer with proper configuration for VersaPRM."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  
    tokenizer.padding_side = 'left' 
    tokenizer.truncation_side = 'left'
    return tokenizer

def build_versaprm_model(model_id="UW-Madison-Lee-Lab/VersaPRM"):
    """Build VersaPRM model and tokenizer."""
    print(f"Loading VersaPRM model: {model_id}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    
    return model, tokenizer, device

def score_completion_versaprm(model, tokenizer, device, prompt, completion, candidate_tokens=[12, 10]):
    """
    Score a single prompt-completion pair using VersaPRM.
    
    Args:
        model: VersaPRM model
        tokenizer: VersaPRM tokenizer
        device: Device to run on
        prompt: Input prompt
        completion: Generated completion
        candidate_tokens: Token IDs for scoring (default: [12, 10] for VersaPRM)
    
    Returns:
        float: Score for the completion
    """
    try:
        # Format input as per VersaPRM requirements
        input_text = prompt + ' \n\n' + completion + ' \n\n\n\n'
        input_id = torch.tensor([tokenizer.encode(input_text)]).to(device)
        
        with torch.no_grad():
            logits = model(input_id).logits[:,:,candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,1] 
            # Get the score for the last token (end of completion)
            final_score = scores[0, -1].item()
            
        return final_score
    except Exception as e:
        print(f"Error scoring completion: {e}")
        return 0.0

def score_completions_versaprm(model, tokenizer, device, completions, prompt, candidate_tokens=[12, 10]):
    """
    Score multiple completions for a single prompt using VersaPRM.
    
    Args:
        model: VersaPRM model
        tokenizer: VersaPRM tokenizer
        device: Device to run on
        completions: List of completions to score
        prompt: Input prompt
        candidate_tokens: Token IDs for scoring
    
    Returns:
        tuple: (scores, best_response, best_idx)
    """
    scores = []
    for completion in completions:
        score = score_completion_versaprm(model, tokenizer, device, prompt, completion, candidate_tokens)
        scores.append(score)
    
    best_idx = int(np.argmax(scores))
    best_response = completions[best_idx]
    
    return scores, best_response, best_idx

def score_generations_versaprm(input_path: str, output_path: str = None, save_results: bool = True):
    """
    Load generations and score them with VersaPRM.
    
    Args:
        input_path: Path to the JSONL file containing generations
        output_path: Path to save the scored results (optional)
        save_results: Whether to save results to file
    
    Returns:
        DataFrame with original data plus VersaPRM scores
    """
    print(f"Loading generations from: {input_path}")
    
    # Load the generations
    df_generations = pd.read_json(input_path, lines=True)
    print(f"Loaded {len(df_generations)} prompts with completions")
    
    # Build VersaPRM model
    model, tokenizer, device = build_versaprm_model()
    
    # Score all completions
    print("Scoring completions with VersaPRM...")
    t0 = time.time()
    
    all_scores = []
    best_responses = []
    score_times = []
    
    for i, row in enumerate(df_generations.iterrows()):
        idx, row_data = row
        completions = row_data["completions"]
        prompt = row_data["prompt"]
        
        if i % 10 == 0:
            print(f"  Processing prompt {i+1}/{len(df_generations)}")
        
        prompt_start = time.time()
        scores, best_response, best_idx = score_completions_versaprm(
            model, tokenizer, device, completions, prompt
        )
        prompt_time = time.time() - prompt_start
        
        all_scores.append(scores)
        best_responses.append(best_response)
        score_times.append(prompt_time)
    
    total_time = time.time() - t0
    
    # Add results to dataframe
    df_generations["versaprm_scores"] = all_scores
    df_generations["best_response"] = best_responses
    df_generations["score_time"] = score_times
    
    # Calculate summary statistics
    avg_score = np.mean([np.mean(scores) for scores in all_scores])
    total_score_time = sum(score_times)
    
    print(f"\nVersaPRM scoring completed!")
    print(f"Total scoring time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(df_generations):.3f}s")
    print(f"Total prompts processed: {len(df_generations)}")
    print(f"Average VersaPRM score: {avg_score:.4f}")
    
    # Save results if requested
    if save_results and output_path:
        print(f"Saving results to: {output_path}")
        df_generations.to_json(output_path, orient='records', lines=True)
    
    return df_generations

def main():
    parser = argparse.ArgumentParser(description="Score generated completions with VersaPRM")
    parser.add_argument("--input_path", required=True, 
                       help="Path to the JSONL file containing generations")
    parser.add_argument("--output_path", 
                       help="Path to save the scored results (optional)")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results to file")
    
    args = parser.parse_args()
    
    # Determine output path if not provided
    if not args.output_path and not args.no_save:
        # Create output path based on input path
        base_path = args.input_path.replace('.jsonl', '')
        args.output_path = f"{base_path}_versaprm_scored.jsonl"
    
    # Score the generations
    df_scored = score_generations_versaprm(
        input_path=args.input_path,
        output_path=args.output_path,
        save_results=not args.no_save
    )
    
    # Print summary
    print("\n" + "="*50)
    print("VERSA PRM SCORING SUMMARY")
    print("="*50)
    print(f"Input file: {args.input_path}")
    print(f"Model: UW-Madison-Lee-Lab/VersaPRM")
    if args.output_path and not args.no_save:
        print(f"Output file: {args.output_path}")
    print(f"Total prompts: {len(df_scored)}")
    
    # Calculate and display score statistics
    all_scores = [score for scores in df_scored['versaprm_scores'] for score in scores]
    if all_scores:
        print(f"Average score: {np.mean(all_scores):.4f}")
        print(f"Min score: {np.min(all_scores):.4f}")
        print(f"Max score: {np.max(all_scores):.4f}")
        print(f"Score std: {np.std(all_scores):.4f}")
    
    return df_scored

if __name__ == "__main__":
    df_result = main()
