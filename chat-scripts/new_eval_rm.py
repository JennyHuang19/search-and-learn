#!/usr/bin/env python3
import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import time

def load_reward_model(rm_model_name: str):
    """Load reward model and tokenizer following the exact pattern from the example."""
    print(f"Loading evaluation reward model: {rm_model_name}")
    
    # Use cuda:0 if available, otherwise cpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Try to load model with flash attention first, fall back if not available
    try:
        print("Attempting to load with Flash Attention 2...")
        model = AutoModelForSequenceClassification.from_pretrained(
            rm_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        print("✓ Successfully loaded with Flash Attention 2")
    except ImportError as e:
        if "flash_attn" in str(e):
            print("Flash Attention 2 not available, falling back to default attention...")
            model = AutoModelForSequenceClassification.from_pretrained(
                rm_model_name,
                torch_dtype=torch.bfloat16,
                device_map=device,
                num_labels=1,
            )
            print("Successfully loaded with default attention")
        else:
            raise e
    
    tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
    
    return model, tokenizer, device

def get_reward_score_for_conversation(model, tokenizer, device, conversation):
    """Get reward score for a conversation using the exact pattern from the example."""
    try:
        # Format the conversation using chat template
        conv_formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
        
        # Remove potential duplicate bos token
        if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
            conv_formatted = conv_formatted[len(tokenizer.bos_token):]
        
        # Tokenize the conversation
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)
        
        # Get the reward score
        with torch.no_grad():
            score = model(**conv_tokenized).logits[0][0].item()
        
        return score
    except Exception as e:
        print(f"Error scoring conversation: {e}")
        return None

def get_reward_score_for_text(model, tokenizer, device, text, prompt):
    """Get reward score for plain text by wrapping it in a conversation format."""
    # Wrap plain text in a simple conversation format
    conversation = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}]
    return get_reward_score_for_conversation(model, tokenizer, device, conversation)

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
    
    # Load reward model
    model, tokenizer, device = load_reward_model(eval_rm_model)
    
    # Score each response
    print(f"Evaluating {response_column} with reward model...")
    t0 = time.time()
    
    # Use tqdm for progress tracking
    eval_scores = []
    for i, response in enumerate(tqdm(df_scored[response_column], desc="Evaluating responses")):
        score = get_reward_score_for_text(model, tokenizer, device, response, df_scored.iloc[i]['prompt'])
        # print the beginning of the question.
        print(f"Question: {df_scored.iloc[i]['prompt'][:100]}")
        eval_scores.append(score)
        print(f"Score for response {i}: {score}")
    
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

def test_with_examples():
    """Test the reward model with the provided examples."""
    print("Testing reward model with provided examples...")
    
    # Load the model
    model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    model, tokenizer, device = load_reward_model(model_name)
    
    # Define the test examples
    prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
    response1 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."
    response2 = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."

    conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
    conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]
    
    # Get scores for both conversations
    print("Scoring conversations...")
    score1 = get_reward_score_for_conversation(model, tokenizer, device, conv1)
    score2 = get_reward_score_for_conversation(model, tokenizer, device, conv2)
    
    print(f"\nResults:")
    print(f"Score for response 1 (correct): {score1}")
    print(f"Score for response 2 (incorrect): {score2}")
    
    if score1 is not None and score2 is not None:
        if score1 > score2:
            print("✓ Correct! Response 1 (correct answer) scored higher than Response 2 (incorrect answer)")
        else:
            print("✗ Unexpected! Response 2 (incorrect answer) scored higher than Response 1 (correct answer)")
    else:
        print("⚠ Could not compare scores due to errors")
    
    return score1, score2

def main():
    parser = argparse.ArgumentParser(description="Evaluate best responses with a reward model")
    parser.add_argument("--input_path", 
                       help="Path to the JSONL file containing scored generations")
    parser.add_argument("--eval_rm_model", default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
                       help="Name of the evaluation reward model to use")
    parser.add_argument("--output_path", 
                       help="Path to save the evaluation results (optional)")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results to file")
    parser.add_argument("--test", action="store_true",
                       help="Run test with provided examples instead of processing input file")
    
    args = parser.parse_args()
    
    # If test flag is set, run the test
    if args.test:
        score1, score2 = test_with_examples()
        return score1, score2
    
    # Otherwise require input_path for normal operation
    if not args.input_path:
        parser.error("--input_path is required unless using --test flag")
    
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
