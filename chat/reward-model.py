import torch
from transformers import AutoTokenizer, pipeline

def main():
    model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B" # Skywork/Skywork-Reward-V2-Llama-3.1-8B, NCSOFT/Llama-3-OffsetBias-RM-8B

    # Load tokenizer and pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm_pipeline = pipeline(
        "sentiment-analysis", # a sequence classification head that outputs logits (originally used to classify positive/negative sentiments).
        model=model_name,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # 0 = first GPU, -1 = CPU
        model_kwargs={"torch_dtype": torch.bfloat16},   # use bf16 for speed/memory
        return_all_scores=True,
        function_to_apply="none"  # don't apply softmax
    )

    def score_response(prompt: str, response: str) -> float: # helper to score one (prompt, response) pair
        """Compute reward model score for a (prompt, response) pair."""
        input_text = prompt + "\n" + response # Many RMs were trained on “prompt+answer” concatenations
        outputs = rm_pipeline(input_text) 
        scores = [o["score"] for o in outputs[0]]
        return max(scores)  # adjust if model outputs multiple reward heads

    # Example usage
    prompt = "What are the benefits of exercise?"
    response = "Exercise improves mental health, physical strength, and overall well-being."
    score = score_response(prompt, response)

    print(f"Score for response: {score:.4f}")

if __name__ == "__main__":
    main()
