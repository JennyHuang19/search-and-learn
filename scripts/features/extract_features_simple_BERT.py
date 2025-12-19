import torch
import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
import os
import argparse

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_bert(device: torch.device) -> tuple[BertModel, BertTokenizer]:
    # Prefer loading from a local directory if it exists
    # model_path = "/u/jhjenny9/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
    # use_local_only = os.path.isdir(model_path)
    # tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    # model = BertModel.from_pretrained(model_path, local_files_only=use_local_only)
    
    # Load BERT model and tokenizer from HuggingFace
    model_name = "bert-base-uncased"
    print(f"Loading BERT model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    # Move model to device
    model.to(device)
    model.eval()   # Enable evaluation mode (no dropout, frozen batch norm)
    
    return model, tokenizer

def extract_features_simple(input_csv, batch_size, output_dir):
    """
    Extract features from beam router training data and save to .npy files.
    
    Args:
        input_csv: DataFrame with columns ['problem', 'N', 'beam_width', 'max_iteration', 'sl']
        model_name: HuggingFace model name
        batch_size: Batch size for processing
        output_dir: Directory to save .npy files
    
    Returns:
        X_train: Feature matrix
        y_train: Target labels
    """

    df = pd.read_csv(input_csv)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model and tokenizer
    print(f"Loading model: BERT")
    device = get_device()
    # my_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    my_model, tokenizer = load_bert(device)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_model = my_model.to(device)
    print(f"Using device: {device}")
    
    all_features = []
    
    # method-specific features
    method_beam_search = df["method_beam_search"].values
    method_maj = df["method_maj"].values
    method_naive = df["method_naive"].values
    method_weighted = df["method_weighted"].values
    beam_sizes = df["N"].values
    beam_widths = df["beam_width"].values
    max_iterations = df["max_iteration"].values

    # problem-specific features
    problems = df["question"].tolist()
    # problem length
    problem_lengths = [len(p) for p in problems]
    
    print(f"Processing {len(problems)} problems in batches of {batch_size}")
    
    for i in tqdm(range(0, len(problems), batch_size)):
        batch_problems = problems[i:i+batch_size]
        
        # Tokenize as a batch
        inputs = tokenizer(batch_problems, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # outputs = my_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            outputs = my_model(**inputs)
        
        # Get last token embeddings for each sample in batch
        # last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
        # batch_embeddings = last_hidden[range(last_hidden.shape[0]), inputs["attention_mask"].sum(1)-1, :]  # [batch, hidden]
        # batch_embeddings = batch_embeddings.cpu().numpy()
        last_hidden = outputs.last_hidden_state  # [batch, seq, hidden] # a bit concerned as to whether this returns the CLS token for a batch of samples.
        batch_embeddings = last_hidden[:, 0, :]  # CLS token is at position 0 for each sample in the batch
        batch_embeddings = batch_embeddings.cpu().numpy()

        
        # Scalar features for this batch
        batch_beam_sizes = beam_sizes[i:i+batch_size]
        batch_beam_widths = beam_widths[i:i+batch_size]
        batch_max_iterations = max_iterations[i:i+batch_size]
        batch_problem_lengths = problem_lengths[i:i+batch_size]
        batch_method_beam_search = method_beam_search[i:i+batch_size]
        batch_method_maj = method_maj[i:i+batch_size]
        batch_method_naive = method_naive[i:i+batch_size]
        batch_method_weighted = method_weighted[i:i+batch_size]
        
        # Concatenate embeddings with scalar features
        batch_features = np.concatenate([
            batch_embeddings,
            np.stack([batch_beam_sizes, batch_beam_widths, batch_max_iterations, batch_problem_lengths, batch_method_beam_search, batch_method_maj, batch_method_naive, batch_method_weighted], axis=1)
        ], axis=1)
        
        all_features.append(batch_features)
    
    # Combine all batches
    X_train = np.concatenate(all_features, axis=0)
    y_train = df["sl"].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    
    # Save to .npy files
    X_path = os.path.join(output_dir, "X.npy")
    y_path = os.path.join(output_dir, "y.npy")
    
    np.save(X_path, X_train)
    np.save(y_path, y_train)
    
    print(f"Features saved to: {X_path}")
    print(f"Targets saved to: {y_path}")
    
    return X_train, y_train

def main():
    parser = argparse.ArgumentParser(description='Extract features from beam router CSV data')
    parser.add_argument('--input_csv', type=str, default='./data/heart/df_test_heart.csv', help='Path to input CSV file')
    parser.add_argument('--model_name', type=str, default='BERT', help='HuggingFace model name (default: BERT)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--output_dir', type=str, default='./data/heart/', help='Output directory for train/test .npy files')

    
    args = parser.parse_args()
    
    print(f"Input CSV: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    
    # Extract features
    X_train, y_train = extract_features_simple(
        input_csv=args.input_csv,
        # model_name=args.model_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    print("Feature extraction completed successfully!")

if __name__ == "__main__":
    main()