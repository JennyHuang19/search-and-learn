import pandas as pd
import numpy as np
import argparse
import os
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler
import pickle

# =========================
# Set Seed for Reproducibility
# =========================
def set_seed(seed=6):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# Set Device
# =========================
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Qwen Model Setup
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
qwen = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def get_embedding(text, model, tokenizer, device):
    """
    Extracts an embedding from the last hidden layer of Qwen.
    Uses average pooling over non-padding tokens (mask-aware mean).
    """
    # Tokenize (returns a dict of tensors on CPU)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Manually move each tensor to the right device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Also move the model to the right device (only once ideally, not in this function!)
    model = model.to(device)
    model.eval()

    # Run model forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    # Last hidden state: shape (batch_size, seq_len, hidden_dim)
    last_hidden = outputs.hidden_states[-1]

    # Attention mask: shape (batch_size, seq_len)
    attention_mask = inputs["attention_mask"]

    # Average pooling over all tokens (excluding padding tokens)
    # Zero out padded tokens, then divide the sum by the count of valid tokens per example.
    masked_embeddings = last_hidden * attention_mask.unsqueeze(-1)
    sum_embeddings = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    avg_embeddings = sum_embeddings / lengths

    return avg_embeddings.squeeze().cpu().numpy()

# =========================
# Improved Feature Processing
# =========================
def process_dataframe_improved(df, method_col_prefix, scaler=None, fit_scaler=True):
    """
    Improved feature processing with:
    1. Feature normalization
    2. Repeated indicator features for signal strength
    3. Separation of embedding and auxiliary features for multi-layer injection
    4. Method information preservation for analysis (for keeping track of losses by method)
    """
    cls_features = []
    auxiliary_features = []
    method_info = []
    

    # Number of times to repeat auxiliary features for signal strength
    AUX_FEATURE_REPETITIONS = 10
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting improved features"):
        if _ % 100 == 0:
            print('Processing question number:', _, 'out of', len(df))
        
        # Get embedding
        cls_vec = get_embedding(row["question"], qwen, tokenizer, device)  # e.g., shape (1536,)
        
        # Extract auxiliary features
        num_samples = row["N"]
        question_length = np.log(row["question_length"])
        method_maj = row[f"{method_col_prefix}majority"]
        method_naive = row[f"{method_col_prefix}naive"]
        method_weighted = row[f"{method_col_prefix}weighted"]
        method_beam_search = row[f"{method_col_prefix}beam_search"]
        
        # Store method information for analysis
        method_name = row.get("method", "unknown")  # Get the actual method used
        method_info.append({
            'method': method_name,
            'method_maj': method_maj,
            'method_naive': method_naive, 
            'method_weighted': method_weighted,
            'method_beam_search': method_beam_search
        })
        
        # Create auxiliary feature vector
        aux_features = np.array([num_samples, question_length, method_maj, method_naive, method_weighted, method_beam_search])
        
        # Repeat auxiliary features to increase signal strength
        repeated_aux_features = np.tile(aux_features, AUX_FEATURE_REPETITIONS)
        
        # Store features separately for multi-layer injection
        cls_features.append(cls_vec)
        auxiliary_features.append(repeated_aux_features)
    
    # Convert to numpy arrays
    X_embeddings = np.array(cls_features)
    X_auxiliary = np.array(auxiliary_features)
    

    # Normalize features using RobustScaler
    if fit_scaler:
        # Fit RobustScaler on training data
        scaler_emb = RobustScaler()
        scaler_aux = RobustScaler()

        X_embeddings_normalized = scaler_emb.fit_transform(X_embeddings)
        X_auxiliary_normalized = scaler_aux.fit_transform(X_auxiliary)

        # Save scalers for later use
        scaler = {
            'embedding': scaler_emb,
            'auxiliary': scaler_aux
        }
    else:
        # Transform test data using fitted scaler
        X_embeddings_normalized = scaler['embedding'].transform(X_embeddings)
        X_auxiliary_normalized = scaler['auxiliary'].transform(X_auxiliary)

    
    # Concatenate for backward compatibility (original format)
    X_combined = np.concatenate([X_embeddings_normalized, X_auxiliary_normalized], axis=1)
    
    return X_combined, X_embeddings_normalized, X_auxiliary_normalized, method_info, scaler


# =========================
# Main Script
# =========================
def main(train_path, test_path, output_dir, seed=6):
    set_seed(seed)  # Ensure reproducibility
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Process training data
    print("Processing training data...")
    X_train_combined, X_train_emb, X_train_aux, method_info_train, scaler = process_dataframe_improved(
        df_train, method_col_prefix="method_", fit_scaler=True
    )
    
    # Process test data using fitted scaler
    print("Processing test data...")
    X_test_combined, X_test_emb, X_test_aux, method_info_test, _ = process_dataframe_improved(
        df_test, method_col_prefix="method_", scaler=scaler, fit_scaler=False
    )
    
    # Get labels
    y_train = df_train["sl"].values.astype(float)
    y_test = df_test["sl"].values.astype(float)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all versions of the data
    np.save(os.path.join(output_dir, "X_train_combined.npy"), X_train_combined)
    np.save(os.path.join(output_dir, "X_train_embeddings.npy"), X_train_emb)
    np.save(os.path.join(output_dir, "X_train_auxiliary.npy"), X_train_aux)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    
    np.save(os.path.join(output_dir, "X_test_combined.npy"), X_test_combined)
    np.save(os.path.join(output_dir, "X_test_embeddings.npy"), X_test_emb)
    np.save(os.path.join(output_dir, "X_test_auxiliary.npy"), X_test_aux)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    # Save scaler for later use
    with open(os.path.join(output_dir, "feature_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save method information for analysis
    with open(os.path.join(output_dir, "method_info_train.pkl"), "wb") as f:
        pickle.dump(method_info_train, f)
    with open(os.path.join(output_dir, "method_info_test.pkl"), "wb") as f:
        pickle.dump(method_info_test, f)

    print(f"\nSaved enhanced features to {output_dir}")
    print(f"Feature dimensions:")
    print(f"  Combined features: {X_train_combined.shape}")
    print(f"  Embedding features: {X_train_emb.shape}")
    print(f"  Auxiliary features: {X_train_aux.shape}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to df_train.csv")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to df_test.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save X/y arrays")
    args = parser.parse_args()

    main(args.train_csv, args.test_csv, args.output_dir)
