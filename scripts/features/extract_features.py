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
# BERT Model Setup
# =========================
print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# local_model_path = "/u/jhjenny9/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
# # tokenizer = BertTokenizer.from_pretrained(local_model_path)
# model = BertModel.from_pretrained(local_model_path).to(device)
# model.eval()
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# =========================
# CLS Embedding Function
# =========================
# def get_cls_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
#     return cls_embedding.squeeze().cpu().numpy()

### =========================
# Qwen Model Setup
# =========================
from transformers import AutoTokenizer, AutoModelForCausalLM
# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
qwen = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True) # try Math PRM embeddings.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def get_embedding(text, model, tokenizer, device): # can try out BERT w/ learnable weights.
    """
    Extracts the average token embedding from the last hidden layer of Qwen.
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

    # Attention mask: shape (batch_size, seq_len, 1)
    mask = inputs["attention_mask"].unsqueeze(-1)

    # Get the last token embedding from the last layer
    # Find the last non-padded token position for each sequence (pooling method - try out variations. pool (mean of every state, or max pool), because no CLS token exists, last hidden layer of all tokens.)
    # predict chance of success from every token (plot of token vs. prediction error, can pool in intermediate layers of the MLP)... then a pooling operation to go from a vector of successes to a single scalar.
    seq_lengths = mask.sum(dim=1).squeeze(-1).long() - 1  # -1 because indexing is 0-based
    batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
    last_token_embeddings = last_hidden[batch_indices, seq_lengths]  # shape (batch_size, hidden_dim)

    return last_token_embeddings.squeeze().cpu().numpy()



# =========================
# Feature Processing
# =========================
def process_dataframe(df, method_col_prefix):
    cls_features = []

    # Scaling factors
    NUM_SAMPLES_SCALE = 1.0
    QUESTION_LENGTH_SCALE = 1.0
    METHOD_SCALE = 1.0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        if _ % 100 == 0:
            print('Processing question number:', _, 'out of', len(df))
        # cls_vec = get_cls_embedding(row["question"])  # e.g., shape (768,)
        cls_vec = get_embedding(row["question"], qwen, tokenizer, device)  # e.g., shape (1536,)

        # Scalar features
        num_samples = row["N"] * NUM_SAMPLES_SCALE # normalize all features (1/largest_N)
        question_length = np.log(row["question_length"]) * QUESTION_LENGTH_SCALE
        method_maj = row[f"{method_col_prefix}majority"] * METHOD_SCALE
        method_naive = row[f"{method_col_prefix}naive"] * METHOD_SCALE
        method_weighted = row[f"{method_col_prefix}weighted"] * METHOD_SCALE
        method_beam_search = row[f"{method_col_prefix}beam_search"] * METHOD_SCALE

        features = np.concatenate([
            cls_vec,
            [num_samples, question_length, method_maj, method_naive, method_weighted, method_beam_search] # repeat features a bunch of times. feed into every layer as input.
        ])
        cls_features.append(features)

    X = np.array(cls_features)
    y = df["sl"].values.astype(float)
    return X, y


# =========================
# Main Script
# =========================
def main(train_path, test_path, output_dir, seed=6):
    set_seed(seed)  # Ensure reproducibility
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train, y_train = process_dataframe(df_train, method_col_prefix="method_")
    X_test, y_test = process_dataframe(df_test, method_col_prefix="method_")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"\nSaved features to {output_dir}")


    # =========================
    # Train Probe and Save Predicted Probabilities
    # =========================
    # print("\n Training 2-layer MLP classifier...")
    # mlp_clf = MLPClassifier(
    #     hidden_layer_sizes=(200, 200),
    #     activation='relu',
    #     max_iter=500,
    #     random_state=seed
    # )
    # mlp_clf.fit(X_train, y_train)

    # # Get predicted probabilities and labels
    # y_proba = mlp_clf.predict_proba(X_test)[:, 1]
    # # Convert probabilities to binary labels
    # y_pred = (y_proba >= 0.5).astype(int)

    # # Accuracy report
    # acc = accuracy_score(y_test, y_pred)
    # print(f"Binary Classification Accuracy: {acc:.4f}")

    # # Save predicted probabilities for later use
    # np.save(os.path.join(output_dir, "test_probs.npy"), y_proba)
    # print(f"Saved predicted probabilities to {os.path.join(output_dir, 'test_probs.npy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to df_train.csv")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to df_test.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save X/y arrays")
    args = parser.parse_args()

    main(args.train_csv, args.test_csv, args.output_dir)
