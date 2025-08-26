import argparse
import json
import os
from typing import Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_bert(model_path: str, device: torch.device, training_mode: bool = True) -> tuple[BertModel, BertTokenizer]:
    # Prefer loading from a local directory if it exists
    use_local_only = os.path.isdir(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
    model = BertModel.from_pretrained(model_path, local_files_only=use_local_only)
    model.to(device)
    
    # Control training mode based on parameter
    if training_mode:
        model.train()  # Enable training mode (dropout, batch norm updates)
    else:
        model.eval()   # Enable evaluation mode (no dropout, frozen batch norm)
    
    return model, tokenizer

def get_cls_embedding(
    text: str,
    model: BertModel,
    tokenizer: BertTokenizer,
    device: torch.device,
    enable_gradients: bool = True,  # New parameter to control gradient computation
) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if enable_gradients: # compute gradients. 
        outputs = model(**inputs) # backpropagate through the model.
    else:
        with torch.no_grad(): # disable gradient calculation for inference.
            outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token at position 0
    return cls_embedding.squeeze().detach().cpu().numpy()

def extract_features_bert(
    df: pd.DataFrame,
    model: BertModel,
    tokenizer: BertTokenizer,
    device: torch.device,
    progress_every: int = 100,
    enable_gradients: bool = False,
    fit_scaler: bool = False,
    scaler: dict = None,
) -> tuple[np.ndarray, dict]:
    # Ensure method columns exist; if missing, fill with 0
    method_cols = [
        "method_majority",
        "method_naive",
        "method_weighted",
        "method_beam_search",
    ]
    for col in method_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Number of times to repeat auxiliary features for signal strength
    AUX_FEATURE_REPETITIONS = 50
    
    features: list[np.ndarray] = []
    cls_features = []
    auxiliary_features = []
    
    for idx, row in df.iterrows():
        if progress_every and idx % progress_every == 0:
            print(f"Processing row {idx} of {len(df)}")

        cls_vec = get_cls_embedding(str(row["question"]), model, tokenizer, device, enable_gradients=enable_gradients)

        n_val = float(row["N"]) if pd.notna(row["N"]) else 0.0
        qlen_val = np.log(float(row["question_length"])) if pd.notna(row["question_length"]) else 0.0

        # Store method information
        m_majority = float(row.get("method_majority", 0.0) or 0.0)
        m_naive = float(row.get("method_naive", 0.0) or 0.0)
        m_weighted = float(row.get("method_weighted", 0.0) or 0.0)
        m_beam = float(row.get("method_beam_search", 0.0) or 0.0)

        # Create auxiliary feature vector
        aux_features = np.array([n_val, qlen_val, m_majority, m_naive, m_weighted, m_beam], dtype=np.float32)
        
        # Repeat auxiliary features to increase signal strength
        repeated_aux_features = np.tile(aux_features, AUX_FEATURE_REPETITIONS)
        
        # Store features separately for multi-layer injection
        cls_features.append(cls_vec)
        auxiliary_features.append(repeated_aux_features)
    
    # Convert to numpy arrays
    X_embeddings = np.array(cls_features)
    X_auxiliary = np.array(auxiliary_features)
    
    # Normalize features using StandardScaler
    if fit_scaler:
        # Fit StandardScaler on training data
        scaler_emb = StandardScaler()
        scaler_aux = StandardScaler()

        X_embeddings_normalized = scaler_emb.fit_transform(X_embeddings)
        X_auxiliary_normalized = scaler_aux.fit_transform(X_auxiliary)

        # Save scalers for later use
        scaler = {
            'embedding': scaler_emb,
            'auxiliary': scaler_aux
        }
    elif scaler is not None:
        # Transform test data using fitted scaler
        X_embeddings_normalized = scaler['embedding'].transform(X_embeddings)
        X_auxiliary_normalized = scaler['auxiliary'].transform(X_auxiliary)
    else:
        # No scaling - use raw features
        X_embeddings_normalized = X_embeddings
        X_auxiliary_normalized = X_auxiliary
    
    # Concatenate for backward compatibility (original format)
    X_combined = np.concatenate([X_embeddings_normalized, X_auxiliary_normalized], axis=1)
    
    # Store metadata
    meta = {
        "scalar_dim": int(6 * AUX_FEATURE_REPETITIONS),  # Updated to reflect repetitions
        "embedding_dim": int(X_embeddings.shape[1]),
        "total_dim": int(X_combined.shape[1]),
        "auxiliary_repetitions": int(AUX_FEATURE_REPETITIONS),
        "scaler": {
            'embedding': {
                'scale_': scaler['embedding'].scale_.tolist() if scaler and 'embedding' in scaler else None,
                'mean_': scaler['embedding'].mean_.tolist() if scaler and 'embedding' in scaler else None,
                'var_': scaler['embedding'].var_.tolist() if scaler and 'embedding' in scaler else None,
                'n_samples_seen_': int(scaler['embedding'].n_samples_seen_) if scaler and 'embedding' in scaler else None
            },
            'auxiliary': {
                'scale_': scaler['auxiliary'].scale_.tolist() if scaler and 'auxiliary' in scaler else None,
                'mean_': scaler['auxiliary'].mean_.tolist() if scaler and 'auxiliary' in scaler else None,
                'var_': scaler['auxiliary'].var_.tolist() if scaler and 'auxiliary' in scaler else None,
                'n_samples_seen_': int(scaler['auxiliary'].n_samples_seen_) if scaler and 'auxiliary' in scaler else None
            }
        } if scaler else None,  # Include the fitted scalers as JSON-serializable dict
    }
    
    return X_combined, meta


def main(
    csv_path: str,
    output_npy: str,
    output_meta_json: Optional[str],
    model_path: str,
    enable_gradients: bool = False,
    fit_scaler: bool = False,
    scaler_path: Optional[str] = None,  # New parameter for scaler file path
    prefer_cuda: bool = True,
):
    device = get_device(prefer_cuda=prefer_cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("Loading model from:", model_path)

    model, tokenizer = load_bert(model_path, device, training_mode=enable_gradients)

    # Load pre-fitted scaler if provided
    scaler = None
    if scaler_path and not fit_scaler:
        try:
            with open(scaler_path, 'r') as f:
                scaler_data = json.load(f)
            # Reconstruct scalers from saved data
            scaler_emb = StandardScaler()
            scaler_aux = StandardScaler()
            scaler_emb.scale_ = np.array(scaler_data['scaler']['embedding']['scale_'])
            scaler_emb.mean_ = np.array(scaler_data['scaler']['embedding']['mean_'])
            scaler_emb.var_ = np.array(scaler_data['scaler']['embedding']['var_'])
            scaler_emb.n_samples_seen_ = scaler_data['scaler']['embedding']['n_samples_seen_']
            scaler_aux.scale_ = np.array(scaler_data['scaler']['auxiliary']['scale_'])
            scaler_aux.mean_ = np.array(scaler_data['scaler']['auxiliary']['mean_'])
            scaler_aux.var_ = np.array(scaler_data['scaler']['auxiliary']['var_'])
            scaler_aux.n_samples_seen_ = scaler_data['scaler']['auxiliary']['n_samples_seen_']
            scaler = {'embedding': scaler_emb, 'auxiliary': scaler_aux}
            print(f"Loaded pre-fitted scaler from {scaler_path}")
        except Exception as e:
            print(f"Warning: Failed to load scaler from {scaler_path}: {e}")
            print("Proceeding without scaler (raw features will be used)")
            scaler = None

    df = pd.read_csv(csv_path)
    X, meta = extract_features_bert(
        df, model, tokenizer, device, 
        enable_gradients=enable_gradients, 
        fit_scaler=fit_scaler,
        scaler=scaler
    )

    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
    np.save(output_npy, X)
    print(f"Saved features: {output_npy} with shape {X.shape}")

    if output_meta_json:
        os.makedirs(os.path.dirname(output_meta_json) or ".", exist_ok=True)
        with open(output_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {output_meta_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract BERT CLS features + scalar features from CSV.")
    parser.add_argument("--csv", dest="csv_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument(
        "--out",
        dest="output_npy",
        type=str,
        required=True,
        help="Path to output .npy file for features",
    )
    parser.add_argument(
        "--meta",
        dest="output_meta_json",
        type=str,
        default=None,
        help="Optional path to save JSON with normalization metadata",
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        type=str,
        default="/u/jhjenny9/.cache/huggingface/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
        help=(
            "Local model directory (preferred) or HF model id. "
            "Defaults to your local bert-base-uncased snapshot."
        ),
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )
    parser.add_argument(
        "--enable-gradients",
        dest="enable_gradients",
        action="store_true",
        help="Enable gradient computation for training/fine-tuning (default: False)",
    )
    parser.add_argument(
        "--fit-scaler",
        dest="fit_scaler",
        action="store_true",
        help="Fit new scalers (default: False)",
    )
    parser.add_argument(
        "--scaler",
        dest="scaler_path",
        type=str,
        help="Path to JSON file containing pre-fitted scaler parameters (for validation/inference)",
    )
    args = parser.parse_args()
    main(
        csv_path=args.csv_path,
        output_npy=args.output_npy,
        output_meta_json=args.output_meta_json,
        model_path=args.model_path,
        prefer_cuda=not args.no_cuda,
        enable_gradients=args.enable_gradients,
        fit_scaler=args.fit_scaler,
        scaler_path=args.scaler_path,
    )


