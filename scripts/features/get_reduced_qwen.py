import argparse
import json
import os
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_qwen(model_name: str, device: torch.device, training_mode: bool = True) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Load Qwen model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    
    # Control training mode based on parameter
    if training_mode:
        model.train()  # Enable training mode (dropout, batch norm updates)
    else:
        model.eval()   # Enable evaluation mode (no dropout, frozen batch norm)
    
    return model, tokenizer


def get_qwen_embedding(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    enable_gradients: bool = True,  # New parameter to control gradient computation
    pooling_method: str = "mean",  # New parameter: "mean" or "max"
) -> np.ndarray:
    """
    Extracts an embedding from all tokens in the last hidden layer
    using either mean or max pooling for a given input text using Qwen model.
    
    Args:
        text (str): Input query
        model (AutoModelForCausalLM): Qwen model
        tokenizer (AutoTokenizer): Corresponding tokenizer
        device (torch.device): Device to run on
        enable_gradients (bool): Whether to enable gradient computation
        pooling_method (str): Pooling method - "mean" or "max" (default: "mean")
    
    Returns:
        np.ndarray: Embedding vector of shape (hidden_dim,)
    """
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)

    if enable_gradients:
        # Run the model to get hidden states with gradients
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    else:
        # Run the model to get hidden states without gradients
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

    # Extract all token embeddings from the last hidden layer and apply pooling
    last_hidden_state = outputs.hidden_states[-1]  # shape: [1, seq_len, hidden_dim]
    
    # Apply pooling method
    if pooling_method == "max":
        # Max pooling over all tokens
        embedding = last_hidden_state.max(dim=1)[0]  # shape: [1, hidden_dim]
    else:
        # Mean pooling over all tokens (default)
        embedding = last_hidden_state.mean(dim=1)    # shape: [1, hidden_dim]

    return embedding.squeeze().detach().cpu().numpy()


def mean_l2(X: np.ndarray) -> float:
    """Compute mean L2 norm of rows in X."""
    return np.mean(np.linalg.norm(X, axis=1))


def safe_n_samples_seen(value):
    """Safely extract n_samples_seen value, handling both scalars and arrays."""
    if value is None:
        return None
    try:
        if hasattr(value, 'item'):
            return int(value.item())
        elif hasattr(value, '__len__') and len(value) == 1:
            return int(value[0])
        else:
            return int(value)
    except (ValueError, TypeError, IndexError):
        return None


def extract_features_qwen(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    progress_every: int = 100,
    enable_gradients: bool = False,
    fit_scaler: bool = False,
    scaler: dict = None,
    pca_components: int = 32,
    aux_weight_multiplier: float = 1.0,
    pooling_method: str = "mean",  # New parameter for pooling method
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
    AUX_FEATURE_REPETITIONS = 1
    
    features: list[np.ndarray] = []
    qwen_features = []
    auxiliary_features = []
    
    for idx, row in df.iterrows():
        if progress_every and idx % progress_every == 0:
            print(f"Processing row {idx} of {len(df)}")

        qwen_vec = get_qwen_embedding(str(row["question"]), model, tokenizer, device, enable_gradients=enable_gradients, pooling_method=pooling_method)

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
        qwen_features.append(qwen_vec)
        auxiliary_features.append(repeated_aux_features)
    
    # Convert to numpy arrays
    X_embeddings = np.array(qwen_features)
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
        if 'embedding' in scaler and hasattr(scaler['embedding'], 'transform'):
            X_embeddings_normalized = scaler['embedding'].transform(X_embeddings)
        else:
            print("Warning: No valid embedding scaler found, using raw features")
            X_embeddings_normalized = X_embeddings
            
        if 'auxiliary' in scaler and hasattr(scaler['auxiliary'], 'transform'):
            X_auxiliary_normalized = scaler['auxiliary'].transform(X_auxiliary)
        else:
            print("Warning: No valid auxiliary scaler found, using raw features")
            X_auxiliary_normalized = X_auxiliary
    else:
        # No scaling - use raw features
        X_embeddings_normalized = X_embeddings
        X_auxiliary_normalized = X_auxiliary
    
    # Apply PCA to reduce Qwen embeddings to specified dimensions
    print(f"PCA components requested: {pca_components}")
    print(f"fit_scaler: {fit_scaler}")
    print(f"scaler has PCA: {'pca' in scaler if scaler else 'No scaler'}")
    print(f"Input embedding dimensions: {X_embeddings_normalized.shape}")
    if scaler:
        print(f"Scaler keys: {list(scaler.keys())}")
        if 'pca' in scaler:
            print(f"PCA in scaler: {scaler['pca'] is not None}")
            if scaler['pca'] and isinstance(scaler['pca'], dict):
                print(f"PCA keys: {list(scaler['pca'].keys())}")
            else:
                print(f"PCA in scaler is not a valid dict: {type(scaler['pca'])}")
    
    # Check if PCA should be applied
    if pca_components is None:
        print("=== NO PCA APPLIED (pca_components is None) ===")
        # No PCA - use original embeddings
        X_emb_reduced = X_embeddings_normalized
        pca = None
        print(f"No PCA applied, using original embedding shape: {X_emb_reduced.shape}")
    elif fit_scaler:
        print("=== FITTING NEW PCA ===")
        pca = PCA(n_components=pca_components, svd_solver="randomized")
        X_emb_reduced = pca.fit_transform(X_embeddings_normalized)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"Reduced embedding shape: {X_emb_reduced.shape}")
    elif scaler is not None and 'pca' in scaler:
        print("=== USING PRE-FITTED PCA ===")
        # Use pre-fitted PCA for test data
        pca_params = scaler['pca']
        print(f"PCA params keys: {list(pca_params.keys())}")
        pca = PCA(n_components=pca_components)
        pca.components_ = np.array(pca_params['components_'])
        pca.mean_ = np.array(pca_params['mean_'])
        if 'explained_variance_' in pca_params:
            pca.explained_variance_ = np.array(pca_params['explained_variance_'])
        if 'explained_variance_ratio' in pca_params:
            pca.explained_variance_ratio_ = np.array(pca_params['explained_variance_ratio'])
        
        # Ensure the PCA object is properly initialized
        pca.n_components_ = pca_components
        pca.n_features_in_ = pca.components_.shape[1]
        
        print(f"PCA reconstruction details:")
        print(f"  - n_components_: {pca.n_components_}")
        print(f"  - n_features_in_: {pca.n_features_in_}")
        print(f"  - components_ shape: {pca.components_.shape}")
        print(f"  - mean_ shape: {pca.mean_.shape}")
        
        X_emb_reduced = pca.transform(X_embeddings_normalized)
        print(f"Applied pre-fitted PCA, reduced embedding shape: {X_emb_reduced.shape}")
        print(f"PCA object attributes: components_={pca.components_.shape}, mean_={pca.mean_.shape}")
    else:
        print("=== NO PCA APPLIED (no scaler or fit_scaler) ===")
        # No PCA - use original embeddings
        X_emb_reduced = X_embeddings_normalized
        pca = None
        print(f"No PCA applied, using original embedding shape: {X_emb_reduced.shape}")
    
    # Group reweighting to balance feature influence
    if fit_scaler:
        # Compute L2 norms of each group
        emb_norm = mean_l2(X_emb_reduced)
        aux_norm = mean_l2(X_auxiliary_normalized)
        
        # Calculate weighting factors to balance the groups
        target_norm = (emb_norm + aux_norm) / 2
        emb_weight = target_norm / emb_norm if emb_norm > 0 else 1.0
        aux_weight = (target_norm / aux_norm) * aux_weight_multiplier if aux_norm > 0 else aux_weight_multiplier
        
        # Apply weights
        X_emb_weighted = X_emb_reduced * emb_weight
        X_aux_weighted = X_auxiliary_normalized * aux_weight
        
        group_norms = {
            'embedding_original': float(emb_norm),
            'auxiliary_original': float(aux_norm),
            'target_norm': float(target_norm)
        }
        weighting_factors = {
            'embedding_weight': float(emb_weight),
            'auxiliary_weight': float(aux_weight)
        }
    else:
        # For test data, use the same weights from training
        if scaler is not None and 'group_norms' in scaler and 'weighting_factors' in scaler:
            emb_weight = scaler['weighting_factors']['embedding_weight']
            aux_weight = scaler['weighting_factors']['auxiliary_weight']
            
            X_emb_weighted = X_emb_reduced * emb_weight
            X_aux_weighted = X_auxiliary_normalized * aux_weight
        else:
            # No weights available, use unweighted features
            X_emb_weighted = X_emb_reduced
            X_aux_weighted = X_auxiliary_normalized
            group_norms = None
            weighting_factors = None
    
    # Concatenate weighted features
    X_combined = np.concatenate([X_emb_weighted, X_aux_weighted], axis=1)
    
    # Verify final dimensions
    print(f"Final feature dimensions:")
    print(f"  - Reduced embeddings: {X_emb_weighted.shape[1]}")
    print(f"  - Auxiliary features: {X_aux_weighted.shape[1]}")
    print(f"  - Combined total: {X_combined.shape[1]}")
    
    if pca_components is None:
        expected_total = X_embeddings.shape[1] + 6 * AUX_FEATURE_REPETITIONS
        print(f"Expected total: {X_embeddings.shape[1]} (original) + {6 * AUX_FEATURE_REPETITIONS} (auxiliary) = {expected_total}")
    else:
        expected_total = pca_components + 6 * AUX_FEATURE_REPETITIONS
        print(f"Expected total: {pca_components} (PCA) + {6 * AUX_FEATURE_REPETITIONS} (auxiliary) = {expected_total}")
    
    # Store metadata
    meta = {
        "scalar_dim": int(6 * AUX_FEATURE_REPETITIONS),  # Updated to reflect repetitions
        "embedding_dim": int(X_embeddings.shape[1]),
        "reduced_embedding_dim": int(X_emb_reduced.shape[1]),
        "total_dim": int(X_combined.shape[1]),
        "auxiliary_repetitions": int(AUX_FEATURE_REPETITIONS),
        "pca_components": pca_components,  # Can be None or int
        "aux_weight_multiplier": float(aux_weight_multiplier),
        "pooling_method": pooling_method,
        "group_norms": group_norms if fit_scaler else None,
        "weighting_factors": weighting_factors if fit_scaler else None,
        "pca": {
            'components_': pca.components_.tolist() if pca and hasattr(pca, 'components_') else None,
            'mean_': pca.mean_.tolist() if pca and hasattr(pca, 'mean_') else None,
            'explained_variance_': pca.explained_variance_.tolist() if pca and hasattr(pca, 'explained_variance_') else None,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist() if pca and hasattr(pca, 'explained_variance_ratio') else None
        } if pca and hasattr(pca, 'components_') and pca.components_ is not None else None,
        "scaler": {
            'embedding': {
                'scale_': scaler['embedding'].scale_.tolist() if scaler and 'embedding' in scaler and hasattr(scaler['embedding'], 'scale_') else None,
                'mean_': scaler['embedding'].mean_.tolist() if scaler and 'embedding' in scaler and hasattr(scaler['embedding'], 'mean_') else None,
                'var_': scaler['embedding'].var_.tolist() if scaler and 'embedding' in scaler and hasattr(scaler['embedding'], 'var_') else None,
                'n_samples_seen_': safe_n_samples_seen(scaler['embedding'].n_samples_seen_) if scaler and 'embedding' in scaler and hasattr(scaler['embedding'], 'n_samples_seen_') else None
            },
            'auxiliary': {
                'scale_': scaler['auxiliary'].scale_.tolist() if scaler and 'auxiliary' in scaler and hasattr(scaler['auxiliary'], 'scale_') else None,
                'mean_': scaler['auxiliary'].mean_.tolist() if scaler and 'auxiliary' in scaler and hasattr(scaler['auxiliary'], 'mean_') else None,
                'var_': scaler['auxiliary'].var_.tolist() if scaler and 'auxiliary' in scaler and hasattr(scaler['auxiliary'], 'var_') else None,
                'n_samples_seen_': safe_n_samples_seen(scaler['auxiliary'].n_samples_seen_) if scaler and 'auxiliary' in scaler and hasattr(scaler['auxiliary'], 'n_samples_seen_') else None
            }
        } if scaler else None,  # Include the fitted scalers as JSON-serializable dict
    }
    
    return X_combined, meta


def main(
    csv_path: str,
    output_npy: str,
    output_meta_json: Optional[str],
    model_name: str,
    enable_gradients: bool = False,
    fit_scaler: bool = False,
    scaler_path: Optional[str] = None,  # New parameter for scaler file path
    prefer_cuda: bool = True,
    pca_components: int = 32,
    aux_weight_multiplier: float = 1.0,
    pooling_method: str = "mean",  # New parameter for pooling method
):
    device = get_device(prefer_cuda=prefer_cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("Loading Qwen model:", model_name)

    model, tokenizer = load_qwen(model_name, device, training_mode=enable_gradients)

    # Load pre-fitted scaler if provided
    scaler = None
    print(f"Scaler loading check:")
    print(f"  - scaler_path: {scaler_path}")
    print(f"  - fit_scaler: {fit_scaler}")
    print(f"  - Condition met: {scaler_path and not fit_scaler}")
    
    if scaler_path and not fit_scaler:
        # Check if file exists
        if not os.path.exists(scaler_path):
            print(f"ERROR: Scaler file does not exist: {scaler_path}")
            print(f"Current working directory: {os.getcwd()}")
            scaler = None
        else:
            try:
                print(f"Attempting to load scaler from: {scaler_path}")
                with open(scaler_path, 'r') as f:
                    scaler_data = json.load(f)
                print(f"Successfully loaded scaler data with keys: {list(scaler_data.keys())}")
                
                # Reconstruct scalers from saved data
                scaler_emb = StandardScaler()
                scaler_aux = StandardScaler()
                scaler_emb.scale_ = np.array(scaler_data['scaler']['embedding']['scale_'])
                scaler_emb.mean_ = np.array(scaler_data['scaler']['embedding']['mean_'])
                scaler_emb.var_ = np.array(scaler_data['scaler']['embedding']['var_'])
                scaler_emb.n_samples_seen_ = safe_n_samples_seen(scaler_data['scaler']['embedding']['n_samples_seen_'])
                scaler_aux.scale_ = np.array(scaler_data['scaler']['auxiliary']['scale_'])
                scaler_aux.mean_ = np.array(scaler_data['scaler']['auxiliary']['mean_'])
                scaler_aux.var_ = np.array(scaler_data['scaler']['auxiliary']['var_'])
                scaler_aux.n_samples_seen_ = safe_n_samples_seen(scaler_data['scaler']['auxiliary']['n_samples_seen_'])
                
                # Create complete scaler dictionary with all components
                scaler = {
                    'embedding': scaler_emb, 
                    'auxiliary': scaler_aux
                }
                
                # Add PCA if available
                if 'pca' in scaler_data and scaler_data['pca']:
                    scaler['pca'] = scaler_data['pca']
                
                # Add group norms and weighting factors if available
                if 'group_norms' in scaler_data:
                    scaler['group_norms'] = scaler_data['group_norms']
                if 'weighting_factors' in scaler_data:
                    scaler['weighting_factors'] = scaler_data['weighting_factors']
                print(f"Loaded pre-fitted scaler from {scaler_path}")
                print(f"PCA components: {scaler_data.get('pca_components', 'Not found')}")
                print(f"Group norms: {'Found' if 'group_norms' in scaler_data else 'Not found'}")
                print(f"Weighting factors: {'Found' if 'weighting_factors' in scaler_data else 'Not found'}")
                print(f"PCA data: {'Found' if 'pca' in scaler_data and scaler_data['pca'] else 'Not found'}")
                if 'pca' in scaler_data and scaler_data['pca'] and isinstance(scaler_data['pca'], dict):
                    print(f"PCA keys: {list(scaler_data['pca'].keys())}")
                    print(f"PCA components shape: {np.array(scaler_data['pca']['components_']).shape if 'components_' in scaler_data['pca'] else 'Not found'}")
                    print(f"PCA mean shape: {np.array(scaler_data['pca']['mean_']).shape if 'mean_' in scaler_data['pca'] else 'Not found'}")
                
                # Verify what's actually in the scaler dictionary
                print(f"Final scaler keys: {list(scaler.keys())}")
                print(f"Scaler has PCA: {'pca' in scaler}")
                if 'pca' in scaler and isinstance(scaler['pca'], dict):
                    print(f"Scaler PCA keys: {list(scaler['pca'].keys())}")
                else:
                    print(f"Scaler PCA is not a valid dict: {type(scaler.get('pca', 'Not found'))}")
            except Exception as e:
                print(f"Warning: Failed to load scaler from {scaler_path}: {e}")
                print("Proceeding without scaler (raw features will be used)")
                scaler = None
    else:
        print(f"Scaler loading skipped - scaler_path: {scaler_path}, fit_scaler: {fit_scaler}")
    
    print(f"Final scaler state: {scaler is not None}")
    if scaler:
        print(f"Final scaler keys: {list(scaler.keys())}")
        if 'pca' in scaler:
            print(f"PCA type: {type(scaler['pca'])}")
            if isinstance(scaler['pca'], dict):
                print(f"PCA keys: {list(scaler['pca'].keys())}")

    df = pd.read_csv(csv_path, low_memory=False)
    X, meta = extract_features_qwen(
        df, model, tokenizer, device, 
        enable_gradients=enable_gradients, 
        fit_scaler=fit_scaler,
        scaler=scaler,
        pca_components=pca_components,
        aux_weight_multiplier=aux_weight_multiplier,
        pooling_method=pooling_method
    )

    os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
    np.save(output_npy, X)
    print(f"Saved features: {output_npy} with shape {X.shape}")
    print(f"Expected dimensions:")
    if pca_components is None:
        print(f"  - PCA components: None (no PCA applied)")
        print(f"  - Auxiliary features: {6 * 1}")  # 6 features × 1 repetition
        print(f"  - Total expected: {X.shape[1]}")
    else:
        print(f"  - PCA components: {pca_components}")
        print(f"  - Auxiliary features: {6 * 1}")  # 6 features × 1 repetition
        print(f"  - Total expected: {pca_components + 6}")

    if output_meta_json:
        os.makedirs(os.path.dirname(output_meta_json) or ".", exist_ok=True)
        with open(output_meta_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta: {output_meta_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract reduced Qwen features + scalar features from CSV.")
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
        dest="model_name",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Qwen model name from Hugging Face Hub (default: Qwen/Qwen2.5-1.5B-Instruct)",
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
    parser.add_argument(
        "--pca-components",
        dest="pca_components",
        type=str,
        default="32",
        help="Number of PCA components for dimensionality reduction. Use 'none' for no PCA (default: 32)",
    )
    parser.add_argument(
        "--aux-weight-multiplier",
        dest="aux_weight_multiplier",
        type=float,
        default=1.0,
        help="Multiplier for auxiliary feature weights during group reweighting (default: 1.0)",
    )
    parser.add_argument(
        "--pooling",
        dest="pooling_method",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="Pooling method for token embeddings: 'mean' or 'max' (default: 'mean')",
    )
    args = parser.parse_args()
    
    # Handle PCA components argument
    if args.pca_components.lower() == 'none':
        pca_components = None
    else:
        try:
            pca_components = int(args.pca_components)
        except ValueError:
            print(f"Error: Invalid PCA components value: {args.pca_components}")
            print("Use an integer or 'none'")
            exit(1)
    
    main(
        csv_path=args.csv_path,
        output_npy=args.output_npy,
        output_meta_json=args.output_meta_json,
        model_name=args.model_name,
        prefer_cuda=not args.no_cuda,
        enable_gradients=args.enable_gradients,
        fit_scaler=args.fit_scaler,
        scaler_path=args.scaler_path,
        pca_components=pca_components,
        aux_weight_multiplier=args.aux_weight_multiplier,
        pooling_method=args.pooling_method,
    )
