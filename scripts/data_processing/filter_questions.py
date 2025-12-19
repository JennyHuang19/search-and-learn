import pandas as pd

def drop_extreme_questions(
    df: pd.DataFrame,
    sl_col: str = "sl",
    n_col: str = "N",
    prefer_group=("sb_idx",),  # fallback to ("question",)
    min_target_N: int = 4,
    target_value: float = 0.0,  # 0.0 for zeros, 1.0 for ones
    frac: float = 0.8,
    eps: float = 1e-10,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Drop a fraction of questions that have extreme soft label values across multiple N values.
    
    This function identifies questions where a significant number of N values have soft labels
    that are close to a target value (e.g., many zeros or many ones), and randomly drops
    a specified fraction of these questions. This is useful for removing questions that
    may be too easy (many ones) or too hard (many zeros) across different sample sizes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the data to filter.
        
    sl_col : str, default="sl"
        Name of the column containing soft label values.
        
    n_col : str, default="N"
        Name of the column containing N values (sample sizes).
        
    prefer_group : tuple, default=("sb_idx",)
        Preferred grouping columns. Falls back to ("question",) if not available.
        Used to group questions for counting extreme values.
        
    min_target_N : int, default=4
        Minimum number of N values that must have the target soft label value
        for a question to be eligible for dropping.
        
    target_value : float, default=0.0
        The target soft label value to count:
        - 0.0: Count questions with many zero soft labels (too hard)
        - 1.0: Count questions with many one soft labels (too easy)
        - Other values: Count questions close to the specified value
        
    frac : float, default=0.8
        Fraction of eligible questions to randomly drop. Must be between 0 and 1.
        
    eps : float, default=1e-12
        Tolerance for determining if a value equals the target_value.
        For target_value=0.0: values <= eps are considered zero
        For target_value=1.0: values >= (1.0-eps) are considered one
        For other values: values within eps of target_value are considered equal
        
    random_state : int or None, default=42
        Random seed for reproducible sampling. Set to None for truly random sampling.
        
    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with extreme questions removed. The original index is reset.
        
    Examples:
    ---------
    # Drop 80% of questions that have at least 4 N values with zero soft labels
    df_filtered = drop_extreme_questions(
        df, target_value=0.0, min_target_N=4, frac=0.8
    )
    
    # Drop 70% of questions that have at least 3 N values with one soft labels
    df_filtered = drop_extreme_questions(
        df, target_value=1.0, min_target_N=3, frac=0.7
    )
    
    # Drop 60% of questions that have at least 5 N values close to 0.5
    df_filtered = drop_extreme_questions(
        df, target_value=0.5, min_target_N=5, frac=0.6, eps=0.1
    )
    
    Notes:
    ------
    - The function groups questions by the specified grouping columns and counts
      how many N values have soft labels close to the target_value.
    - Questions are eligible for dropping if they have at least min_target_N
      N values meeting the target criteria.
    - A random fraction (frac) of eligible questions are dropped.
    - The function handles NaN values in grouping columns by converting them to
      a special "__NA__" string for matching purposes.
    - The original DataFrame is not modified; a copy is returned.
    """
    # pick grouping columns
    if all(c in df.columns for c in prefer_group):
        group_cols = list(prefer_group)
    elif "question" in df.columns:
        group_cols = ["question"]
    else:
        raise KeyError(f"Missing grouping cols {prefer_group} and 'question'")
    if n_col not in df.columns:
        raise KeyError(f"Missing '{n_col}' column")

    out = df.copy()
    # ensure numeric and non-nullable
    out[sl_col] = pd.to_numeric(out[sl_col], errors="coerce").fillna(0.0).astype(float)
    out[n_col] = pd.to_numeric(out[n_col], errors="coerce")

    # compute per-(question, N) max sl (treat NaN N as not countable)
    sub = (
        out.dropna(subset=[n_col])
           .groupby(group_cols + [n_col], dropna=False)[sl_col]
           .max()
           .reset_index(name="sl_max")
    )
    
    # Check if values are close to target_value (allowing for small numerical differences)
    if target_value == 0:
        sub["is_target_for_N"] = (sub["sl_max"] <= eps)
    elif target_value == 1:
        sub["is_target_for_N"] = (sub["sl_max"] >= (1.0 - eps))
    else:
        # For other target values, check if within eps tolerance
        sub["is_target_for_N"] = (abs(sub["sl_max"] - target_value) <= eps)

    # per-question count of N with target softlabel value
    target_count = (
        sub.groupby(group_cols, dropna=False)["is_target_for_N"]
           .sum()
           .reset_index(name=f"num_target_{target_value}_N")
    )

    # questions eligible to drop: at least min_target_N target-Ns
    eligible = target_count[target_count[f"num_target_{target_value}_N"] >= min_target_N]
    if eligible.empty:
        return out.reset_index(drop=True)

    # sample fraction of eligible questions to drop
    drop_groups = eligible.sample(frac=frac, random_state=random_state)[group_cols]

    # robust match even with NaNs in keys
    tmp_all = out[group_cols].copy()
    tmp_drop = drop_groups.copy()
    for c in group_cols:
        tmp_all[c] = tmp_all[c].astype(object).where(~tmp_all[c].isna(), "__NA__")
        tmp_drop[c] = tmp_drop[c].astype(object).where(~tmp_drop[c].isna(), "__NA__")

    mi_all = pd.MultiIndex.from_frame(tmp_all, names=group_cols)
    mi_drop = pd.MultiIndex.from_frame(tmp_drop, names=group_cols)

    drop_mask = pd.Series(mi_all.isin(mi_drop), index=out.index)
    keep_mask = ~drop_mask

    return out.loc[keep_mask].reset_index(drop=True)
