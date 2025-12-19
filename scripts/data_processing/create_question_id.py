# This script is used to create a mapping of questions to indices (sb_idx).
import re, unicodedata, json
import pandas as pd

def normalize_question(text: str) -> str:
    if pd.isna(text):
        return None
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def build_q_to_id(df: pd.DataFrame, question_col: str = "question", start: int = 0) -> dict[str, int]:
    keys = df[question_col].map(normalize_question).dropna().unique()
    keys = sorted(keys)
    return {q: i + start for i, q in enumerate(keys)}

def apply_q_to_id(df: pd.DataFrame, q_to_id: dict[str, int], question_col: str = "question") -> pd.DataFrame:
    out = df.copy()
    out["_q_key"] = out[question_col].map(normalize_question)
    out["sb_idx"] = out["_q_key"].map(q_to_id).astype("Int64")
    out.drop(columns=["_q_key"], inplace=True)
    return out

def extend_mapping_with_df(q_to_id: dict[str, int], df: pd.DataFrame, question_col: str = "question") -> dict[str, int]:
    next_id = (max(q_to_id.values()) + 1) if q_to_id else 0
    keys = df[question_col].map(normalize_question).dropna().unique()
    new_keys = sorted(set(keys) - set(q_to_id.keys()))
    for k in new_keys:
        q_to_id[k] = next_id
        next_id += 1
    return q_to_id

def save_mapping(q_to_id: dict[str, int], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(q_to_id, f, ensure_ascii=False)

def load_mapping(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}
