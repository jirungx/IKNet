import re
from collections import defaultdict
from difflib import get_close_matches
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Subword merge and keyword cleaning utilities
# ---------------------------------------------------------
def merge_subword_tokens(token_list: Sequence[str]) -> List[str]:
    """
    Merge WordPiece subwords into whole words.

    Args:
        token_list (Sequence[str]): Tokenized sequence with possible '##' subwords.

    Returns:
        List[str]: List of merged tokens without WordPiece markers.
    """
    merged: List[str] = []
    buffer = ""
    for token in token_list:
        if token.startswith("##"):
            buffer += token[2:]
        else:
            if buffer:
                merged.append(buffer)
            buffer = token
    if buffer:
        merged.append(buffer)
    return merged


def restore_keyword(subword: str, vocab_keys: Sequence[str], topn: int = 1) -> str:
    """
    Heuristically restore a subword to a most plausible full word using fuzzy match.

    Args:
        subword (str): Subword token (already merged if needed).
        vocab_keys (Sequence[str]): Vocabulary or known keyword list to match against.
        topn (int): How many close matches to consider; the best will be returned.

    Returns:
        str: Restored word if a close match is found; otherwise returns the input subword.
    """
    candidates = [w for w in vocab_keys if subword in w and len(w) > len(subword)]
    matches = get_close_matches(subword, candidates, n=topn, cutoff=0.8)
    return matches[0] if matches else subword


def clean_keyword(word: str) -> str:
    """
    Normalize a keyword by trimming quotes and removing non-alphanumeric characters
    except hyphens.

    Args:
        word (str): Raw keyword.

    Returns:
        str: Cleaned keyword.
    """
    word = str(word).strip()
    word = word.replace("\u201c", "").replace("\u201d", "").replace('"', "").replace("'", "")
    word = re.sub(r"[^a-zA-Z0-9\-]", "", word)
    return word.strip()


# ---------------------------------------------------------
# Daily keyword SHAP table extraction
# ---------------------------------------------------------
def extract_daily_keyword_shap(
    shap_values_array: np.ndarray,
    tokens_data: Sequence[Sequence[str]],
    dates: Sequence[pd.Timestamp],
    technical_names: Sequence[str],
    word2idx: Dict[str, int] | None = None,
    num_keywords: int = 17,
    embedding_dim: int = 768,
) -> pd.DataFrame:
    """
    Build a per-day table of top-N keywords by mean SHAP magnitude (name-based aggregation).

    Assumptions about the SHAP vector layout per sample:
        [ technical_features (num_tech * time_steps), keyword_0 (embedding_dim), keyword_1 (embedding_dim), ... ]

    For each sample i:
      1) Merge WordPiece subwords in tokens_data[i]
      2) Optionally restore each merged token to a full word using word2idx keys
      3) Clean tokens
      4) For the k-th token block, take the mean SHAP across its embedding_dim slice
      5) Aggregate by token string (mean over duplicates)
      6) Keep top `num_keywords` by absolute mean SHAP

    Args:
        shap_values_array (np.ndarray): Array of shape [num_samples, num_features].
        tokens_data (Sequence[Sequence[str]]): Token sequences (WordPiece) per sample.
        dates (Sequence[pd.Timestamp]): Dates aligned to samples (len == num_samples).
        technical_names (Sequence[str]): Names of technical features (for counting only).
        word2idx (Dict[str, int] | None): Optional vocabulary for restoring words.
        num_keywords (int): Number of keywords to keep per row.
        embedding_dim (int): Embedding dimension per keyword block.

    Returns:
        pd.DataFrame: A dataframe with columns:
            - "date"
            - "keyword_1", "shap_value_1", ..., "keyword_{num_keywords}", "shap_value_{num_keywords}"
    """
    num_samples, num_features = shap_values_array.shape
    num_tech = len(technical_names)
    time_steps = 10  # keep consistent with the original pipeline

    daily_shap_records: List[Dict[str, object]] = []

    for i in range(num_samples):
        # 1) Merge subwords
        row_tokens = merge_subword_tokens(tokens_data[i])
        date = dates[i]
        keyword_shaps: Dict[str, List[float]] = defaultdict(list)

        # 2) Iterate over token blocks
        for k, raw_word in enumerate(row_tokens):
            # Optional restoration using provided vocabulary keys
            word = restore_keyword(raw_word, list(word2idx.keys())) if word2idx else raw_word
            word = clean_keyword(word)
            if not word:
                continue

            # Compute the SHAP slice corresponding to this token block
            start_idx = num_tech * time_steps + k * embedding_dim
            end_idx = start_idx + embedding_dim
            if end_idx <= num_features:
                shap_val = float(np.mean(shap_values_array[i, start_idx:end_idx]))
                keyword_shaps[word].append(shap_val)

        # 3) Aggregate SHAP by token and keep top-N by absolute value
        if keyword_shaps:
            word_mean_shaps = {w: float(np.mean(vs)) for w, vs in keyword_shaps.items()}
            top_items = sorted(
                word_mean_shaps.items(), key=lambda x: abs(x[1]), reverse=True
            )[:num_keywords]
        else:
            top_items = []

        # 4) Build the output row with fixed number of columns
        row: Dict[str, object] = {"date": date}
        for idx, (w, s) in enumerate(top_items):
            row[f"keyword_{idx + 1}"] = w
            row[f"shap_value_{idx + 1}"] = s
        for idx in range(len(top_items), num_keywords):
            row[f"keyword_{idx + 1}"] = ""
            row[f"shap_value_{idx + 1}"] = 0.0

        daily_shap_records.append(row)

    return pd.DataFrame(daily_shap_records)
