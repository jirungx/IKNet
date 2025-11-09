import os
import re
import logging
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from difflib import get_close_matches
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------
# Font setup (best-effort)
# ---------------------------------------------------------
def _setup_font(preferred_fonts: Sequence[str] = ("arial", "helvetica", "dejavu sans")) -> None:
    """
    Best-effort font configuration for plots. Tries a small list of common sans-serif fonts.
    """
    try:
        font_files = fm.findSystemFonts(fontpaths=None, fontext="ttf")
        chosen_path = None
        for font_name in preferred_fonts:
            target = font_name.replace(" ", "").lower()
            for f_path in font_files:
                base = os.path.basename(f_path).replace(" ", "").lower()
                if target in base:
                    chosen_path = f_path
                    break
            if chosen_path:
                break

        if chosen_path:
            prop = fm.FontProperties(fname=chosen_path)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.size"] = 12
            plt.rcParams["axes.unicode_minus"] = False
            logging.info("Font set to: %s (%s)", prop.get_name(), chosen_path)
        else:
            logging.warning("Preferred font not found; using matplotlib default.")
    except Exception as e:
        logging.warning("Error setting font: %s", str(e))


_setup_font()


# ---------------------------------------------------------
# Token utilities
# ---------------------------------------------------------
def merge_subword_tokens(token_list: Sequence[str]) -> List[str]:
    """
    Merge WordPiece subwords into whole words.

    Args:
        token_list (Sequence[str]): Tokens possibly containing '##' prefixes.

    Returns:
        List[str]: Merged tokens without WordPiece markers.
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
    Heuristically restore a subword to a plausible full token using fuzzy matching.

    Args:
        subword (str): Merged subword.
        vocab_keys (Sequence[str]): Candidate vocabulary keys.
        topn (int): Number of close matches to consider.

    Returns:
        str: Best match if available; otherwise returns `subword`.
    """
    candidates = [w for w in vocab_keys if subword in w and len(w) > len(subword)]
    matches = get_close_matches(subword, candidates, n=topn, cutoff=0.8)
    return matches[0] if matches else subword


def clean_keyword(word: str) -> str:
    """
    Normalize a keyword by trimming quotes and removing non-alphanumeric characters.

    Args:
        word (str): Input token.

    Returns:
        str: Cleaned token consisting of [a-zA-Z0-9] only.
    """
    word = str(word).strip()
    word = word.replace("\u201c", "").replace("\u201d", "").replace('"', "").replace("'", "")
    return re.sub(r"[^a-zA-Z0-9]", "", word).strip()


# ---------------------------------------------------------
# SHAP aggregation and visualization
# ---------------------------------------------------------
def extract_top_n_feature_shap(
    shap_values_array: np.ndarray,
    tokens_data: Sequence[Sequence[str]],
    technical_names: Sequence[str],
    test_year: int,
    top_n: int = 10,
    save_path: str | None = None,
    word2idx: Dict[str, int] | None = None,
    embedding_dim: int = 768,
) -> pd.DataFrame:
    """
    Aggregate SHAP values over technical features and per-day keyword embeddings,
    select global top-N by mean absolute contribution, and optionally save a plot.

    Assumptions about SHAP layout per sample:
        [ technical blocks (num_tech * time_steps), keyword_0 (D), keyword_1 (D), ... ]

    Args:
        shap_values_array (np.ndarray): SHAP matrix [num_samples, num_features].
        tokens_data (Sequence[Sequence[str]]): Per-sample token sequences (WordPiece).
        technical_names (Sequence[str]): Technical feature names, order-aligned with data.
        test_year (int): Year label for reporting.
        top_n (int): Number of global features to visualize and report.
        save_path (str | None): If provided, saves bar plot to this PNG path.
        word2idx (Dict[str, int] | None): Vocab keys for optional token restoration.
        embedding_dim (int): Keyword embedding dimension (default 768).

    Returns:
        pd.DataFrame: Table with top-N features and statistics.
    """
    time_steps = 10

    # Exclude 'close' from technical features
    technical_names = [name for name in technical_names if name.lower() != "close"]
    num_tech = len(technical_names)
    num_samples, num_features = shap_values_array.shape
    num_keywords = (num_features - (num_tech * time_steps)) // embedding_dim

    # Alias for human-readable labels
    TECHNICAL_NAME_ALIAS: Dict[str, str] = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "sma_20": "SMA",
        "ema_20": "EMA",
        "rsi_14": "RSI",
        "macd": "MACD",
        "signal_line": "Signal Line",
        "macd_diff": "MACD Difference",
        "bollinger_upper": "Bollinger Upper",
        "bollinger_lower": "Bollinger Lower",
        "bollinger_width": "Bollinger Width",
        "volatility_ratio": "Volatility Ratio",
        "volume_change": "Volume Change",
        "close_vs_sma": "SMA Relative Difference",
    }

    tech_set = set(TECHNICAL_NAME_ALIAS.get(name, name) for name in technical_names)
    tokens_data = [merge_subword_tokens(row) for row in tokens_data]

    feature_abs_contributions: Dict[str, float] = defaultdict(float)
    feature_signed_contributions: Dict[str, float] = defaultdict(float)
    keyword_counts: Dict[str, int] = defaultdict(int)

    # Per-sample aggregation
    for n in range(num_samples):
        current_keywords = tokens_data[n] if isinstance(tokens_data[n], list) else []

        # Technical blocks: average over time steps
        for i in range(num_tech):
            idx_start = i * time_steps
            idx_end = (i + 1) * time_steps
            feat_name = technical_names[i]
            alias_name = TECHNICAL_NAME_ALIAS.get(feat_name, feat_name)

            shap_val = float(np.mean(shap_values_array[n, idx_start:idx_end]))
            feature_abs_contributions[alias_name] += abs(shap_val)
            feature_signed_contributions[alias_name] += shap_val

        # Keyword blocks: average over embedding dimension
        for k in range(num_keywords):
            idx_start = num_tech * time_steps + k * embedding_dim
            idx_end = idx_start + embedding_dim
            if k < len(current_keywords):
                raw_word = current_keywords[k]
                word = restore_keyword(raw_word, list(word2idx.keys())) if word2idx else raw_word
                word = clean_keyword(word)
                if not word:
                    continue

                keyword_shap = shap_values_array[n, idx_start:idx_end]
                feature_abs_contributions[word] += float(np.mean(np.abs(keyword_shap)))
                feature_signed_contributions[word] += float(np.mean(keyword_shap))
                keyword_counts[word] += 1

    # Normalize: technical by sample count, keywords by occurrence count
    final_abs_shaps: Dict[str, float] = {}
    final_signed_shaps: Dict[str, float] = {}
    for feat in feature_abs_contributions:
        if feat in tech_set:
            final_abs_shaps[feat] = feature_abs_contributions[feat] / max(num_samples, 1)
            final_signed_shaps[feat] = feature_signed_contributions[feat] / max(num_samples, 1)
        else:
            count = max(keyword_counts[feat], 1)
            final_abs_shaps[feat] = feature_abs_contributions[feat] / count
            final_signed_shaps[feat] = feature_signed_contributions[feat] / count

    # Keyword rank aliases based on global importance
    keyword_only = [(feat, val) for feat, val in final_abs_shaps.items() if feat not in tech_set]
    keyword_only_sorted = sorted(keyword_only, key=lambda x: x[1], reverse=True)
    keyword_rank_mapping = {feat: f"Keyword-{i + 1}" for i, (feat, _) in enumerate(keyword_only_sorted)}

    # Global top-N by mean |SHAP|
    top_features = sorted(final_abs_shaps.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Visualization
    try:
        if top_features:
            plt.figure(figsize=(10, max(5, len(top_features) * 0.35 + 1)))
            y_pos = np.arange(len(top_features))
            abs_values = [v for _, v in top_features]
            bar_labels = [keyword_rank_mapping[f] if f not in tech_set else f for f, _ in top_features]

            plt.barh(y_pos, abs_values, align="center")  # no explicit color to keep style-neutral
            plt.yticks(y_pos, bar_labels, fontsize=20)
            plt.gca().invert_yaxis()
            plt.xlabel("Mean Absolute SHAP Value", fontsize=20)
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.xticks(fontsize=20)
            plt.grid(axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight", dpi=400)
                logging.info("Plot saved: %s", save_path)
    except Exception as e:
        logging.exception("Plot error: %s", str(e))
    finally:
        plt.close()

    # Export table
    out_rows: List[Dict[str, object]] = []
    for feat, _ in top_features:
        is_keyword = feat not in tech_set
        feature_name = keyword_rank_mapping[feat] if is_keyword else feat
        out_rows.append(
            {
                "year": test_year,
                "feature": feature_name,
                "actual_keyword": feat if is_keyword else "",
                "type": "keyword" if is_keyword else "technical",
                "mean_abs_shap": final_abs_shaps[feat],
                "mean_signed_shap": final_signed_shaps[feat],
            }
        )

    # Save keyword rank-to-token mapping next to the figure
    if save_path:
        mapping_path = save_path.replace(".png", "_keyword_mapping.csv")
        top_k_for_map = 17  # keep consistent with default report
        full_mapping = [{"Keyword-Rank": keyword_rank_mapping[k], "Token": k} for k, _ in keyword_only_sorted[:top_k_for_map]]
        while len(full_mapping) < top_k_for_map:
            full_mapping.append({"Keyword-Rank": f"Keyword-{len(full_mapping) + 1}", "Token": "N/A"})
        pd.DataFrame(full_mapping).to_csv(mapping_path, index=False)
        logging.info("SHAP-based keyword mapping saved: %s", mapping_path)

    return pd.DataFrame(out_rows)
