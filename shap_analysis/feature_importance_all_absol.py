import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from difflib import get_close_matches
from matplotlib.ticker import MaxNLocator
import re

# --- 한글 폰트 설정 ---
try:
    font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    korean_font_path = None
    preferred_fonts = ['arial', 'helvetica', 'dejavu sans']
    for font_name in preferred_fonts:
        for f_path in font_files:
            base_name = os.path.basename(f_path).lower().replace(' ', '')
            if font_name.replace(' ', '') in base_name:
                korean_font_path = f_path
                break
        if korean_font_path:
            break
    if korean_font_path:
        prop = fm.FontProperties(fname=korean_font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Font set to: {prop.get_name()} ({korean_font_path})")
    else:
        print("Warning: Preferred font not found for plots.")
except Exception as e:
    print(f"Warning: Error setting font - {e}")

# --- subword 병합 함수 ---
def merge_subword_tokens(token_list):
    merged = []
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

# --- 단어 복원 함수 ---
def restore_keyword(subword, vocab_keys, topn=1):
    candidates = [w for w in vocab_keys if subword in w and len(w) > len(subword)]
    matches = get_close_matches(subword, candidates, n=topn, cutoff=0.8)
    return matches[0] if matches else subword

# --- 클리닝 함수 ---
def clean_keyword(word):
    word = str(word).strip()
    word = word.replace("\u201c", "").replace("\u201d", "").replace('"', "").replace("'", "")
    return re.sub(r"[^a-zA-Z0-9]", "", word).strip()

# --- 불용어 정의 ---
STOPWORDS = {
    "co", "the", "for", "and", "but", "of", "on", "chatte", "tapp", "magnif", "bett", "payx",
    "rosen", "esca", "reu", "techno", "ticker", "thursday", "steam", "guil", "scar", "pharm", "subp", "stran",
    "depressi", "thal", "msci", "dipp", "conci", "spec", "cheese", "stretche", "ninth", "seven", "behav", "legg",
    "vant", "divergen"
}

# --- SHAP 분석 함수 ---
def extract_top_n_feature_shap(
    shap_values_array,
    tokens_data,
    technical_names,
    test_year,
    top_n=10,
    save_path=None,
    word2idx=None,
    embedding_dim=768
):
    time_steps = 10

    # close 제외
    technical_names = [name for name in technical_names if name.lower() != "close"]
    num_tech = len(technical_names)
    num_samples, num_features = shap_values_array.shape
    num_keywords = (num_features - (num_tech * time_steps)) // embedding_dim

    TECHNICAL_NAME_ALIAS = {
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
        "close_vs_sma": "SMA Relative Difference"
    }


    tech_set = set(TECHNICAL_NAME_ALIAS.get(name, name) for name in technical_names)
    tokens_data = [merge_subword_tokens(row) for row in tokens_data]

    feature_abs_contributions = defaultdict(float)
    feature_signed_contributions = defaultdict(float)
    keyword_counts = defaultdict(int)

    for n in range(num_samples):
        current_keywords = tokens_data[n] if isinstance(tokens_data[n], list) else []

        for i in range(num_tech):
            idx_start = i * time_steps
            idx_end = (i + 1) * time_steps
            feat_name = technical_names[i]
            alias_name = TECHNICAL_NAME_ALIAS.get(feat_name, feat_name)
            shap_val = np.mean(shap_values_array[n, idx_start:idx_end])
            feature_abs_contributions[alias_name] += abs(shap_val)
            feature_signed_contributions[alias_name] += shap_val

        for k in range(num_keywords):
            idx_start = num_tech * time_steps + k * embedding_dim
            idx_end = idx_start + embedding_dim
            if k < len(current_keywords):
                raw_word = current_keywords[k]
                word = restore_keyword(raw_word, list(word2idx.keys())) if word2idx else raw_word
                word = clean_keyword(word)
                if len(word) < 2 or word.lower() in STOPWORDS or word.lower() == "pad":
                    continue
                keyword_shap = shap_values_array[n, idx_start:idx_end]
                feature_abs_contributions[word] += np.mean(np.abs(keyword_shap))
                feature_signed_contributions[word] += np.mean(keyword_shap)
                keyword_counts[word] += 1

    final_abs_shaps = {}
    final_signed_shaps = {}
    for feat in feature_abs_contributions:
        if feat in tech_set:
            final_abs_shaps[feat] = feature_abs_contributions[feat] / num_samples
            final_signed_shaps[feat] = feature_signed_contributions[feat] / num_samples
        else:
            count = keyword_counts[feat]
            final_abs_shaps[feat] = feature_abs_contributions[feat] / count if count > 0 else 0
            final_signed_shaps[feat] = feature_signed_contributions[feat] / count if count > 0 else 0

    keyword_only = [(feat, val) for feat, val in final_abs_shaps.items() if feat not in tech_set]
    keyword_only_sorted = sorted(keyword_only, key=lambda x: x[1], reverse=True)
    keyword_rank_mapping = {feat: f"Keyword-{i+1}" for i, (feat, _) in enumerate(keyword_only_sorted)}

    top_features = sorted(final_abs_shaps.items(), key=lambda x: x[1], reverse=True)[:top_n]

    try:
        plt.figure(figsize=(10, max(5, len(top_features) * 0.35 + 1)))
        y_pos = np.arange(len(top_features))
        abs_values = [v for _, v in top_features]
        bar_labels = [
            keyword_rank_mapping[f] if f not in tech_set else f
            for f, _ in top_features
        ]
        plt.barh(y_pos, abs_values, color='blue', align='center')
        plt.yticks(y_pos, bar_labels, fontsize=20)
        plt.gca().invert_yaxis()
        plt.xlabel("Mean Absolute SHAP Value", fontsize=20)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.xticks(fontsize=20)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=400)
            print(f"Plot saved: {save_path}")

    except Exception as e:
        print(f"[Plot Error] {e}")
    finally:
        plt.close()

    out_df = []
    for i, (feat, _) in enumerate(top_features):
        is_keyword = feat not in tech_set
        feature_name = keyword_rank_mapping[feat] if is_keyword else feat
        out_df.append({
            "year": test_year,
            "feature": feature_name,
            "actual_keyword": feat if is_keyword else "",
            "type": "keyword" if is_keyword else "technical",
            "mean_abs_shap": final_abs_shaps[feat],
            "mean_signed_shap": final_signed_shaps[feat]
        })

    if save_path:
        mapping_path = save_path.replace(".png", "_keyword_mapping.csv")
        full_mapping = [
            {"Keyword-Rank": keyword_rank_mapping[k], "Token": k}
            for k, _ in keyword_only_sorted[:17]
        ]
        while len(full_mapping) < 17:
            full_mapping.append({
                "Keyword-Rank": f"Keyword-{len(full_mapping)+1}",
                "Token": "N/A"
            })
        pd.DataFrame(full_mapping).to_csv(mapping_path, index=False)
        print(f"[논문용] SHAP 기반 Keyword mapping saved to: {mapping_path}")

    return pd.DataFrame(out_df)
