import pandas as pd
import numpy as np
from collections import defaultdict
from difflib import get_close_matches
import re

# --- 서브워드 병합 및 정제 함수 ---
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

def restore_keyword(subword, vocab_keys, topn=1):
    candidates = [w for w in vocab_keys if subword in w and len(w) > len(subword)]
    matches = get_close_matches(subword, candidates, n=topn, cutoff=0.8)
    return matches[0] if matches else subword

def clean_keyword(word):
    word = str(word).strip()
    word = word.replace("\u201c", "").replace("\u201d", "").replace('"', "").replace("'", "")
    word = re.sub(r"[^a-zA-Z0-9\-]", "", word)
    return word.strip()

STOPWORDS = {"co", "the", "for", "and", "but", "of", "on", "chatte", "tapp", "magnif", "bett", "payx",
             "rosen", "esca", "reu", "techno", "ticker", "thursday", "steam", "guil", "scar", "pharm",
             "subp", "stran", "depressi", "thal", "msci", "dipp", "conci", "spec", "cheese", "stretche",
             "ninth", "seven", "behav", "legg", "vant", "divergen"}

# === 날짜별 SHAP 테이블 생성 함수 (이름 기반 통합 버전) ===
def extract_daily_keyword_shap(
    shap_values_array,
    tokens_data,
    dates,
    technical_names,
    word2idx=None,
    num_keywords=13,
    embedding_dim=768
):
    num_samples, num_features = shap_values_array.shape
    num_tech = len(technical_names)
    time_steps = 10

    daily_shap_records = []

    for i in range(num_samples):
        row_tokens = merge_subword_tokens(tokens_data[i])
        date = dates[i]
        keyword_shaps = defaultdict(list)

        for k, raw_word in enumerate(row_tokens):
            word = restore_keyword(raw_word, list(word2idx.keys())) if word2idx else raw_word
            word = clean_keyword(word)
            if len(word) <= 3 or word.lower() in STOPWORDS or not word.isalpha():
                continue

            start_idx = num_tech * time_steps + k * embedding_dim
            end_idx = start_idx + embedding_dim
            if end_idx <= num_features:
                shap_val = np.mean(shap_values_array[i, start_idx:end_idx])
                keyword_shaps[word].append(shap_val)

        # 평균 SHAP으로 정리 후 상위 n개 추출
        word_mean_shaps = {w: np.mean(vs) for w, vs in keyword_shaps.items()}
        top_items = sorted(word_mean_shaps.items(), key=lambda x: abs(x[1]), reverse=True)[:num_keywords]

        row = {"date": date}
        for idx, (w, s) in enumerate(top_items):
            row[f"keyword_{idx+1}"] = w
            row[f"shap_value_{idx+1}"] = s
        for idx in range(len(top_items), num_keywords):
            row[f"keyword_{idx+1}"] = ""
            row[f"shap_value_{idx+1}"] = 0.0

        daily_shap_records.append(row)

    return pd.DataFrame(daily_shap_records)
