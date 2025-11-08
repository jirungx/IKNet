# iknet/shap_analysis/run_shap_daily.py
# -*- coding: utf-8 -*-
"""
Daily keyword-level SHAP generator for IKNet
- Loads trained IKNet checkpoints per rolling window (train_start -> test_year)
- Uses FinBERT token embedding matrix on the fly (no need for precomputed pkl here)
- Produces a per-day keyword SHAP CSV per test year

Usage example:
iknet-shap-daily \
  --price-csv dataset/snp500_dataset.csv \
  --tokens-csv tokens/snp_topk25_tokens.csv \
  --outdir outputs_shap/daily \
  --start-year 2018 --end-year 2024 \
  --num-keywords 25 --time-steps 30 --hidden-size 384
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import shap
import joblib
from typing import List, Tuple

# 상대 임포트 (중요)
from ..modules.model import IKNet
from .shap_keyword_utils_daily import extract_daily_keyword_shap

# 디바이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# I/O helpers
# ---------------------------
def load_data(price_path: str, token_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"[ERROR] price csv not found: {price_path}")
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"[ERROR] tokens csv not found: {token_path}")

    price_df = pd.read_csv(price_path)
    token_df = pd.read_csv(token_path)

    # 날짜형 변환
    price_df["date"] = pd.to_datetime(price_df["date"])
    token_df["date"] = pd.to_datetime(token_df["date"])

    # 토큰 컬럼 정규화
    if "filtered_keywords" not in token_df.columns:
        if "tokens" in token_df.columns:
            token_df = token_df.rename(columns={"tokens": "filtered_keywords"})
        else:
            raise KeyError("Tokens column not found. Expected one of ['filtered_keywords', 'tokens'].")

    # 머지 (일자 기준)
    df = (
        pd.merge(price_df, token_df[["date", "filtered_keywords"]], on="date", how="inner")
        .dropna(subset=["filtered_keywords"])
        .reset_index(drop=True)
    )
    return df, price_df


def ensure_dir(p: str) -> None:
    os.makedirs(p or ".", exist_ok=True)


# ---------------------------
# FinBERT embedding matrix
# ---------------------------
def setup_embedding(model_name: str = "yiyanghkust/finbert-tone"):
    from transformers import AutoModel, AutoTokenizer

    finbert_model = AutoModel.from_pretrained(model_name)
    finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        embedding_matrix = finbert_model.embeddings.word_embeddings.weight.detach().cpu()
    word2idx = finbert_tokenizer.get_vocab()
    unk_idx = word2idx.get("[UNK]", 0)
    pad_idx = word2idx.get("[PAD]", 0)
    return embedding_matrix, word2idx, unk_idx, pad_idx


def keywords_to_embeddings(
    keywords, embedding_matrix: torch.Tensor, word2idx: dict,
    unk_idx: int, pad_idx: int, num_keywords: int, embedding_dim: int
) -> torch.Tensor:
    if isinstance(keywords, str):
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    elif isinstance(keywords, list):
        keyword_list = [str(kw).strip() for kw in keywords if str(kw).strip()]
    else:
        keyword_list = []

    indices = [word2idx.get(w, unk_idx) for w in keyword_list[:num_keywords]]
    while len(indices) < num_keywords:
        indices.append(pad_idx)

    emb = embedding_matrix[indices]
    if emb.shape != (num_keywords, embedding_dim):
        return torch.zeros((num_keywords, embedding_dim), dtype=embedding_matrix.dtype)
    return emb


def create_price_tensor(price_data_np: np.ndarray, time_steps: int) -> torch.Tensor | None:
    if len(price_data_np) < time_steps:
        return None
    price_tensor = torch.tensor(price_data_np, dtype=torch.float32).unfold(0, time_steps, 1)
    return price_tensor.permute(0, 2, 1)  # [B, T, F]


def create_keyword_tensor(
    df_part: pd.DataFrame, time_steps: int, embedding_matrix: torch.Tensor,
    word2idx: dict, unk_idx: int, pad_idx: int, num_keywords: int, embedding_dim: int
) -> torch.Tensor | None:
    token_data = df_part["filtered_keywords"].tolist()
    if len(token_data) < time_steps:
        return None
    token_seq = token_data[time_steps - 1:]  # align with rolling windows
    embeddings = [
        keywords_to_embeddings(kws, embedding_matrix, word2idx, unk_idx, pad_idx, num_keywords, embedding_dim)
        for kws in token_seq
    ]
    embeddings = [emb for emb in embeddings if emb.shape == (num_keywords, embedding_dim)]
    if not embeddings:
        return None
    return torch.stack(embeddings)  # [B, K, D]


# ---------------------------
# Core routine
# ---------------------------
def run_shap_daily(
    price_csv: str,
    tokens_csv: str,
    outdir: str,
    start_year: int,
    end_year: int,
    num_keywords: int = 25,
    time_steps: int = 30,
    hidden_size: int = 384,
    horizons: List[int] = [1],
    embedding_dim: int = 768,
    finbert_model_name: str = "yiyanghkust/finbert-tone",
    bg_samples: int = 100,
    nsamples_kernel: int = 350,
    model_dir: str = "saved_models",
    scaler_dir: str = "saved_models",
) -> None:
    print(f"[INFO] Using device: {DEVICE}")
    ensure_dir(outdir)

    # Data
    df, price_df = load_data(price_csv, tokens_csv)
    feature_cols = [c for c in price_df.columns if c != "date"]
    price_feature_count = len(feature_cols)
    technical_names = feature_cols.copy()

    # Embedding resources
    embedding_matrix, word2idx, UNK_IDX, PAD_IDX = setup_embedding(finbert_model_name)

    # SHAP loop
    for horizon in horizons:
        for test_year in range(start_year, end_year + 1):
            train_start = test_year - 3

            model_path = os.path.join(model_dir, f"IKNet_{train_start}_{test_year}_k{num_keywords}.pt")
            scaler_path = os.path.join(scaler_dir, f"scaler_x_{train_start}_{test_year}_k{num_keywords}.pkl")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                print(f"[WARN] skip: missing model/scaler for {train_start}->{test_year} (k={num_keywords})")
                continue

            # Build test slice
            test_df = df[df["date"].dt.year == test_year].copy()
            if len(test_df) < time_steps:
                print(f"[WARN] skip: not enough rows for test_year={test_year}")
                continue

            scaler_x = joblib.load(scaler_path)

            # Tensors
            test_features_np = scaler_x.transform(test_df[feature_cols].values)
            x_price_tensor = create_price_tensor(test_features_np, time_steps)
            x_emb_tensor = create_keyword_tensor(
                test_df, time_steps, embedding_matrix, word2idx, UNK_IDX, PAD_IDX, num_keywords, embedding_dim
            )
            if x_price_tensor is None or x_emb_tensor is None:
                print(f"[WARN] skip: failed to build tensors for test_year={test_year}")
                continue

            # Align lengths
            num_samples = min(len(x_price_tensor), len(x_emb_tensor))
            x_price_tensor = x_price_tensor[:num_samples]
            x_emb_tensor = x_emb_tensor[:num_samples]

            # Tokens/dates for post-processing
            tokens_df = test_df.iloc[time_steps - 1: time_steps - 1 + num_samples]
            tokens_for_shap = tokens_df["filtered_keywords"].apply(
                lambda x: [t.strip() for t in str(x).split(",") if t.strip()]
            ).tolist()
            dates_for_shap = tokens_df["date"].tolist()

            # Flatten input features for KernelExplainer
            x_price_np = x_price_tensor.numpy().reshape(num_samples, -1)
            x_emb_np = x_emb_tensor.numpy().reshape(num_samples, -1)
            x_combined_np = np.concatenate([x_price_np, x_emb_np], axis=1)

            # Background samples
            bg_n = min(bg_samples, num_samples)
            bg_idx = np.random.choice(num_samples, bg_n, replace=False)
            background_data = x_combined_np[bg_idx]

            # Wrapped model
            model = IKNet(
                input_size=price_feature_count, output_size=1, num_keywords=num_keywords,
                embedding_dim=embedding_dim, hidden_size=hidden_size
            ).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()

            def wrapped_model(x: np.ndarray) -> np.ndarray:
                xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
                x_price = xt[:, :time_steps * price_feature_count].reshape(-1, time_steps, price_feature_count)
                x_emb = xt[:, time_steps * price_feature_count:].reshape(-1, num_keywords, embedding_dim)
                with torch.no_grad():
                    out = model(x_price, x_emb)  # [B, 1]
                return out.detach().cpu().numpy()

            # SHAP
            explainer = shap.KernelExplainer(wrapped_model, background_data)
            shap_values = explainer.shap_values(x_combined_np, nsamples=nsamples_kernel)
            shap_values_np = shap_values[0] if isinstance(shap_values, list) else shap_values

            # Squeeze to [N, F]
            if shap_values_np.ndim == 3 and shap_values_np.shape[2] == 1:
                shap_values_np = np.squeeze(shap_values_np, axis=2)
            elif shap_values_np.ndim != 2:
                raise ValueError(f"[ERROR] Unexpected SHAP shape: {shap_values_np.shape}")

            shap_values_np = np.nan_to_num(shap_values_np)

            # Post-process per-day keyword SHAP
            daily_df = extract_daily_keyword_shap(
                shap_values_array=shap_values_np,
                tokens_data=tokens_for_shap,
                dates=dates_for_shap,
                technical_names=technical_names,
                word2idx=word2idx,
                num_keywords=num_keywords
            )

            ensure_dir(outdir)
            save_path = os.path.join(outdir, f"keyword_shap_table_{test_year}_k{num_keywords}.csv")
            daily_df.to_csv(save_path, index=False)
            print(f"[OK] Saved: {save_path}")

    print("[DONE] Daily keyword SHAP complete.")


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IKNet - Daily Keyword SHAP")
    ap.add_argument("--price-csv", required=True, help="CSV with price/tech features (must include 'date')")
    ap.add_argument("--tokens-csv", required=True, help="CSV with columns: date, filtered_keywords or tokens")
    ap.add_argument("--outdir", default="outputs_shap/daily")

    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--horizons", type=str, default="1", help="comma-separated, currently only 1 supported")

    ap.add_argument("--num-keywords", type=int, default=25)
    ap.add_argument("--time-steps", type=int, default=30)
    ap.add_argument("--hidden-size", type=int, default=384)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--finbert-model", type=str, default="yiyanghkust/finbert-tone")

    ap.add_argument("--bg-samples", type=int, default=100)
    ap.add_argument("--nsamples-kernel", type=int, default=350)

    ap.add_argument("--model-dir", type=str, default="saved_models")
    ap.add_argument("--scaler-dir", type=str, default="saved_models")
    return ap.parse_args()


def main():
    args = parse_args()
    horizons = [int(x) for x in str(args.horizons).split(",") if str(x).strip()]
    run_shap_daily(
        price_csv=args.price_csv,
        tokens_csv=args.tokens_csv,
        outdir=args.outdir,
        start_year=args.start_year,
        end_year=args.end_year,
        num_keywords=args.num_keywords,
        time_steps=args.time_steps,
        hidden_size=args.hidden_size,
        horizons=horizons,
        embedding_dim=args.embedding_dim,
        finbert_model_name=args.finbert_model,
        bg_samples=args.bg_samples,
        nsamples_kernel=args.nsamples_kernel,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
    )


if __name__ == "__main__":
    main()
