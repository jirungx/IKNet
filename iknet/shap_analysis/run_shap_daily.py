# ik_mwfnet/shap_analysis/run_shap_daily.py
# -*- coding: utf-8 -*-
"""
Daily keyword-level SHAP generator for IKNet.

- Loads trained IKNet checkpoints per rolling window (train_start -> test_year)
- Uses FinBERT token embedding matrix on the fly (no need for precomputed PKL here)
- Produces a per-day keyword SHAP CSV per test year

Example:
iknet-shap-daily \
  --price-csv dataset/price_features.csv \
  --tokens-csv tokens/topk25_tokens.csv \
  --outdir outputs_shap/daily \
  --start-year 2018 --end-year 2024 \
  --num-keywords 17 --time-steps 10 --hidden-size 384 \
  --model-dir outputs/train_run_001/models \
  --scaler-dir outputs/train_run_001/scalers
"""

from __future__ import annotations

import os
import gc
import argparse
import logging
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import shap
import torch

# Local imports
from ..modules.model import IKNet
from .shap_keyword_utils_daily import extract_daily_keyword_shap


# ---------------------------------------------------------
# Device & logger
# ---------------------------------------------------------
# Use CPU for SHAP analysis to avoid OOM; KernelExplainer is CPU-friendly
DEVICE = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------
def load_data(price_path: str, token_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price features and token CSV, normalize date columns, and inner-join by date.

    If 'filtered_keywords' is absent in tokens CSV, it falls back to 'tokens'.

    Args:
        price_path (str): CSV path with columns including 'date' and feature columns.
        token_path (str): CSV path with columns ['date', 'filtered_keywords'] or ['date', 'tokens'].

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (merged_df, raw_price_df)
            - merged_df has columns from price_df and a 'filtered_keywords' column.
            - raw_price_df is returned to report feature column order and count.
    """
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Price CSV not found: {price_path}")
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Tokens CSV not found: {token_path}")

    price_df = pd.read_csv(price_path)
    token_df = pd.read_csv(token_path)

    price_df["date"] = pd.to_datetime(price_df["date"])
    token_df["date"] = pd.to_datetime(token_df["date"])

    if "filtered_keywords" not in token_df.columns:
        if "tokens" in token_df.columns:
            token_df = token_df.rename(columns={"tokens": "filtered_keywords"})
        else:
            raise KeyError("Tokens column not found. Expected one of ['filtered_keywords', 'tokens'].")

    merged = (
        pd.merge(price_df, token_df[["date", "filtered_keywords"]], on="date", how="inner")
        .dropna(subset=["filtered_keywords"])
        .reset_index(drop=True)
    )
    return merged, price_df


def ensure_dir(p: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(p or ".", exist_ok=True)


# ---------------------------------------------------------
# FinBERT embedding matrix
# ---------------------------------------------------------
def setup_embedding(model_name: str = "yiyanghkust/finbert-tone"):
    """
    Load FinBERT to retrieve its token embedding matrix and vocabulary.

    Args:
        model_name (str): HF model id for FinBERT.

    Returns:
        Tuple[torch.Tensor, dict, int, int]:
            - embedding_matrix: [vocab_size, embedding_dim] on CPU
            - word2idx: tokenizer vocabulary mapping token -> index
            - unk_idx: index for [UNK]
            - pad_idx: index for [PAD]
    """
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
    keywords,
    embedding_matrix: torch.Tensor,
    word2idx: dict,
    unk_idx: int,
    pad_idx: int,
    num_keywords: int,
    embedding_dim: int,
) -> torch.Tensor:
    """
    Convert a list or comma-separated string of keywords to an embedding matrix [K, D].

    Missing/short lists are padded with PAD token; unknown tokens map to UNK.

    Args:
        keywords: list[str] or comma-separated str of keywords.
        embedding_matrix (torch.Tensor): Token embedding matrix [V, D].
        word2idx (dict): Vocab mapping token -> index.
        unk_idx (int): Index for unknown token.
        pad_idx (int): Index for padding token.
        num_keywords (int): K (number of keywords to keep).
        embedding_dim (int): D (embedding dimension).

    Returns:
        torch.Tensor: [num_keywords, embedding_dim]
    """
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


def create_price_tensor(price_data_np: np.ndarray, time_steps: int) -> Optional[torch.Tensor]:
    """
    Create sliding-window price tensor [B, T, F] from normalized feature array.

    Args:
        price_data_np (np.ndarray): 2D array [N, F] of normalized price features.
        time_steps (int): Window length T.

    Returns:
        Optional[torch.Tensor]: [B, T, F] or None if not enough rows.
    """
    if len(price_data_np) < time_steps:
        return None
    price_tensor = torch.tensor(price_data_np, dtype=torch.float32).unfold(0, time_steps, 1)
    return price_tensor.permute(0, 2, 1)  # [B, T, F]


def create_keyword_tensor(
    df_part: pd.DataFrame,
    time_steps: int,
    embedding_matrix: torch.Tensor,
    word2idx: dict,
    unk_idx: int,
    pad_idx: int,
    num_keywords: int,
    embedding_dim: int,
) -> Optional[torch.Tensor]:
    """
    Create per-day keyword embedding tensor [B, K, D] aligned with sliding windows.

    Args:
        df_part (pd.DataFrame): Must include 'filtered_keywords'.
        time_steps (int): Window length T.
        embedding_matrix (torch.Tensor): [V, D] embedding table.
        word2idx (dict): Vocab mapping token -> index.
        unk_idx (int): Index for unknown token.
        pad_idx (int): Index for padding token.
        num_keywords (int): K (keywords per day).
        embedding_dim (int): D (embedding dimension).

    Returns:
        Optional[torch.Tensor]: [B, K, D] or None if not enough rows or empty tokens.
    """
    token_data = df_part["filtered_keywords"].tolist()
    if len(token_data) < time_steps:
        return None

    token_seq = token_data[time_steps - 1 :]  # align with rolling start
    embeddings = [
        keywords_to_embeddings(kws, embedding_matrix, word2idx, unk_idx, pad_idx, num_keywords, embedding_dim)
        for kws in token_seq
    ]
    embeddings = [emb for emb in embeddings if emb.shape == (num_keywords, embedding_dim)]
    if not embeddings:
        return None
    return torch.stack(embeddings)  # [B, K, D]


# ---------------------------------------------------------
# Core routine
# ---------------------------------------------------------
def run_shap_daily(
    price_csv: str,
    tokens_csv: str,
    outdir: str,
    start_year: int,
    end_year: int,
    num_keywords: int = 17,
    time_steps: int = 30,
    hidden_size: int = 384,
    horizons: List[int] = [1],
    embedding_dim: int = 768,
    finbert_model_name: str = "yiyanghkust/finbert-tone",
    bg_samples: int = 300,
    nsamples_kernel: int = 500,
    batch_size: int = 10,
    model_dir: str = "saved_models",
    scaler_dir: str = "saved_models",
) -> None:
    """
    Compute per-day keyword SHAP tables for each test year within [start_year, end_year].

    Outputs:
        {outdir}/keyword_shap_table_{test_year}_h{horizon}_k{num_keywords}.csv
    """
    logging.info("Using device: %s", DEVICE)
    ensure_dir(outdir)

    # Load and prepare data
    df, price_df = load_data(price_csv, tokens_csv)
    feature_cols = [c for c in price_df.columns if c != "date"]
    price_feature_count = len(feature_cols)
    technical_names = feature_cols.copy()

    # Prepare FinBERT embedding resources
    embedding_matrix, word2idx, UNK_IDX, PAD_IDX = setup_embedding(finbert_model_name)
    logging.info("Loaded FinBERT embeddings: vocab=%d, dim=%d", embedding_matrix.shape[0], embedding_matrix.shape[1])

    # Iterate horizons and test years
    for horizon in horizons:
        for test_year in range(start_year, end_year + 1):
            train_start = test_year - 3
            tag = f"{train_start}_{test_year}_h{horizon}_k{num_keywords}"

            model_path = os.path.join(model_dir, f"IKNet_{tag}.pt")
            scaler_x_path = os.path.join(scaler_dir, f"scaler_x_{tag}.pkl")

            logging.info("============================================================")
            logging.info("[Year %d] model=%s | scaler=%s", test_year, model_path, scaler_x_path)

            if not os.path.exists(model_path) or not os.path.exists(scaler_x_path):
                logging.warning("Skip: missing model or scaler for %d -> %d (h=%d, k=%d)", train_start, test_year, horizon, num_keywords)
                continue

            # Test slice
            test_df = df[df["date"].dt.year == test_year].copy()
            if len(test_df) < time_steps:
                logging.warning("Skip: not enough rows for test_year=%d", test_year)
                continue

            scaler_x = joblib.load(scaler_x_path)

            # Build tensors
            test_features_np = scaler_x.transform(test_df[feature_cols].values)
            x_price_tensor = create_price_tensor(test_features_np, time_steps)
            x_emb_tensor = create_keyword_tensor(
                test_df, time_steps, embedding_matrix, word2idx, UNK_IDX, PAD_IDX, num_keywords, embedding_dim
            )
            if x_price_tensor is None or x_emb_tensor is None:
                logging.warning("Skip: failed to build tensors for test_year=%d", test_year)
                continue

            # Align lengths
            num_samples = min(len(x_price_tensor), len(x_emb_tensor))
            x_price_tensor = x_price_tensor[:num_samples]
            x_emb_tensor = x_emb_tensor[:num_samples]
            logging.info("Created tensors: %d samples", num_samples)

            # Prepare tokens/dates for post-processing
            tokens_df = test_df.iloc[time_steps - 1 : time_steps - 1 + num_samples]
            tokens_for_shap = tokens_df["filtered_keywords"].apply(
                lambda x: [t.strip() for t in str(x).split(",") if t.strip()]
            ).tolist()
            dates_for_shap = tokens_df["date"].tolist()

            # Flatten features for KernelExplainer
            x_price_np = x_price_tensor.numpy().reshape(num_samples, -1)
            x_emb_np = x_emb_tensor.numpy().reshape(num_samples, -1)
            x_combined_np = np.concatenate([x_price_np, x_emb_np], axis=1)

            # Background samples for KernelExplainer
            bg_n = min(bg_samples, num_samples)
            bg_idx = np.random.choice(num_samples, bg_n, replace=False)
            background_data = x_combined_np[bg_idx]
            logging.info("Background samples: %d", len(background_data))

            # Load model (CPU)
            logging.info("Loading model on CPU...")
            model = IKNet(
                input_size=price_feature_count,
                output_size=1,
                num_keywords=num_keywords,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
            ).cpu()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()

            def wrapped_model(x: np.ndarray) -> np.ndarray:
                """
                Wrapper for SHAP KernelExplainer: x -> model outputs.
                x is a 2D array [B, T*F + K*D] combining price and keyword embeddings.
                """
                xt = torch.tensor(x, dtype=torch.float32)  # CPU tensor
                x_price = xt[:, : time_steps * price_feature_count].reshape(-1, time_steps, price_feature_count)
                x_emb = xt[:, time_steps * price_feature_count :].reshape(-1, num_keywords, embedding_dim)
                with torch.no_grad():
                    out = model(x_price, x_emb)  # [B, 1]
                return out.numpy()

            # SHAP computation in batches
            logging.info("Computing SHAP values (nsamples=%d, batch_size=%d)...", nsamples_kernel, batch_size)
            explainer = shap.KernelExplainer(wrapped_model, background_data)

            all_shap_values: List[np.ndarray] = []
            num_batches = (num_samples + batch_size - 1) // batch_size

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_data = x_combined_np[batch_start:batch_end]

                batch_num = batch_start // batch_size + 1
                logging.info("  Batch %d/%d (samples %d-%d)", batch_num, num_batches, batch_start, batch_end - 1)

                batch_shap = explainer.shap_values(batch_data, nsamples=nsamples_kernel)
                # If model has a single output, shap_values may be a list with one element
                all_shap_values.append(batch_shap[0] if isinstance(batch_shap, list) else batch_shap)

                # Free memory periodically
                gc.collect()

            # Concatenate batch results -> [N, F]
            shap_values_np = np.concatenate(all_shap_values, axis=0)

            # Squeeze or validate shape
            if shap_values_np.ndim == 3 and shap_values_np.shape[2] == 1:
                shap_values_np = np.squeeze(shap_values_np, axis=2)
            elif shap_values_np.ndim != 2:
                raise ValueError(f"Unexpected SHAP shape: {shap_values_np.shape}")

            shap_values_np = np.nan_to_num(shap_values_np)
            logging.info("SHAP values shape: %s", tuple(shap_values_np.shape))

            # Post-process per-day keyword SHAP
            logging.info("Extracting daily keyword SHAP...")
            daily_df = extract_daily_keyword_shap(
                shap_values_array=shap_values_np,
                tokens_data=tokens_for_shap,
                dates=dates_for_shap,
                technical_names=technical_names,
                word2idx=word2idx,
                num_keywords=num_keywords,
            )

            ensure_dir(outdir)
            save_path = os.path.join(outdir, f"keyword_shap_table_{test_year}_h{horizon}_k{num_keywords}.csv")
            daily_df.to_csv(save_path, index=False)
            logging.info("Saved: %s", save_path)

    logging.info("Daily keyword SHAP complete.")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IKNet - Daily Keyword SHAP")
    ap.add_argument("--price-csv", required=True, help="CSV with price/tech features (must include 'date')")
    ap.add_argument("--tokens-csv", required=True, help="CSV with columns: date, filtered_keywords or tokens")
    ap.add_argument("--outdir", default="outputs_shap/daily")

    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--horizons", type=str, default="1", help="Comma-separated horizons; currently only 1 is supported")

    ap.add_argument("--num-keywords", type=int, default=17)
    ap.add_argument("--time-steps", type=int, default=10)
    ap.add_argument("--hidden-size", type=int, default=384)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--finbert-model", type=str, default="yiyanghkust/finbert-tone")

    ap.add_argument("--bg-samples", type=int, default=300, help="Background samples for SHAP (default: 300)")
    ap.add_argument("--nsamples-kernel", type=int, default=500, help="SHAP kernel samples (default: 500)")
    ap.add_argument("--batch-size", type=int, default=10, help="Batch size for SHAP processing (default: 10)")

    # Paths consistent with train_cli artifact layout
    ap.add_argument("--model-dir", type=str, default="outputs/train_run_001/models")
    ap.add_argument("--scaler-dir", type=str, default="outputs/train_run_001/scalers")
    return ap.parse_args()


def main() -> None:
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
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
    )


if __name__ == "__main__":
    main()
