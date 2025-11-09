# feature_importance.py  (unified paths: outputs/train_run_001/{models,scalers,preds})
from __future__ import annotations

import os
import sys
import gc
import json
import argparse
import logging
from typing import Optional, List, Tuple

import joblib
import shap
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Make parent directory importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.model import IKNet
from shap_analysis.feature_importance_all_absol import extract_top_n_feature_shap


# ---------------------------------------------------------
# Device & logging
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------
embedding_dim = 768
num_keywords = 17
time_steps = 10
hidden_size = 384
horizons = [1]

top_n = 15
N_SAMPLES_KERNEL = 500
N_BACKGROUND_SAMPLES = 300
BATCH_SIZE = 10


# ---------------------------------------------------------
# Argparse
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IK-MWFNet Feature Importance via SHAP (KernelExplainer)")
    ap.add_argument(
        "--run-dir",
        default="outputs/train_run_001",
        help="Root run directory (e.g., outputs/train_run_001)",
    )
    ap.add_argument(
        "--price-csv",
        default=os.environ.get("IKNET_PRICE_CSV", "dataset/price_features.csv"),
        help="CSV path for price/technical features (must include 'date')",
    )
    ap.add_argument(
        "--tokens-csv",
        default=os.environ.get("IKNET_TOKENS_CSV", "tokens/topk25_tokens.csv"),
        help="CSV path for tokens (columns: date, filtered_keywords or tokens)",
    )
    ap.add_argument(
        "--save-subdir",
        default="preds",
        help="Subdirectory under run-dir to save outputs (e.g., preds or figs)",
    )
    ap.add_argument(
        "--include-close",
        action="store_true",
        help="If set, keep the 'close' column among technical features (excluded by default)",
    )
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024)
    return ap.parse_args()


# ---------------------------------------------------------
# Data I/O
# ---------------------------------------------------------
def load_data(price_path: str, token_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price and token CSV, normalize date columns, and merge on date.

    If 'filtered_keywords' is absent, falls back to 'tokens'.
    """
    if not os.path.exists(price_path) or not os.path.exists(token_path):
        raise FileNotFoundError(f"Missing file(s): {price_path}, {token_path}")

    price_df = pd.read_csv(price_path)
    token_df = pd.read_csv(token_path)

    price_df["date"] = pd.to_datetime(price_df["date"])
    token_df["date"] = pd.to_datetime(token_df["date"])

    if "filtered_keywords" not in token_df.columns:
        if "tokens" in token_df.columns:
            token_df = token_df.rename(columns={"tokens": "filtered_keywords"})
        else:
            raise KeyError("Expected one of ['filtered_keywords', 'tokens'] in the tokens CSV.")

    df = (
        pd.merge(price_df, token_df, on="date", how="inner")
        .dropna(subset=["filtered_keywords"])
        .reset_index(drop=True)
    )
    return df, price_df


# ---------------------------------------------------------
# FinBERT embedding utilities
# ---------------------------------------------------------
def setup_embedding(model_name: str = "yiyanghkust/finbert-tone") -> tuple[torch.Tensor, dict, int, int]:
    """
    Load FinBERT to retrieve the token embedding matrix and the tokenizer vocab.
    Returns (embedding_matrix [V, D], word2idx, unk_idx, pad_idx).
    """
    finbert_model = AutoModel.from_pretrained(model_name)
    finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        emb = finbert_model.embeddings.word_embeddings.weight.detach().cpu()
    word2idx = finbert_tokenizer.get_vocab()
    unk_idx = word2idx.get("[UNK]", 0)
    pad_idx = word2idx.get("[PAD]", 0)
    return emb, word2idx, unk_idx, pad_idx


def keywords_to_embeddings(
    keywords,
    embedding_matrix: torch.Tensor,
    word2idx: dict,
    unk_idx: int,
    pad_idx: int,
    num_keywords_: int,
    embedding_dim_: int,
) -> torch.Tensor:
    """
    Convert a keyword list (or comma-separated string) to a [K, D] embedding tensor.
    Unknown tokens map to UNK; short lists are padded with PAD.
    """
    if isinstance(keywords, str):
        keyword_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    elif isinstance(keywords, list):
        keyword_list = [str(kw).strip() for kw in keywords if str(kw).strip()]
    else:
        keyword_list = []

    indices = [word2idx.get(w, unk_idx) for w in keyword_list[:num_keywords_]]
    while len(indices) < num_keywords_:
        indices.append(pad_idx)

    embeddings = embedding_matrix[indices]
    if embeddings.shape != (num_keywords_, embedding_dim_):
        return torch.zeros((num_keywords_, embedding_dim_), dtype=embedding_matrix.dtype)
    return embeddings


# ---------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------
def create_price_tensor(price_data_np: np.ndarray, time_steps_: int) -> Optional[torch.Tensor]:
    """
    Build [B, T, F] sliding windows from normalized price/technical features [N, F].
    """
    if len(price_data_np) < time_steps_:
        return None
    price_tensor = torch.tensor(price_data_np, dtype=torch.float32).unfold(0, time_steps_, 1)
    return price_tensor.permute(0, 2, 1)  # [B, T, F]


def create_keyword_tensor(
    df_part: pd.DataFrame,
    time_steps_: int,
    embedding_matrix: torch.Tensor,
    word2idx: dict,
    unk_idx: int,
    pad_idx: int,
    num_keywords_: int,
    embedding_dim_: int,
) -> Optional[torch.Tensor]:
    """
    Build [B, K, D] per-day keyword embeddings aligned with the sliding-window targets.
    """
    token_data = df_part["filtered_keywords"].tolist()
    if len(token_data) < time_steps_:
        return None

    token_seq = token_data[time_steps_ - 1 :]
    embeddings = [
        keywords_to_embeddings(
            kws, embedding_matrix, word2idx, unk_idx, pad_idx, num_keywords_, embedding_dim_
        )
        for kws in token_seq
    ]
    embeddings = [emb for emb in embeddings if emb.shape == (num_keywords_, embedding_dim_)]
    if not embeddings:
        return None
    return torch.stack(embeddings)  # [B, K, D]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main() -> None:
    args = parse_args()

    run_dir = args.run_dir
    model_dir = os.path.join(run_dir, "models")
    scaler_dir = os.path.join(run_dir, "scalers")
    out_dir = os.path.join(run_dir, args.save_subdir)
    os.makedirs(out_dir, exist_ok=True)

    logging.info("=" * 60)
    logging.info("RUN_DIR:    %s", run_dir)
    logging.info("MODEL_DIR:  %s", model_dir)
    logging.info("SCALER_DIR: %s", scaler_dir)
    logging.info("OUT_DIR:    %s", out_dir)
    logging.info("=" * 60)

    # Load data
    df, price_df = load_data(price_path=args.price_csv, token_path=args.tokens_csv)
    logging.info("Merged data rows: %d", len(df))

    # Feature columns (optionally drop 'close')
    feature_cols = [c for c in price_df.columns if c != "date"]
    if not args.include_close:
        feature_cols = [c for c in feature_cols if c.lower() != "close"]
    technical_names = feature_cols.copy()
    price_feature_count = len(feature_cols)
    logging.info("Using %d technical features: %s", price_feature_count, technical_names)

    # FinBERT embeddings
    logging.info("Loading FinBERT embeddings...")
    embedding_matrix, word2idx, UNK_IDX, PAD_IDX = setup_embedding()
    logging.info("Embedding matrix: %s", tuple(embedding_matrix.shape))

    processed_years: List[int] = []
    skipped_years: List[int] = []

    # Year loop
    for horizon in horizons:
        for test_year in range(args.start_year, args.end_year + 1):
            train_start = test_year - 3

            model_path = os.path.join(model_dir, f"IKNet_{train_start}_{test_year}_h{horizon}_k{num_keywords}.pt")
            scaler_path = os.path.join(scaler_dir, f"scaler_x_{train_start}_{test_year}_h{horizon}_k{num_keywords}.pkl")

            logging.info("=" * 60)
            logging.info("[Year %d] Checking files...", test_year)
            logging.info("  Model:  %s (%s)", model_path, "exists" if os.path.exists(model_path) else "missing")
            logging.info("  Scaler: %s (%s)", scaler_path, "exists" if os.path.exists(scaler_path) else "missing")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logging.warning("  Skip year %d (missing model or scaler)", test_year)
                skipped_years.append(test_year)
                continue

            logging.info("  Processing year %d...", test_year)

            scaler_x = joblib.load(scaler_path)
            test_df = df[df["date"].dt.year == test_year].copy()
            logging.info("     Test rows: %d", len(test_df))

            if len(test_df) < time_steps:
                logging.warning("     Insufficient rows for T=%d (got %d)", time_steps, len(test_df))
                skipped_years.append(test_year)
                continue

            # Build tensors
            test_features_np = scaler_x.transform(test_df[feature_cols].values)
            x_price_tensor = create_price_tensor(test_features_np, time_steps)
            x_emb_tensor = create_keyword_tensor(
                test_df, time_steps, embedding_matrix, word2idx, UNK_IDX, PAD_IDX, num_keywords, embedding_dim
            )

            if x_price_tensor is None or x_emb_tensor is None:
                logging.warning("     Failed to create tensors")
                skipped_years.append(test_year)
                continue

            num_samples = min(len(x_price_tensor), len(x_emb_tensor))
            x_price_tensor = x_price_tensor[:num_samples]
            x_emb_tensor = x_emb_tensor[:num_samples]
            logging.info("     Created tensors: %d samples", num_samples)

            tokens_df = test_df.iloc[time_steps - 1 : time_steps - 1 + num_samples]
            tokens_for_shap = tokens_df["filtered_keywords"].apply(
                lambda x: [t.strip() for t in str(x).split(",") if t.strip()]
            ).tolist()

            x_price_np = x_price_tensor.numpy().reshape(num_samples, -1)
            x_emb_np = x_emb_tensor.numpy().reshape(num_samples, -1)
            x_combined_np = np.concatenate([x_price_np, x_emb_np], axis=1)

            # Background for KernelExplainer
            bg_idx = np.random.choice(num_samples, min(N_BACKGROUND_SAMPLES, num_samples), replace=False)
            background_data = x_combined_np[bg_idx]
            logging.info("     Background samples: %d", len(background_data))

            # Load model on CPU
            logging.info("     Loading model on CPU (memory efficient)...")
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
                SHAP wrapper. Input x is [B, T*F + K*D], output is [B, 1].
                """
                xt = torch.tensor(x, dtype=torch.float32)
                x_price = xt[:, : time_steps * price_feature_count].reshape(-1, time_steps, price_feature_count)
                x_emb = xt[:, time_steps * price_feature_count :].reshape(-1, num_keywords, embedding_dim)
                with torch.no_grad():
                    out = model(x_price, x_emb)  # [B, 1]
                return out.numpy()

            # SHAP in batches
            logging.info("     Computing SHAP values (nsamples=%d, batch_size=%d)...", N_SAMPLES_KERNEL, BATCH_SIZE)
            explainer = shap.KernelExplainer(wrapped_model, background_data)

            all_shap_values: List[np.ndarray] = []
            num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_start in range(0, num_samples, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, num_samples)
                batch_data = x_combined_np[batch_start:batch_end]

                batch_no = batch_start // BATCH_SIZE + 1
                logging.info("       Batch %d/%d (samples %d-%d)", batch_no, num_batches, batch_start, batch_end - 1)

                batch_shap = explainer.shap_values(batch_data, nsamples=N_SAMPLES_KERNEL)
                all_shap_values.append(batch_shap[0] if isinstance(batch_shap, list) else batch_shap)

                gc.collect()

            # Concatenate to [N, F]
            shap_values_np = np.concatenate(all_shap_values, axis=0)

            # Squeeze if needed
            if shap_values_np.ndim == 3 and shap_values_np.shape[2] == 1:
                shap_values_np = np.squeeze(shap_values_np, axis=2)
            elif shap_values_np.ndim != 2:
                raise ValueError(f"Unexpected SHAP array shape: {shap_values_np.shape}")

            shap_values_np = np.nan_to_num(shap_values_np)
            logging.info("     SHAP values shape: %s", tuple(shap_values_np.shape))

            # Save figure path under run dir
            save_path = os.path.join(out_dir, f"Feature_Importance_{test_year}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            logging.info("     Extracting top-%d features...", top_n)
            _ = extract_top_n_feature_shap(
                shap_values_array=shap_values_np,
                tokens_data=tokens_for_shap,
                technical_names=technical_names,
                test_year=test_year,
                top_n=top_n,
                save_path=save_path,
                word2idx=word2idx,
                embedding_dim=embedding_dim,
            )

            processed_years.append(test_year)
            logging.info("     Completed year %d", test_year)

    # Summary
    logging.info("=" * 60)
    logging.info("SHAP Analysis Summary")
    logging.info("  Processed: %d years - %s", len(processed_years), processed_years)
    logging.info("  Skipped:   %d years - %s", len(skipped_years), skipped_years)
    logging.info("=" * 60)

    if not processed_years:
        logging.warning("No years were processed. Please check:")
        logging.warning("  1) Model files exist in: %s", model_dir)
        logging.warning("  2) Scaler files exist in: %s", scaler_dir)
        logging.warning("  3) File naming: IKNet_{train_start}_{test_year}_h{horizon}_k{num_keywords}.pt")
        logging.warning("  4) Data availability for the specified years")


if __name__ == "__main__":
    main()
