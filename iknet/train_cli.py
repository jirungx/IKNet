# -*- coding: utf-8 -*-
import argparse
import os
import csv
import sys
import json
import time
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from .modules.rolling_utils import split_by_rolling_window, normalize_and_sequence
from .modules.model import IKNet
from .modules.train import train_model
from .modules.predict import predict_model
from .modules.metrics_utils import compute_metrics, print_metrics
from .config import DEVICE


# ---------------------------
# Logging
# ---------------------------
def setup_logger(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Embedding cache loader
# ---------------------------
def load_embedding_cache(pkl_path: str):
    import joblib
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Embedding cache not found: {pkl_path}")
    cache = joblib.load(pkl_path)
    if not isinstance(cache, dict):
        raise ValueError(f"Unexpected cache type: {type(cache)}; expected dict[date_str]->ndarray(K,768)")
    return cache


def get_cached_embeddings(
    date_series: pd.Series,
    embedding_cache: dict,
    top_k: int,
    embedding_dim: int = 768,
) -> torch.Tensor:
    """
    date_series: pandas Series of datetime64
    embedding_cache: {'YYYY-MM-DD': np.ndarray [K, embedding_dim]}
    returns: torch.FloatTensor [B, top_k, embedding_dim]
    """
    embs = []
    for date in date_series:
        key = pd.to_datetime(date).strftime("%Y-%m-%d")
        if key in embedding_cache:
            arr = embedding_cache[key]
            if arr.shape[0] >= top_k:
                embs.append(torch.tensor(arr[:top_k], dtype=torch.float32))
            else:
                pad = np.zeros((top_k, arr.shape[1]), dtype=np.float32)
                pad[:arr.shape[0], :arr.shape[1]] = arr
                embs.append(torch.tensor(pad, dtype=torch.float32))
        else:
            embs.append(torch.zeros((top_k, embedding_dim), dtype=torch.float32))
    return torch.stack(embs, dim=0)


# ---------------------------
# Argparse
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="IKNet training CLI (rolling windows).")

    # Data and features
    ap.add_argument("--price-csv", required=True, help="CSV with columns: date, features incl. close")
    ap.add_argument("--feature-cols", required=True, help="Comma-separated feature columns (include close)")

    # Windowing
    ap.add_argument("--time-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, nargs="+", default=[1], help="One or more forecast horizons")
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-years", type=int, default=1)
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2024)

    # Keywords / embeddings
    ap.add_argument("--use-keywords", action="store_true", help="Use precomputed FinBERT keyword embeddings")
    ap.add_argument("--embedding-pkl", default="precomputed_embeddings/finbert_embeddings_k25.pkl",
                    help="joblib pkl of {date: [K,768]} dict")
    ap.add_argument("--topk", type=int, default=17, help="number of keywords per day")
    ap.add_argument("--embedding-dim", type=int, default=768)

    # Model + training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=384)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)

    # Output
    ap.add_argument("--outdir", default="outputs", help="Base directory to save results")
    ap.add_argument("--results-filename", default="results.csv", help="CSV filename for metrics inside outdir")

    # Saving toggles
    ap.add_argument("--save-models", action="store_true", help="Save model weights per window")
    ap.add_argument("--model-dir", type=str, default=None, help="Where to store model weights; defaults to outdir/models")
    ap.add_argument("--save-scalers", action="store_true", help="Save fitted scalers per window")
    ap.add_argument("--scaler-dir", type=str, default=None, help="Where to store scalers; defaults to outdir/scalers")
    ap.add_argument("--save-preds", action="store_true", help="Save per-window predictions as CSV")
    ap.add_argument("--preds-dir", type=str, default=None, help="Where to store predictions; defaults to outdir/preds")

    return ap.parse_args()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    setup_logger(args.outdir)
    set_seed(args.seed)

    logging.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False))

    # Prepare dirs
    os.makedirs(args.outdir, exist_ok=True)
    model_dir = args.model_dir or os.path.join(args.outdir, "models")
    scaler_dir = args.scaler_dir or os.path.join(args.outdir, "scalers")
    preds_dir = args.preds_dir or os.path.join(args.outdir, "preds")
    if args.save_models:
        os.makedirs(model_dir, exist_ok=True)
    if args.save_scalers:
        os.makedirs(scaler_dir, exist_ok=True)
    if args.save_preds:
        os.makedirs(preds_dir, exist_ok=True)

    # Load data
    price_df = pd.read_csv(args.price_csv)
    if "date" not in price_df.columns:
        raise ValueError("price_csv must contain a 'date' column")
    price_df["date"] = pd.to_datetime(price_df["date"])

    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    for col in feature_cols:
        if col not in price_df.columns:
            raise ValueError(f"Feature column '{col}' not found in price_csv")

    windows = split_by_rolling_window(
        price_df, args.train_years, args.test_years, args.start_year, args.end_year
    )
    logging.info("Rolling windows: %d", len(windows))

    # Embeddings (optional)
    if args.use_keywords:
        embedding_cache = load_embedding_cache(args.embedding_pkl)
        logging.info("Loaded embedding cache with %d dates", len(embedding_cache))
    else:
        embedding_cache = None
        logging.info("Keyword embeddings disabled; will use zero tensors")

    # Results writer
    result_csv = os.path.join(args.outdir, args.results_filename)
    with open(result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_span", "test_year", "horizon", "time_steps", "RMSE", "MAE", "SMAPE", "R2"])

        for h in args.horizon:
            for (train_start, test_start, train_df, test_df) in windows:
                try:
                    # Build sequences and scalers
                    X_train, y_train, X_test, y_test, scaler_x, scaler_y = normalize_and_sequence(
                        train_df.copy(), test_df.copy(), feature_cols, args.time_steps, h
                    )

                    # Align dates with sequences
                    date_train = train_df["date"].iloc[
                        args.time_steps - 1 : args.time_steps - 1 + len(X_train)
                    ].reset_index(drop=True)
                    date_test = test_df["date"].iloc[
                        args.time_steps - 1 : args.time_steps - 1 + len(X_test)
                    ].reset_index(drop=True)

                    # Build embeddings if required
                    if args.use_keywords:
                        X_emb_train = get_cached_embeddings(date_train, embedding_cache, args.topk, args.embedding_dim)
                        X_emb_test = get_cached_embeddings(date_test, embedding_cache, args.topk, args.embedding_dim)
                    else:
                        X_emb_train = torch.zeros((len(X_train), args.topk, args.embedding_dim), dtype=torch.float32)
                        X_emb_test = torch.zeros((len(X_test), args.topk, args.embedding_dim), dtype=torch.float32)

                    # Tensors
                    X_train_t = torch.tensor(X_train, dtype=torch.float32)
                    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                    X_test_t = torch.tensor(X_test, dtype=torch.float32)
                    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

                    # Model
                    input_size = X_train_t.shape[-1]
                    model = IKNet(
                        input_size=input_size,
                        num_keywords=args.topk,
                        embedding_dim=args.embedding_dim,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        output_size=1,
                        dropout=args.dropout,
                    )

                    # Train
                    model = train_model(
                        model,
                        X_train_t, X_emb_train,
                        y_train_t,
                        device=str(DEVICE),
                        epochs=args.epochs,
                        lr=args.lr,
                    )

                    # Predict
                    y_pred = predict_model(model, X_test_t, X_emb_test, device=str(DEVICE))

                    # Inverse scale predictions and targets
                    y_true_inv = scaler_y.inverse_transform(y_test_t.numpy().reshape(-1, 1)).reshape(-1)
                    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)

                    # Metrics
                    metrics = compute_metrics(y_true_inv, y_pred_inv)
                    print_metrics(metrics, label=f"{train_start}-{test_start-1} -> {test_start} h={h}")
                    writer.writerow([
                        f"{train_start}-{test_start-1}", test_start, h, args.time_steps,
                        round(metrics["RMSE"], 3), round(metrics["MAE"], 3),
                        round(metrics["SMAPE"], 3), round(metrics["R2"], 3),
                    ])

                    # Save model / scalers / preds
                    tag = f"{train_start}_{test_start}_h{h}_k{args.topk}"

                    if args.save_models:
                        torch.save(model.state_dict(), os.path.join(model_dir, f"IKNet_{tag}.pt"))

                    if args.save_scalers:
                        import joblib
                        joblib.dump(scaler_x, os.path.join(scaler_dir, f"scaler_x_{tag}.pkl"))
                        joblib.dump(scaler_y, os.path.join(scaler_dir, f"scaler_y_{tag}.pkl"))

                    if args.save_preds:
                        df_pred = pd.DataFrame({
                            "date": test_df["date"].iloc[
                                args.time_steps + h - 1 : args.time_steps + h - 1 + len(y_true_inv)
                            ].reset_index(drop=True),
                            "y_true": y_true_inv,
                            "y_pred": y_pred_inv,
                        })
                        df_pred.to_csv(os.path.join(preds_dir, f"preds_{tag}.csv"), index=False)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logging.error(f"[ERROR] Window {train_start}-{test_start-1} -> {test_start}, h={h}: {e}")

    logging.info("Done. Results saved to %s", result_csv)


if __name__ == "__main__":
    main()
