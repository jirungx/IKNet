# -*- coding: utf-8 -*-
import os
import csv
import joblib
import random
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict, Optional

# 패키지 상대 임포트 (중요!)
from .modules.rolling_utils import split_by_rolling_window, normalize_and_sequence
from .modules.model import IKNet
from .modules.train import train_model
from .modules.predict import predict_model
from .modules.metrics_utils import compute_metrics, print_metrics
from .config import DEVICE


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Embedding cache utilities
# ---------------------------
def load_embedding_cache(pkl_path: str) -> Dict[str, np.ndarray]:
    """Load precomputed FinBERT keyword embeddings by date (key = 'YYYY-MM-DD')."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Embedding cache not found: {pkl_path}")
    cache = joblib.load(pkl_path)
    if not isinstance(cache, dict):
        raise ValueError(f"Unexpected cache type: {type(cache)}")
    return cache


def get_cached_embeddings(
    dates: pd.Series,
    cache: Dict[str, np.ndarray],
    top_k: int,
    embedding_dim: int = 768,
) -> torch.Tensor:
    """
    dates: pandas Series of datetime64[ns]
    cache: { 'YYYY-MM-DD': np.ndarray [K, 768] }
    returns: torch.Tensor [B, K, 768]
    """
    embs: List[torch.Tensor] = []
    for dt in dates:
        key = pd.to_datetime(dt).strftime("%Y-%m-%d")
        if key in cache:
            arr = cache[key][:top_k]
            embs.append(torch.tensor(arr, dtype=torch.float32))
        else:
            embs.append(torch.zeros((top_k, embedding_dim), dtype=torch.float32))
    return torch.stack(embs)  # [B, K, 768]


def get_keywords_batch(
    df: pd.DataFrame,
    token_df: pd.DataFrame,
    time_steps: int,
    num_samples: int,
) -> List[List[str]]:
    """
    token_df must have columns: date, tokens (comma-separated)
    returns: list of token list per sample
    """
    # 날짜 정규화
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    if not np.issubdtype(token_df["date"].dtype, np.datetime64):
        token_df = token_df.copy()
        token_df["date"] = pd.to_datetime(token_df["date"])

    date_targets = df["date"].iloc[time_steps - 1 : time_steps - 1 + num_samples].reset_index(drop=True)

    batch_keywords: List[List[str]] = []
    for date in date_targets:
        mask = token_df["date"] == pd.to_datetime(date)
        if mask.any():
            row = token_df.loc[mask].iloc[0]
            tokens_str = row.get("tokens", "")
            if pd.notna(tokens_str) and str(tokens_str).strip():
                tokens = [t.strip() for t in str(tokens_str).split(",") if t.strip()]
            else:
                tokens = []
        else:
            tokens = []
        batch_keywords.append(tokens)
    return batch_keywords


# ---------------------------
# Main experiment runner (optional)
# ---------------------------
def run_experiment(
    price_csv: str,
    tokens_csv: str,
    embedding_pkl: str,
    out_metrics_csv: str = "results/IKNet.csv",
    out_pred_dir: str = "results/IKNet_preds",
    save_dir: str = "saved_models",
    time_steps: int = 10,
    horizons: List[int] = [1],
    train_years: int = 3,
    test_years: int = 1,
    num_keywords: int = 17,
    seed: int = 10,
) -> None:
    """
    A self-contained training+evaluation loop. Safe to import (no side effects).
    """
    set_seed(seed)
    os.makedirs(os.path.dirname(out_metrics_csv) or ".", exist_ok=True)
    os.makedirs(out_pred_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    price_df = pd.read_csv(price_csv)
    tokens_df = pd.read_csv(tokens_csv)
    price_df["date"] = pd.to_datetime(price_df["date"])
    tokens_df["date"] = pd.to_datetime(tokens_df["date"])

    feature_cols = [c for c in price_df.columns if c != "date"]
    windows = split_by_rolling_window(price_df, train_years, test_years)
    embedding_cache = load_embedding_cache(embedding_pkl)

    with open(out_metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_years", "test_year", "horizon", "time_steps", "RMSE", "MAE", "SMAPE", "R2"])

        for horizon in horizons:
            for train_start, test_start, train_df, test_df in windows:
                try:
                    X_train, y_train, X_test, y_test, scaler_x, scaler_y = normalize_and_sequence(
                        train_df, test_df, feature_cols, time_steps, horizon
                    )

                    # Build embedding tensors from cache
                    date_train = train_df["date"].iloc[time_steps - 1 : time_steps - 1 + len(X_train)].reset_index(drop=True)
                    X_emb_train = get_cached_embeddings(date_train, embedding_cache, top_k=num_keywords)

                    date_test = test_df["date"].iloc[time_steps - 1 : time_steps - 1 + len(X_test)].reset_index(drop=True)
                    X_emb_test = get_cached_embeddings(date_test, embedding_cache, top_k=num_keywords)

                    # Torch tensors
                    X_train = torch.tensor(X_train, dtype=torch.float32)
                    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                    X_test = torch.tensor(X_test, dtype=torch.float32)
                    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

                    # Train
                    model = IKNet(input_size=X_train.shape[2], output_size=1, num_keywords=num_keywords)
                    model = train_model(model, X_train, X_emb_train, y_train, device=DEVICE)

                    # Save model & scalers
                    model_name = f"IKNet_{train_start}_{test_start}_k{num_keywords}.pt"
                    torch.save(model.state_dict(), os.path.join(save_dir, model_name))

                    joblib.dump(scaler_x, os.path.join(save_dir, f"scaler_x_{train_start}_{test_start}_k{num_keywords}.pkl"))
                    joblib.dump(scaler_y, os.path.join(save_dir, f"scaler_y_{train_start}_{test_start}_k{num_keywords}.pkl"))

                    # Predict
                    pred = predict_model(model, X_test, X_emb_test, device=DEVICE)

                    # Inverse transform
                    y_true = scaler_y.inverse_transform(y_test.view(-1, 1).cpu().numpy()).flatten()
                    y_pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()

                    # Save per-date predictions
                    dates = test_df["date"].iloc[time_steps + horizon - 1 : time_steps + horizon - 1 + len(y_true)].reset_index(drop=True)
                    df_result = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
                    df_result.to_csv(os.path.join(out_pred_dir, f"IKNet_{test_start}_k{num_keywords}.csv"), index=False)

                    # Metrics
                    metrics = compute_metrics(y_true, y_pred)
                    print_metrics(metrics, label=f"{train_start}-{test_start-1} --> {test_start} h={horizon}")

                    writer.writerow([
                        f"{train_start}-{test_start - 1}", test_start, horizon, time_steps,
                        round(metrics["RMSE"], 3),
                        round(metrics["MAE"], 3),
                        round(metrics["SMAPE"], 3),
                        round(metrics["R2"], 3),
                    ])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[ERROR] {train_start}-{test_start} 에러: {e}")


if __name__ == "__main__":
    # 예시 실행: 실제로는 iknet-train에서 자체 argparse를 쓰므로 여기선 아무것도 안 함.
    pass
