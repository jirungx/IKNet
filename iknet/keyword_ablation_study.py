import os
import csv
import random
import joblib
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


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across python, numpy, and torch.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(11)


# ---------------------------------------------------------
# File paths and experiment settings
# ---------------------------------------------------------
PRICE_PATH = "dataset/snp500_dataset.csv"
TOKEN_PATH = "tokens/snp_topk25_tokens.csv"
EMBEDDING_CACHE_PATH = "precomputed_embeddings/finbert_embeddings_k25.pkl"
OUTPUT_PATH = "results/IKNet_k_ablation.csv"

SAVE_DIR = "saved_models/IKNet"
os.makedirs(SAVE_DIR, exist_ok=True)

# Rolling-window settings
TRAIN_YEARS, TEST_YEARS = 3, 1
TIME_STEPS = 10
HORIZONS = [1]
KEYWORD_LIST = [9, 11, 13, 15, 17, 19, 21]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
price_df = pd.read_csv(PRICE_PATH)
token_df = pd.read_csv(TOKEN_PATH)
embedding_cache = joblib.load(EMBEDDING_CACHE_PATH)

price_df["date"] = pd.to_datetime(price_df["date"])
token_df["date"] = pd.to_datetime(token_df["date"])

feature_cols = [col for col in price_df.columns if col != "date"]
windows: List[Tuple[int, int, pd.DataFrame, pd.DataFrame]] = split_by_rolling_window(
    price_df, TRAIN_YEARS, TEST_YEARS
)


# ---------------------------------------------------------
# Embedding cache helper
# ---------------------------------------------------------
def get_cached_embeddings(date_list: pd.Series, top_k: int) -> torch.Tensor:
    """
    Return a stacked tensor of FinBERT embeddings for given dates.

    If a date is missing from the cache, a zero tensor is used as fallback.

    Args:
        date_list (pd.Series): Series of pandas Timestamps.
        top_k (int): Number of top tokens to use per day.

    Returns:
        torch.Tensor: Shape [B, K, 768]
    """
    embs = []
    for date in date_list:
        key = date.strftime("%Y-%m-%d")
        if key in embedding_cache:
            # Use cached embedding, possibly slicing to top_k
            embs.append(torch.tensor(embedding_cache[key][:top_k]))
        else:
            # Fallback zero-padding when cache miss
            embs.append(torch.zeros((top_k, 768)))
    return torch.stack(embs)  # [B, K, 768]


# ---------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------
def main() -> None:
    """
    Run k-ablation experiments with rolling windows and record metrics.
    Trains IKNet for each (num_keywords, horizon, window) combination,
    saves model and scalers, writes per-window predictions and summary metrics.
    """
    # Basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    os.makedirs("results/IKNet_preds", exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["num_keywords", "train_years", "test_year", "horizon", "time_steps", "RMSE", "MAE", "SMAPE", "R2"]
        )

        for num_keywords in KEYWORD_LIST:
            logging.info(f"===== Start experiment: num_keywords={num_keywords} =====")

            for horizon in HORIZONS:
                for train_start, test_start, train_df, test_df in windows:
                    try:
                        # Normalize and frame sequences
                        X_train, y_train, X_test, y_test, scaler_x, scaler_y = normalize_and_sequence(
                            train_df, test_df, feature_cols, TIME_STEPS, horizon
                        )

                        # Align dates with framed sequences
                        date_train = train_df["date"].iloc[
                            TIME_STEPS - 1 : TIME_STEPS - 1 + len(X_train)
                        ].reset_index(drop=True)
                        date_test = test_df["date"].iloc[
                            TIME_STEPS - 1 : TIME_STEPS - 1 + len(X_test)
                        ].reset_index(drop=True)

                        # Get token embeddings per day
                        X_emb_train = get_cached_embeddings(date_train, top_k=num_keywords)
                        X_emb_test = get_cached_embeddings(date_test, top_k=num_keywords)

                        # To tensors
                        X_train = torch.tensor(X_train, dtype=torch.float32)
                        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                        X_test = torch.tensor(X_test, dtype=torch.float32)
                        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

                        # Build and train model
                        model = IKNet(
                            input_size=X_train.shape[2],
                            output_size=1,
                            num_keywords=num_keywords,
                        )
                        model = train_model(model, X_train, X_emb_train, y_train, device=DEVICE)

                        # Save artifacts
                        torch.save(
                            model.state_dict(),
                            f"{SAVE_DIR}/IKNet_{train_start}_{test_start}_k{num_keywords}.pt",
                        )
                        joblib.dump(
                            scaler_x,
                            f"{SAVE_DIR}/scaler_x_{train_start}_{test_start}_k{num_keywords}.pkl",
                        )
                        joblib.dump(
                            scaler_y,
                            f"{SAVE_DIR}/scaler_y_{train_start}_{test_start}_k{num_keywords}.pkl",
                        )

                        # Predict
                        pred = predict_model(model, X_test, X_emb_test, device=DEVICE)

                        # Inverse-scale predictions and targets
                        y_true = scaler_y.inverse_transform(y_test.view(-1, 1).cpu().numpy()).flatten()
                        y_pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()

                        # Save per-date predictions for the test window
                        dates = test_df["date"].iloc[
                            TIME_STEPS + horizon - 1 : TIME_STEPS + horizon - 1 + len(y_true)
                        ].reset_index(drop=True)

                        result_df = pd.DataFrame(
                            {"date": dates, "y_true": y_true, "y_pred": y_pred}
                        )
                        result_df.to_csv(
                            f"results/IKNet_preds/IKNet_{test_start}_k{num_keywords}.csv",
                            index=False,
                        )

                        # Record metrics
                        metrics = compute_metrics(y_true, y_pred)
                        print_metrics(
                            metrics,
                            label=f"[k={num_keywords}] {train_start}-{test_start - 1} -> {test_start}",
                        )

                        writer.writerow(
                            [
                                num_keywords,
                                f"{train_start}-{test_start - 1}",
                                test_start,
                                horizon,
                                TIME_STEPS,
                                round(metrics["RMSE"], 3),
                                round(metrics["MAE"], 3),
                                round(metrics["SMAPE"], 3),
                                round(metrics["R2"], 3),
                            ]
                        )
                    except Exception as e:
                        # Keep a full traceback in logs without interrupting other windows
                        logging.exception(
                            "[ERROR] %s-%s, k=%s failed with error: %s",
                            train_start,
                            test_start,
                            num_keywords,
                            str(e),
                        )


if __name__ == "__main__":
    main()
