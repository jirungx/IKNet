
import argparse, os, csv, json
import pandas as pd
import torch
import numpy as np
from .modules.rolling_utils import split_by_rolling_window, normalize_and_sequence
from .modules.embedding_utils import FinBERTEmbedder
from .modules.model import IKNet
from .modules.train import train_model
from .modules.predict import predict_model
from .modules.metrics_utils import compute_metrics, print_metrics
from . import core_main as core  # for helper funcs defined there if any
from .config import DEVICE

def parse_args():
    ap = argparse.ArgumentParser(description="IKNet training CLI (rolling windows).")
    ap.add_argument("--price-csv", required=True, help="CSV with columns: date, features incl. close")
    ap.add_argument("--feature-cols", required=True, help="Comma-separated feature columns (include close)")
    ap.add_argument("--time-steps", type=int, default=30)
    ap.add_argument("--horizon", type=int, nargs="+", default=[1], help="One or more forecast horizons")
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-years", type=int, default=1)
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--use-keywords", action="store_true", help="Use precomputed FinBERT keyword embeddings")
    ap.add_argument("--embedding-pkl", default="precomputed_embeddings/finbert_embeddings_k25.pkl", help="joblib pkl of {date: [K,768]} dict")
    ap.add_argument("--topk", type=int, default=13, help="number of keywords per day")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-size", type=int, default=384)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="outputs", help="Where to save results")
    return ap.parse_args()

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_cached_embeddings(date_series, embedding_cache, top_k):
    embs = []
    for date in date_series:
        key = pd.to_datetime(date).strftime("%Y-%m-%d")
        if key in embedding_cache:
            arr = embedding_cache[key]
            if arr.shape[0] >= top_k:
                embs.append(torch.tensor(arr[:top_k]))
            else:
                pad = np.zeros((top_k, arr.shape[1]), dtype=arr.dtype)
                pad[:arr.shape[0]] = arr
                embs.append(torch.tensor(pad))
        else:
            embs.append(torch.zeros((top_k, 768)))
    return torch.stack(embs)

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    price_df = pd.read_csv(args.price_csv)
    price_df["date"] = pd.to_datetime(price_df["date"])
    feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    windows = split_by_rolling_window(price_df, args.train_years, args.test_years, args.start_year, args.end_year)

    if args.use_keywords:
        import joblib
        embedding_cache = joblib.load(args.embedding_pkl)
    else:
        embedding_cache = None

    result_csv = os.path.join(args.outdir, "results.csv")
    with open(result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_span","test_year","horizon","time_steps","RMSE","MAE","SMAPE","R2"])

        for h in args.horizon:
            for (train_start, test_start, train_df, test_df) in windows:
                try:
                    X_train, y_train, X_test, y_test, scaler_x, scaler_y = normalize_and_sequence(
                        train_df.copy(), test_df.copy(), feature_cols, args.time_steps, h
                    )
                    # Align dates with sequences
                    date_train = train_df["date"].iloc[args.time_steps - 1 : args.time_steps - 1 + len(X_train)].reset_index(drop=True)
                    date_test = test_df["date"].iloc[args.time_steps - 1 : args.time_steps - 1 + len(X_test)].reset_index(drop=True)

                    # Build embeddings if required
                    if args.use_keywords:
                        X_emb_train = get_cached_embeddings(date_train, embedding_cache, args.topk)
                        X_emb_test = get_cached_embeddings(date_test, embedding_cache, args.topk)
                    else:
                        # If not using keywords, create zero embeddings matching topk
                        X_emb_train = torch.zeros((len(X_train), args.topk, 768))
                        X_emb_test = torch.zeros((len(X_test), args.topk, 768))

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
                        embedding_dim=768,
                        hidden_size=args.hidden_size,
                        num_layers=args.num_layers,
                        output_size=1,
                        dropout=args.dropout
                    )

                    model = train_model(
                        model,
                        X_train_t, X_emb_train,
                        y_train_t,
                        device=str(DEVICE),
                        epochs=args.epochs,
                        lr=args.lr
                    )

                    y_pred = predict_model(model, X_test_t, X_emb_test, device=str(DEVICE))
                    # Inverse scale predictions and targets
                    y_true_inv = scaler_y.inverse_transform(y_test_t.numpy().reshape(-1,1)).reshape(-1)
                    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1,1)).reshape(-1)

                    metrics = compute_metrics(y_true_inv, y_pred_inv)
                    print_metrics(metrics, label=f"{train_start}-{test_start-1} -> {test_start} h={h}")

                    writer.writerow([
                        f"{train_start}-{test_start-1}", test_start, h, args.time_steps,
                        round(metrics["RMSE"],3), round(metrics["MAE"],3),
                        round(metrics["SMAPE"],3), round(metrics["R2"],3)
                    ])

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[ERROR] Window {train_start}-{test_start-1} -> {test_start}, h={h}: {e}")

if __name__ == "__main__":
    main()
