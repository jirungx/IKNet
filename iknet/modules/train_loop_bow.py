import logging
import numpy as np
import pandas as pd
import torch
from modules.train import train_model
from modules.predict import predict_model
from modules.metrics_utils import compute_metrics, print_metrics
from modules.rolling_utils import normalize_and_sequence

logger = logging.getLogger(__name__)


def run_bow_model_pipeline(windows, token_df, feature_cols, time_steps, horizon,
                           device, model_class, model_kwargs, writer):
    """
    Train/evaluate a model using price sequences + BoW tokens over rolling windows.

    Args:
        windows: iterable of (train_start, test_start, train_df, test_df)
        token_df (pd.DataFrame): must contain columns ['date', 'tokens']
        feature_cols (list[str]): price/technical feature columns
        time_steps (int): sequence window length
        horizon (int): forecast horizon
        device (torch.device): training device
        model_class: model class to instantiate
        model_kwargs (dict): kwargs passed to model_class
        writer (csv.writer): CSV writer to record metrics
    """
    token_df["date"] = pd.to_datetime(token_df["date"])

    for train_start, test_start, train_df, test_df in windows:
        try:
            # Sequence building & normalization
            X_train, y_train, X_test, y_test, scaler_y = normalize_and_sequence(
                train_df, test_df, feature_cols, time_steps, horizon
            )

            def make_bow_tensor(df, topk=3, vocab=None, num_samples=None):
                """
                Build a BoW tensor aligned to sequence targets.
                - One-hot by presence of top-k tokens for the day.
                """
                bow_vecs = []
                dates = df["date"].iloc[time_steps + horizon - 1:].reset_index(drop=True)
                if num_samples:
                    # Align length to the number of sliding sequences
                    dates = dates[:num_samples]

                for date in dates:
                    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
                    row = token_df[token_df["date"] == date_str]
                    bow = np.zeros(len(vocab))
                    if len(row) > 0:
                        tokens = row.iloc[0]["tokens"].split(",")
                        for token in tokens[:topk]:
                            token = token.strip()
                            if token in vocab:
                                bow[vocab[token]] = 1
                    bow_vecs.append(bow)
                return torch.tensor(np.array(bow_vecs), dtype=torch.float32)

            # Build vocabulary from the tokens column
            vocab_set = set()
            for row in token_df["tokens"]:
                vocab_set.update(t.strip() for t in row.split(","))
            vocab = {word: idx for idx, word in enumerate(sorted(vocab_set))}

            # BoW tensors (train/test)
            X_bow_train = make_bow_tensor(train_df, vocab=vocab, num_samples=len(X_train))
            X_bow_test = make_bow_tensor(test_df, vocab=vocab, num_samples=len(X_test))

            # Convert arrays to tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)

            # Train & predict
            model = model_class(input_size=X_train.shape[2], bow_dim=len(vocab), **model_kwargs)
            model = train_model(model, X_train, X_bow_train, y_train, device=device)
            pred = predict_model(model, X_test, X_bow_test, device=device)

            # Inverse scale and compute metrics
            y_true = scaler_y.inverse_transform(y_test.view(-1, 1)).flatten()
            y_pred = scaler_y.inverse_transform(pred).flatten()

            metrics = compute_metrics(y_true, y_pred)
            print_metrics(metrics, label=f"h={horizon}, ts={time_steps}")

            # Write one row per window
            writer.writerow([
                f"{train_start}-{test_start - 1}", test_start, horizon, time_steps,
                round(metrics["RMSE"], 3),
                round(metrics["MAE"], 3),
                round(metrics["SMAPE"], 3),
                round(metrics["R2"], 3)
            ])

        except Exception as e:
            # Log the full traceback and continue with the next window
            logger.exception("Pipeline error at window %s-%s: %s", train_start, test_start, str(e))
