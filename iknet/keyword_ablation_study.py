import torch
import pandas as pd
import csv
import numpy as np
from .modules.rolling_utils import split_by_rolling_window, normalize_and_sequence
from .modules.model import IKNet
from .modules.train import train_model
from .modules.predict import predict_model
from .modules.metrics_utils import compute_metrics, print_metrics
from .config import DEVICE
import random
import os
import joblib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(11)

# 파일 경로
price_path = "dataset/snp500_dataset.csv"
token_path = "tokens/snp_topk25_tokens.csv"
embedding_cache_path = "precomputed_embeddings/finbert_embeddings_k25.pkl"
output_path = "results/IKNet_k_ablation.csv"
SAVE_DIR = "saved_models/IKNet"
os.makedirs(SAVE_DIR, exist_ok=True)

# 실험 설정
train_years, test_years = 3, 1
time_steps = 10
horizons = [1]
keyword_list = [9, 11, 13, 15, 17, 19, 21]

# 데이터 로딩
price_df = pd.read_csv(price_path)
token_df = pd.read_csv(token_path)
embedding_cache = joblib.load(embedding_cache_path)
price_df["date"] = pd.to_datetime(price_df["date"])
token_df["date"] = pd.to_datetime(token_df["date"])

feature_cols = [col for col in price_df.columns if col != "date"]
windows = split_by_rolling_window(price_df, train_years, test_years)

# 임베딩 캐시 함수
def get_cached_embeddings(date_list, top_k):
    embs = []
    for date in date_list:
        key = date.strftime("%Y-%m-%d")
        if key in embedding_cache:
            embs.append(torch.tensor(embedding_cache[key][:top_k]))
        else:
            embs.append(torch.zeros((top_k, 768)))
    return torch.stack(embs)  # [B, K, 768]


def main():
    # 실험 루프
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["num_keywords", "train_years", "test_year", "horizon", "time_steps", "RMSE", "MAE", "SMAPE", "R2"])

        for num_keywords in keyword_list:
            print(f"\n===== 실험 시작: num_keywords = {num_keywords} =====")

            for horizon in horizons:
                for train_start, test_start, train_df, test_df in windows:
                    try:
                        # 시계열 입력 처리
                        X_train, y_train, X_test, y_test, scaler_x, scaler_y = normalize_and_sequence(
                            train_df, test_df, feature_cols, time_steps, horizon
                        )

                        date_train = train_df["date"].iloc[time_steps - 1 : time_steps - 1 + len(X_train)].reset_index(drop=True)
                        date_test = test_df["date"].iloc[time_steps - 1 : time_steps - 1 + len(X_test)].reset_index(drop=True)

                        X_emb_train = get_cached_embeddings(date_train, top_k=num_keywords)
                        X_emb_test = get_cached_embeddings(date_test, top_k=num_keywords)

                        # 텐서로 변환
                        X_train = torch.tensor(X_train, dtype=torch.float32)
                        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                        X_test = torch.tensor(X_test, dtype=torch.float32)
                        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

                        # 모델 학습
                        model = IKNet(input_size=X_train.shape[2], output_size=1, num_keywords=num_keywords)
                        model = train_model(model, X_train, X_emb_train, y_train, device=DEVICE)

                        # 저장
                        torch.save(model.state_dict(), f"{SAVE_DIR}/IKNet_{train_start}_{test_start}_k{num_keywords}.pt")
                        joblib.dump(scaler_x, f"{SAVE_DIR}/scaler_x_{train_start}_{test_start}_k{num_keywords}.pkl")
                        joblib.dump(scaler_y, f"{SAVE_DIR}/scaler_y_{train_start}_{test_start}_k{num_keywords}.pkl")

                        # 예측
                        pred = predict_model(model, X_test, X_emb_test, device=DEVICE)
                        y_true = scaler_y.inverse_transform(y_test.view(-1, 1).cpu().numpy()).flatten()
                        y_pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()

                        # 날짜별 저장
                        dates = test_df["date"].iloc[time_steps + horizon - 1 : time_steps + horizon - 1 + len(y_true)].reset_index(drop=True)
                        result_df = pd.DataFrame({"date": dates, "y_true": y_true, "y_pred": y_pred})
                        os.makedirs("results/IKNet_preds", exist_ok=True)
                        result_df.to_csv(f"results/IKNet_preds/IKNet_{test_start}_k{num_keywords}.csv", index=False)

                        # 성능 기록
                        metrics = compute_metrics(y_true, y_pred)
                        print_metrics(metrics, label=f"[k={num_keywords}] {train_start}-{test_start - 1} → {test_start}")
                        writer.writerow([
                            num_keywords,
                            f"{train_start}-{test_start - 1}", test_start, horizon, time_steps,
                            round(metrics["RMSE"], 3),
                            round(metrics["MAE"], 3),
                            round(metrics["SMAPE"], 3),
                            round(metrics["R2"], 3)
                        ])
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[ERROR] {train_start}-{test_start}, k={num_keywords}: {e}")

if __name__ == '__main__':
    main()