# run_shap_by_year_back_to_utils.py (기술 지표 + 키워드 SHAP 통합 버전, Close 포함 + SHAP shape 처리 추가)

import os
import torch
import pandas as pd
import numpy as np
import shap
import sys
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModel, AutoTokenizer
from ..modules.model import IKNet
from .feature_importance_all_absol import extract_top_n_feature_shap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

embedding_dim = 768
num_keywords = 21
time_steps = 10
hidden_size = 256
horizons = [1]
top_n = 15
N_SAMPLES_KERNEL = 350
N_BACKGROUND_SAMPLES = 50
SCALER_SAVE_DIR = "saved_models"

def load_data(price_path=os.environ.get("IKNET_PRICE_CSV","dataset/snp500_dataset.csv"), token_path=os.environ.get("IKNET_TOKENS_CSV","tokens/snp_topk25_tokens.csv")):
    if not os.path.exists(price_path) or not os.path.exists(token_path):
        raise FileNotFoundError(f"파일 없음: {price_path}, {token_path}")
    price_df = pd.read_csv(price_path); token_df = pd.read_csv(token_path)
    price_df["date"] = pd.to_datetime(price_df["date"])
    token_df["date"] = pd.to_datetime(token_df["date"])
    if 'filtered_keywords' not in token_df.columns:
        if 'tokens' in token_df.columns:
            token_df.rename(columns={'tokens': 'filtered_keywords'}, inplace=True)
        else:
            sys.exit("filtered_keywords 또는 tokens 컬럼이 필요합니다.")
    df = pd.merge(price_df, token_df, on="date").dropna(subset=['filtered_keywords']).reset_index(drop=True)
    return df, price_df

def setup_embedding(model_name="yiyanghkust/finbert-tone"):
    finbert_model = AutoModel.from_pretrained(model_name)
    finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        embedding_matrix = finbert_model.embeddings.word_embeddings.weight.detach().cpu()
    word2idx = finbert_tokenizer.get_vocab()
    unk_idx = word2idx.get("[UNK]", 0)
    pad_idx = word2idx.get("[PAD]", 0)
    return embedding_matrix, word2idx, unk_idx, pad_idx

def keywords_to_embeddings(keywords, embedding_matrix, word2idx, unk_idx, pad_idx, num_keywords, embedding_dim):
    if isinstance(keywords, str):
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    elif isinstance(keywords, list):
        keyword_list = [str(kw).strip() for kw in keywords if str(kw).strip()]
    else:
        keyword_list = []
    indices = [word2idx.get(w, unk_idx) for w in keyword_list[:num_keywords]]
    while len(indices) < num_keywords:
        indices.append(pad_idx)
    embeddings = embedding_matrix[indices]
    if embeddings.shape != (num_keywords, embedding_dim):
        return torch.zeros((num_keywords, embedding_dim), dtype=embedding_matrix.dtype)
    return embeddings

def create_price_tensor(price_data_np, time_steps):
    if len(price_data_np) < time_steps: return None
    price_tensor = torch.tensor(price_data_np, dtype=torch.float32).unfold(0, time_steps, 1)
    return price_tensor.permute(0, 2, 1)

def create_keyword_tensor(df_part, time_steps, embedding_matrix, word2idx, unk_idx, pad_idx, num_keywords, embedding_dim):
    token_data = df_part["filtered_keywords"].tolist()
    if len(token_data) < time_steps: return None
    token_seq = token_data[time_steps - 1 : ]
    embeddings = [keywords_to_embeddings(kws, embedding_matrix, word2idx, unk_idx, pad_idx, num_keywords, embedding_dim) for kws in token_seq]
    embeddings = [emb for emb in embeddings if emb.shape == (num_keywords, embedding_dim)]
    if not embeddings: return None
    return torch.stack(embeddings)

def main():

        df, price_df = load_data()
        feature_cols = [col for col in price_df.columns if col != "date"] 
        technical_names = feature_cols.copy()
        price_feature_count = len(feature_cols)
        embedding_matrix, word2idx, UNK_IDX, PAD_IDX = setup_embedding()
        selected_years = [2021]
        # range(2018, 2025)
        for horizon in horizons:
            for test_year in range(2018, 2025):
                train_start = test_year - 3
                model_path = f"saved_models/IKNet_{train_start}_{test_year}_k{num_keywords}.pt"
                scaler_path = os.path.join(SCALER_SAVE_DIR, f"scaler_x_{train_start}_{test_year}_k{num_keywords}.pkl")

                if not os.path.exists(model_path) or not os.path.exists(scaler_path): continue

                scaler_x = joblib.load(scaler_path)
                test_df = df[df["date"].dt.year == test_year].copy()
                if len(test_df) < time_steps: continue

                test_features_np = scaler_x.transform(test_df[feature_cols].values)
                x_price_tensor = create_price_tensor(test_features_np, time_steps)
                x_emb_tensor = create_keyword_tensor(test_df, time_steps, embedding_matrix, word2idx, UNK_IDX, PAD_IDX, num_keywords, embedding_dim)

                if x_price_tensor is None or x_emb_tensor is None: continue

                num_samples = min(len(x_price_tensor), len(x_emb_tensor))
                x_price_tensor = x_price_tensor[:num_samples]
                x_emb_tensor = x_emb_tensor[:num_samples]

                tokens_df = test_df.iloc[time_steps - 1 : time_steps - 1 + num_samples]
                tokens_for_shap = tokens_df["filtered_keywords"].apply(lambda x: [t.strip() for t in str(x).split(',') if t.strip()]).tolist()

                x_price_np = x_price_tensor.numpy().reshape(num_samples, -1)
                x_emb_np = x_emb_tensor.numpy().reshape(num_samples, -1)
                x_combined_np = np.concatenate([x_price_np, x_emb_np], axis=1)

                background_data = x_combined_np[np.random.choice(num_samples, min(N_BACKGROUND_SAMPLES, num_samples), replace=False)]

                def wrapped_model(x):
                    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
                    x_price = x[:, :time_steps * price_feature_count].reshape(-1, time_steps, price_feature_count)
                    x_emb = x[:, time_steps * price_feature_count:].reshape(-1, num_keywords, embedding_dim)
                    with torch.no_grad():
                        return model(x_price, x_emb).cpu().numpy()

                model = IKNet(
                    input_size=price_feature_count, output_size=1, num_keywords=num_keywords,
                    embedding_dim=embedding_dim, hidden_size=hidden_size
                ).to(DEVICE)
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval()

                explainer = shap.KernelExplainer(wrapped_model, background_data)
                shap_values = explainer.shap_values(x_combined_np, nsamples=N_SAMPLES_KERNEL)

                shap_values_np = shap_values[0] if isinstance(shap_values, list) else shap_values
                if shap_values_np.ndim == 3 and shap_values_np.shape[2] == 1:
                    shap_values_np = np.squeeze(shap_values_np, axis=2)
                elif shap_values_np.ndim != 2:
                    raise ValueError(f"Unexpected SHAP array shape: {shap_values_np.shape}")

                shap_values_np = np.nan_to_num(shap_values_np)

                save_path = f"Feature_Importance_Final/Feature_Importance_{test_year}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                _ = extract_top_n_feature_shap(
                    shap_values_array=shap_values_np,
                    tokens_data=tokens_for_shap,
                    technical_names=technical_names,
                    test_year=test_year,
                    top_n=top_n,
                    save_path=save_path,
                    word2idx=word2idx
                )

        print("SHAP 분석 완료.")

if __name__ == "__main__":
    main()