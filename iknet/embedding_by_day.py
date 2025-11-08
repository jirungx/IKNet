import pandas as pd
import torch
import os
from tqdm import tqdm
from .modules.embedding_utils import FinBERTEmbedder
from .config import DEVICE
import joblib

# 데이터 경로
token_path = "tokens/snp_topk25_tokens.csv"
token_df = pd.read_csv(token_path)
token_df["date"] = pd.to_datetime(token_df["date"])

# FinBERT 모델 초기화
embedder = FinBERTEmbedder(device=DEVICE)
top_k = 25

embedding_dict = {}

for _, row in tqdm(token_df.iterrows(), total=len(token_df)):
    date = row["date"].strftime("%Y-%m-%d")
    tokens_str = row["tokens"] if pd.notna(row["tokens"]) else ""
    tokens = tokens_str.split(",") if tokens_str else []
    emb_tensor = embedder.get_keyword_embedding_tensor([tokens], top_k=top_k)  # shape: [1, K, 768]
    embedding_dict[date] = emb_tensor.squeeze(0).cpu().numpy()  # [K, 768]

# 저장
os.makedirs("precomputed_embeddings", exist_ok=True)
joblib.dump(embedding_dict, "precomputed_embeddings/finbert_embeddings_k25.pkl")
print("임베딩 저장 완료")
