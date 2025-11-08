import pandas as pd
import torch
import os
from tqdm import tqdm
from modules.embedding_utils import FinBERTEmbedder
from config import DEVICE
import joblib

# ❶ 캐시 임베딩 불러오기
embedding_cache = joblib.load("precomputed_embeddings/finbert_embeddings_k25.pkl")

# ❷ 새 함수: date 리스트를 받아서 임베딩 텐서 반환
def get_cached_embeddings(date_list, top_k=13):
    embs = []
    for date in date_list:
        key = date.strftime("%Y-%m-%d")
        if key in embedding_cache:
            embs.append(torch.tensor(embedding_cache[key]))  # [K, 768]
        else:
            embs.append(torch.zeros((top_k, 768)))  # fallback: zero padding
    return torch.stack(embs)  # [B, K, 768]
