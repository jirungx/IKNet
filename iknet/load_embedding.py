import os
import joblib
import torch
import pandas as pd
from tqdm import tqdm
from modules.embedding_utils import FinBERTEmbedder
from config import DEVICE


# =========================================================
# 1. Load precomputed FinBERT embeddings from cache
# =========================================================
embedding_cache = joblib.load("precomputed_embeddings/finbert_embeddings_k25.pkl")


def get_cached_embeddings(date_list: list, top_k: int = 13) -> torch.Tensor:
    """
    Retrieve cached FinBERT embeddings for a list of dates.

    Args:
        date_list (list[datetime.date]): List of dates to retrieve embeddings for.
        top_k (int): Number of top tokens (K) per day. Defaults to 13.

    Returns:
        torch.Tensor: A tensor of shape [B, K, 768], where
            - B: number of dates
            - K: top-k tokens
            - 768: FinBERT embedding dimension
    """
    embs = []
    for date in date_list:
        key = date.strftime("%Y-%m-%d")
        if key in embedding_cache:
            # Retrieve embedding from cache
            embs.append(torch.tensor(embedding_cache[key]))
        else:
            # Use zero-padding if embedding not found
            embs.append(torch.zeros((top_k, 768)))
    return torch.stack(embs)
