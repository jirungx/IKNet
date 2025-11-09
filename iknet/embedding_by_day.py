import os
import joblib
import pandas as pd
import torch
from tqdm import tqdm
from .modules.embedding_utils import FinBERTEmbedder
from .config import DEVICE


# ---------------------------------------------------------
# 1. Load token list (top-k keywords per date)
# ---------------------------------------------------------
TOKEN_PATH = "tokens/topk25_tokens.csv"
token_df = pd.read_csv(TOKEN_PATH)
token_df["date"] = pd.to_datetime(token_df["date"])

# ---------------------------------------------------------
# 2. Initialize FinBERT embedder
# ---------------------------------------------------------
embedder = FinBERTEmbedder(device=DEVICE)
TOP_K = 25

embedding_dict = {}

# ---------------------------------------------------------
# 3. Generate embeddings for each date
# ---------------------------------------------------------
for _, row in tqdm(token_df.iterrows(), total=len(token_df), desc="Embedding tokens"):
    date_str = row["date"].strftime("%Y-%m-%d")
    tokens_str = row["tokens"] if pd.notna(row["tokens"]) else ""
    tokens = [tok.strip() for tok in tokens_str.split(",") if tok.strip()]
    if not tokens:
        # Skip empty token lists
        continue

    # Compute FinBERT embeddings for given tokens
    emb_tensor = embedder.get_keyword_embedding_tensor([tokens], top_k=TOP_K)  # [1, K, 768]
    embedding_dict[date_str] = emb_tensor.squeeze(0).cpu().numpy()  # [K, 768]

# ---------------------------------------------------------
# 4. Save precomputed embeddings
# ---------------------------------------------------------
os.makedirs("precomputed_embeddings", exist_ok=True)
output_path = "precomputed_embeddings/finbert_embeddings_k25.pkl"
joblib.dump(embedding_dict, output_path)

print(f"Precomputed embeddings saved to: {output_path}")
