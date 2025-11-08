
import argparse, os, pandas as pd, joblib
from tqdm import tqdm
from .modules.embedding_utils import FinBERTEmbedder
from .config import DEVICE

def main():
    ap = argparse.ArgumentParser(description="Precompute FinBERT keyword embeddings per date")
    ap.add_argument("--tokens-csv", required=True, help="CSV with columns: date,tokens (comma-separated)")
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--out-pkl", default="precomputed_embeddings/finbert_embeddings_k25.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.tokens_csv)
    df["date"] = pd.to_datetime(df["date"])
    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)

    embedder = FinBERTEmbedder(device=DEVICE)
    cache = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        date = row["date"].strftime("%Y-%m-%d")
        tokens_str = row.get("tokens", "")
        tokens = [] if pd.isna(tokens_str) or not tokens_str else [t.strip() for t in str(tokens_str).split(",") if t.strip()]
        emb = embedder.get_keyword_embedding_tensor([tokens], top_k=args.topk).squeeze(0).cpu().numpy()
        cache[date] = emb

    joblib.dump(cache, args.out_pkl)
    print(f"Saved embeddings to {args.out_pkl}")

if __name__ == "__main__":
    main()
