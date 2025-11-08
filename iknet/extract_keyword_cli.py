import argparse, os, pandas as pd
from .extract_keyword_from_news import process_news_top_tokens_only

def main():
    ap = argparse.ArgumentParser(description="Extract top-K salient tokens per date from news")
    ap.add_argument("--news-csv", required=True, help="CSV with columns: date,text")
    ap.add_argument("--out-csv", default="tokens/snp_topk25_tokens.csv")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df = pd.read_csv(args.news_csv)
    process_news_top_tokens_only(df, top_k=args.topk, save_token_path=args.out_csv)

if __name__ == "__main__":
    main()
