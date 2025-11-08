# iknet/shap_analysis/feature_importance_cli.py
import argparse, os
from . import feature_importance as fi

def main():
    ap = argparse.ArgumentParser(description="IKNet global feature importance (SHAP)")
    ap.add_argument("--price-csv", default="dataset/snp500_dataset.csv")
    ap.add_argument("--tokens-csv", default="tokens/snp_topk25_tokens.csv")
    ap.add_argument("--embedding-pkl", default="precomputed_embeddings/finbert_embeddings_k25.pkl")
    ap.add_argument("--outdir", default="outputs_shap/fi")
    args = ap.parse_args()

    # 프로젝트 루트로 이동 (이 파일 기준 상위 2단계)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(project_root)

    # load_data를 네 인자로 고정하는 래퍼로 교체
    orig_load_data = fi.load_data
    def patched_load_data(price_path=None, token_path=None):
        # feature_importance.py가 인자를 없이 부를 때도 네 인자를 강제 사용
        return orig_load_data(price_path=args.price_csv, token_path=args.tokens_csv)
    fi.load_data = patched_load_data

    # 필요하면 outdir도 ENV로 통일 (feature_importance.py가 기본 outdir을 쓴다면 이 값이 동일해서 문제 없음)
    os.environ["IKNET_OUTDIR"] = args.outdir
    os.environ["IKNET_EMBED_PKL"] = args.embedding_pkl

    # 기존 메인 실행
    fi.main()

if __name__ == "__main__":
    main()
