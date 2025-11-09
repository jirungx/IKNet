# iknet/shap_analysis/feature_importance_cli.py
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

from . import feature_importance as fi


def main() -> None:
    """
    Thin CLI wrapper around `feature_importance.main()`.

    It:
      1) Parses CLI args for paths and options,
      2) Moves CWD to project root (two levels up from this file),
      3) Patches `fi.load_data` so it always uses the provided CSV paths,
      4) Patches `fi.parse_args` to feed the parsed args directly,
      5) Calls `fi.main()`.
    """
    ap = argparse.ArgumentParser(description="IKNet global feature importance (SHAP) CLI")
    ap.add_argument(
        "--run-dir",
        default="outputs/train_run_001",
        help="Root directory containing models/scalers/results (e.g., outputs/train_run_001)",
    )
    ap.add_argument(
        "--price-csv",
        default="dataset/price_features.csv",
        help="CSV path for price/technical features (must include 'date')",
    )
    ap.add_argument(
        "--tokens-csv",
        default="tokens/topk25_tokens.csv",
        help="CSV path for keywords (must contain 'filtered_keywords' or 'tokens')",
    )
    ap.add_argument(
        "--save-subdir",
        default="figs",
        help="Subdirectory under RUN_DIR to save outputs (default: figs)",
    )
    ap.add_argument(
        "--include-close",
        action="store_true",
        help="If set, keep the 'close' feature (excluded by default)",
    )
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--end-year", type=int, default=2024)
    args = ap.parse_args()

    # Move to project root (two levels up from this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(project_root)

    # Patch load_data to always use current CLI-provided CSV paths
    orig_load_data = fi.load_data

    def patched_load_data(price_path=None, token_path=None):
        return orig_load_data(price_path=args.price_csv, token_path=args.tokens_csv)

    fi.load_data = patched_load_data

    # Patch parse_args to return our current CLI args directly
    def patched_parse_args():
        return SimpleNamespace(
            run_dir=args.run_dir,
            price_csv=args.price_csv,
            tokens_csv=args.tokens_csv,
            save_subdir=args.save_subdir,
            include_close=args.include_close,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    fi.parse_args = patched_parse_args

    # Delegate to the underlying implementation
    fi.main()


if __name__ == "__main__":
    main()
