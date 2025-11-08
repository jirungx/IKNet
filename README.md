# IKNet (CLI-ready)

Terminal-friendly packaging of your existing IKNet implementation. This preserves your modules and adds lightweight CLIs.

## Install (editable)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 1) Precompute keyword embeddings (optional)
Input CSV format:
```
date,tokens
2018-01-02,boost,surge,recovery
2018-01-03,fall,crash,drop
```
Run:
```bash
iknet-embed --tokens-csv tokens/snp_topk25_tokens.csv --topk 25 --out-pkl precomputed_embeddings/finbert_embeddings_k25.pkl
```

## 2) Train with rolling windows
```bash
iknet-train \
  --price-csv data/price_features.csv \
  --feature-cols "open,high,low,close,volume,sma_20,ema_20,rsi_14,macd,signal_line,bollinger_upper,bollinger_lower,volatility_ratio,volume_change,macd_diff,bollinger_width,close_vs_sma" \
  --time-steps 30 \
  --horizon 1 5 10 \
  --train-years 3 --test-years 1 \
  --start-year 2015 --end-year 2024 \
  --use-keywords \
  --embedding-pkl precomputed_embeddings/finbert_embeddings_k25.pkl \
  --topk 13 \
  --epochs 200 --lr 1e-3 \
  --outdir outputs_cli
```

### Output
`outputs_cli/results.csv` with RMSE/MAE/SMAPE/R2 per window and horizon.

## Project layout
```
IKNet-CLI/
  pyproject.toml
  requirements.txt
  src/iknet/
    __init__.py
    core_main.py        # your original main (kept for reference)
    modules/            # your original implementation
    train_cli.py        # new: terminal entry
    embed_cli.py        # new: embeddings precompute
```

> Note: The CLI expects a `date` column in `--price-csv`, and the features listed by `--feature-cols` must include `close`. For keyword mode, supply a joblib pkl mapping YYYY-MM-DD -> [K, 768] arrays (use `iknet-embed` to build it).
