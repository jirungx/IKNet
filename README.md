# Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators

<p align="center">
  <b>Jinwoong Kim<sup>1</sup> · Sangjin Park<sup>1*</sup></b><br>
  <sup>1</sup>Graduate School of Industrial Data Engineering, Hanyang University, Seoul, Republic of Korea
</p>

****  
[![Paper](https://img.shields.io/badge/Paper-ACM_Digital_Library-red?style=for-the-badge)](https://dl.acm.org/doi/10.1145/3768292.3770343)
[![Conference](https://img.shields.io/badge/Conference-ICAIF_2025-blue?style=for-the-badge)](https://icaif25.org/)
[![Publisher](https://img.shields.io/badge/Publisher-ACM-black?style=for-the-badge)](https://www.acm.org/)

<img width="4944" height="3738" alt="Figure1_Architecture" src="https://github.com/user-attachments/assets/be8e54ee-98f1-456b-865a-69ed89480bf5" />
<p align="center"><em>Figure 1. Overall architecture of IKNet integrating keyword sentiment and technical indicators.</em></p>


## Abstract

The increasing influence of unstructured external information, such as news articles, on stock prices has attracted growing attention in financial markets. Despite recent advances, most existing news-based forecasting models represent all articles using sentiment scores or average embeddings that capture the general tone but fail to provide quantitative, context-aware explanations of the impacts of public sentiment on predictions. To address this limitation, we propose an interpretable keyword-guided network (IKNet), which is an explainable forecasting framework that models the semantic association between individual news keywords and stock price movements. The IKNet identifies salient keywords via FinBERT-based contextual analysis, processes each embedding through a separate nonlinear projection layer, and integrates their representations with the time-series data of technical indicators to forecast next-day closing prices. By applying Shapley Additive Explanations, the model generates quantifiable and interpretable attributions for the contribution of each keyword to predictions. Empirical evaluations of S&P 500 data from 2015 to 2024 demonstrate that IKNet outperforms baselines, including recurrent neural networks and transformer models, reducing RMSE by up to 32.9% and improving cumulative returns by 18.5%. Moreover, IKNet enhances transparency by offering contextualized explanations of volatility events driven by public sentiment.

<br>

## Motivation

- Financial news sentiment increasingly drives markets, especially during political or economic events.
  
- Traditional models rely on numbers, overlooking emotional and public opinion effects.

- Prior sentiment models reduce news to sentiment scores, missing word-level nuances.

- We propose IKNet, an interpretable keyword-guided model integrating FinBERT sentiment with technical indicators for next-day price prediction.

<br>

## Contribution

- IKNet integrates keyword sentiment with technical indicators to enhance stock prediction accuracy, interpretability, and profitability.

- Our keyword-level sentiment representation enables explainability of how specific terms affect market trends.

- By visualizing semantic causality between opinion keywords and future stock prices, it enables efficient investment strategies.

<br>

## Build & Run Guide

### 1. Environment Setup
```bash
# Create and activate environment
conda create -n iknet python=3.10 -y
conda activate iknet

# Install required packages
pip install -r requirements.txt

# Install IKNet package (for CLI commands)
pip install -e .
```

---


### 2. Project Structure
```
IKNet/
├── iknet/
│   ├── modules/                  # Core model, training, utils
│   ├── shap_analysis/            # SHAP explainability modules
│   └── ...
├── dataset/                      # (Empty) Store technical indicators CSVs
│   └── .gitkeep
├── tokens/                       # (Empty) Store extracted news tokens
│   └── .gitkeep
├── precomputed_embeddings/       # (Empty) Store cached FinBERT embeddings
│   └── .gitkeep
├── outputs/                      # Training outputs (auto-generated)
├── outputs_shap/                 # SHAP analysis results (auto-generated)
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

### 3. Commands Overview

#### 🔹 3.1 Extract News Keywords
```bash
iknet-extract-keywords \
  --news-csv dataset/snp500_news.csv \
  --out-csv tokens/snp_topk25_tokens.csv \
  --topk 25
```

#### 🔹 3.2 Generate Keyword Embeddings (FinBERT)
```bash
iknet-embed \
  --tokens-csv tokens/snp_topk25_tokens.csv \
  --out-pkl precomputed_embeddings/finbert_embeddings_k25.pkl
```

#### 🔹 3.3 Train Model (IKNet)
```bash
iknet-train \
  --price-csv dataset/snp500_dataset.csv \
  --feature-cols "open,high,low,close,volume,sma_20,ema_20,rsi_14,macd,signal_line,bollinger_upper,bollinger_lower,volatility_ratio,volume_change,macd_diff,bollinger_width,close_vs_sma" \
  --time-steps 30 \
  --horizon 1 \
  --train-years 3 --test-years 1 \
  --start-year 2015 --end-year 2024 \
  --use-keywords \
  --embedding-pkl precomputed_embeddings/finbert_embeddings_k25.pkl \
  --topk 25 \
  --outdir outputs/train_run_001
```

#### 🔹 3.4 Daily SHAP Analysis
```bash
iknet-shap-daily \
  --price-csv dataset/snp500_dataset.csv \
  --tokens-csv tokens/snp_topk25_tokens.csv \
  --embedding-pkl precomputed_embeddings/finbert_embeddings_k25.pkl \
  --outdir outputs_shap/daily
```

#### 🔹 3.5 Global Feature Importance
```bash
iknet-shap-fi \
  --price-csv dataset/snp500_dataset.csv \
  --tokens-csv tokens/snp_topk25_tokens.csv \
  --embedding-pkl precomputed_embeddings/finbert_embeddings_k25.pkl \
  --outdir outputs_shap/fi
```

---


### 4. Data Folders (empty by default)
| Folder | Purpose | Notes |
|---------|----------|-------|
| `dataset/` | CSVs with price & technical indicators | not tracked in Git |
| `tokens/` | Extracted top-k keywords per day | not tracked in Git |
| `precomputed_embeddings/` | Cached FinBERT embedding `.pkl` | not tracked in Git |

> Each folder contains a `.gitkeep` file to preserve the structure in GitHub.

---

<br>

## Citation
```bibtex
@inproceedings{kim2025iknet,
  title={Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators},
  author={Jinwoong Kim and Sangjin Park},
  booktitle={Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF 2025)},
  year={2025},
  publisher={ACM}
}
```
