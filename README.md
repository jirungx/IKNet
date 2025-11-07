# IKNet 
**Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators**  
[📄 ICAIF 2025 Paper](https://doi.org/10.1145/3768292.3770343)

---
<br>

## 🧠 Overview
**IKNet** is an interpretable deep learning model that integrates **FinBERT-based keyword embeddings** from financial news with **technical indicators** to predict the next-day closing price of the S&P 500 index.  
Unlike prior models that average document embeddings or rely on sentiment scores, IKNet provides **keyword-level explanations** of how public sentiment influences price movement using **SHAP (Shapley Additive Explanations)**.

---
<br>

## 📘 Publication
> **IKNet: Interpretable Stock Price Prediction via Keyword-Guided Integration of News and Technical Indicators**  
> *Jinwoong Kim* and *Sangjin Park*  
> *6th ACM International Conference on AI in Finance (ICAIF 2025), Singapore, November 15–18, 2025.*  
> [DOI: 10.1145/3768292.3770343](https://doi.org/10.1145/3768292.3770343)

---
<br>

## 🧩 Model Architecture
<img width="4944" height="3738" alt="Figure1_Architecture" src="https://github.com/user-attachments/assets/bf803c1a-a611-4ce7-bb5a-25f0ea9bd27b" />

**Figure 1**: Architecture of the proposed IKNet model.
<br><br>

The IKNet model consists of four main modules:

(a) **FinBERT-based Keyword Extraction** – identifies contextually salient keywords from daily financial news using gradient-based saliency.

(b) **Keyword Encoder** – processes extracted keyword embeddings via independent MLP projection and GRU sequence encoder.

(c) **Technical Indicator Encoder** – encodes structured OHLCV-based features using a Bi-LSTM network.

(d) **Feature Fusion & Prediction** – concatenates both representations for next-day price forecasting.

---



