import os
import re
import logging
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------
# Device & FinBERT setup
# ---------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained(
    "yiyanghkust/finbert-tone"
).to(DEVICE)
model.eval()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------
# Token saliency extraction
# ---------------------------------------------------------
def get_salient_tokens_only(text: str, top_k: int = 5) -> List[str]:
    """
    Extract top-k salient tokens from a text using FinBERT gradients.
    - Uses a backward hook on the embedding layer.
    - Merges WordPiece subwords and averages their saliency.
    - Returns unique alphabetic tokens (lowercased), up to top_k.

    Args:
        text (str): Input text.
        top_k (int): Number of salient tokens to return.

    Returns:
        List[str]: List of salient tokens (lowercased), length <= top_k.
    """
    # Prepare inputs
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    grads = {}

    # Backward hook to capture gradients on embeddings
    def hook_fn(module, grad_input, grad_output):
        grads["emb_grad"] = grad_output[0]

    # NOTE: FinBERT is BERT-based; embedding layer is at model.bert.embeddings
    handle = model.bert.embeddings.register_full_backward_hook(hook_fn)

    try:
        # Forward
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=-1)
        score = logits[0, pred_label]

        # Backward on the predicted logit
        model.zero_grad(set_to_none=True)
        score.backward()

        if "emb_grad" not in grads:
            return []

        # Saliency per token (L2 norm of embedding grad)
        grad = grads["emb_grad"].squeeze(0)        # [seq_len, hidden]
        saliency = grad.norm(dim=1)                # [seq_len]

        # Recover tokens (skip special tokens)
        tokens: List[str] = []
        saliencies: List[float] = []
        for idx in range(input_ids.shape[1]):
            tok_id = input_ids[0, idx].item()
            tok = tokenizer.convert_ids_to_tokens(tok_id)
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            tokens.append(tok)
            saliencies.append(float(saliency[idx].item()))

        # Merge WordPiece subwords: accumulate chars and average saliency
        merged_tokens: List[str] = []
        merged_scores: List[float] = []
        buffer = ""
        buffer_scores: List[float] = []

        for tok, sc in zip(tokens, saliencies):
            if tok.startswith("##"):
                buffer += tok[2:]
                buffer_scores.append(sc)
            else:
                if buffer:
                    merged_tokens.append(buffer)
                    merged_scores.append(float(np.mean(buffer_scores)))
                buffer = tok
                buffer_scores = [sc]

        if buffer:
            merged_tokens.append(buffer)
            merged_scores.append(float(np.mean(buffer_scores)))

        # Rank by absolute saliency, enforce uniqueness and alphabetic constraint
        token_score_pairs = sorted(
            zip(merged_tokens, merged_scores),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        picked: List[str] = []
        seen = set()
        for tok, _ in token_score_pairs:
            # Keep alphabetic-only tokens; normalize to lowercase
            if not tok.isalpha():
                continue
            low = tok.lower()
            if low in seen:
                continue
            picked.append(low)
            seen.add(low)
            if len(picked) == top_k:
                break

        return picked

    except Exception as e:
        logging.exception("Failed to extract gradients for saliency: %s", str(e))
        return []

    finally:
        # Always remove hook to avoid memory leaks
        try:
            handle.remove()
        except Exception:
            pass


# ---------------------------------------------------------
# Per-day processing & CSV saving
# ---------------------------------------------------------
def process_news_top_tokens_only(
    news_df: pd.DataFrame, top_k: int = 5, save_token_path: str = "tokens/snp_topk25_tokens.csv"
) -> Dict[pd.Timestamp, List[str]]:
    """
    For each date, extract salient tokens from all news texts and keep top-k by frequency.

    Args:
        news_df (pd.DataFrame): Must contain columns ["date", "text"].
        top_k (int): Number of tokens to keep per date after frequency aggregation.
        save_token_path (str): Output CSV path with columns ["date", "tokens"].

    Returns:
        Dict[pd.Timestamp, List[str]]: Mapping from date to list of tokens.
    """
    if "date" not in news_df.columns or "text" not in news_df.columns:
        raise ValueError("news_df must contain 'date' and 'text' columns.")

    os.makedirs(os.path.dirname(save_token_path), exist_ok=True)
    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.date

    daily_tokens: Dict[pd.Timestamp, List[str]] = {}

    for date in tqdm(news_df["date"].unique(), desc="Extracting tokens"):
        day_texts: Iterable[str] = news_df[news_df["date"] == date]["text"]
        collected: List[str] = []

        for text in day_texts:
            try:
                tokens = get_salient_tokens_only(text, top_k=top_k)
                collected.extend(tokens)
            except Exception as e:
                # Continue processing other samples for this date
                logging.exception("Token extraction error on %s: %s", str(date), str(e))
                continue

        if collected:
            counter = Counter(collected)
            most_common = [tok for tok, _ in counter.most_common(top_k)]
            daily_tokens[date] = most_common

    save_daily_tokens_to_csv(daily_tokens, save_token_path)
    return daily_tokens


def save_daily_tokens_to_csv(daily_tokens: Dict[pd.Timestamp, List[str]], filepath: str) -> None:
    """
    Save date-wise salient tokens to a CSV file.

    Args:
        daily_tokens (Dict[date, List[str]]): Mapping date -> tokens list.
        filepath (str): Output CSV path.
    """
    rows = [{"date": date, "tokens": ", ".join(tokens)} for date, tokens in daily_tokens.items()]
    pd.DataFrame(rows).to_csv(filepath, index=False)
