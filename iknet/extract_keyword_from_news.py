import os
import re
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 디바이스 설정
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FinBERT 모델 로드
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(DEVICE)
model.eval()

def get_salient_tokens_only(text, top_k=5):
    """
    FinBERT gradient 기반으로 중요 토큰 top_k개 추출 (subword 병합 및 중요도 평균 포함)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    grads = {}

    def hook_fn(module, grad_input, grad_output):
        grads["emb_grad"] = grad_output[0]

    h = model.bert.embeddings.register_full_backward_hook(hook_fn)

    try:
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits)
        score = logits[0, pred_label]

        model.zero_grad()
        score.backward()
        h.remove()

        if "emb_grad" not in grads:
            return []

        grad = grads["emb_grad"].squeeze(0)
        saliency = grad.norm(dim=1)

        # 원래 토큰 + saliency 추출
        tokens = []
        saliencies = []
        for idx in range(input_ids.shape[1]):
            token_id = input_ids[0, idx].item()
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            tokens.append(token)
            saliencies.append(saliency[idx].item())

        # subword 병합 + saliency 평균
        merged_tokens = []
        merged_scores = []
        buffer = ""
        buffer_scores = []

        for tok, score in zip(tokens, saliencies):
            if tok.startswith("##"):
                buffer += tok[2:]
                buffer_scores.append(score)
            else:
                if buffer:
                    merged_tokens.append(buffer)
                    merged_scores.append(np.mean(buffer_scores))
                buffer = tok
                buffer_scores = [score]

        if buffer:
            merged_tokens.append(buffer)
            merged_scores.append(np.mean(buffer_scores))

        # 상위 top-k 단어 선택 (중복 없이, 알파벳만)
        token_score_pairs = list(zip(merged_tokens, merged_scores))
        token_score_pairs = sorted(token_score_pairs, key=lambda x: abs(x[1]), reverse=True)

        tokens_used = []
        seen = set()
        for tok, _ in token_score_pairs:
            if not tok.isalpha():
                continue
            lower_tok = tok.lower()
            if lower_tok in seen:
                continue
            tokens_used.append(lower_tok)
            seen.add(lower_tok)
            if len(tokens_used) == top_k:
                break

        return tokens_used

    except Exception as e:
        h.remove()
        print(f"Gradient 추출 실패: {e}")
        return []

def process_news_top_tokens_only(news_df, top_k=5, save_token_path="tokens/snp_topk25_tokens.csv"):
    """
    뉴스에서 날짜별로 중요 토큰 top-k개만 추출하여 저장 (빈도 기준으로 top-k 선정)
    """
    os.makedirs(os.path.dirname(save_token_path), exist_ok=True)
    news_df['date'] = pd.to_datetime(news_df['date']).dt.date
    daily_tokens = {}

    for date in tqdm(news_df['date'].unique(), desc="토큰 추출 중"):
        day_texts = news_df[news_df['date'] == date]['text']
        day_tokens = []

        for text in day_texts:
            try:
                tokens = get_salient_tokens_only(text, top_k=top_k)
                day_tokens.extend(tokens)
            except Exception as e:
                print(f"[{date}] 오류 발생: {e}")
                continue

        if day_tokens:
            counter = Counter(day_tokens)
            most_common_tokens = [token for token, _ in counter.most_common(top_k)]
            daily_tokens[date] = most_common_tokens

    save_daily_tokens_to_csv(daily_tokens, save_token_path)
    return daily_tokens

def save_daily_tokens_to_csv(daily_tokens, filepath):
    """
    날짜별 중요 토큰들을 CSV로 저장
    """
    rows = [{"date": date, "tokens": ", ".join(tokens)} for date, tokens in daily_tokens.items()]
    pd.DataFrame(rows).to_csv(filepath, index=False)
