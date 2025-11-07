from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


class FinBERTEmbedder:
    def __init__(self, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)
        self.device = device
        self.model.eval()

    def get_keyword_embedding_tensor(self, keywords_batch, top_k=5):
        """
        keywords_batch: List[List[str]]  # e.g., [["boost", "surge", "recovery"], ["fall", "crash", "drop"], ...]
        returns: torch.Tensor of shape [B, K, 768]
        """
        batch_embeddings = []

        for keywords in tqdm(keywords_batch, desc="Extracting FinBERT embeddings"):
            emb_list = []
            for word in keywords[:top_k]:
                inputs = self.tokenizer(word, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
                emb_list.append(cls_emb.squeeze(0).cpu())

            while len(emb_list) < top_k:
                emb_list.append(torch.zeros(768))  # padding

            stacked = torch.stack(emb_list)  # [K, 768]
            batch_embeddings.append(stacked)

        return torch.stack(batch_embeddings)  # [B, K, 768]