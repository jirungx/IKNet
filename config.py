# config.py
import torch

# 현재 사용 가능한 디바이스 설정 (GPU 우선)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")