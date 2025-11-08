# config.py
import torch
import os, sys

# 현재 파일 기준으로 프로젝트 루트 경로 계산
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ★ 핵심: sys.path에 루트 추가 (이게 없으면 import 탐색이 안 됨)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ★ 자동 루트 이동 (상대 경로 실행 시 dataset/... 인식)
os.chdir(PROJECT_ROOT)

# 현재 사용 가능한 디바이스 설정 (GPU 우선)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
