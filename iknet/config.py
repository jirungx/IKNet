# config.py
import os
import sys
import torch


# ---------------------------------------------------------
# 1. Determine project root directory
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add project root to sys.path for module imports
# (Required when executing scripts from subdirectories)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Automatically move working directory to the project root
# (Ensures relative paths like dataset/... work consistently)
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------
# 2. Device configuration
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
