from pathlib import Path

import torch

ROOT_DIR = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

DEFAULT_CONFIG_FILE = ROOT_DIR / "config.yaml"
