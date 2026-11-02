"""Check the GPU and paths required by the Wan reproduction runner."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import flex_attention  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
wan_repo = Path(os.environ.get("WAN_REPO", ROOT / "Wan2.1"))
ckpt_dir = Path(os.environ.get("CKPT_DIR", ROOT / "models/Wan2.1-T2V-1.3B"))

print(f"Python/Torch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
print(f"Wan generate.py: {wan_repo / 'generate.py'} ({(wan_repo / 'generate.py').is_file()})")
print(f"Model directory: {ckpt_dir} ({ckpt_dir.is_dir()})")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

if not (wan_repo / "generate.py").is_file():
    raise SystemExit("Missing official Wan2.1/generate.py; set WAN_REPO.")
if not ckpt_dir.is_dir():
    raise SystemExit("Missing model directory; set CKPT_DIR.")
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("A CUDA GPU with BF16 support is required for generation.")
