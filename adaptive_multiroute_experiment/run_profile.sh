#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
OUT="$ROOT/adaptive_multiroute_experiment/profiling"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code"

"$PYTHON" -u "$ROOT/adaptive_multiroute_experiment/profile_attention_hotspots.py" \
  --wan-repo "$ROOT/Wan2.1" \
  --sta-strategy "$ROOT/sta_calibration_50/strategies/sta_strategy_mse_0p1.json" \
  --adaptive-strategy "$ROOT/adaptive_multiroute_experiment/strategy_energy_safe.json" \
  --profile-out "$OUT" --profile-step 23 --profile-layers 6,23,28 \
  --profile-repeats 10 \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_steps 50 --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
  --offload_model True --t5_cpu \
  --ckpt_dir /model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B \
  --prompt "A red sports car drives quickly from left to right while the camera pans to follow it." \
  --save_file "$OUT/unused.mp4"
