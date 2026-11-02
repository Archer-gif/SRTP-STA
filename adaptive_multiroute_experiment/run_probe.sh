#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
CODE="$ROOT/correspondence_sta_experiment/code"
ADAPT="$ROOT/adaptive_multiroute_experiment"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
export PYTHONPATH="$ROOT:$CODE:$ADAPT/code"

"$PYTHON" -u "$ADAPT/code/probe_adaptive_real.py" \
  --wan-repo "$ROOT/Wan2.1" \
  --base-strategy "$ROOT/sta_calibration_50/strategies/sta_strategy_mse_0p1.json" \
  --safe-strategy "$ADAPT/strategy_energy_safe.json" \
  --balanced-strategy "$ADAPT/strategy_energy_balanced.json" \
  --probe-out "$ADAPT/probe/adaptive_energy_layer27.json" \
  --probe-steps 14,23,33,40 --probe-layers 27 \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_steps 50 --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
  --offload_model True --t5_cpu \
  --ckpt_dir /model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B \
  --prompt "A red sports car drives quickly from left to right while the camera pans to follow it." \
  --save_file "$ADAPT/probe/unused.mp4"
