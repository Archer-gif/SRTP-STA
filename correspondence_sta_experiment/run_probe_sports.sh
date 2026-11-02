#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
PROBE_STEPS=${PROBE_STEPS:-40}
PROBE_LAYERS=${PROBE_LAYERS:-5,15,25}
PROBE_NAME=${PROBE_NAME:-gated_sports_step40}
COMPACT=${COMPACT:-}

"$PYTHON" -u "$ROOT/correspondence_sta_experiment/code/probe_correspondence_sta.py" \
  --strategy "$ROOT/sta_calibration_50/strategies/sta_strategy_mse_0p1.json" \
  --probe-out "$ROOT/correspondence_sta_experiment/probe/$PROBE_NAME.json" \
  --probe-steps "$PROBE_STEPS" --probe-layers "$PROBE_LAYERS" \
  $COMPACT \
  --wan-repo "$ROOT/Wan2.1" --task t2v-1.3B --size '832*480' \
  --frame_num 81 --sample_steps 50 --sample_guide_scale 6 --sample_shift 8 \
  --base_seed 1234 --offload_model True --t5_cpu \
  --ckpt_dir /model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B \
  --prompt "A red sports car drives quickly from left to right while the camera pans to follow it." \
  --save_file "$ROOT/correspondence_sta_experiment/probe/unused.mp4"
