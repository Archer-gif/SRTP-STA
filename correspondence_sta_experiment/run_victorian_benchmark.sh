#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
OUT="$ROOT/correspondence_sta_experiment/benchmark_victorian"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
STRATEGY="$ROOT/sta_calibration_50/strategies/sta_strategy_mse_0p1.json"
MODEL=/model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B
PROMPT="A 3D model of a 1800s victorian house."
GATE_SIGMA=${GATE_SIGMA:-0.5}
mkdir -p "$OUT"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code"

/usr/bin/time -f '%e' -o "$OUT/sta_flex.wall_seconds.txt" \
  "$PYTHON" -u "$ROOT/correspondence_sta_experiment/code/run_correspondence_wan.py" \
  --sta-only --sta-strategy "$STRATEGY" --wan-repo "$ROOT/Wan2.1" \
  --sample_steps 50 --fixed-anchor-count 0 \
  --attention-metrics-out "$OUT/sta_flex.attention.json" \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
  --offload_model True --t5_cpu --ckpt_dir "$MODEL" --prompt "$PROMPT" \
  --save_file "$OUT/sta_flex.mp4" > "$OUT/sta_flex.log" 2>&1

/usr/bin/time -f '%e' -o "$OUT/tc_sta.wall_seconds.txt" \
  "$PYTHON" -u "$ROOT/correspondence_sta_experiment/code/run_correspondence_wan.py" \
  --wan-repo "$ROOT/Wan2.1" --sta-strategy "$STRATEGY" \
  --sample_steps 50 --attention-metrics-out "$OUT/tc_sta.attention.json" \
  --dynamic-anchor-count 4 --dynamic-temporal-lags 2 --dynamic-radius 1 \
  --fixed-anchor-count 8 --fixed-temporal-lags 3 --fixed-radius 2 \
  --fixed-gate-sigma "$GATE_SIGMA" --late-guard-step 35 --late-guard-layer 25 \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
  --offload_model True --t5_cpu --ckpt_dir "$MODEL" --prompt "$PROMPT" \
  --save_file "$OUT/tc_sta.mp4" > "$OUT/tc_sta.log" 2>&1
