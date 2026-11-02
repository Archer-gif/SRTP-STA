#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
OUT="$ROOT/adaptive_multiroute_experiment/benchmark_victorian"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
STRATEGY="$ROOT/adaptive_multiroute_experiment/strategy_energy_safe.json"
MODEL=/model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B
PROMPT="A 3D model of a 1800s victorian house."
mkdir -p "$OUT"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code"

/usr/bin/time -f '%e' -o "$OUT/adaptive_production.wall_seconds.txt" \
  "$PYTHON" -u "$ROOT/correspondence_sta_experiment/code/run_correspondence_wan.py" \
  --wan-repo "$ROOT/Wan2.1" --sta-strategy "$STRATEGY" \
  --sample_steps 50 --attention-metrics-out "$OUT/adaptive_production.attention.json" \
  --dynamic-anchor-count 4 --dynamic-temporal-lags 2 --dynamic-radius 1 \
  --fixed-anchor-count 8 --fixed-temporal-lags 3 --fixed-radius 2 \
  --fixed-gate-sigma 0.5 --late-guard-step 35 --late-guard-layer 25 \
  --route-policy adaptive --route-margin-high 0.35 --route-margin-low 0.15 \
  --production-optimized \
  --task t2v-1.3B --size '832*480' --frame_num 81 \
  --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
  --offload_model True --t5_cpu --ckpt_dir "$MODEL" --prompt "$PROMPT" \
  --save_file "$OUT/adaptive_production.mp4" \
  > "$OUT/adaptive_production.log" 2>&1
