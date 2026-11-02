#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
OUT="$ROOT/adaptive_multiroute_experiment/benchmark_multi_prompt"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
STRATEGY="$ROOT/sta_calibration_50/strategies/sta_strategy_mse_0p1.json"
MODEL=/model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B
WAN="$ROOT/Wan2.1"
names=(sports_car station_walk ocean_lighthouse)

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code"

for name in "${names[@]}"; do
  dir="$OUT/$name"
  prompt="$(<"$dir/prompt.txt")"
  if [[ -s "$dir/sta_production.mp4" ]]; then
    continue
  fi
  /usr/bin/time -f '%e' -o "$dir/sta_production.wall_seconds.txt" \
    "$PYTHON" -u "$ROOT/correspondence_sta_experiment/code/run_correspondence_wan.py" \
    --sta-only --production-optimized \
    --wan-repo "$WAN" --sta-strategy "$STRATEGY" \
    --sample_steps 50 --fixed-anchor-count 0 \
    --attention-metrics-out "$dir/sta_production.attention.json" \
    --task t2v-1.3B --size '832*480' --frame_num 81 \
    --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
    --offload_model True --t5_cpu --ckpt_dir "$MODEL" --prompt "$prompt" \
    --save_file "$dir/sta_production.mp4" \
    > "$dir/sta_production.log" 2>&1
done
