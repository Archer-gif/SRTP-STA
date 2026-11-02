#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/wan_srtp
OUT="$ROOT/adaptive_multiroute_experiment/benchmark_multi_prompt"
PYTHON=/usr/local/miniconda3/envs/py312/bin/python
STRATEGY="$ROOT/adaptive_multiroute_experiment/strategy_energy_safe.json"
MODEL=/model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B
WAN="$ROOT/Wan2.1"
mkdir -p "$OUT"

names=(sports_car station_walk ocean_lighthouse)
prompts=(
  "A red sports car drives quickly from left to right while the camera pans to follow it."
  "A woman in a red coat walks through a crowded train station while the camera tracks beside her and people briefly occlude her."
  "The camera flies low over detailed ocean waves toward a white lighthouse during sunrise, with reflections moving across the water."
)

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code"

for index in "${!names[@]}"; do
  name="${names[$index]}"
  prompt="${prompts[$index]}"
  dir="$OUT/$name"
  mkdir -p "$dir"
  printf '%s\n' "$prompt" > "$dir/prompt.txt"

  if [[ ! -s "$dir/full.mp4" ]]; then
    cd "$WAN"
    /usr/bin/time -f '%e' -o "$dir/full.wall_seconds.txt" \
      "$PYTHON" -u generate.py \
      --task t2v-1.3B --size '832*480' --frame_num 81 --sample_steps 50 \
      --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
      --offload_model True --t5_cpu --ckpt_dir "$MODEL" --prompt "$prompt" \
      --save_file "$dir/full.mp4" > "$dir/full.log" 2>&1
  fi

  if [[ ! -s "$dir/adaptive_production.mp4" ]]; then
    cd "$ROOT"
    /usr/bin/time -f '%e' -o "$dir/adaptive_production.wall_seconds.txt" \
      "$PYTHON" -u "$ROOT/correspondence_sta_experiment/code/run_correspondence_wan.py" \
      --wan-repo "$WAN" --sta-strategy "$STRATEGY" \
      --sample_steps 50 --attention-metrics-out "$dir/adaptive_production.attention.json" \
      --dynamic-anchor-count 4 --dynamic-temporal-lags 2 --dynamic-radius 1 \
      --fixed-anchor-count 8 --fixed-temporal-lags 3 --fixed-radius 2 \
      --fixed-gate-sigma 0.5 --late-guard-step 35 --late-guard-layer 25 \
      --route-policy adaptive --route-margin-high 0.35 --route-margin-low 0.15 \
      --production-optimized \
      --task t2v-1.3B --size '832*480' --frame_num 81 \
      --sample_guide_scale 6 --sample_shift 8 --base_seed 1234 \
      --offload_model True --t5_cpu --ckpt_dir "$MODEL" --prompt "$prompt" \
      --save_file "$dir/adaptive_production.mp4" \
      > "$dir/adaptive_production.log" 2>&1
  fi
done
