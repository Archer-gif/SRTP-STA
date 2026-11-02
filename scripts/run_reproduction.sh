#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAN_REPO="${WAN_REPO:-$ROOT/Wan2.1}"
CKPT_DIR="${CKPT_DIR:-$ROOT/models/Wan2.1-T2V-1.3B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-$ROOT/results/victorian}"
PROMPT="${PROMPT:-A 3D model of a 1800s victorian house.}"
SEED="${SEED:-1234}"
METHOD="${1:-}"

if [[ ! "$METHOD" =~ ^(full|sta|tc_sta|adaptive)$ ]]; then
  echo "Usage: bash scripts/run_reproduction.sh {full|sta|tc_sta|adaptive}" >&2
  exit 2
fi
if [[ ! -f "$WAN_REPO/generate.py" || ! -d "$CKPT_DIR" ]]; then
  echo "Set WAN_REPO and CKPT_DIR to the official Wan2.1 code and model paths." >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "Install GNU time (/usr/bin/time) for comparable wall timings." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
common=(
  --task t2v-1.3B --size '832*480' --frame_num 81
  --sample_steps 50 --sample_guide_scale 6 --sample_shift 8
  --base_seed "$SEED" --offload_model True --t5_cpu
  --ckpt_dir "$CKPT_DIR" --prompt "$PROMPT"
  --save_file "$OUT_DIR/$METHOD.mp4"
)

if [[ "$METHOD" == full ]]; then
  (cd "$WAN_REPO" && /usr/bin/time -f '%e' -o "$OUT_DIR/$METHOD.wall_seconds.txt" \
    "$PYTHON_BIN" -u generate.py "${common[@]}" \
    > "$OUT_DIR/$METHOD.log" 2>&1)
  exit 0
fi

strategy="$ROOT/sta_calibration_50/strategies/sta_strategy_mse_0p1.json"
options=(--production-optimized)
case "$METHOD" in
  sta)
    options+=(--sta-only --fixed-anchor-count 0)
    ;;
  tc_sta)
    options+=(--route-policy top1)
    ;;
  adaptive)
    strategy="$ROOT/adaptive_multiroute_experiment/strategy_energy_safe.json"
    options+=(--route-policy adaptive --route-margin-high 0.35 \
      --route-margin-low 0.15)
    ;;
esac

export PYTHONPATH="$ROOT:$ROOT/correspondence_sta_experiment/code${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
/usr/bin/time -f '%e' -o "$OUT_DIR/$METHOD.wall_seconds.txt" \
  "$PYTHON_BIN" -u "$ROOT/correspondence_sta_experiment/code/run_correspondence_wan.py" \
  --wan-repo "$WAN_REPO" --sta-strategy "$strategy" \
  --attention-metrics-out "$OUT_DIR/$METHOD.attention.json" \
  --dynamic-anchor-count 4 --dynamic-temporal-lags 2 --dynamic-radius 1 \
  --fixed-anchor-count 8 --fixed-temporal-lags 3 --fixed-radius 2 \
  --fixed-gate-sigma 0.5 --late-guard-step 35 --late-guard-layer 25 \
  "${options[@]}" "${common[@]}" > "$OUT_DIR/$METHOD.log" 2>&1
