"""Run official Wan generation with correspondence-augmented STA attention."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--sta-strategy", type=Path, required=True)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--attention-metrics-out", type=Path, required=True)
    parser.add_argument("--dynamic-anchor-count", type=int, default=4)
    parser.add_argument("--dynamic-temporal-lags", type=int, default=2)
    parser.add_argument("--dynamic-radius", type=int, default=1)
    parser.add_argument("--fixed-anchor-count", type=int, default=8)
    parser.add_argument("--fixed-temporal-lags", type=int, default=3)
    parser.add_argument("--fixed-radius", type=int, default=2)
    parser.add_argument("--fixed-gate-sigma", type=float, default=0.5)
    parser.add_argument("--late-guard-step", type=int, default=35)
    parser.add_argument("--late-guard-layer", type=int, default=25)
    parser.add_argument("--route-policy", choices=("top1", "adaptive"),
                        default="top1")
    parser.add_argument("--route-margin-high", type=float, default=0.20)
    parser.add_argument("--route-margin-low", type=float, default=0.08)
    parser.add_argument("--production-optimized", action="store_true")
    parser.add_argument("--sta-only", action="store_true")
    args, other = parser.parse_known_args()
    generate = args.wan_repo.resolve() / "generate.py"
    if not generate.is_file():
        parser.error(f"official Wan generate.py not found at {generate}")
    for name in ("dynamic_anchor_count", "dynamic_temporal_lags", "dynamic_radius",
                 "fixed_anchor_count", "fixed_temporal_lags", "fixed_radius"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan.modules.model as wan_model
    from correspondence_bridge import install_correspondence_attention
    from wan_sta_reference import load_sta_strategy

    strategy = load_sta_strategy(args.sta_strategy, steps=args.sample_steps,
                                 allow_uncalibrated=False)
    parameters = {
        "dynamic_anchor_count": args.dynamic_anchor_count,
        "dynamic_temporal_lags": args.dynamic_temporal_lags,
        "dynamic_radius": args.dynamic_radius,
        "fixed_anchor_count": args.fixed_anchor_count,
        "fixed_temporal_lags": args.fixed_temporal_lags,
        "fixed_radius": args.fixed_radius,
        "fixed_gate_sigma": args.fixed_gate_sigma,
        "late_guard_step": args.late_guard_step,
        "late_guard_layer": args.late_guard_layer,
        "route_policy": args.route_policy,
        "route_margin_high": args.route_margin_high,
        "route_margin_low": args.route_margin_low,
        "production_optimized": args.production_optimized,
        "sta_only": args.sta_only,
    }
    state = install_correspondence_attention(
        wan_model, total_steps=args.sample_steps, strategy=strategy, **parameters)
    sys.argv = [str(generate), "--sample_steps", str(args.sample_steps), *other]
    started = perf_counter()
    try:
        runpy.run_path(str(generate), run_name="__main__")
    finally:
        wall_seconds = perf_counter() - started
        args.attention_metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.attention_metrics_out.write_text(json.dumps({
            "method": ("stable-shape STA Flex baseline" if args.sta_only else
                       "production-optimized adaptive correspondence anchors"
                       if args.production_optimized else
                       "STA-protected correspondence anchors"),
            "parameters": parameters,
            "total_steps": args.sample_steps,
            "model_calls": state.model_calls,
            "attention_calls": state.attention_calls,
            "denoise_seconds": state.denoise_seconds,
            "runner_wall_seconds": wall_seconds,
            "steps": state.step_metrics,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Correspondence bridge calls: model={state.model_calls}, "
              f"attention={state.attention_calls}, denoise={state.denoise_seconds}",
              flush=True)


if __name__ == "__main__":
    main()
