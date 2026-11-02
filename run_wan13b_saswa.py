"""Launch official Wan generate.py with the opt-in SASWA reference bridge.

Example, from the Wan checkout with this repository on PYTHONPATH:
  python /root/wan_srtp/run_wan13b_saswa.py --saswa-mode base \
      --task t2v-1.3B --size '832*480' --frame_num 81 ...

This reference implementation is for integration/correctness checks only.
Full 50-step timing awaits a vectorized or fused SASWA GPU backend.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--saswa-mode", choices=("base", "progressive", "hybrid", "sta", "sta_search"), required=True)
    parser.add_argument("--saswa-backend", choices=("reference", "grouped", "flex"), default="reference")
    parser.add_argument("--sta-strategy", type=Path)
    parser.add_argument("--allow-uncalibrated-sta", action="store_true")
    parser.add_argument("--sta-search-out", type=Path)
    parser.add_argument("--sta-search-skip-steps", type=int, default=12)
    parser.add_argument("--wan-repo", type=Path, default=Path.cwd())
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--saswa-preflight-only", action="store_true")
    parser.add_argument("--saswa-first-attention-only", action="store_true")
    parser.add_argument("--saswa-probe-step", type=int)
    parser.add_argument("--attention-metrics-out", type=Path)
    args, other = parser.parse_known_args()
    if args.sample_steps <= 0:
        parser.error("--sample_steps must be positive")
    generate = args.wan_repo.resolve() / "generate.py"
    if not generate.is_file():
        parser.error(f"official Wan generate.py not found at {generate}")
    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan.modules.model as wan_model
    from wan_saswa_integration import (
        SASWAFirstAttentionDone, SASWAPreflightDone, install_wan_saswa)
    from wan_sta_reference import load_sta_strategy

    if args.saswa_mode in ("sta", "hybrid") and args.sta_strategy is None:
        parser.error("--sta-strategy is required for STA and hybrid modes")
    if args.saswa_mode == "sta_search" and args.sta_search_out is None:
        parser.error("--sta-search-out is required for STA calibration")
    strategy = (load_sta_strategy(args.sta_strategy, steps=args.sample_steps,
                                  allow_uncalibrated=args.allow_uncalibrated_sta)
                if args.sta_strategy else None)

    state = install_wan_saswa(wan_model, mode=args.saswa_mode,
                              total_steps=args.sample_steps,
                              preflight_only=args.saswa_preflight_only,
                              first_attention_only=args.saswa_first_attention_only,
                              sta_strategy=strategy,
                              backend=args.saswa_backend,
                              probe_step=args.saswa_probe_step,
                              search_skip_steps=args.sta_search_skip_steps
                              if args.saswa_mode == "sta_search" else 0)
    sys.argv = [str(generate), "--sample_steps", str(args.sample_steps), *other]
    print(f"Wan attention bridge: mode={state.mode}, backend={state.backend}, "
          f"steps={state.total_steps}",
          flush=True)
    try:
        runpy.run_path(str(generate), run_name="__main__")
    except (SASWAPreflightDone, SASWAFirstAttentionDone) as done:
        print(done, flush=True)
    finally:
        if state.search_accum is not None:
            from wan_sta_calibration import write_search_rows
            write_search_rows(args.sta_search_out, state.search_accum,
                              prompt_count=1)
        if args.attention_metrics_out:
            args.attention_metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.attention_metrics_out.write_text(json.dumps({
                "mode": state.mode, "backend": state.backend,
                "total_steps": state.total_steps,
                "model_calls": state.model_calls,
                "attention_calls": state.attention_calls,
                "steps": {step: {
                    key: int(value.item()) if hasattr(value, "item") else value
                    for key, value in metric.items()}
                    for step, metric in state.step_metrics.items()},
            }, indent=2) + "\n", encoding="utf-8")
        print(f"SASWA bridge calls: model={state.model_calls}, "
              f"attention={state.attention_calls}", flush=True)


if __name__ == "__main__":
    main()
