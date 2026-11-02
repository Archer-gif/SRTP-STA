"""Probe adaptive multi-route TC-STA variants on real Wan activations."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

import torch

from correspondence_hybrid_flex import correspondence_hybrid_flex_attention
from wan_sta_reference import load_sta_strategy


_step: ContextVar[int | None] = ContextVar("adaptive_probe_step", default=None)


class ProbeFinished(Exception):
    pass


@dataclass
class State:
    steps: set[int]
    layers: set[int]
    strategies: dict[str, dict]
    model_calls: int = 0
    rows: list[dict] = field(default_factory=list)


def _metrics(candidate: torch.Tensor, teacher: torch.Tensor) -> dict:
    error = (candidate.float() - teacher.float()).square().sum((0, 1, 3))
    energy = teacher.float().square().sum((0, 1, 3)).clamp_min(1e-12)
    head_error = error / energy
    return {
        "relative_mse": float(error.sum().item() / energy.sum().item()),
        "head_relative_mse": [float(value) for value in head_error.tolist()],
    }


def _route_summary(info: dict) -> dict:
    histograms = info["route_cardinality_histograms"]
    totals = [sum(head[index] for head in histograms) for index in range(3)]
    count = sum(totals)
    return {
        "route_cardinality": totals,
        "average_routes": ((totals[0] + 2 * totals[1] + 3 * totals[2]) / count
                           if count else 0.0),
        "expanded_tile_percent": (100 * (totals[1] + totals[2]) / count
                                  if count else 0.0),
    }


def install(model_module, state: State) -> None:
    model_class = model_module.WanModel
    attention_class = model_module.WanSelfAttention
    original_model_forward = model_class.forward

    def model_forward(self, *args, **kwargs):
        step = state.model_calls // 2
        token = _step.set(step)
        for index, block in enumerate(self.blocks):
            block.self_attn._adaptive_probe_layer = index
        try:
            return original_model_forward(self, *args, **kwargs)
        finally:
            _step.reset(token)
            state.model_calls += 1

    def attention_forward(self, x, seq_lens, grid_sizes, freqs):
        step = _step.get()
        layer = getattr(self, "_adaptive_probe_layer", None)
        grid = tuple(int(value) for value in grid_sizes[0].tolist())
        if step is None or layer is None or x.shape != (1, prod(grid), 1536):
            raise RuntimeError("invalid Wan probe state")
        batch, length = x.shape[:2]
        q = self.norm_q(self.q(x)).view(batch, length, 12, 128)
        k = self.norm_k(self.k(x)).view(batch, length, 12, 128)
        v = self.v(x).view(batch, length, 12, 128)
        q_raw = model_module.rope_apply(q, grid_sizes, freqs)
        k_raw = model_module.rope_apply(k, grid_sizes, freqs)
        teacher = model_module.flash_attention(
            q=q_raw, k=k_raw, v=v, k_lens=seq_lens, window_size=self.window_size)

        if step in state.steps and layer in state.layers and state.model_calls % 2 == 0:
            q_half, k_half = q_raw.to(v.dtype), k_raw.to(v.dtype)
            features = x.reshape(batch, *grid, 1536)
            fixed_count = 0 if step >= 35 and layer >= 25 else 8
            variants = [
                ("tc_top1", "base", "top1", 0.20, 0.08),
                ("adaptive_20_08", "base", "adaptive", 0.20, 0.08),
                ("adaptive_35_15", "base", "adaptive", 0.35, 0.15),
                ("adaptive_energy_safe", "safe", "adaptive", 0.35, 0.15),
                ("adaptive_energy_balanced", "balanced", "adaptive", 0.35, 0.15),
            ]
            row = {"step": step, "layer": layer, "methods": {}}
            for name, strategy_name, policy, high, low in variants:
                strategy = state.strategies[strategy_name]

                def method():
                    return correspondence_hybrid_flex_attention(
                        q_half, k_half, v, features, step=step, layer=layer,
                        strategy=strategy, token_grid=grid,
                        fixed_anchor_count=fixed_count, fixed_gate_sigma=0.5,
                        route_policy=policy, route_margin_high=high,
                        route_margin_low=low)

                method()
                torch.cuda.synchronize()
                started = time.perf_counter()
                result, info = method()
                torch.cuda.synchronize()
                row["methods"][name] = {
                    **_metrics(result, teacher), **_route_summary(info),
                    "milliseconds": 1000 * (time.perf_counter() - started),
                    "full_heads": info["full_heads"],
                    "final_qk_pairs": info["final_qk_pairs"],
                    "executed_qk_pairs": info["executed_qk_pairs"],
                    "accepted_fixed_anchors": info["accepted_fixed_anchors"],
                }
            state.rows.append(row)
            print(json.dumps(row), flush=True)
            if step == max(state.steps) and layer == max(state.layers):
                raise ProbeFinished
        return self.o(teacher.flatten(2))

    attention_class.forward = attention_forward
    model_class.forward = model_forward


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--base-strategy", type=Path, required=True)
    parser.add_argument("--safe-strategy", type=Path, required=True)
    parser.add_argument("--balanced-strategy", type=Path, required=True)
    parser.add_argument("--probe-out", type=Path, required=True)
    parser.add_argument("--probe-steps", default="23,33,40")
    parser.add_argument("--probe-layers", default="6,9,15,25")
    args, other = parser.parse_known_args()
    strategies = {
        "base": load_sta_strategy(args.base_strategy, steps=50, allow_uncalibrated=False),
        "safe": load_sta_strategy(args.safe_strategy, steps=50,
                                   allow_uncalibrated=False),
        "balanced": load_sta_strategy(args.balanced_strategy, steps=50,
                                       allow_uncalibrated=False),
    }
    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan.modules.model as model_module
    state = State({int(x) for x in args.probe_steps.split(",")},
                  {int(x) for x in args.probe_layers.split(",")}, strategies)
    install(model_module, state)
    sys.argv = [str(args.wan_repo / "generate.py"), *other]
    try:
        runpy.run_path(str(args.wan_repo / "generate.py"), run_name="__main__")
    except ProbeFinished:
        pass
    finally:
        args.probe_out.parent.mkdir(parents=True, exist_ok=True)
        args.probe_out.write_text(json.dumps({
            "steps": sorted(state.steps), "layers": sorted(state.layers),
            "model_calls": state.model_calls, "rows": state.rows,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
