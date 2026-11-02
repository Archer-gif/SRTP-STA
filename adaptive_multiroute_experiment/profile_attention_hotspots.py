"""Profile representative real Wan QKV attention calls on CUDA."""

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
from torch.profiler import ProfilerActivity, profile

from correspondence_hybrid_flex import correspondence_hybrid_flex_attention
from wan_sta_reference import load_sta_strategy


_step: ContextVar[int | None] = ContextVar("profile_step", default=None)


class ProfileFinished(Exception):
    pass


@dataclass
class State:
    target_step: int
    layers: list[int]
    repeats: int
    strategies: dict[str, dict]
    out_dir: Path
    model_calls: int = 0
    rows: list[dict] = field(default_factory=list)


def _device_time_us(event) -> float:
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _self_device_time_us(event) -> float:
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def profile_method(name: str, method, repeats: int, out_dir: Path,
                   step: int, layer: int) -> dict:
    for _ in range(3):
        method()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=False,
    ) as prof:
        for _ in range(repeats):
            method()
        torch.cuda.synchronize()
    wall_ms = 1000 * (time.perf_counter() - started) / repeats

    prefix = f"step{step:02d}_layer{layer:02d}_{name}"
    prof.export_chrome_trace(str(out_dir / f"{prefix}.trace.json.gz"))
    (out_dir / f"{prefix}.table.txt").write_text(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=80), encoding="utf-8")

    phases = []
    operations = []
    for event in prof.key_averages():
        row = {
            "name": event.key,
            "calls": int(event.count),
            "device_total_us": _device_time_us(event),
            "self_device_total_us": _self_device_time_us(event),
            "cpu_total_us": float(event.cpu_time_total),
            "self_cpu_total_us": float(event.self_cpu_time_total),
        }
        if event.key.startswith("tcsta/"):
            phases.append(row)
        elif row["self_device_total_us"] > 0:
            operations.append(row)
    phases.sort(key=lambda row: row["device_total_us"], reverse=True)
    operations.sort(key=lambda row: row["self_device_total_us"], reverse=True)
    return {
        "method": name, "step": step, "layer": layer, "repeats": repeats,
        "wall_ms_per_call": wall_ms,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "phases": phases,
        "top_cuda_operations": operations[:30],
    }


def install(model_module, state: State) -> None:
    model_class = model_module.WanModel
    attention_class = model_module.WanSelfAttention
    original_model_forward = model_class.forward

    def model_forward(self, *args, **kwargs):
        step = state.model_calls // 2
        token = _step.set(step)
        for index, block in enumerate(self.blocks):
            block.self_attn._profile_layer = index
        try:
            return original_model_forward(self, *args, **kwargs)
        finally:
            _step.reset(token)
            state.model_calls += 1

    def attention_forward(self, x, seq_lens, grid_sizes, freqs):
        step = _step.get()
        layer = getattr(self, "_profile_layer", None)
        grid = tuple(int(value) for value in grid_sizes[0].tolist())
        batch, length = x.shape[:2]
        q = self.norm_q(self.q(x)).view(batch, length, 12, 128)
        k = self.norm_k(self.k(x)).view(batch, length, 12, 128)
        v = self.v(x).view(batch, length, 12, 128)
        q = model_module.rope_apply(q, grid_sizes, freqs).to(v.dtype)
        k = model_module.rope_apply(k, grid_sizes, freqs).to(v.dtype)
        teacher = model_module.flash_attention(
            q=q, k=k, v=v, k_lens=seq_lens, window_size=self.window_size)

        should_profile = (
            step == state.target_step and layer in state.layers and
            state.model_calls % 2 == 0 and x.shape == (1, prod(grid), 1536))
        if should_profile:
            features = x.reshape(batch, *grid, 1536)
            variants = {
                "adaptive": dict(
                    strategy=state.strategies["adaptive"],
                    fixed_anchor_count=8, fixed_temporal_lags=3,
                    fixed_radius=2, fixed_gate_sigma=0.5,
                    route_policy="adaptive", route_margin_high=0.35,
                    route_margin_low=0.15),
                "sta": dict(
                    strategy=state.strategies["sta"], dynamic_settings=(),
                    fixed_anchor_count=0, route_policy="top1"),
            }
            for name, options in variants.items():
                def method(options=options):
                    return correspondence_hybrid_flex_attention(
                        q, k, v, features, step=step, layer=layer,
                        token_grid=grid, collect_diagnostics=False,
                        production_optimized=True, profile_ranges=True,
                        **options)[0]

                row = profile_method(
                    name, method, state.repeats, state.out_dir, step, layer)
                state.rows.append(row)
                print(json.dumps({
                    "method": name, "step": step, "layer": layer,
                    "wall_ms_per_call": row["wall_ms_per_call"],
                    "phases": row["phases"],
                }), flush=True)
            if layer == max(state.layers):
                raise ProfileFinished
        return self.o(teacher.flatten(2))

    attention_class.forward = attention_forward
    model_class.forward = model_forward


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--sta-strategy", type=Path, required=True)
    parser.add_argument("--adaptive-strategy", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("--profile-step", type=int, default=23)
    parser.add_argument("--profile-layers", default="6,23,28")
    parser.add_argument("--profile-repeats", type=int, default=10)
    args, other = parser.parse_known_args()
    args.profile_out.mkdir(parents=True, exist_ok=True)
    strategies = {
        "sta": load_sta_strategy(
            args.sta_strategy, steps=50, allow_uncalibrated=False),
        "adaptive": load_sta_strategy(
            args.adaptive_strategy, steps=50, allow_uncalibrated=False),
    }
    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan.modules.model as model_module
    state = State(
        args.profile_step,
        [int(value) for value in args.profile_layers.split(",")],
        args.profile_repeats, strategies, args.profile_out)
    install(model_module, state)
    sys.argv = [str(args.wan_repo / "generate.py"), *other]
    try:
        runpy.run_path(str(args.wan_repo / "generate.py"), run_name="__main__")
    except ProfileFinished:
        pass
    finally:
        (args.profile_out / "profile_summary.json").write_text(json.dumps({
            "target_step": state.target_step,
            "layers": state.layers,
            "repeats": state.repeats,
            "model_calls": state.model_calls,
            "profiles": state.rows,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
