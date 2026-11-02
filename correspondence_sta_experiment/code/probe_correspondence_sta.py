"""Evaluate correspondence-anchored STA variants on real Wan activations."""

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

from correspondence_hybrid_flex import (
    correspondence_hybrid_flex_attention, sta_temporal_anchor_flex_attention,
)
from wan_hybrid_flex import hybrid_wan13b_flex_attention
from wan_sta_flex import sta_wan13b_flex_attention
from wan_sta_reference import load_sta_strategy


_step: ContextVar[int | None] = ContextVar("correspondence_probe_step", default=None)


class ProbeFinished(Exception):
    pass


@dataclass
class State:
    steps: set[int]
    layers: set[int]
    strategy: dict
    model_calls: int = 0
    rows: list[dict] = field(default_factory=list)
    compact: bool = False


def _video(x: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    batch, _, heads, dim = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch, heads, *grid, dim)


def _tile_error(candidate: torch.Tensor, teacher: torch.Tensor,
                grid: tuple[int, int, int], tile=(3, 5, 4)) -> tuple[torch.Tensor, torch.Tensor]:
    error = (_video(candidate, grid).float() - _video(teacher, grid).float()).square()
    energy = _video(teacher, grid).float().square()
    batch, heads, _, _, _, dim = error.shape
    gt, gh, gw = (n // width for n, width in zip(grid, tile))

    def reduce(x: torch.Tensor) -> torch.Tensor:
        return (x.reshape(batch, heads, gt, tile[0], gh, tile[1], gw, tile[2], dim)
                .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
                .reshape(batch, heads, gt * gh * gw, -1).sum(-1))
    return reduce(error), reduce(energy)


def _metrics(error: torch.Tensor, energy: torch.Tensor) -> dict:
    relative = error / energy.clamp_min(1e-12)
    return {
        "relative_mse": float(error.sum().item() / energy.sum().item()),
        "tile_relative_mse_mean": float(relative.mean().item()),
        "tile_relative_mse_p95": float(torch.quantile(relative.flatten(), 0.95).item()),
    }


def _timed(callable_):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = callable_()
    torch.cuda.synchronize()
    return value, 1000 * (time.perf_counter() - started)


def install(model_module, state: State):
    model_class = model_module.WanModel
    attention_class = model_module.WanSelfAttention
    original_model_forward = model_class.forward

    def model_forward(self, *args, **kwargs):
        step = state.model_calls // 2
        token = _step.set(step)
        for index, block in enumerate(self.blocks):
            block.self_attn._correspondence_probe_layer = index
        try:
            return original_model_forward(self, *args, **kwargs)
        finally:
            _step.reset(token)
            state.model_calls += 1

    def attention_forward(self, x, seq_lens, grid_sizes, freqs):
        step = _step.get()
        layer = getattr(self, "_correspondence_probe_layer", None)
        grid = tuple(int(value) for value in grid_sizes[0].tolist())
        if step is None or layer is None or x.shape != (1, prod(grid), 1536):
            raise RuntimeError("invalid Wan probe state or geometry")
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
            row = {"step": step, "layer": layer, "methods": {}}
            errors = []
            energy = None

            methods = [
                ("sta", lambda: sta_wan13b_flex_attention(
                    q_half, k_half, v, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid)),
                ("hybrid", lambda: hybrid_wan13b_flex_attention(
                    q_half, k_half, v, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid)),
                ("anchor_qk", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk")),
                ("hybrid_anchor_qk_k2", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=2)),
                ("hybrid_anchor_qk_k4", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=4)),
                ("hybrid_anchor_qk_k8", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=8)),
                ("hybrid_anchor_gate_m0p5", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=8, fixed_gate_sigma=-0.5)),
                ("hybrid_anchor_gate_0", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=8, fixed_gate_sigma=0.0)),
                ("hybrid_anchor_gate_0p5", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=8, fixed_gate_sigma=0.5)),
                ("hybrid_anchor_gate_1", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="qk",
                    fixed_anchor_count=8, fixed_gate_sigma=1.0)),
                ("anchor_feature", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="feature")),
                ("anchor_combined_0p25", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="combined",
                    feature_weight=0.25)),
                ("anchor_combined_0p5", lambda: correspondence_hybrid_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, rank_mode="combined",
                    feature_weight=0.5)),
                ("sta_anchor_qk_k2", lambda: sta_temporal_anchor_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, anchor_count=2,
                    rank_mode="qk")),
                ("sta_anchor_qk_k4", lambda: sta_temporal_anchor_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, anchor_count=4,
                    rank_mode="qk")),
                ("sta_anchor_qk_k8", lambda: sta_temporal_anchor_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, anchor_count=8,
                    rank_mode="qk")),
                ("sta_anchor_feature_k4", lambda: sta_temporal_anchor_flex_attention(
                    q_half, k_half, v, features, step=step, layer=layer,
                    strategy=state.strategy, token_grid=grid, anchor_count=4,
                    rank_mode="feature")),
            ]
            if state.compact:
                keep = {"hybrid", "hybrid_anchor_gate_0p5", "hybrid_anchor_gate_1"}
                methods = [(name, method) for name, method in methods if name in keep]
            for name, method in methods:
                (result, info), elapsed = _timed(method)
                error, current_energy = _tile_error(result, teacher, grid)
                energy = current_energy if energy is None else energy
                errors.append(error)
                row["methods"][name] = {
                    **_metrics(error, energy), "milliseconds": elapsed,
                    "final_qk_pairs": int(info["final_qk_pairs"]),
                    "candidate_qk_pairs": int(info.get("candidate_qk_pairs", 0)),
                    "accepted_fixed_anchors": int(info.get("accepted_fixed_anchors", 0)),
                    "dynamic_heads": int(info.get("dynamic_heads", 0)),
                }
            if not state.compact:
                oracle_error, oracle_choice = torch.stack(errors).min(0)
                row["oracle_over_methods"] = {
                    **_metrics(oracle_error, energy),
                    "choice_histogram": [int(value) for value in torch.bincount(
                        oracle_choice.flatten(), minlength=len(methods)).tolist()],
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
    parser.add_argument("--wan-repo", type=Path, default=Path("/root/wan_srtp/Wan2.1"))
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--probe-out", type=Path, required=True)
    parser.add_argument("--probe-steps", default="25")
    parser.add_argument("--probe-layers", default="5,15,25")
    parser.add_argument("--compact", action="store_true")
    args, other = parser.parse_known_args()
    steps = {int(value) for value in args.probe_steps.split(",")}
    layers = {int(value) for value in args.probe_layers.split(",")}
    strategy = load_sta_strategy(args.strategy, steps=50, allow_uncalibrated=False)
    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan.modules.model as model_module
    state = State(steps, layers, strategy, compact=args.compact)
    install(model_module, state)
    sys.argv = [str(args.wan_repo / "generate.py"), *other]
    try:
        runpy.run_path(str(args.wan_repo / "generate.py"), run_name="__main__")
    except ProbeFinished:
        pass
    finally:
        args.probe_out.parent.mkdir(parents=True, exist_ok=True)
        args.probe_out.write_text(json.dumps({
            "steps": sorted(steps), "layers": sorted(layers),
            "model_calls": state.model_calls, "rows": state.rows,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.probe_out} with {len(state.rows)} rows", flush=True)


if __name__ == "__main__":
    main()
