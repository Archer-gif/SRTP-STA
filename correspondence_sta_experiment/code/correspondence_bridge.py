"""Isolated Wan bridge for correspondence-augmented STA experiments."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from math import prod
from time import perf_counter

import torch

from correspondence_hybrid_flex import correspondence_hybrid_flex_attention


_sampling_step: ContextVar[int | None] = ContextVar("correspondence_step", default=None)


@dataclass
class CorrespondenceState:
    total_steps: int
    model_calls: int = 0
    attention_calls: int = 0
    denoise_started_at: float | None = None
    denoise_finished_at: float | None = None
    step_metrics: dict[int, dict] = field(default_factory=dict)

    @property
    def denoise_seconds(self) -> float | None:
        if self.denoise_started_at is None or self.denoise_finished_at is None:
            return None
        return self.denoise_finished_at - self.denoise_started_at


def install_correspondence_attention(
    model_module, *, total_steps: int, strategy: dict,
    dynamic_anchor_count: int = 4, dynamic_temporal_lags: int = 2,
    dynamic_radius: int = 1, fixed_anchor_count: int = 8,
    fixed_temporal_lags: int = 3, fixed_radius: int = 2,
    fixed_gate_sigma: float | None = 0.5,
    late_guard_step: int = 35, late_guard_layer: int = 25,
    route_policy: str = "top1", route_margin_high: float = 0.20,
    route_margin_low: float = 0.08,
    production_optimized: bool = False,
    sta_only: bool = False,
) -> CorrespondenceState:
    """Patch only Wan 1.3B self-attention; cross-attention stays official."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    model_class = model_module.WanModel
    attn_class = model_module.WanSelfAttention
    if getattr(model_class, "_correspondence_bridge_installed", False):
        raise RuntimeError("correspondence bridge is already installed")

    original_model_forward = model_class.forward
    state = CorrespondenceState(total_steps=total_steps)

    def model_forward(self, *args, **kwargs):
        if getattr(self, "model_type", None) != "t2v":
            raise ValueError("correspondence bridge supports Wan T2V only")
        if len(self.blocks) != 30 or self.dim != 1536 or self.num_heads != 12:
            raise ValueError("correspondence bridge requires Wan2.1-T2V-1.3B")
        step = state.model_calls // 2
        if step >= total_steps:
            raise RuntimeError("more WanModel calls than expected")
        if state.model_calls == 0:
            state.denoise_started_at = perf_counter()
        for index, block in enumerate(self.blocks):
            block.self_attn._correspondence_layer_index = index
        token = _sampling_step.set(step)
        try:
            result = original_model_forward(self, *args, **kwargs)
        finally:
            _sampling_step.reset(token)
        state.model_calls += 1
        if state.model_calls == total_steps * 2:
            state.denoise_finished_at = perf_counter()
        return result

    def attention_forward(self, x, seq_lens, grid_sizes, freqs):
        step = _sampling_step.get()
        layer = getattr(self, "_correspondence_layer_index", None)
        if step is None or layer is None:
            raise RuntimeError("self-attention called outside tracked generation")
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 1536:
            raise ValueError("bridge requires batch-one [B,L,1536] input")
        grid = getattr(self, "_correspondence_token_grid", None)
        if grid is None:
            grid = tuple(int(value) for value in grid_sizes[0].tolist())
            self._correspondence_token_grid = grid
        if tuple(grid_sizes.shape) != (1, 3) or prod(grid) != x.shape[1]:
            raise ValueError("padded or inconsistent token grid is unsupported")
        if ((not production_optimized and int(seq_lens[0]) != x.shape[1])
                or self.num_heads != 12 or self.head_dim != 128):
            raise ValueError("unsupported Wan attention geometry")

        batch, length = x.shape[:2]
        q = self.norm_q(self.q(x)).view(batch, length, 12, 128)
        k = self.norm_k(self.k(x)).view(batch, length, 12, 128)
        v = self.v(x).view(batch, length, 12, 128)
        q = model_module.rope_apply(q, grid_sizes, freqs).to(v.dtype)
        k = model_module.rope_apply(k, grid_sizes, freqs).to(v.dtype)
        features = x.reshape(batch, *grid, x.shape[-1])
        late_guard_active = step >= late_guard_step and layer >= late_guard_layer
        effective_fixed_anchor_count = 0 if late_guard_active else fixed_anchor_count
        method_options = {"dynamic_settings": ()} if sta_only else {}
        result, diagnostics = correspondence_hybrid_flex_attention(
            q, k, v, features, step=step, layer=layer, strategy=strategy,
            token_grid=grid, anchor_count=dynamic_anchor_count,
            temporal_lags=dynamic_temporal_lags, radius=dynamic_radius,
            fixed_anchor_count=effective_fixed_anchor_count,
            fixed_temporal_lags=fixed_temporal_lags, fixed_radius=fixed_radius,
            fixed_gate_sigma=fixed_gate_sigma, route_policy=route_policy,
            route_margin_high=route_margin_high,
            route_margin_low=route_margin_low,
            collect_diagnostics=not production_optimized,
            production_optimized=production_optimized, **method_options)

        state.attention_calls += 1
        if production_optimized:
            return self.o(result.flatten(2))
        metric = state.step_metrics.setdefault(step, {
            "attention_calls": 0, "candidate_tile_scores": 0,
            "accepted_fixed_anchors": 0,
            "eligible_fixed_anchor_slots": 0,
            "late_guard_calls": 0,
            "final_qk_pairs": 0, "executed_qk_pairs": 0,
            "attention_kernel_calls": 0, "layers": {}})
        metric["attention_calls"] += 1
        metric["candidate_tile_scores"] += diagnostics["candidate_qk_pairs"]
        metric["accepted_fixed_anchors"] += diagnostics["accepted_fixed_anchors"]
        metric["eligible_fixed_anchor_slots"] += diagnostics[
            "eligible_fixed_anchor_slots"]
        metric["late_guard_calls"] += int(late_guard_active)
        metric["final_qk_pairs"] += diagnostics["final_qk_pairs"]
        metric["executed_qk_pairs"] += diagnostics["executed_qk_pairs"]
        metric["attention_kernel_calls"] += diagnostics["attention_kernel_calls"]
        layer_metric = metric["layers"].setdefault(str(layer), {
            "calls": 0, "head_modes": diagnostics["head_modes"],
            "head_qk_pairs": [0] * 12,
            "route_histograms": [[0, 0, 0] for _ in range(12)],
            "route_cardinality_histograms": [[0, 0, 0] for _ in range(12)]})
        layer_metric["calls"] += 1
        layer_metric["head_qk_pairs"] = [
            old + int(new) for old, new in
            zip(layer_metric["head_qk_pairs"], diagnostics["head_qk_pairs"])]
        layer_metric["route_histograms"] = [
            [old + int(new) for old, new in zip(old_head, new_head)]
            for old_head, new_head in zip(
                layer_metric["route_histograms"], diagnostics["route_histograms"])]
        layer_metric["route_cardinality_histograms"] = [
            [old + int(new) for old, new in zip(old_head, new_head)]
            for old_head, new_head in zip(
                layer_metric["route_cardinality_histograms"],
                diagnostics["route_cardinality_histograms"])]
        return self.o(result.flatten(2))

    attn_class.forward = attention_forward
    model_class.forward = model_forward
    model_class._correspondence_bridge_installed = True
    return state
