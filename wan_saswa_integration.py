"""Opt-in Wan2.1-T2V-1.3B self-attention bridge for SASWA prototypes.

Install before generation. The bridge patches only WanSelfAttention.forward;
Wan text cross-attention subclasses implement their own forward methods.
This is a correctness bridge, not a fast inference kernel.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from math import prod
from typing import Literal

import torch

from wan13b import saswa_wan13b_flat_attention
from wan13b_progressive import progressive_wan13b_flat_attention
from wan_sta_reference import sta_wan13b_flat_attention
from wan_sta_calibration import DEFAULT_CANDIDATES, sampled_candidate_errors


_sampling_step: ContextVar[int | None] = ContextVar("wan_saswa_step", default=None)


@dataclass
class BridgeState:
    mode: Literal["base", "progressive", "hybrid", "sta", "sta_search"]
    total_steps: int
    preflight_only: bool = False
    first_attention_only: bool = False
    model_calls: int = 0
    attention_calls: int = 0
    last_diagnostics: dict | None = None
    search_accum: dict | None = None
    search_skip_steps: int = 0
    backend: Literal["reference", "grouped", "flex"] = "reference"
    probe_step: int | None = None
    step_metrics: dict[int, dict] = field(default_factory=dict)


class SASWAPreflightDone(Exception):
    """Intentional stop after checking the first real Wan self-attention input."""


class SASWAFirstAttentionDone(Exception):
    """Intentional stop after one real Wan SASWA attention invocation."""


def install_wan_saswa(model_module, *, mode: Literal["base", "progressive", "hybrid", "sta", "sta_search"],
                      total_steps: int, preflight_only: bool = False,
                      first_attention_only: bool = False,
                      sta_strategy: dict | None = None,
                      search_skip_steps: int = 0,
                      backend: Literal["reference", "grouped", "flex"] = "reference",
                      probe_step: int | None = None) -> BridgeState:
    """Patch official Wan model classes before constructing/running WanT2V.

    WanT2V calls WanModel twice per denoising step (CFG conditional and null).
    Only single-GPU, batch-one, unpadded 1.3B T2V is supported here. Reinstalling
    in the same process is rejected so step tracking cannot silently drift.
    """
    if mode not in ("base", "progressive", "hybrid", "sta", "sta_search") or total_steps <= 0:
        raise ValueError("invalid bridge mode or total_steps")
    if backend not in ("reference", "grouped", "flex"):
        raise ValueError("unknown SASWA backend")
    if probe_step is not None and (
            not first_attention_only or mode not in ("base", "progressive", "hybrid", "sta") or
            not 0 <= probe_step < total_steps):
        raise ValueError("probe_step requires a first-attention probe")
    if mode in ("sta", "hybrid") and sta_strategy is None:
        raise ValueError("STA and hybrid modes require a Wan 1.3B mask strategy")
    if preflight_only and first_attention_only:
        raise ValueError("select only one SASWA probe mode")
    if not 0 <= search_skip_steps <= total_steps:
        raise ValueError("invalid STA search skip count")
    model_class = model_module.WanModel
    attn_class = model_module.WanSelfAttention
    if getattr(model_class, "_saswa_bridge_installed", False):
        raise RuntimeError("SASWA bridge already installed in this process")

    original_model_forward = model_class.forward
    state = BridgeState(mode=mode, total_steps=total_steps,
                        preflight_only=preflight_only,
                        first_attention_only=first_attention_only,
                        search_accum={} if mode == "sta_search" else None,
                        search_skip_steps=search_skip_steps, backend=backend,
                        probe_step=probe_step)

    def model_forward(self, *args, **kwargs):
        if getattr(self, "model_type", None) != "t2v":
            raise ValueError("SASWA bridge currently supports Wan T2V only")
        if len(self.blocks) != 30 or self.dim != 1536 or self.num_heads != 12:
            raise ValueError("SASWA bridge requires Wan2.1-T2V-1.3B")
        step = state.model_calls // 2
        if step >= state.total_steps:
            raise RuntimeError("more WanModel calls than expected for CFG steps")
        token = _sampling_step.set(step)
        # The original WanModel.forward iterates self.blocks in order.
        for index, block in enumerate(self.blocks):
            if hasattr(block, "self_attn"):
                block.self_attn._saswa_layer_index = index
        try:
            result = original_model_forward(self, *args, **kwargs)
        finally:
            _sampling_step.reset(token)
        state.model_calls += 1
        return result

    def attention_forward(self, x, seq_lens, grid_sizes, freqs):
        step = _sampling_step.get()
        if step is None:
            raise RuntimeError("Wan self-attention called outside tracked WanModel.forward")
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 1536:
            raise ValueError("SASWA bridge requires batch-one [B,L,1536] input")
        grid = tuple(int(value) for value in grid_sizes[0].tolist())
        if tuple(grid_sizes.shape) != (1, 3) or prod(grid) != x.shape[1]:
            raise ValueError("padded or inconsistent Wan token grid is unsupported")
        if int(seq_lens[0]) != x.shape[1] or self.num_heads != 12 or self.head_dim != 128:
            raise ValueError("padded sequence or unsupported attention head geometry")
        batch, length = x.shape[:2]
        q = self.norm_q(self.q(x)).view(batch, length, 12, 128)
        k = self.norm_k(self.k(x)).view(batch, length, 12, 128)
        v = self.v(x).view(batch, length, 12, 128)
        q = model_module.rope_apply(q, grid_sizes, freqs)
        k = model_module.rope_apply(k, grid_sizes, freqs)
        q_raw, k_raw = q, k
        # Official Wan flash_attention converts RoPE's float32 Q/K to V's
        # half dtype before QK. Progressive full-stage SDPA also requires it.
        q, k = q.to(v.dtype), k.to(v.dtype)
        if state.preflight_only:
            raise SASWAPreflightDone(
                f"real Wan QKV ready: grid={grid}, q={tuple(q.shape)}, "
                f"dtype={q.dtype}, device={q.device}, step={step}")
        if mode == "sta_search":
            layer = getattr(self, "_saswa_layer_index", None)
            if layer is None:
                raise RuntimeError("STA search layer index was not set")
            result = model_module.flash_attention(
                q=q_raw, k=k_raw, v=v, k_lens=seq_lens,
                window_size=self.window_size)
            if step >= state.search_skip_steps:
                losses = sampled_candidate_errors(
                    q, k, v, result, token_grid=grid)
                for window in DEFAULT_CANDIDATES:
                    for head in range(12):
                        key = (step, layer, head, window)
                        old = state.search_accum.setdefault(key, [0.0, 0.0])
                        old[0] += float(losses[window][head, 0])
                        old[1] += float(losses[window][head, 1])
            diagnostics = {"stage": step, "layer": layer,
                           "search_windows": len(DEFAULT_CANDIDATES)}
        elif mode in ("sta", "hybrid"):
            layer = getattr(self, "_saswa_layer_index", None)
            if layer is None:
                raise RuntimeError("STA attention layer index was not set")
            sta_step = probe_step if probe_step is not None else step
            if mode == "hybrid":
                if backend != "flex":
                    raise ValueError("hybrid mode requires the Flex backend")
                from wan_hybrid_flex import hybrid_wan13b_flex_attention
                result, diagnostics = hybrid_wan13b_flex_attention(
                    q, k, v, step=sta_step, layer=layer,
                    strategy=sta_strategy, token_grid=grid)
            elif backend == "flex":
                from wan_sta_flex import sta_wan13b_flex_attention
                result, diagnostics = sta_wan13b_flex_attention(
                    q, k, v, step=sta_step, layer=layer,
                    strategy=sta_strategy, token_grid=grid)
            else:
                result, diagnostics = sta_wan13b_flat_attention(
                    q, k, v, step=sta_step, layer=layer,
                    strategy=sta_strategy, token_grid=grid)
        else:
            adapter = (saswa_wan13b_flat_attention if mode == "base"
                       else progressive_wan13b_flat_attention)
            adapter_args = dict(step=probe_step if probe_step is not None else step,
                                total_steps=total_steps, token_grid=grid)
            if backend in ("grouped", "flex"):
                adapter_args["backend"] = backend
            result, diagnostics = adapter(q, k, v, x, **adapter_args)
        state.attention_calls += 1
        state.last_diagnostics = diagnostics
        metric_step = probe_step if probe_step is not None else step
        metric = state.step_metrics.setdefault(
            metric_step, {"attention_calls": 0, "candidate_qk_pairs": 0,
                          "final_qk_pairs": 0, "attention_kernel_calls": 0,
                          "executed_qk_pairs": 0, "padding_qk_pairs": 0,
                          "stage": diagnostics.get("stage"),
                          "attention_mode": diagnostics.get("mode", mode)})
        metric["attention_calls"] += 1
        metric["candidate_qk_pairs"] += diagnostics.get("candidate_qk_pairs", 0)
        metric["final_qk_pairs"] += diagnostics.get("final_qk_pairs", 0)
        metric["executed_qk_pairs"] += diagnostics.get(
            "executed_qk_pairs", diagnostics.get("final_qk_pairs", 0))
        metric["padding_qk_pairs"] += diagnostics.get("padding_qk_pairs", 0)
        metric["attention_kernel_calls"] += diagnostics.get("attention_kernel_calls", 1)
        if "head_qk_pairs" in diagnostics:
            layer_key = str(diagnostics["layer"])
            layer_metric = metric.setdefault("layers", {}).setdefault(
                layer_key, {"calls": 0, "head_qk_pairs": [0] * 12,
                            "head_modes": diagnostics["head_modes"],
                            "chosen_histograms": [[0, 0, 0] for _ in range(12)]})
            layer_metric["calls"] += 1
            layer_metric["head_qk_pairs"] = [
                old + int(new) for old, new in
                zip(layer_metric["head_qk_pairs"], diagnostics["head_qk_pairs"])]
            layer_metric["chosen_histograms"] = [
                [old + int(new) for old, new in zip(old_head, new_head)]
                for old_head, new_head in zip(
                    layer_metric["chosen_histograms"],
                    diagnostics["chosen_histograms"])]
        if state.first_attention_only:
            if not torch.isfinite(result).all():
                raise ValueError("first SASWA attention output contains NaN/Inf")
            raise SASWAFirstAttentionDone(
                f"real bridged attention finished: mode={mode}, grid={grid}, "
                f"model_step={step}, mask_step="
                f"{probe_step if probe_step is not None else step}, "
                f"output={tuple(result.shape)}, diagnostics_stage="
                f"{diagnostics.get('stage')}")
        return self.o(result.flatten(2))

    # Mutate only after all static checks; Wan's cross-attention overrides
    # forward and therefore continues to use the unmodified official path.
    attn_class.forward = attention_forward
    model_class.forward = model_forward
    model_class._saswa_bridge_installed = True
    return state
