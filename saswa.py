"""Reference implementation of stage and content adaptive sliding tile attention.

This is an inference-only, correctness-oriented PyTorch implementation. Inputs
are video-token Q/K/V tensors, not raw video pixels. It is not an optimized STA
kernel and its Python loops should not be used for latency comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, prod, sqrt
from typing import Literal

import torch
from torch import Tensor


Size3 = tuple[int, int, int]


@dataclass(frozen=True)
class StageConfig:
    """Nominal window widths in (time, height, width) tile order.

    The three extensions add equal numbers of tiles away from video borders.
    Border clipping can make their actual added tile counts differ.
    """

    base: Size3
    time: Size3
    height: Size3
    width: Size3
    expand_fraction: float = 1.0
    detail_lambda: float = 0.0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        windows = (self.base, self.time, self.height, self.width)
        if any(len(w) != 3 or any(n <= 0 for n in w) for w in windows):
            raise ValueError("Every window must contain three positive tile counts")
        if not 0.0 <= self.expand_fraction <= 1.0:
            raise ValueError("expand_fraction must be between 0 and 1")
        if self.detail_lambda < 0 or self.threshold < 0:
            raise ValueError("detail_lambda and threshold must be nonnegative")
        for axis, extended in enumerate(windows[1:]):
            if any(extended[j] != self.base[j] for j in range(3) if j != axis):
                raise ValueError("Each extension may grow along only its named axis")
            if extended[axis] <= self.base[axis]:
                raise ValueError("An extended window must exceed the base window")
        if len({prod(w) - prod(self.base) for w in windows[1:]}) != 1:
            raise ValueError("The three extensions must add equal numbers of tiles")

    @property
    def windows(self) -> tuple[Size3, Size3, Size3, Size3]:
        return self.base, self.time, self.height, self.width


@dataclass(frozen=True)
class SASWAConfig:
    tile_size: Size3
    stages: tuple[StageConfig, ...]
    spatial_scale: float = 1.0
    temporal_scale: float = 1.0
    spatial_weight: float = 0.5
    routing_mode: Literal["pooled", "exact"] = "pooled"

    def __post_init__(self) -> None:
        if len(self.tile_size) != 3 or any(n <= 0 for n in self.tile_size):
            raise ValueError("tile_size must have three positive entries")
        if not self.stages:
            raise ValueError("At least one stage is required")
        if self.spatial_scale <= 0 or self.temporal_scale <= 0:
            raise ValueError("Feature-difference scales must be positive")
        if not 0 <= self.spatial_weight <= 1:
            raise ValueError("spatial_weight must be between 0 and 1")
        if self.routing_mode not in ("pooled", "exact"):
            raise ValueError("routing_mode must be 'pooled' or 'exact'")


def _detail_weight(x: Tensor, config: SASWAConfig, stage: StageConfig) -> Tensor:
    """Return [batch, query_tile] weights from adjacent feature differences."""
    x = x.float()
    batch, frames, height, width, _ = x.shape
    dt, dh, dw = config.tile_size
    spatial_h = x.new_zeros(batch, frames, height, width)
    spatial_w = torch.zeros_like(spatial_h)
    temporal = torch.zeros_like(spatial_h)
    spatial_h[:, :, 1:] = (x[:, :, 1:] - x[:, :, :-1]).square().mean(-1)
    spatial_w[:, :, :, 1:] = (x[:, :, :, 1:] - x[:, :, :, :-1]).square().mean(-1)
    temporal[:, 1:] = (x[:, 1:] - x[:, :-1]).square().mean(-1)
    ds = ((spatial_h + spatial_w) / (2 * config.spatial_scale)).clamp(0, 1)
    dt_score = (temporal / config.temporal_scale).clamp(0, 1)
    detail = config.spatial_weight * ds + (1 - config.spatial_weight) * dt_score
    detail = detail.reshape(batch, frames // dt, dt, height // dh, dh, width // dw, dw)
    return 1 + stage.detail_lambda * detail.mean((2, 4, 6)).reshape(batch, -1)


def _window_tile_ids(center: Size3, width: Size3, grid: Size3) -> list[int]:
    # Keep the nominal center. Positions outside the video are discarded;
    # the window is never translated inward to replace missing keys.
    starts = [max(0, c - w // 2) for c, w in zip(center, width)]
    stops = [min(dim, c - w // 2 + w) for c, w, dim in zip(center, width, grid)]
    return [
        (t * grid[1] + h) * grid[2] + w
        for t in range(starts[0], stops[0])
        for h in range(starts[1], stops[1])
        for w in range(starts[2], stops[2])
    ]


def _tile_tokens(grid: Size3, tile: Size3, device: torch.device) -> list[Tensor]:
    frames, height, width = (g * n for g, n in zip(grid, tile))
    token_ids = torch.arange(frames * height * width, device=device).reshape(frames, height, width)
    return [
        token_ids[t * tile[0] : (t + 1) * tile[0],
                  h * tile[1] : (h + 1) * tile[1],
                  w * tile[2] : (w + 1) * tile[2]].reshape(-1)
        for t in range(grid[0])
        for h in range(grid[1])
        for w in range(grid[2])
    ]


def _pool_tokens_by_tile(x: Tensor, grid: Size3, tile: Size3) -> Tensor:
    """Mean-pool Q or K *after* RoPE to [batch, heads, tiles, dim]."""
    batch, heads, _, _, _, dim = x.shape
    gt, gh, gw = grid
    dt, dh, dw = tile
    return (x.reshape(batch, heads, gt, dt, gh, dh, gw, dw, dim)
            .permute(0, 1, 2, 4, 6, 3, 5, 7, 8)
            .reshape(batch, heads, gt * gh * gw, dt * dh * dw, dim)
            .float().mean(-2))


@torch.no_grad()
def fixed_tile_attention(q: Tensor, k: Tensor, v: Tensor, tile_size: Size3,
                         window: Size3) -> tuple[Tensor, int]:
    """Simple fixed-window baseline using the same tile geometry as SASWA.

    Returns the output and exact QK pair count. This is a reference baseline,
    not STA's optimized kernel or its searched per-head configuration.
    """
    if q.ndim != 6 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share [batch, heads, time, height, width, d]")
    batch, heads, frames, height, width, dim = q.shape
    if any(n % t for n, t in zip((frames, height, width), tile_size)):
        raise ValueError("Video-token dimensions must be divisible by tile_size")
    grid = tuple(n // t for n, t in zip((frames, height, width), tile_size))
    if any(w > g for w, g in zip(window, grid)):
        raise ValueError("Window exceeds the tile grid")
    tile_tokens = _tile_tokens(grid, tile_size, q.device)
    n_tokens = frames * height * width
    q_flat, k_flat, v_flat = (
        tensor.reshape(batch, heads, n_tokens, dim) for tensor in (q, k, v)
    )
    output = torch.empty_like(q_flat)
    pairs_per_head = 0
    for tile_id, query_ids in enumerate(tile_tokens):
        center = (tile_id // (grid[1] * grid[2]),
                  tile_id // grid[2] % grid[1], tile_id % grid[2])
        key_tiles = _window_tile_ids(center, window, grid)
        key_ids = torch.cat([tile_tokens[j] for j in key_tiles])
        pairs_per_head += len(query_ids) * len(key_ids)
        logits = torch.matmul(q_flat[:, :, query_ids].float(),
                              k_flat[:, :, key_ids].float().transpose(-1, -2)) / sqrt(dim)
        probabilities = logits.softmax(-1)
        output[:, :, query_ids] = torch.matmul(
            probabilities, v_flat[:, :, key_ids].float()
        ).to(v.dtype)
    return output.reshape_as(q), pairs_per_head * batch * heads


@torch.no_grad()
def saswa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    features: Tensor,
    config: SASWAConfig,
    step: int,
    total_steps: int,
) -> tuple[Tensor, dict[str, Tensor | int]]:
    """Compute SASWA for Q/K/V shaped [batch, heads, time, height, width, d].

    ``features`` is the current hidden state [batch, time, height, width, channels].
    ``step`` is zero-based sampling order; larger values mean later denoising.
    Selection is separate for each batch item and attention head. Direction IDs
    are 0=base, 1=time, 2=height, 3=width.
    """
    if q.ndim != 6 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share shape [batch, heads, time, height, width, d]")
    if q.device != k.device or q.device != v.device or q.device != features.device:
        raise ValueError("All inputs must be on the same device")
    batch, heads, frames, height, width, dim = q.shape
    if features.shape[:4] != (batch, frames, height, width) or features.ndim != 5:
        raise ValueError("features must have shape [batch, time, height, width, channels]")
    if not 0 <= step < total_steps or total_steps <= 0:
        raise ValueError("step must be in [0, total_steps)")
    if any(n % t for n, t in zip((frames, height, width), config.tile_size)):
        raise ValueError("Video-token dimensions must be divisible by tile_size")

    stage_id = min(step * len(config.stages) // total_steps, len(config.stages) - 1)
    stage = config.stages[stage_id]
    grid = tuple(n // t for n, t in zip((frames, height, width), config.tile_size))
    if any(any(w > g for w, g in zip(window, grid)) for window in stage.windows):
        raise ValueError("Every candidate window must fit within the tile grid")

    tile_tokens = _tile_tokens(grid, config.tile_size, q.device)
    tile_count = len(tile_tokens)
    key_indices: list[list[Tensor]] = []
    gain_masks: list[Tensor] = []
    union_tile_ids: list[Tensor] = []
    for tile_id in range(tile_count):
        center = (tile_id // (grid[1] * grid[2]),
                  tile_id // grid[2] % grid[1], tile_id % grid[2])
        candidate_tiles = [_window_tile_ids(center, w, grid) for w in stage.windows]
        base_tiles = set(candidate_tiles[0])
        union_tiles = sorted(set().union(*candidate_tiles))
        key_indices.append([torch.cat([tile_tokens[j] for j in ids]) for ids in candidate_tiles])
        union_tile_ids.append(torch.tensor(union_tiles, dtype=torch.long, device=q.device))
        if config.routing_mode == "exact":
            key_indices[-1].append(torch.cat([tile_tokens[j] for j in union_tiles]))
        mask = torch.tensor(
            [[j in set(ids) - base_tiles for j in union_tiles] for ids in candidate_tiles[1:]],
            dtype=torch.bool, device=q.device,
        )
        if config.routing_mode == "exact":
            mask = mask.repeat_interleave(prod(config.tile_size), dim=1)
        gain_masks.append(mask)

    n_tokens = frames * height * width
    q_flat = q.reshape(batch, heads, n_tokens, dim)
    k_flat = k.reshape(batch, heads, n_tokens, dim)
    v_flat = v.reshape(batch, heads, n_tokens, dim)
    # With every eligible tile expanded and no positive threshold, detail
    # weights cannot change the chosen direction: they scale all three gains
    # equally. Skip the feature scan in this quality-first configuration.
    if stage.expand_fraction == 1.0 and stage.threshold == 0.0:
        omega = torch.ones(batch, tile_count, device=q.device, dtype=torch.float32)
    else:
        omega = _detail_weight(features, config, stage)
    scores = torch.empty(batch, heads, tile_count, 3, device=q.device, dtype=torch.float32)

    if config.routing_mode == "pooled":
        pooled_q = _pool_tokens_by_tile(q, grid, config.tile_size)
        pooled_k = _pool_tokens_by_tile(k, grid, config.tile_size)

    # Pooled routing scores one Q/K pair per tile pair. Exact routing remains
    # available as an ablation, and scores every token pair in the union.
    for tile_id, query_ids in enumerate(tile_tokens):
        if config.routing_mode == "pooled":
            logits = torch.matmul(
                pooled_q[:, :, tile_id].unsqueeze(-2),
                pooled_k[:, :, union_tile_ids[tile_id]].transpose(-1, -2),
            ).squeeze(-2) / sqrt(dim)
            probabilities = logits.softmax(-1)
            gains = (probabilities.unsqueeze(-2) * gain_masks[tile_id][None, None]).sum(-1)
            scores[:, :, tile_id] = gains * omega[:, None, tile_id, None]
        else:
            union_ids = key_indices[tile_id][4]
            logits = torch.matmul(q_flat[:, :, query_ids].float(),
                                  k_flat[:, :, union_ids].float().transpose(-1, -2)) / sqrt(dim)
            probabilities = logits.softmax(-1)
            gains = (probabilities.unsqueeze(-2) * gain_masks[tile_id][None, None, None]).sum(-1)
            scores[:, :, tile_id] = gains.mean(-2) * omega[:, None, tile_id, None]

    best_gain, best_direction = scores.max(-1)
    chosen = torch.zeros(batch, heads, tile_count, dtype=torch.long, device=q.device)
    budget = floor(stage.expand_fraction * tile_count)
    if budget == tile_count:
        chosen = torch.where(best_gain > stage.threshold, best_direction + 1, chosen)
    elif budget:
        for b in range(batch):
            for head in range(heads):
                eligible = torch.nonzero(best_gain[b, head] > stage.threshold).flatten()
                if eligible.numel():
                    top = best_gain[b, head, eligible].topk(min(budget, eligible.numel())).indices
                    ids = eligible[top]
                    chosen[b, head, ids] = best_direction[b, head, ids] + 1

    output = torch.empty_like(q_flat)
    # Re-normalize softmax inside each selected final window.
    for b in range(batch):
        for head in range(heads):
            for tile_id, query_ids in enumerate(tile_tokens):
                keys = key_indices[tile_id][int(chosen[b, head, tile_id])]
                logits = torch.matmul(q_flat[b, head, query_ids].float(),
                                      k_flat[b, head, keys].float().T) / sqrt(dim)
                probabilities = logits.softmax(-1)
                output[b, head, query_ids] = torch.matmul(
                    probabilities, v_flat[b, head, keys].float()
                ).to(v.dtype)

    diagnostics: dict[str, Tensor | int] = {
        "stage": stage_id,
        "routing_mode": config.routing_mode,
        "chosen": chosen,
        "scores": scores,
        "detail_weight": omega,
        "candidate_qk_pairs": (
            sum(len(ids) for ids in union_tile_ids) if config.routing_mode == "pooled"
            else sum(len(ids[4]) * len(qids) for ids, qids in zip(key_indices, tile_tokens))
        ) * batch * heads,
        "final_qk_pairs": sum(
            len(key_indices[tile_id][int(chosen[b, head, tile_id])]) * len(tile_tokens[tile_id])
            for b in range(batch) for head in range(heads) for tile_id in range(tile_count)
        ),
    }
    return output.reshape_as(q), diagnostics
