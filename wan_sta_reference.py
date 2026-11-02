"""Wan 1.3B STA-style sliding tile reference with per-step/layer/head masks.

This preserves STA's dense query/key tiles and mask-table structure. It is not
the archived FastVideo ThunderKittens kernel and has no calibrated masks built
in. A strategy must be fitted on Wan 1.3B teacher activations before results
can be reported as a calibrated STA baseline.
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from saswa import fixed_tile_attention
from wan13b import WAN13B_HEADS, WAN13B_HEAD_DIM


def load_sta_strategy(path: str | Path, *, steps: int,
                      allow_uncalibrated: bool = False) -> dict:
    strategy = json.loads(Path(path).read_text(encoding="utf-8"))
    if strategy.get("model") != "Wan2.1-T2V-1.3B" or strategy.get("total_steps") != steps:
        raise ValueError("STA strategy model/step count does not match this run")
    if not strategy.get("calibrated", False) and not allow_uncalibrated:
        raise ValueError("STA mask table is uncalibrated; only use it for smoke tests")
    tile = strategy.get("tile_size")
    if tile != [3, 5, 4]:
        raise ValueError("STA strategy requires 3x5x4 DiT-token tiles")
    masks = strategy.get("masks")
    if not isinstance(masks, list) or len(masks) != steps:
        raise ValueError("STA strategy must contain one mask table per step")
    for step in masks:
        if not isinstance(step, list) or len(step) != 30:
            raise ValueError("STA strategy must contain 30 layers per step")
        for layer in step:
            if not isinstance(layer, list) or len(layer) != WAN13B_HEADS:
                raise ValueError("STA strategy must contain 12 heads per layer")
            for mask in layer:
                if mask != "full" and (not isinstance(mask, list) or len(mask) != 3
                                       or any(not isinstance(n, int) or n < 1 for n in mask)):
                    raise ValueError("STA masks must be 'full' or positive [t,h,w]")
    return strategy


@torch.no_grad()
def sta_wan13b_flat_attention(q: Tensor, k: Tensor, v: Tensor, *,
                              step: int, layer: int, strategy: dict,
                              token_grid: tuple[int, int, int]) -> tuple[Tensor, dict]:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,L,H,D]")
    batch, length, heads, dim = q.shape
    if batch != 1 or length != prod(token_grid) or heads != WAN13B_HEADS or dim != WAN13B_HEAD_DIM:
        raise ValueError("STA QKV does not match Wan 1.3B geometry")
    if not 0 <= step < strategy["total_steps"] or not 0 <= layer < 30:
        raise ValueError("STA step/layer out of range")
    tile = tuple(strategy["tile_size"])
    grid_tiles = tuple(size // width for size, width in zip(token_grid, tile))
    if any(size % width for size, width in zip(token_grid, tile)):
        raise ValueError("STA tile size must divide token grid")
    masks = strategy["masks"][step][layer]
    qv, kv, vv = [x.permute(0, 2, 1, 3).reshape(batch, heads, *token_grid, dim)
                  for x in (q, k, v)]
    output = torch.empty_like(qv)
    qk_pairs = 0
    for mask in {tuple(m) if isinstance(m, list) else m for m in masks}:
        selected = [head for head, setting in enumerate(masks)
                    if (tuple(setting) if isinstance(setting, list) else setting) == mask]
        if not selected:
            continue
        if mask == "full":
            qs, ks, vs = [x[:, selected].reshape(batch, len(selected), length, dim)
                          for x in (qv, kv, vv)]
            result = scaled_dot_product_attention(qs, ks, vs)
            output[:, selected] = result.reshape(batch, len(selected), *token_grid, dim)
            qk_pairs += batch * len(selected) * length * length
        else:
            if any(width > size for width, size in zip(mask, grid_tiles)):
                raise ValueError(f"STA mask {mask} exceeds grid {grid_tiles}")
            result, pairs = fixed_tile_attention(
                qv[:, selected], kv[:, selected], vv[:, selected], tile, mask)
            output[:, selected] = result
            qk_pairs += pairs
    flat = output.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
    return flat, {"step": step, "layer": layer, "masks": masks,
                  "final_qk_pairs": qk_pairs}
