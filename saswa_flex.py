"""Whole-layer block-sparse FlexAttention for Wan SASWA tile geometry.

Each 60-token video tile occupies one 64-token attention block. Four padded
key positions are masked, and all query tiles run in one FlexAttention call.
This is an optional experimental backend; benchmark after compilation.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod, sqrt

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from saswa import SASWAConfig, StageConfig, _pool_tokens_by_tile, _window_tile_ids
from saswa_grouped import _geometry


BLOCK = 64
KERNEL_OPTIONS = {"BLOCK_M": BLOCK, "BLOCK_N": BLOCK}


@lru_cache(maxsize=24)
def _layout(grid: tuple[int, int, int], tile: tuple[int, int, int],
            windows: tuple[tuple[int, int, int], ...], device_name: str):
    if prod(tile) != 60:
        raise ValueError("Flex backend currently requires 60-token tiles")
    tile_grid = tuple(n // size for n, size in zip(grid, tile))
    device = torch.device(device_name)
    token_grid = torch.arange(prod(grid)).reshape(grid)
    centers = [(t, h, w) for t in range(tile_grid[0])
               for h in range(tile_grid[1]) for w in range(tile_grid[2])]
    tile_tokens = []
    for t, h, w in centers:
        tile_tokens.append(token_grid[
            t * tile[0]:(t + 1) * tile[0],
            h * tile[1]:(h + 1) * tile[1],
            w * tile[2]:(w + 1) * tile[2]].reshape(-1))
    packed = torch.stack([torch.cat((ids, ids.new_zeros(BLOCK - 60)))
                          for ids in tile_tokens]).reshape(-1).to(device)
    choices = [[_window_tile_ids(center, window, tile_grid)
                for center in centers] for window in windows]
    max_keys = max(len(keys) for variant in choices for keys in variant)
    indices = torch.zeros(len(windows), len(centers), max_keys,
                          dtype=torch.int32)
    counts = torch.zeros(len(windows), len(centers), dtype=torch.int32)
    adjacency = torch.zeros(len(windows), len(centers), len(centers),
                            dtype=torch.bool)
    for direction, variant in enumerate(choices):
        for query_tile, keys in enumerate(variant):
            indices[direction, query_tile, :len(keys)] = torch.tensor(keys,
                                                                       dtype=torch.int32)
            counts[direction, query_tile] = len(keys)
            adjacency[direction, query_tile, keys] = True
    return (packed, indices.to(device), counts.to(device), adjacency.to(device),
            tile_grid, tuple(int(counts[direction].sum())
                             for direction in range(len(windows))))


def _pack(x: Tensor, packed_ids: Tensor) -> Tensor:
    batch, heads, frames, height, width, dim = x.shape
    return x.reshape(batch, heads, frames * height * width, dim).index_select(
        2, packed_ids).contiguous()


def _block_mask(indices: Tensor, counts: Tensor, adjacency: Tensor,
                batch: int, heads: int, tiles: int) -> BlockMask:
    indices = indices.expand(batch, heads, -1, -1).contiguous()
    counts = counts.expand(batch, heads, -1).contiguous()
    adjacency = adjacency.expand(batch, heads, -1, -1)

    def mask_mod(b: Tensor, h: Tensor, query: Tensor, key: Tensor) -> Tensor:
        return adjacency[b, h, query // BLOCK, key // BLOCK] & (key % BLOCK < 60)

    return BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=(BLOCK, BLOCK),
        mask_mod=mask_mod, seq_lengths=(tiles * BLOCK, tiles * BLOCK),
        compute_q_blocks=False)


@lru_cache(maxsize=2)
def _compiled_flex(device_type: str):
    # PyTorch 2.13's FlexAttention BlockMask lowering is stable with the
    # default compile mode used by the validated A800 path. Its max-autotune
    # mode currently rejects some valid small BlockMasks during compilation.
    return torch.compile(flex_attention) if device_type == "cuda" else flex_attention


@lru_cache(maxsize=12)
def _fixed_mask(grid: tuple[int, int, int], tile: tuple[int, int, int],
                window: tuple[int, int, int], device_name: str,
                batch: int, heads: int) -> BlockMask:
    _, indices, counts, adjacency, tile_grid, _ = _layout(
        grid, tile, (window,), device_name)
    return _block_mask(indices[0], counts[0], adjacency[0],
                       batch, heads, prod(tile_grid))


@torch.no_grad()
def fixed_flex_attention(q: Tensor, k: Tensor, v: Tensor,
                         tile_size: tuple[int, int, int],
                         window: tuple[int, int, int]) -> tuple[Tensor, int]:
    if q.ndim != 6 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,H,T,H,W,D]")
    batch, heads, frames, height, width, dim = q.shape
    grid = frames, height, width
    if any(n % t for n, t in zip(grid, tile_size)):
        raise ValueError("token grid must divide into tiles")
    packed_ids, key_ids, key_counts, _, tile_grid, pair_counts = _layout(
        grid, tile_size, (window,), str(q.device))
    if any(w > g for w, g in zip(window, tile_grid)):
        raise ValueError("window exceeds tile grid")
    tiles = prod(tile_grid)
    block_mask = _fixed_mask(grid, tile_size, window, str(q.device), batch, heads)
    result = _compiled_flex(q.device.type)(_pack(q, packed_ids), _pack(k, packed_ids),
                              _pack(v, packed_ids), block_mask=block_mask,
                              kernel_options=KERNEL_OPTIONS)
    output = torch.empty(batch, heads, prod(grid), dim, device=q.device, dtype=q.dtype)
    output.index_copy_(2, packed_ids.reshape(tiles, BLOCK)[:, :60].reshape(-1),
                       result.reshape(batch, heads, tiles, BLOCK, dim)[:, :, :, :60]
                       .reshape(batch, heads, prod(grid), dim))
    pairs = pair_counts[0] * batch * heads * 60 * 60
    return output.reshape_as(q), pairs


@torch.no_grad()
def saswa_flex_attention(q: Tensor, k: Tensor, v: Tensor, features: Tensor,
                         config: SASWAConfig, step: int, total_steps: int
                         ) -> tuple[Tensor, dict]:
    if q.ndim != 6 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,H,T,H,W,D]")
    batch, heads, frames, height, width, dim = q.shape
    grid = frames, height, width
    if features.shape[:4] != (batch, *grid) or features.device != q.device:
        raise ValueError("feature shape/device does not match QKV")
    if any(n % t for n, t in zip(grid, config.tile_size)):
        raise ValueError("token grid must divide into tiles")
    if not 0 <= step < total_steps:
        raise ValueError("invalid step")
    stage_id = min(step * len(config.stages) // total_steps, len(config.stages) - 1)
    stage: StageConfig = config.stages[stage_id]
    if (config.routing_mode != "pooled" or stage.expand_fraction != 1.0
            or stage.threshold != 0.0):
        raise ValueError("Flex backend supports pooled routing, rho=1, threshold=0")
    packed_ids, key_ids, key_counts, adjacency, tile_grid, _ = _layout(
        grid, config.tile_size, stage.windows, str(q.device))
    if any(any(w > g for w, g in zip(window, tile_grid))
           for window in stage.windows):
        raise ValueError("window exceeds tile grid")
    tiles = prod(tile_grid)
    pooled_q = _pool_tokens_by_tile(q, tile_grid, config.tile_size)
    pooled_k = _pool_tokens_by_tile(k, tile_grid, config.tile_size)
    # Cached geometry is built once; routing and selected block indices remain
    # on the GPU during each attention call.
    _, _, union_ids, valid, gains, union_count = _geometry(
        grid, config.tile_size, stage, str(q.device))
    logits = (pooled_q.unsqueeze(-2) * pooled_k[:, :, union_ids]).sum(-1) / sqrt(dim)
    probabilities = logits.masked_fill(~valid[None, None], -torch.inf).softmax(-1)
    scores = (probabilities.unsqueeze(-2) * gains[None, None]).sum(-1)
    best_gain, direction = scores.max(-1)
    chosen = torch.where(best_gain > 0, direction + 1, 0)
    tile_positions = torch.arange(tiles, device=q.device)[None, None, :]
    selected_ids = key_ids[chosen.long(), tile_positions]
    selected_counts = key_counts[chosen.long(), tile_positions]
    selected_adjacency = adjacency[chosen.long(), tile_positions]
    block_mask = _block_mask(selected_ids, selected_counts, selected_adjacency,
                             batch, heads, tiles)
    result = _compiled_flex(q.device.type)(_pack(q, packed_ids), _pack(k, packed_ids),
                              _pack(v, packed_ids), block_mask=block_mask,
                              kernel_options=KERNEL_OPTIONS)
    output = torch.empty(batch, heads, prod(grid), dim, device=q.device, dtype=q.dtype)
    output.index_copy_(2, packed_ids.reshape(tiles, BLOCK)[:, :60].reshape(-1),
                       result.reshape(batch, heads, tiles, BLOCK, dim)[:, :, :, :60]
                       .reshape(batch, heads, prod(grid), dim))
    final_pairs = selected_counts.sum() * (60 * 60)
    return output.reshape_as(q), {
        "stage": stage_id, "routing_mode": "pooled", "chosen": chosen,
        "scores": scores, "candidate_qk_pairs": union_count * batch * heads,
        "final_qk_pairs": final_pairs, "backend": "flex_blocks"}
