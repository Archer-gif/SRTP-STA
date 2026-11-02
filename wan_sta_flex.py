"""Whole-layer FlexAttention backend for calibrated Wan 1.3B STA masks.

Full heads use one dense SDPA call. Every local head, even when the searched
window differs by head, is packed into one block-sparse FlexAttention call.
This removes the reference backend's Python loop over 546 query tiles.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from saswa_flex import BLOCK, KERNEL_OPTIONS, _block_mask, _compiled_flex, _layout, _pack
from wan13b import WAN13B_HEAD_DIM, WAN13B_HEADS


Window = tuple[int, int, int]


@lru_cache(maxsize=64)
def _searched_local_mask(
    grid: tuple[int, int, int],
    tile: tuple[int, int, int],
    head_windows: tuple[Window, ...],
    device_name: str,
    batch: int,
):
    """Build one head-specific BlockMask for all local heads in a layer.

    The cache size retains both CFG passes for the current 30-layer Wan step
    without retaining all 1,500 step/layer masks for the complete generation.
    """
    unique = tuple(dict.fromkeys(head_windows))
    packed, indices, counts, adjacency, tile_grid, pair_counts = _layout(
        grid, tile, unique, device_name)
    choice = torch.tensor([unique.index(window) for window in head_windows],
                          device=indices.device, dtype=torch.long)
    tiles = prod(tile_grid)
    positions = torch.arange(tiles, device=indices.device)[None, :]
    selected_indices = indices[choice[:, None], positions]
    selected_counts = counts[choice[:, None], positions]
    selected_adjacency = adjacency[choice[:, None], positions]
    mask = _block_mask(
        selected_indices[None], selected_counts[None], selected_adjacency[None],
        batch, len(head_windows), tiles)
    # _layout counts key tiles per query tile. Each dense tile pair is 60x60.
    pairs = batch * 60 * 60 * sum(pair_counts[unique.index(window)]
                                  for window in head_windows)
    return packed, mask, pairs


@torch.no_grad()
def sta_wan13b_flex_attention(
    q: Tensor, k: Tensor, v: Tensor, *, step: int, layer: int,
    strategy: dict, token_grid: tuple[int, int, int],
) -> tuple[Tensor, dict]:
    """Apply one calibrated STA layer with at most two attention calls."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,L,H,D]")
    batch, length, heads, dim = q.shape
    if length != prod(token_grid) or heads != WAN13B_HEADS or dim != WAN13B_HEAD_DIM:
        raise ValueError("STA QKV does not match Wan 1.3B geometry")
    if not 0 <= step < strategy["total_steps"] or not 0 <= layer < 30:
        raise ValueError("STA step/layer out of range")
    tile = tuple(strategy["tile_size"])
    if prod(tile) != 60 or any(size % width for size, width in zip(token_grid, tile)):
        raise ValueError("Flex STA requires divisible 60-token tiles")

    settings = strategy["masks"][step][layer]
    full_heads = [index for index, setting in enumerate(settings) if setting == "full"]
    local_heads = [index for index, setting in enumerate(settings) if setting != "full"]
    qv, kv, vv = [x.permute(0, 2, 1, 3).reshape(batch, heads, *token_grid, dim)
                  for x in (q, k, v)]
    output = torch.empty_like(qv)
    qk_pairs = 0

    if full_heads:
        dense = [x[:, full_heads].reshape(batch, len(full_heads), length, dim)
                 for x in (qv, kv, vv)]
        result = scaled_dot_product_attention(*dense)
        output[:, full_heads] = result.reshape(
            batch, len(full_heads), *token_grid, dim)
        qk_pairs += batch * len(full_heads) * length * length

    if local_heads:
        windows = tuple(tuple(settings[index]) for index in local_heads)
        if any(any(width > size // tile_width for width, size, tile_width in
                   zip(window, token_grid, tile)) for window in windows):
            raise ValueError("STA local mask exceeds the tile grid")
        packed_ids, block_mask, local_pairs = _searched_local_mask(
            token_grid, tile, windows, str(q.device), batch)
        local = [x[:, local_heads] for x in (qv, kv, vv)]
        result = _compiled_flex(q.device.type)(
            _pack(local[0], packed_ids), _pack(local[1], packed_ids),
            _pack(local[2], packed_ids), block_mask=block_mask,
            kernel_options=KERNEL_OPTIONS)
        tiles = length // 60
        unpacked = torch.empty(batch, len(local_heads), length, dim,
                               device=q.device, dtype=q.dtype)
        unpacked.index_copy_(
            2, packed_ids.reshape(tiles, BLOCK)[:, :60].reshape(-1),
            result.reshape(batch, len(local_heads), tiles, BLOCK, dim)[:, :, :, :60]
                  .reshape(batch, len(local_heads), length, dim))
        output[:, local_heads] = unpacked.reshape(
            batch, len(local_heads), *token_grid, dim)
        qk_pairs += local_pairs

    flat = output.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
    return flat, {
        "step": step, "layer": layer, "masks": settings,
        "full_heads": len(full_heads), "local_heads": len(local_heads),
        "attention_kernel_calls": int(bool(full_heads)) + int(bool(local_heads)),
        "final_qk_pairs": qk_pairs, "backend": "sta_flex_blocks",
    }
