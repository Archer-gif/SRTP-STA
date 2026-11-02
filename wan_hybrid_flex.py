"""STA-protected heads with SASWA routing for safely local small heads."""

from __future__ import annotations

from math import prod, sqrt

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention

from saswa import StageConfig, _pool_tokens_by_tile
from saswa_flex import BLOCK, KERNEL_OPTIONS, _block_mask, _compiled_flex, _layout, _pack
from saswa_grouped import _geometry
from wan13b import WAN13B_HEAD_DIM, WAN13B_HEADS


Window = tuple[int, int, int]
DEFAULT_DYNAMIC_SETTINGS = ((1, 2, 2), (2, 2, 2))
DEFAULT_DYNAMIC_STAGE = StageConfig(
    base=(1, 2, 2), time=(2, 2, 2), height=(1, 4, 2),
    width=(1, 2, 4), expand_fraction=1.0, detail_lambda=0.0,
)
PRODUCTION_WINDOWS = ((5, 3, 5), (3, 3, 3), (2, 2, 2),
                      (1, 4, 2), (1, 2, 4))


@torch.no_grad()
def hybrid_wan13b_flex_attention(
    q: Tensor, k: Tensor, v: Tensor, *, step: int, layer: int,
    strategy: dict, token_grid: tuple[int, int, int],
    dynamic_settings: tuple[Window, ...] = DEFAULT_DYNAMIC_SETTINGS,
    dynamic_stage: StageConfig = DEFAULT_DYNAMIC_STAGE,
) -> tuple[Tensor, dict]:
    """Run protected full heads and all local heads in at most two calls."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,L,H,D]")
    batch, length, heads, dim = q.shape
    if length != prod(token_grid) or heads != WAN13B_HEADS or dim != WAN13B_HEAD_DIM:
        raise ValueError("QKV does not match Wan 1.3B geometry")
    if not 0 <= step < strategy["total_steps"] or not 0 <= layer < 30:
        raise ValueError("hybrid step/layer out of range")
    tile = tuple(strategy["tile_size"])
    if prod(tile) != 60 or any(size % width for size, width in zip(token_grid, tile)):
        raise ValueError("hybrid Flex requires divisible 60-token tiles")

    settings = strategy["masks"][step][layer]
    full_heads = [index for index, setting in enumerate(settings) if setting == "full"]
    local_heads = [index for index, setting in enumerate(settings) if setting != "full"]
    dynamic_set = {tuple(window) for window in dynamic_settings}
    dynamic_heads = [index for index in local_heads if tuple(settings[index]) in dynamic_set]
    fixed_heads = [index for index in local_heads if index not in dynamic_heads]
    qv, kv, vv = [x.permute(0, 2, 1, 3).reshape(batch, heads, *token_grid, dim)
                  for x in (q, k, v)]
    output = torch.empty_like(qv)
    head_pairs = [0] * heads
    candidate_pairs = 0
    chosen_histograms = [[0, 0, 0] for _ in range(heads)]

    if full_heads:
        dense = [x[:, full_heads].reshape(batch, len(full_heads), length, dim)
                 for x in (qv, kv, vv)]
        result = scaled_dot_product_attention(*dense)
        output[:, full_heads] = result.reshape(
            batch, len(full_heads), *token_grid, dim)
        for head in full_heads:
            head_pairs[head] = batch * length * length

    if local_heads:
        final_dynamic_windows = dynamic_stage.windows[1:]
        static_windows = tuple(dict.fromkeys(
            tuple(settings[head]) for head in fixed_heads))
        routed_windows = final_dynamic_windows if dynamic_heads else ()
        # Production calls must keep both the head dimension and BlockMask
        # metadata shape stable across all 30 layers. Otherwise Dynamo reaches
        # its recompile limit and FlexAttention falls back to dense eager work.
        tile_grid = tuple(size // width for size, width in zip(token_grid, tile))
        stable_windows = tuple(window for window in PRODUCTION_WINDOWS
                               if all(width <= size for width, size in
                                      zip(window, tile_grid)))
        windows = tuple(dict.fromkeys(
            (*stable_windows, *static_windows, *routed_windows)))
        packed_ids, indices, counts, adjacency, tile_grid, _ = _layout(
            token_grid, tile, windows, str(q.device))
        if any(any(width > size for width, size in zip(window, tile_grid))
               for window in windows):
            raise ValueError("hybrid local window exceeds tile grid")
        tiles = prod(tile_grid)
        choice = torch.empty(batch, len(local_heads), tiles, device=q.device,
                             dtype=torch.long)
        local_position = {head: index for index, head in enumerate(local_heads)}
        window_position = {window: index for index, window in enumerate(windows)}

        for head in fixed_heads:
            choice[:, local_position[head]].fill_(
                window_position[tuple(settings[head])])

        if dynamic_heads:
            dynamic_q = _pool_tokens_by_tile(qv[:, dynamic_heads], tile_grid, tile)
            dynamic_k = _pool_tokens_by_tile(kv[:, dynamic_heads], tile_grid, tile)
            _, _, union_ids, valid, gains, union_count = _geometry(
                token_grid, tile, dynamic_stage, str(q.device))
            logits = ((dynamic_q.unsqueeze(-2) * dynamic_k[:, :, union_ids]).sum(-1)
                      / sqrt(dim))
            probabilities = logits.masked_fill(
                ~valid[None, None], -torch.inf).softmax(-1)
            scores = (probabilities.unsqueeze(-2) * gains[None, None]).sum(-1)
            direction = scores.argmax(-1)
            direction_to_window = torch.tensor(
                [window_position[window] for window in final_dynamic_windows],
                device=q.device)
            dynamic_choice = direction_to_window[direction]
            for dynamic_index, head in enumerate(dynamic_heads):
                choice[:, local_position[head]] = dynamic_choice[:, dynamic_index]
                histogram = torch.bincount(direction[:, dynamic_index].reshape(-1),
                                           minlength=3)
                chosen_histograms[head] = [int(value) for value in histogram.tolist()]
            candidate_pairs = union_count * batch * len(dynamic_heads)

        positions = torch.arange(tiles, device=q.device)[None, None, :]
        selected_indices = indices[choice, positions]
        selected_counts = counts[choice, positions]
        selected_adjacency = adjacency[choice, positions]
        padded_heads = local_heads + [local_heads[0]] * (heads - len(local_heads))
        if len(local_heads) < heads:
            padding = choice[:, :1].expand(-1, heads - len(local_heads), -1)
            padded_choice = torch.cat((choice, padding), dim=1)
            selected_indices = indices[padded_choice, positions]
            selected_counts = counts[padded_choice, positions]
            selected_adjacency = adjacency[padded_choice, positions]
        block_mask = _block_mask(selected_indices, selected_counts,
                                 selected_adjacency, batch, heads, tiles)
        local = [x[:, padded_heads] for x in (qv, kv, vv)]
        result = _compiled_flex(q.device.type)(
            _pack(local[0], packed_ids), _pack(local[1], packed_ids),
            _pack(local[2], packed_ids), block_mask=block_mask,
            kernel_options=KERNEL_OPTIONS)
        unpacked = torch.empty(batch, len(local_heads), length, dim,
                               device=q.device, dtype=q.dtype)
        unpacked.index_copy_(
            2, packed_ids.reshape(tiles, BLOCK)[:, :60].reshape(-1),
            result[:, :len(local_heads)].reshape(
                batch, len(local_heads), tiles, BLOCK, dim)[:, :, :, :60]
                  .reshape(batch, len(local_heads), length, dim))
        output[:, local_heads] = unpacked.reshape(
            batch, len(local_heads), *token_grid, dim)
        per_local_head = selected_counts[:, :len(local_heads)].sum(dim=(0, 2)) * (60 * 60)
        for local_index, head in enumerate(local_heads):
            head_pairs[head] = int(per_local_head[local_index].item())
        executed_local_pairs = int(selected_counts.sum().item()) * (60 * 60)
    else:
        executed_local_pairs = 0

    head_modes = []
    for head, setting in enumerate(settings):
        if setting == "full":
            head_modes.append("full")
        elif head in dynamic_heads:
            head_modes.append("dynamic_small")
        else:
            head_modes.append("fixed_" + "x".join(map(str, setting)))
    flat = output.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
    return flat, {
        "step": step, "layer": layer, "masks": settings,
        "full_heads": len(full_heads), "fixed_heads": len(fixed_heads),
        "dynamic_heads": len(dynamic_heads), "head_modes": head_modes,
        "head_qk_pairs": head_pairs, "chosen_histograms": chosen_histograms,
        "candidate_qk_pairs": candidate_pairs,
        "final_qk_pairs": sum(head_pairs),
        "executed_qk_pairs": (len(full_heads) * batch * length * length +
                              executed_local_pairs),
        "padding_qk_pairs": (len(full_heads) * batch * length * length +
                             executed_local_pairs - sum(head_pairs)),
        "attention_kernel_calls": int(bool(full_heads)) + int(bool(local_heads)),
        "backend": "sta_protected_saswa_flex",
    }
