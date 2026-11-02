"""STA-protected attention with local cross-time correspondence anchors."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from math import prod, sqrt

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.functional import scaled_dot_product_attention
from torch.profiler import record_function

from saswa import _pool_tokens_by_tile, _window_tile_ids
from saswa_flex import BLOCK, KERNEL_OPTIONS, _block_mask, _compiled_flex, _layout, _pack
from wan13b import WAN13B_HEAD_DIM, WAN13B_HEADS
from wan_hybrid_flex import DEFAULT_DYNAMIC_SETTINGS, PRODUCTION_WINDOWS


Window = tuple[int, int, int]
BASE_WINDOW = (1, 2, 2)
HEIGHT_WINDOW = (1, 4, 2)
WIDTH_WINDOW = (1, 2, 4)


def _profile_phase(name: str, enabled: bool):
    return record_function(name) if enabled else nullcontext()


@lru_cache(maxsize=32)
def _index_range(length: int, device_name: str) -> Tensor:
    return torch.arange(length, device=torch.device(device_name))


def _packed_block_mask(indices: Tensor, counts: Tensor, batch: int,
                       heads: int, tiles: int) -> BlockMask:
    """Build a dynamic BlockMask directly from packed KV block IDs."""
    indices = indices.expand(batch, heads, -1, -1).contiguous()
    counts = counts.expand(batch, heads, -1).contiguous()

    def mask_mod(_batch: Tensor, _head: Tensor, _query: Tensor,
                 key: Tensor) -> Tensor:
        return key % BLOCK < 60

    return BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=(BLOCK, BLOCK), mask_mod=mask_mod,
        seq_lengths=(tiles * BLOCK, tiles * BLOCK), compute_q_blocks=False)


@lru_cache(maxsize=8)
def _anchor_geometry(grid: tuple[int, int, int], tile: tuple[int, int, int],
                     device_name: str, temporal_lags: int, radius: int):
    tile_grid = tuple(n // width for n, width in zip(grid, tile))
    gt, gh, gw = tile_grid
    tiles = prod(tile_grid)
    _, indices, counts, adjacency, _, _ = _layout(
        grid, tile, (BASE_WINDOW, HEIGHT_WINDOW, WIDTH_WINDOW), device_name)
    candidates: list[list[int]] = []
    unions: list[list[int]] = []
    for query in range(tiles):
        qt = query // (gh * gw)
        qh = query // gw % gh
        qw = query % gw
        local = []
        for lag in range(1, temporal_lags + 1):
            kt = qt - lag
            if kt < 0:
                continue
            for dh in range(-radius, radius + 1):
                for dw in range(-radius, radius + 1):
                    kh, kw = qh + dh, qw + dw
                    if 0 <= kh < gh and 0 <= kw < gw:
                        local.append((kt * gh + kh) * gw + kw)
        candidates.append(local)
        unions.append(sorted(set(local).union(
            *[set(torch.where(adjacency[variant, query].cpu())[0].tolist())
              for variant in range(3)])))

    max_candidates = max(map(len, candidates))
    max_union = max(map(len, unions))
    device = torch.device(device_name)
    candidate_ids = torch.zeros(tiles, max_candidates, dtype=torch.long, device=device)
    candidate_valid = torch.zeros(tiles, max_candidates, dtype=torch.bool, device=device)
    union_ids = torch.zeros(tiles, max_union, dtype=torch.long, device=device)
    union_valid = torch.zeros(tiles, max_union, dtype=torch.bool, device=device)
    candidate_positions = torch.zeros(tiles, max_candidates, dtype=torch.long, device=device)
    static_membership = torch.zeros(3, tiles, max_union, dtype=torch.bool, device=device)
    for query, (local, union) in enumerate(zip(candidates, unions)):
        candidate_ids[query, :len(local)] = torch.tensor(local, device=device)
        candidate_valid[query, :len(local)] = True
        union_ids[query, :len(union)] = torch.tensor(union, device=device)
        union_valid[query, :len(union)] = True
        position = {key: index for index, key in enumerate(union)}
        if local:
            candidate_positions[query, :len(local)] = torch.tensor(
                [position[key] for key in local], device=device)
        for variant in range(3):
            keys = torch.where(adjacency[variant, query].cpu())[0].tolist()
            static_membership[variant, query, [position[key] for key in keys]] = True
    return (indices, counts, candidate_ids, candidate_valid, union_ids,
            union_valid, candidate_positions, static_membership, tile_grid)


@lru_cache(maxsize=12)
def _temporal_candidates(grid: tuple[int, int, int], tile: tuple[int, int, int],
                         device_name: str, temporal_lags: int, radius: int):
    tile_grid = tuple(n // width for n, width in zip(grid, tile))
    gt, gh, gw = tile_grid
    tiles = prod(tile_grid)
    rows = []
    for query in range(tiles):
        qt = query // (gh * gw)
        qh = query // gw % gh
        qw = query % gw
        keys = []
        for lag in range(1, temporal_lags + 1):
            kt = qt - lag
            if kt < 0:
                continue
            for dh in range(-radius, radius + 1):
                for dw in range(-radius, radius + 1):
                    kh, kw = qh + dh, qw + dw
                    if 0 <= kh < gh and 0 <= kw < gw:
                        keys.append((kt * gh + kh) * gw + kw)
        rows.append(keys)
    width = max(map(len, rows))
    device = torch.device(device_name)
    ids = torch.zeros(tiles, width, dtype=torch.long, device=device)
    valid = torch.zeros(tiles, width, dtype=torch.bool, device=device)
    for query, keys in enumerate(rows):
        ids[query, :len(keys)] = torch.tensor(keys, device=device)
        valid[query, :len(keys)] = True
    return ids, valid, tile_grid


def _pool_features(features: Tensor, tile_grid: tuple[int, int, int],
                   tile: tuple[int, int, int]) -> Tensor:
    batch, _, _, _, channels = features.shape
    gt, gh, gw = tile_grid
    dt, dh, dw = tile
    return (features.reshape(batch, gt, dt, gh, dh, gw, dw, channels)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(batch, gt * gh * gw, dt * dh * dw, channels)
            .float().mean(-2))


def _compact_indices(ids: Tensor, valid: Tensor, tiles: int) -> tuple[Tensor, Tensor]:
    marked = torch.where(valid, ids, torch.full_like(ids, tiles))
    ordered = marked.sort(-1).values
    counts = valid.sum(-1, dtype=torch.int32)
    return ordered.clamp_max(tiles - 1).to(torch.int32), counts


@torch.no_grad()
def correspondence_hybrid_flex_attention(
    q: Tensor, k: Tensor, v: Tensor, features: Tensor, *,
    step: int, layer: int, strategy: dict,
    token_grid: tuple[int, int, int],
    dynamic_settings: tuple[Window, ...] = DEFAULT_DYNAMIC_SETTINGS,
    temporal_lags: int = 2, radius: int = 1, anchor_count: int = 4,
    rank_mode: str = "qk", feature_weight: float = 0.5,
    fixed_anchor_count: int = 0, fixed_temporal_lags: int = 3,
    fixed_radius: int = 2, fixed_gate_sigma: float | None = None,
    route_policy: str = "top1", route_margin_high: float = 0.20,
    route_margin_low: float = 0.08,
    collect_diagnostics: bool = True, production_optimized: bool = False,
    optimize_route_union: bool | None = None,
    optimize_block_mask: bool | None = None,
    profile_ranges: bool = False,
) -> tuple[Tensor, dict]:
    """Use local semantic anchors only inside STA-approved small heads."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,L,H,D]")
    batch, length, heads, dim = q.shape
    if (length != prod(token_grid) or heads != WAN13B_HEADS
            or dim != WAN13B_HEAD_DIM):
        raise ValueError("QKV does not match Wan 1.3B geometry")
    if rank_mode not in ("qk", "feature", "combined"):
        raise ValueError("rank_mode must be qk, feature, or combined")
    if route_policy not in ("top1", "adaptive"):
        raise ValueError("route_policy must be top1 or adaptive")
    if not 0 <= route_margin_low <= route_margin_high:
        raise ValueError("route confidence margins are inconsistent")
    if optimize_route_union is None:
        optimize_route_union = production_optimized
    if optimize_block_mask is None:
        optimize_block_mask = production_optimized
    tile = tuple(strategy["tile_size"])
    settings = strategy["masks"][step][layer]
    full_heads = [i for i, setting in enumerate(settings) if setting == "full"]
    local_heads = [i for i, setting in enumerate(settings) if setting != "full"]
    dynamic_set = {tuple(window) for window in dynamic_settings}
    dynamic_heads = [i for i in local_heads if tuple(settings[i]) in dynamic_set]
    fixed_heads = [i for i in local_heads if i not in dynamic_heads]
    qv, kv, vv = [x.permute(0, 2, 1, 3).reshape(batch, heads, *token_grid, dim)
                  for x in (q, k, v)]
    output = torch.empty_like(qv)
    head_pairs = [0] * heads
    route_histograms = [[0, 0, 0] for _ in range(heads)]
    route_cardinality_histograms = [[0, 0, 0] for _ in range(heads)]
    candidate_pairs = 0
    accepted_fixed_anchors = 0
    eligible_fixed_anchor_slots = 0

    if full_heads:
        with _profile_phase("tcsta/dense_attention", profile_ranges):
            dense = [x[:, full_heads].reshape(batch, len(full_heads), length, dim)
                     for x in (qv, kv, vv)]
            result = scaled_dot_product_attention(*dense)
            output[:, full_heads] = result.reshape(
                batch, len(full_heads), *token_grid, dim)
        for head in full_heads:
            head_pairs[head] = batch * length * length

    executed_local_pairs = 0
    if local_heads:
        stable_windows = tuple(window for window in (*PRODUCTION_WINDOWS, BASE_WINDOW)
                               if all(width <= size // tile_width
                                      for width, size, tile_width in
                                      zip(window, token_grid, tile)))
        (packed_ids, window_indices, window_counts, window_adjacency,
         tile_grid, _) = _layout(
            token_grid, tile, stable_windows, str(q.device))
        tiles = prod(tile_grid)
        base_max_keys = window_indices.shape[-1]
        max_keys = base_max_keys + fixed_anchor_count
        if route_policy == "adaptive" and dynamic_heads:
            max_keys = max(max_keys, 3 * (base_max_keys + anchor_count))
        selected_ids = torch.zeros(batch, len(local_heads), tiles, max_keys,
                                   dtype=torch.int32, device=q.device)
        selected_counts = torch.zeros(batch, len(local_heads), tiles,
                                      dtype=torch.int32, device=q.device)
        local_position = {head: index for index, head in enumerate(local_heads)}
        window_position = {window: index for index, window in enumerate(stable_windows)}

        for head in fixed_heads:
            position = local_position[head]
            variant = window_position[tuple(settings[head])]
            selected_ids[:, position, :, :base_max_keys] = window_indices[variant]
            selected_counts[:, position] = window_counts[variant]

        if fixed_heads and fixed_anchor_count:
            fixed_phase = _profile_phase(
                "tcsta/fixed_anchor_routing", profile_ranges)
            fixed_phase.__enter__()
            fixed_positions = [local_position[head] for head in fixed_heads]
            fixed_variants = torch.tensor(
                [window_position[tuple(settings[head])] for head in fixed_heads],
                dtype=torch.long, device=q.device)
            fixed_base_ids = window_indices[fixed_variants]
            fixed_base_counts = window_counts[fixed_variants]
            fixed_base_adjacency = window_adjacency[fixed_variants]
            temporal_ids, temporal_valid, fixed_tile_grid = _temporal_candidates(
                token_grid, tile, str(q.device), fixed_temporal_lags, fixed_radius)
            if fixed_tile_grid != tile_grid:
                raise AssertionError("fixed anchor geometry mismatch")
            temporal_expanded = temporal_ids[None].expand(len(fixed_heads), -1, -1)
            eligible = (temporal_valid[None] &
                        ~fixed_base_adjacency.gather(-1, temporal_expanded))
            with _profile_phase("tcsta/fixed_pool_qk", profile_ranges):
                fixed_q = _pool_tokens_by_tile(qv[:, fixed_heads], tile_grid, tile)
                fixed_k = _pool_tokens_by_tile(kv[:, fixed_heads], tile_grid, tile)
            with _profile_phase("tcsta/fixed_candidate_qk", profile_ranges):
                ranking = ((fixed_q.unsqueeze(-2) * fixed_k[:, :, temporal_ids]).sum(-1)
                           / sqrt(dim)).masked_fill(~eligible[None], -torch.inf)
            with _profile_phase("tcsta/fixed_anchor_topk", profile_ranges):
                top_count = min(fixed_anchor_count, temporal_ids.shape[-1])
                top_positions = ranking.topk(top_count, dim=-1).indices
                top_valid = eligible[None].expand(batch, -1, -1, -1).gather(
                    -1, top_positions)
            if collect_diagnostics:
                eligible_fixed_anchor_slots = int(top_valid.sum().item())
            with _profile_phase("tcsta/fixed_gate", profile_ranges):
                if fixed_gate_sigma is not None:
                    head_ids = torch.arange(
                        len(fixed_heads), device=q.device)[:, None, None]
                    base_k = fixed_k[:, head_ids, fixed_base_ids]
                    base_scores = ((fixed_q.unsqueeze(-2) * base_k).sum(-1) /
                                   sqrt(dim))
                    base_valid = (
                        _index_range(base_max_keys, str(q.device))[None, None] <
                        fixed_base_counts[..., None])
                    denominator = base_valid.sum(-1).clamp_min(1)[None]
                    base_mean = (base_scores * base_valid[None]).sum(-1) / denominator
                    centered = (base_scores - base_mean[..., None]).masked_fill(
                        ~base_valid[None], 0)
                    base_std = (centered.square().sum(-1) / denominator).sqrt()
                    threshold = base_mean + fixed_gate_sigma * base_std
                    top_scores = ranking.gather(-1, top_positions)
                    top_valid &= top_scores >= threshold[..., None]
                top_ids = temporal_ids[None, None].expand(
                    batch, len(fixed_heads), -1, -1).gather(-1, top_positions)
            if collect_diagnostics:
                accepted_fixed_anchors = int(top_valid.sum().item())
            with _profile_phase("tcsta/fixed_route_union", profile_ranges):
                expanded_base = fixed_base_ids[None].expand(batch, -1, -1, -1)
                base_valid = (
                    _index_range(base_max_keys, str(q.device))[None, None, None] <
                    fixed_base_counts[None, :, :, None])
                augmented_ids, augmented_counts = _compact_indices(
                    torch.cat((expanded_base, top_ids), -1),
                    torch.cat((base_valid.expand_as(expanded_base), top_valid), -1),
                    tiles)
            for fixed_index, position in enumerate(fixed_positions):
                selected_ids[:, position, :, :augmented_ids.shape[-1]] = augmented_ids[:, fixed_index]
                selected_counts[:, position] = augmented_counts[:, fixed_index]
            if collect_diagnostics:
                candidate_pairs += int(
                    batch * len(fixed_heads) * temporal_valid.sum().item())
            fixed_phase.__exit__(None, None, None)

        if dynamic_heads:
            dynamic_phase = _profile_phase(
                "tcsta/dynamic_routing_total", profile_ranges)
            dynamic_phase.__enter__()
            (small_indices, small_counts, temporal_ids, temporal_valid, union_ids,
             union_valid, temporal_positions, static_membership,
             anchor_tile_grid) = _anchor_geometry(
                token_grid, tile, str(q.device), temporal_lags, radius)
            if anchor_tile_grid != tile_grid:
                raise AssertionError("anchor geometry mismatch")
            with _profile_phase("tcsta/dynamic_pool_qk", profile_ranges):
                pooled_q = _pool_tokens_by_tile(
                    qv[:, dynamic_heads], tile_grid, tile)
                pooled_k = _pool_tokens_by_tile(
                    kv[:, dynamic_heads], tile_grid, tile)
            with _profile_phase("tcsta/dynamic_candidate_qk", profile_ranges):
                logits = ((pooled_q.unsqueeze(-2) * pooled_k[:, :, union_ids]).sum(-1)
                          / sqrt(dim))
                probabilities = logits.masked_fill(
                    ~union_valid[None, None], -torch.inf).softmax(-1)
                temporal_logits = logits.gather(
                    -1, temporal_positions[None, None].expand(
                        batch, len(dynamic_heads), -1, -1))

            if rank_mode in ("feature", "combined"):
                pooled_features = torch.nn.functional.normalize(
                    _pool_features(features, tile_grid, tile), dim=-1)
                feature_scores = (pooled_features.unsqueeze(-2) *
                                  pooled_features[:, temporal_ids]).sum(-1)
                if rank_mode == "feature":
                    ranking = feature_scores[:, None].expand(
                        -1, len(dynamic_heads), -1, -1)
                else:
                    centered = temporal_logits - temporal_logits.masked_fill(
                        ~temporal_valid[None, None], 0).sum(-1, keepdim=True) / (
                            temporal_valid.sum(-1).clamp_min(1)[None, None, :, None])
                    ranking = centered + feature_weight * feature_scores[:, None]
            else:
                ranking = temporal_logits
            with _profile_phase("tcsta/dynamic_anchor_topk", profile_ranges):
                ranking = ranking.masked_fill(
                    ~temporal_valid[None, None], -torch.inf)
                top_count = min(anchor_count, temporal_ids.shape[-1])
                top_positions = ranking.topk(top_count, dim=-1).indices
                top_valid = temporal_valid[None, None].expand(
                    batch, len(dynamic_heads), -1, -1).gather(-1, top_positions)
                top_ids = temporal_ids[None, None].expand(
                    batch, len(dynamic_heads), -1, -1).gather(-1, top_positions)
                top_union_positions = temporal_positions[None, None].expand(
                    batch, len(dynamic_heads), -1, -1).gather(-1, top_positions)

            base_width = small_indices.shape[-1]
            base_ids = small_indices[0][None, None].expand(
                batch, len(dynamic_heads), -1, -1)
            base_valid = (_index_range(base_width, str(q.device))[None, None, None] <
                          small_counts[0][None, None, :, None])
            temporal_choice_ids, temporal_choice_counts = _compact_indices(
                torch.cat((base_ids, top_ids), dim=-1),
                torch.cat((base_valid.expand_as(base_ids), top_valid), dim=-1), tiles)

            anchor_membership = torch.zeros(
                batch, len(dynamic_heads), tiles, union_ids.shape[-1],
                dtype=torch.int8, device=q.device)
            anchor_membership.scatter_add_(
                -1, top_union_positions, top_valid.to(torch.int8))
            dynamic_membership = (
                static_membership[0][None, None].expand(
                    batch, len(dynamic_heads), -1, -1) |
                anchor_membership.bool())
            route_membership = torch.stack((
                dynamic_membership,
                static_membership[1][None, None].expand(
                    batch, len(dynamic_heads), -1, -1),
                static_membership[2][None, None].expand(
                    batch, len(dynamic_heads), -1, -1)), dim=2)
            route_scores = (probabilities.unsqueeze(2) * route_membership).sum(-1)
            route = route_scores.argmax(2)

            route_width = max(base_max_keys, temporal_choice_ids.shape[-1])
            route_ids = torch.zeros(batch, len(dynamic_heads), 3, tiles,
                                    route_width,
                                    dtype=torch.int32, device=q.device)
            route_counts = torch.zeros(batch, len(dynamic_heads), 3, tiles,
                                       dtype=torch.int32, device=q.device)
            route_ids[:, :, 0, :, :temporal_choice_ids.shape[-1]] = temporal_choice_ids
            route_counts[:, :, 0] = temporal_choice_counts
            for route_index, window in enumerate((HEIGHT_WINDOW, WIDTH_WINDOW), start=1):
                variant = window_position[window]
                route_ids[:, :, route_index, :, :base_max_keys] = window_indices[variant]
                route_counts[:, :, route_index] = window_counts[variant]
            positions = _index_range(tiles, str(q.device))[None, None, :]
            if route_policy == "top1":
                chosen_ids = route_ids[
                    torch.arange(batch, device=q.device)[:, None, None],
                    torch.arange(len(dynamic_heads), device=q.device)[None, :, None],
                    route, positions]
                chosen_counts = route_counts[
                    torch.arange(batch, device=q.device)[:, None, None],
                    torch.arange(len(dynamic_heads), device=q.device)[None, :, None],
                    route, positions]
                route_cardinality = torch.ones_like(route)
            else:
                sorted_scores, route_order = route_scores.sort(2, descending=True)
                confidence = ((sorted_scores[:, :, 0] - sorted_scores[:, :, 1]) /
                              sorted_scores[:, :, 0].abs().clamp_min(1e-6))
                route_cardinality = (1 + (confidence < route_margin_high).long() +
                                     (confidence < route_margin_low).long())
                ranks = _index_range(3, str(q.device))[None, None, :, None]
                ordered_enabled = ranks < route_cardinality[:, :, None]
                route_enabled = torch.zeros_like(ordered_enabled)
                route_enabled.scatter_(2, route_order, ordered_enabled)
                option_valid = (
                    _index_range(route_width, str(q.device))[None, None, None, None] <
                    route_counts[..., None])
                flat_ids = route_ids.permute(0, 1, 3, 2, 4).reshape(
                    batch, len(dynamic_heads), tiles, -1)
                flat_valid = (option_valid & route_enabled[..., None]).permute(
                    0, 1, 3, 2, 4).reshape(batch, len(dynamic_heads), tiles, -1)
                if optimize_route_union:
                    with _profile_phase(
                            "tcsta/dynamic_route_union", profile_ranges):
                        marked = torch.where(
                            flat_valid, flat_ids, torch.full_like(flat_ids, tiles))
                        ordered = marked.sort(-1).values
                        unique_valid = ordered < tiles
                        unique_valid[..., 1:] &= (
                            ordered[..., 1:] != ordered[..., :-1])
                        compact_ids, chosen_counts = _compact_indices(
                            ordered.clamp_max(tiles - 1), unique_valid, tiles)
                        chosen_ids = compact_ids[..., :max_keys]
                else:
                    union_counts = torch.zeros(
                        batch, len(dynamic_heads), tiles, tiles,
                        dtype=torch.int8, device=q.device)
                    union_counts.scatter_add_(
                        -1, flat_ids.long(), flat_valid.to(torch.int8))
                    union_adjacency = union_counts.bool()
                    tile_ids = _index_range(
                        tiles, str(q.device))[None, None, None]
                    chosen_ids = torch.where(
                        union_adjacency, tile_ids,
                        torch.full_like(tile_ids, tiles)
                    ).sort(-1).values[..., :max_keys].clamp_max(
                        tiles - 1).to(torch.int32)
                    chosen_counts = union_adjacency.sum(-1, dtype=torch.int32)
            for dynamic_index, head in enumerate(dynamic_heads):
                position = local_position[head]
                selected_ids[:, position, :, :chosen_ids.shape[-1]] = chosen_ids[:, dynamic_index]
                selected_counts[:, position] = chosen_counts[:, dynamic_index]
                if collect_diagnostics:
                    histogram = torch.bincount(
                        route[:, dynamic_index].reshape(-1), minlength=3)
                    route_histograms[head] = [
                        int(value) for value in histogram.tolist()]
                    cardinality = torch.bincount(
                        route_cardinality[:, dynamic_index].reshape(-1) - 1,
                        minlength=3)
                    route_cardinality_histograms[head] = [
                        int(value) for value in cardinality.tolist()]
            candidate_pairs += int(batch * len(dynamic_heads) * tiles * union_ids.shape[-1])
            dynamic_phase.__exit__(None, None, None)

        padded_heads = local_heads + [local_heads[0]] * (heads - len(local_heads))
        if len(local_heads) < heads:
            pad = heads - len(local_heads)
            selected_ids = torch.cat((selected_ids, selected_ids[:, :1].expand(-1, pad, -1, -1)), 1)
            selected_counts = torch.cat((selected_counts, selected_counts[:, :1].expand(-1, pad, -1)), 1)
        if optimize_block_mask:
            with _profile_phase("tcsta/block_mask_build", profile_ranges):
                block_mask = _packed_block_mask(
                    selected_ids, selected_counts, batch, heads, tiles)
        else:
            valid_blocks = (
                _index_range(max_keys, str(q.device))[None, None, None] <
                selected_counts[..., None])
            adjacency_counts = torch.zeros(batch, heads, tiles, tiles,
                                           dtype=torch.int8, device=q.device)
            adjacency_counts.scatter_add_(
                -1, selected_ids.long(), valid_blocks.to(torch.int8))
            adjacency = adjacency_counts.bool()
            block_mask = _block_mask(selected_ids, selected_counts, adjacency,
                                     batch, heads, tiles)
        with _profile_phase("tcsta/pack_qkv", profile_ranges):
            local = [x[:, padded_heads] for x in (qv, kv, vv)]
            packed_qkv = [_pack(value, packed_ids) for value in local]
        with _profile_phase("tcsta/flex_attention", profile_ranges):
            result = _compiled_flex(q.device.type)(
                *packed_qkv, block_mask=block_mask,
                kernel_options=KERNEL_OPTIONS)
        unpacked = torch.empty(batch, len(local_heads), length, dim,
                               device=q.device, dtype=q.dtype)
        with _profile_phase("tcsta/unpack_output", profile_ranges):
            unpacked.index_copy_(
                2, packed_ids.reshape(tiles, BLOCK)[:, :60].reshape(-1),
                result[:, :len(local_heads)].reshape(
                    batch, len(local_heads), tiles, BLOCK, dim)[:, :, :, :60]
                .reshape(batch, len(local_heads), length, dim))
            output[:, local_heads] = unpacked.reshape(
                batch, len(local_heads), *token_grid, dim)
        if collect_diagnostics:
            actual_counts = selected_counts[:, :len(local_heads)]
            per_head = actual_counts.sum((0, 2)) * (60 * 60)
            for local_index, head in enumerate(local_heads):
                head_pairs[head] = int(per_head[local_index].item())
            executed_local_pairs = int(selected_counts.sum().item()) * 60 * 60

    flat = output.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
    return flat, {
        "step": step, "layer": layer, "full_heads": len(full_heads),
        "fixed_heads": len(fixed_heads), "dynamic_heads": len(dynamic_heads),
        "head_modes": ["full" if value == "full" else str(tuple(value))
                       for value in settings],
        "head_qk_pairs": head_pairs, "route_histograms": route_histograms,
        "route_cardinality_histograms": route_cardinality_histograms,
        "candidate_qk_pairs": candidate_pairs, "final_qk_pairs": sum(head_pairs),
        "accepted_fixed_anchors": accepted_fixed_anchors,
        "eligible_fixed_anchor_slots": eligible_fixed_anchor_slots,
        "executed_qk_pairs": len(full_heads) * batch * length * length + executed_local_pairs,
        "attention_kernel_calls": int(bool(full_heads)) + int(bool(local_heads)),
        "backend": f"correspondence_sta_{rank_mode}_fixedk{fixed_anchor_count}",
    }


@torch.no_grad()
def sta_temporal_anchor_flex_attention(
    q: Tensor, k: Tensor, v: Tensor, features: Tensor, *,
    step: int, layer: int, strategy: dict,
    token_grid: tuple[int, int, int], anchor_count: int = 4,
    temporal_lags: int = 3, radius: int = 2,
    rank_mode: str = "qk",
) -> tuple[Tensor, dict]:
    """Keep every STA window and append a few local cross-time anchors."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,L,H,D]")
    batch, length, heads, dim = q.shape
    if length != prod(token_grid) or heads != 12 or dim != 128:
        raise ValueError("QKV does not match Wan 1.3B geometry")
    if rank_mode not in ("qk", "feature") or anchor_count < 1:
        raise ValueError("invalid anchor mode or count")
    tile = tuple(strategy["tile_size"])
    settings = strategy["masks"][step][layer]
    full_heads = [i for i, setting in enumerate(settings) if setting == "full"]
    local_heads = [i for i, setting in enumerate(settings) if setting != "full"]
    qv, kv, vv = [x.permute(0, 2, 1, 3).reshape(batch, heads, *token_grid, dim)
                  for x in (q, k, v)]
    output = torch.empty_like(qv)
    head_pairs = [0] * heads

    if full_heads:
        dense = [x[:, full_heads].reshape(batch, len(full_heads), length, dim)
                 for x in (qv, kv, vv)]
        result = scaled_dot_product_attention(*dense)
        output[:, full_heads] = result.reshape(batch, len(full_heads), *token_grid, dim)
        for head in full_heads:
            head_pairs[head] = batch * length * length

    candidate_pairs = 0
    executed_local_pairs = 0
    if local_heads:
        windows = tuple(dict.fromkeys(tuple(settings[head]) for head in local_heads))
        packed_ids, indices, counts, base_adjacency, tile_grid, _ = _layout(
            token_grid, tile, windows, str(q.device))
        tiles = prod(tile_grid)
        choice = torch.tensor([windows.index(tuple(settings[head])) for head in local_heads],
                              dtype=torch.long, device=q.device)
        base_ids = indices[choice]
        base_counts = counts[choice]
        selected_base_adjacency = base_adjacency[choice]
        candidate_ids, candidate_valid, candidate_grid = _temporal_candidates(
            token_grid, tile, str(q.device), temporal_lags, radius)
        if candidate_grid != tile_grid:
            raise AssertionError("candidate geometry mismatch")
        candidate_ids_expanded = candidate_ids[None].expand(len(local_heads), -1, -1)
        already_inside = selected_base_adjacency.gather(-1, candidate_ids_expanded)
        eligible = candidate_valid[None] & ~already_inside

        pooled_q = _pool_tokens_by_tile(qv[:, local_heads], tile_grid, tile)
        pooled_k = _pool_tokens_by_tile(kv[:, local_heads], tile_grid, tile)
        qk_scores = ((pooled_q.unsqueeze(-2) * pooled_k[:, :, candidate_ids]).sum(-1)
                     / sqrt(dim))
        if rank_mode == "feature":
            pooled_features = torch.nn.functional.normalize(
                _pool_features(features, tile_grid, tile), dim=-1)
            feature_scores = (pooled_features.unsqueeze(-2) *
                              pooled_features[:, candidate_ids]).sum(-1)
            ranking = feature_scores[:, None].expand(-1, len(local_heads), -1, -1)
        else:
            ranking = qk_scores
        ranking = ranking.masked_fill(~eligible[None], -torch.inf)
        top_count = min(anchor_count, candidate_ids.shape[-1])
        top_positions = ranking.topk(top_count, dim=-1).indices
        top_valid = eligible[None].expand(batch, -1, -1, -1).gather(-1, top_positions)
        top_ids = candidate_ids[None, None].expand(
            batch, len(local_heads), -1, -1).gather(-1, top_positions)

        base_width = base_ids.shape[-1]
        expanded_base_ids = base_ids[None].expand(batch, -1, -1, -1)
        base_valid = (torch.arange(base_width, device=q.device)[None, None, None] <
                      base_counts[None, :, :, None])
        selected_ids, selected_counts = _compact_indices(
            torch.cat((expanded_base_ids, top_ids), dim=-1),
            torch.cat((base_valid.expand_as(expanded_base_ids), top_valid), dim=-1), tiles)
        max_keys = selected_ids.shape[-1]
        valid_blocks = (torch.arange(max_keys, device=q.device)[None, None, None] <
                        selected_counts[..., None])
        adjacency_counts = torch.zeros(batch, len(local_heads), tiles, tiles,
                                       dtype=torch.int8, device=q.device)
        adjacency_counts.scatter_add_(-1, selected_ids.long(), valid_blocks.to(torch.int8))
        adjacency = adjacency_counts.bool()
        padded_heads = local_heads + [local_heads[0]] * (heads - len(local_heads))
        if len(local_heads) < heads:
            pad = heads - len(local_heads)
            selected_ids = torch.cat((selected_ids, selected_ids[:, :1].expand(-1, pad, -1, -1)), 1)
            selected_counts = torch.cat((selected_counts, selected_counts[:, :1].expand(-1, pad, -1)), 1)
            adjacency = torch.cat((adjacency, adjacency[:, :1].expand(-1, pad, -1, -1)), 1)
        block_mask = _block_mask(selected_ids, selected_counts, adjacency,
                                 batch, heads, tiles)
        local = [x[:, padded_heads] for x in (qv, kv, vv)]
        result = _compiled_flex(q.device.type)(
            _pack(local[0], packed_ids), _pack(local[1], packed_ids),
            _pack(local[2], packed_ids), block_mask=block_mask,
            kernel_options=KERNEL_OPTIONS)
        unpacked = torch.empty(batch, len(local_heads), length, dim,
                               dtype=q.dtype, device=q.device)
        unpacked.index_copy_(
            2, packed_ids.reshape(tiles, BLOCK)[:, :60].reshape(-1),
            result[:, :len(local_heads)].reshape(
                batch, len(local_heads), tiles, BLOCK, dim)[:, :, :, :60]
            .reshape(batch, len(local_heads), length, dim))
        output[:, local_heads] = unpacked.reshape(batch, len(local_heads), *token_grid, dim)
        actual_counts = selected_counts[:, :len(local_heads)]
        per_head = actual_counts.sum((0, 2)) * (60 * 60)
        for local_index, head in enumerate(local_heads):
            head_pairs[head] = int(per_head[local_index].item())
        executed_local_pairs = int(selected_counts.sum().item()) * 60 * 60
        candidate_pairs = int(batch * len(local_heads) * candidate_valid.sum().item())

    flat = output.reshape(batch, heads, length, dim).permute(0, 2, 1, 3)
    return flat, {
        "step": step, "layer": layer, "full_heads": len(full_heads),
        "local_heads": len(local_heads), "anchor_count": anchor_count,
        "candidate_qk_pairs": candidate_pairs, "head_qk_pairs": head_pairs,
        "final_qk_pairs": sum(head_pairs),
        "executed_qk_pairs": len(full_heads) * batch * length * length + executed_local_pairs,
        "attention_kernel_calls": int(bool(full_heads)) + int(bool(local_heads)),
        "backend": f"sta_temporal_anchor_{rank_mode}_k{anchor_count}",
    }
