"""Grouped PyTorch SDPA backend for default pooled, full-budget SASWA.

This batches query tiles with the same actual key length, so one SDPA call
handles many (batch, head, tile) groups. It keeps the reference algorithm's
clipped borders, pooled direction choice, and final-window softmax semantics.
It is an intermediate GPU backend; measure it before claiming speedups.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from math import prod, sqrt

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from saswa import SASWAConfig, StageConfig, _window_tile_ids, _pool_tokens_by_tile


@lru_cache(maxsize=12)
def _geometry(grid: tuple[int, int, int], tile: tuple[int, int, int],
              stage: StageConfig, device_name: str):
    device = torch.device(device_name)
    tile_grid = tuple(n // t for n, t in zip(grid, tile))
    token_ids = torch.arange(prod(grid), dtype=torch.long).reshape(grid)
    tile_tokens = []
    centers = []
    for t in range(tile_grid[0]):
        for h in range(tile_grid[1]):
            for w in range(tile_grid[2]):
                centers.append((t, h, w))
                tile_tokens.append(tuple(token_ids[
                    t * tile[0]:(t + 1) * tile[0],
                    h * tile[1]:(h + 1) * tile[1],
                    w * tile[2]:(w + 1) * tile[2]].reshape(-1).tolist()))
    window_keys = []
    unions = []
    gains = []
    for center in centers:
        candidates = [_window_tile_ids(center, window, tile_grid)
                      for window in stage.windows]
        window_keys.append(tuple(
            tuple(token for key_tile in ids for token in tile_tokens[key_tile])
            for ids in candidates))
        union = sorted(set().union(*candidates))
        unions.append(union)
        base = set(candidates[0])
        gains.append([[key in set(ids) - base for key in union]
                      for ids in candidates[1:]])
    max_union = max(map(len, unions))
    union_ids = torch.zeros(len(unions), max_union, dtype=torch.long, device=device)
    valid = torch.zeros(len(unions), max_union, dtype=torch.bool, device=device)
    gain_mask = torch.zeros(len(unions), 3, max_union, dtype=torch.bool, device=device)
    for index, union in enumerate(unions):
        width = len(union)
        union_ids[index, :width] = torch.tensor(union, device=device)
        valid[index, :width] = True
        gain_mask[index, :, :width] = torch.tensor(gains[index], device=device)
    return (tuple(tile_tokens), tuple(window_keys), union_ids, valid,
            gain_mask, sum(map(len, unions)))


@torch.no_grad()
def saswa_grouped_attention(q: Tensor, k: Tensor, v: Tensor, features: Tensor,
                            config: SASWAConfig, step: int, total_steps: int,
                            *, groups_per_call: int = 32) -> tuple[Tensor, dict]:
    if q.ndim != 6 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("QKV must share [B,H,T,H,W,D]")
    batch, heads, frames, height, width, dim = q.shape
    grid = (frames, height, width)
    if features.shape[:4] != (batch, *grid) or features.device != q.device:
        raise ValueError("feature shape/device does not match QKV")
    if any(n % t for n, t in zip(grid, config.tile_size)):
        raise ValueError("token grid must divide into tiles")
    if not 0 <= step < total_steps or groups_per_call < 1:
        raise ValueError("invalid step or groups_per_call")
    stage_id = min(step * len(config.stages) // total_steps, len(config.stages) - 1)
    stage = config.stages[stage_id]
    tile_grid = tuple(n // t for n, t in zip(grid, config.tile_size))
    if any(any(w > g for w, g in zip(window, tile_grid)) for window in stage.windows):
        raise ValueError("SASWA window exceeds tile grid")
    if (config.routing_mode != "pooled" or stage.expand_fraction != 1.0
            or stage.threshold != 0.0):
        raise ValueError("grouped backend supports pooled routing, rho=1, threshold=0")
    (tile_tokens, window_keys, union_ids, valid,
     gain_mask, union_count) = _geometry(grid, config.tile_size, stage,
                                        str(q.device))
    pooled_q = _pool_tokens_by_tile(q, tile_grid, config.tile_size)
    pooled_k = _pool_tokens_by_tile(k, tile_grid, config.tile_size)
    key_means = pooled_k[:, :, union_ids]
    logits = (pooled_q.unsqueeze(-2) * key_means).sum(-1) / sqrt(dim)
    probabilities = logits.masked_fill(~valid[None, None], -torch.inf).softmax(-1)
    scores = (probabilities.unsqueeze(-2) * gain_mask[None, None]).sum(-1)
    best_gain, best_direction = scores.max(-1)
    chosen = torch.where(best_gain > 0, best_direction + 1, 0)

    n_tokens = prod(grid)
    q_flat, k_flat, v_flat = [x.reshape(batch, heads, n_tokens, dim)
                             for x in (q, k, v)]
    output = torch.empty_like(q_flat)
    buckets = defaultdict(list)
    selected_cpu = chosen.cpu().tolist()
    for b in range(batch):
        for head in range(heads):
            for tile_id, direction in enumerate(selected_cpu[b][head]):
                keys = window_keys[tile_id][direction]
                buckets[len(keys)].append((b, head, tile_id, keys))
    final_pairs = 0
    for key_count, group in buckets.items():
        for start in range(0, len(group), groups_per_call):
            part = group[start:start + groups_per_call]
            b_idx = torch.tensor([item[0] for item in part], device=q.device)
            h_idx = torch.tensor([item[1] for item in part], device=q.device)
            q_idx = torch.tensor([tile_tokens[item[2]] for item in part], device=q.device)
            k_idx = torch.tensor([item[3] for item in part], device=q.device)
            query = q_flat[b_idx[:, None], h_idx[:, None], q_idx]
            key = k_flat[b_idx[:, None], h_idx[:, None], k_idx]
            value = v_flat[b_idx[:, None], h_idx[:, None], k_idx]
            result = scaled_dot_product_attention(
                query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1))[:, 0]
            output[b_idx[:, None], h_idx[:, None], q_idx] = result
            final_pairs += len(part) * prod(config.tile_size) * key_count
    diagnostics = {"stage": stage_id, "routing_mode": "pooled",
                   "chosen": chosen, "scores": scores,
                   "candidate_qk_pairs": union_count * batch * heads,
                   "final_qk_pairs": final_pairs,
                   "backend": "grouped_sdpa"}
    return output.reshape_as(q), diagnostics
