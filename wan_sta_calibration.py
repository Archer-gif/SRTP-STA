"""Sampled real-activation mask search for Wan 1.3B STA.

Scores candidate sliding-tile masks against the full-attention output without
storing full QKV activations. Losses are estimates over fixed sampled query
tiles, not the original STA paper's full-output 16-prompt calibration.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import prod, sqrt
from pathlib import Path

import torch

from saswa import _tile_tokens, _window_tile_ids


DEFAULT_CANDIDATES = ((1, 2, 2), (2, 2, 2), (3, 3, 3), (5, 3, 5))


@torch.no_grad()
def sampled_candidate_errors(q, k, v, teacher, *, token_grid,
                             candidates=DEFAULT_CANDIDATES,
                             sample_tiles=8):
    """Return per-head (SSE, teacher energy) for candidate masks.

    Inputs are unpadded [1,seq,12,128], after Wan's QK norm and RoPE.
    Tiles are spread deterministically across the full video grid.
    """
    tile_size = (3, 5, 4)
    if q.shape != k.shape or q.shape != v.shape or q.shape != teacher.shape:
        raise ValueError("QKV/teacher shapes differ")
    if q.shape[0] != 1 or q.shape[1] != prod(token_grid) or q.shape[2:] != (12, 128):
        raise ValueError("expected unpadded Wan 1.3B QKV")
    if any(n % t for n, t in zip(token_grid, tile_size)):
        raise ValueError("token grid must divide into STA tiles")
    tile_grid = tuple(n // t for n, t in zip(token_grid, tile_size))
    if any(any(w > g for w, g in zip(window, tile_grid)) for window in candidates):
        raise ValueError("STA candidate window exceeds tile grid")
    tile_tokens = _tile_tokens(tile_grid, tile_size, q.device)
    count = len(tile_tokens)
    ids = sorted({min(count - 1, (i * count + count // 2) // sample_tiles)
                  for i in range(sample_tiles)})
    qh, kh, vh, th = [x[0].permute(1, 0, 2).float()
                      for x in (q, k, v, teacher)]
    losses = {window: torch.zeros(12, 2, dtype=torch.float64)
              for window in candidates}
    for tile_id in ids:
        center = (tile_id // (tile_grid[1] * tile_grid[2]),
                  tile_id // tile_grid[2] % tile_grid[1],
                  tile_id % tile_grid[2])
        query_ids = tile_tokens[tile_id]
        target = th[:, query_ids]
        energy = target.square().sum(dim=(1, 2)).double().cpu()
        for window in candidates:
            key_tiles = _window_tile_ids(center, window, tile_grid)
            key_ids = torch.cat([tile_tokens[key] for key in key_tiles])
            logits = torch.matmul(qh[:, query_ids], kh[:, key_ids].transpose(-1, -2)) / sqrt(128)
            result = torch.matmul(logits.softmax(-1), vh[:, key_ids])
            losses[window][:, 0] += (result - target).square().sum(dim=(1, 2)).double().cpu()
            losses[window][:, 1] += energy
    return losses


def write_search_rows(path: str | Path, accumulated: dict, *, prompt_count: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "layer", "head", "window", "sse", "teacher_energy",
                         "prompt_count"))
        for (step, layer, head, window), (sse, energy) in sorted(accumulated.items()):
            writer.writerow((step, layer, head, ",".join(map(str, window)),
                             sse, energy, prompt_count))


def select_strategy(csv_paths: list[Path], *, total_steps: int, keep_full_steps: int,
                    max_relative_mse: float, out: Path) -> dict:
    """Select the smallest passing mask per step/layer/head, full otherwise."""
    if not 0 <= keep_full_steps <= total_steps or max_relative_mse < 0:
        raise ValueError("invalid calibration budget")
    totals = defaultdict(lambda: [0.0, 0.0])
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key = (int(row["step"]), int(row["layer"]), int(row["head"]),
                       tuple(int(n) for n in row["window"].split(",")))
                totals[key][0] += float(row["sse"])
                totals[key][1] += float(row["teacher_energy"])
    masks = [[[] for _ in range(30)] for _ in range(total_steps)]
    missing = 0
    for step in range(total_steps):
        for layer in range(30):
            for head in range(12):
                if step < keep_full_steps:
                    masks[step][layer].append("full")
                    continue
                candidates = []
                for window in DEFAULT_CANDIDATES:
                    observed = totals.get((step, layer, head, window))
                    if observed and observed[1] > 0:
                        candidates.append((prod(window), window,
                                           observed[0] / observed[1]))
                if len(candidates) != len(DEFAULT_CANDIDATES):
                    missing += 1
                    masks[step][layer].append("full")
                    continue
                passing = sorted((cost, window) for cost, window, error in candidates
                                 if error <= max_relative_mse)
                masks[step][layer].append(list(passing[0][1]) if passing else "full")
    if missing:
        raise ValueError(f"calibration is missing {missing} step/layer/head entries")
    strategy = {"model": "Wan2.1-T2V-1.3B", "total_steps": total_steps,
                "tile_size": [3, 5, 4], "calibrated": True,
                "keep_full_steps": keep_full_steps,
                "max_relative_mse": max_relative_mse,
                "calibration_files": [str(path) for path in csv_paths],
                "candidates": [list(w) for w in DEFAULT_CANDIDATES],
                "masks": masks}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strategy, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return strategy
