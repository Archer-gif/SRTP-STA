"""Derive conservative-to-fast STA mask variants from one 16-prompt search."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from functools import lru_cache
from pathlib import Path

from saswa import _window_tile_ids
from wan_sta_calibration import select_strategy


TILE_GRID = (7, 6, 13)
TOKENS_PER_TILE = 60
TOTAL_TOKENS = 32760
THRESHOLDS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.5)


@lru_cache(maxsize=8)
def pairs_per_head(mask: tuple[int, int, int] | str) -> int:
    if mask == "full":
        return TOTAL_TOKENS**2
    return sum(TOKENS_PER_TILE**2 * len(_window_tile_ids(
        (t, h, w), mask, TILE_GRID))
        for t in range(TILE_GRID[0])
        for h in range(TILE_GRID[1])
        for w in range(TILE_GRID[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = sorted(args.calibration_dir.glob("prompt_*.csv"))
    if len(inputs) != 16:
        parser.error(f"expected 16 completed calibration CSVs, found {len(inputs)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    full_cost = 50 * 30 * 12 * pairs_per_head("full")
    for threshold in THRESHOLDS:
        label = str(threshold).replace(".", "p")
        path = args.out_dir / f"sta_strategy_mse_{label}.json"
        strategy = select_strategy(inputs, total_steps=50, keep_full_steps=12,
                                   max_relative_mse=threshold, out=path)
        counts = Counter(tuple(mask) if isinstance(mask, list) else mask
                         for step in strategy["masks"] for layer in step
                         for mask in layer)
        estimated_cost = sum(count * pairs_per_head(mask)
                             for mask, count in counts.items())
        rows.append({"threshold": threshold, "full_head_settings": counts["full"],
                     "local_head_settings": 18000 - counts["full"],
                     "qk_ratio_to_full": estimated_cost / full_cost,
                     "strategy": str(path)})
    summary = args.out_dir / "sta_strategy_sweep.csv"
    with summary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(summary)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
