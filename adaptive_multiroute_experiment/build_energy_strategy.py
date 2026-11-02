"""Demote full heads only when their error contribution to the whole layer is small."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--calibration-glob", required=True)
    parser.add_argument("--mean-layer-limit", type=float, required=True)
    parser.add_argument("--p90-layer-limit", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    strategy = json.loads(args.base.read_text(encoding="utf-8"))
    contributions: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    files = sorted(glob.glob(args.calibration_glob))

    for filename in files:
        rows = []
        with open(filename, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["window"] == "5,3,5":
                    rows.append(row)
        layer_energy = defaultdict(float)
        for row in rows:
            layer_energy[int(row["step"]), int(row["layer"])] += float(
                row["teacher_energy"])
        for row in rows:
            key = int(row["step"]), int(row["layer"]), int(row["head"])
            contributions[key].append(
                float(row["sse"]) / layer_energy[key[:2]])

    demoted = []
    for step in range(strategy["keep_full_steps"], strategy["total_steps"]):
        for layer in range(30):
            for head in range(12):
                if strategy["masks"][step][layer][head] != "full":
                    continue
                values = np.asarray(contributions[step, layer, head])
                mean = float(values.mean())
                p90 = float(np.quantile(values, 0.9))
                if mean <= args.mean_layer_limit and p90 <= args.p90_layer_limit:
                    strategy["masks"][step][layer][head] = [5, 3, 5]
                    demoted.append({"step": step, "layer": layer, "head": head,
                                    "mean_layer_contribution": mean,
                                    "p90_layer_contribution": p90})

    strategy["energy_aware_full_demotion"] = {
        "base_strategy": str(args.base), "window": [5, 3, 5],
        "mean_layer_limit": args.mean_layer_limit,
        "p90_layer_limit": args.p90_layer_limit,
        "count": len(demoted), "entries": demoted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(strategy, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: demoted {len(demoted)} low-energy full heads")


if __name__ == "__main__":
    main()
