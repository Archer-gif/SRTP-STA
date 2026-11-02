"""Build conservative STA strategies that demote only robustly local full heads."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SAFE_WINDOW = [5, 3, 5]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--calibration-glob", required=True)
    parser.add_argument("--mean-limit", type=float, required=True)
    parser.add_argument("--p90-limit", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    strategy = json.loads(args.base.read_text(encoding="utf-8"))
    observations: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    calibration_files = sorted(glob.glob(args.calibration_glob))
    for filename in calibration_files:
        with open(filename, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["window"] != "5,3,5":
                    continue
                key = int(row["step"]), int(row["layer"]), int(row["head"])
                observations[key].append(
                    float(row["sse"]) / float(row["teacher_energy"]))

    demoted = []
    for step in range(strategy["keep_full_steps"], strategy["total_steps"]):
        for layer in range(30):
            for head in range(12):
                if strategy["masks"][step][layer][head] != "full":
                    continue
                values = np.asarray(observations[(step, layer, head)])
                if len(values) != len(calibration_files):
                    raise RuntimeError(f"missing calibration values for {(step, layer, head)}")
                mean = float(values.mean())
                p90 = float(np.quantile(values, 0.9))
                if mean <= args.mean_limit and p90 <= args.p90_limit:
                    strategy["masks"][step][layer][head] = SAFE_WINDOW
                    demoted.append({"step": step, "layer": layer, "head": head,
                                    "mean_relative_mse": mean, "p90_relative_mse": p90})

    strategy["selective_full_demotion"] = {
        "base_strategy": str(args.base), "window": SAFE_WINDOW,
        "mean_limit": args.mean_limit, "p90_limit": args.p90_limit,
        "count": len(demoted), "entries": demoted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(strategy, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: demoted {len(demoted)} full heads")


if __name__ == "__main__":
    main()
