"""Export complete per-step/layer/head TC-STA attention accounting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LENGTH_SQUARED = 32760**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail = []
    step_rows = []
    for step_text, step in sorted(data["steps"].items(), key=lambda item: int(item[0])):
        step_index = int(step_text)
        step_rows.append({
            "step": step_index,
            "final_qk_pairs": step["final_qk_pairs"],
            "executed_qk_pairs": step["executed_qk_pairs"],
            "candidate_tile_scores": step["candidate_tile_scores"],
            "accepted_fixed_anchors": step["accepted_fixed_anchors"],
            "eligible_fixed_anchor_slots": step["eligible_fixed_anchor_slots"],
            "late_guard_calls": step["late_guard_calls"],
        })
        for layer_text, layer in sorted(
                step["layers"].items(), key=lambda item: int(item[0])):
            for head in range(12):
                calls = int(layer["calls"])
                routes = layer["route_histograms"][head]
                detail.append({
                    "step": step_index, "layer": int(layer_text), "head": head,
                    "sta_mode": layer["head_modes"][head], "cfg_calls": calls,
                    "final_qk_pairs": layer["head_qk_pairs"][head],
                    "attention_percent_of_full":
                        100 * layer["head_qk_pairs"][head] / (calls * LENGTH_SQUARED),
                    "route_temporal": routes[0], "route_height": routes[1],
                    "route_width": routes[2],
                })

    for name, rows in (("step_layer_head_attention.csv", detail),
                       ("step_summary.csv", step_rows)):
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
