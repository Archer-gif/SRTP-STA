"""Compare guarded adaptive multi-route attention with paired baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

experiment_root = Path(__file__).resolve().parents[1] / "correspondence_sta_experiment"
sys.path.insert(0, str(experiment_root / "code" if
                       (experiment_root / "code").is_dir() else experiment_root))
from analyze_benchmark import (FULL_QK, make_contact_sheet, quality, read_frames,
                               total, write_attention_table)


def cardinality(metrics: dict) -> dict:
    hist = [0, 0, 0]
    for step in metrics["steps"].values():
        for layer in step["layers"].values():
            for head in layer.get("route_cardinality_histograms", []):
                hist = [old + int(new) for old, new in zip(hist, head)]
    count = sum(hist)
    return {
        "one_route_tiles": hist[0],
        "two_route_tiles": hist[1],
        "three_route_tiles": hist[2],
        "expanded_tile_percent": 100 * (hist[1] + hist[2]) / count if count else 0,
        "average_routes": sum((index + 1) * value for index, value in enumerate(hist)) /
                          count if count else 0,
    }


def make_chart(qualities: dict, walls: dict, qk: dict, out: Path) -> None:
    labels = ["STA 0.1", "TC-STA", "Adaptive guarded", "Adaptive speed"]
    colors = [(120, 110, 105), (90, 145, 40), (205, 115, 35), (175, 70, 145)]
    panels = [
        ("SSIM to full", [qualities[name]["ssim_to_full"] for name in labels]),
        ("Wall time (s)", [walls[name] for name in labels]),
        ("QK vs full (%)", [qk[name] for name in labels]),
    ]
    chart = np.full((520, 1500, 3), 250, dtype=np.uint8)
    cv2.putText(chart, "Adaptive multi-route TC-STA paired benchmark",
                (365, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2,
                cv2.LINE_AA)
    for panel, (title, values) in enumerate(panels):
        x0, y0, width, height = 30 + panel * 490, 90, 450, 320
        cv2.putText(chart, title, (x0 + 95, y0 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (30, 30, 30), 2, cv2.LINE_AA)
        low, high = min(values), max(values)
        margin = max((high - low) * 0.8, abs(high) * 0.01, 1e-6)
        floor, ceiling = max(0, low - margin), high + margin
        cv2.line(chart, (x0, y0 + height), (x0 + width, y0 + height),
                 (170, 170, 170), 1)
        for index, (label, value, color) in enumerate(zip(labels, values, colors)):
            bar_x = x0 + 20 + index * 108
            bar_h = int((value - floor) / (ceiling - floor) * height)
            cv2.rectangle(chart, (bar_x, y0 + height - bar_h),
                          (bar_x + 64, y0 + height), color, -1)
            cv2.putText(chart, f"{value:.3f}",
                        (bar_x - 1, y0 + height - bar_h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1,
                        cv2.LINE_AA)
            short_label = label.replace("Adaptive ", "A-")
            cv2.putText(chart, short_label, (bar_x - 5, y0 + height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (35, 35, 35), 1,
                        cv2.LINE_AA)
    cv2.imwrite(str(out), chart)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--sta", type=Path, required=True)
    parser.add_argument("--tc", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--speed", type=Path, required=True)
    parser.add_argument("--sta-attention", type=Path, required=True)
    parser.add_argument("--tc-attention", type=Path, required=True)
    parser.add_argument("--adaptive-attention", type=Path, required=True)
    parser.add_argument("--speed-attention", type=Path, required=True)
    parser.add_argument("--sta-wall", type=float, required=True)
    parser.add_argument("--tc-wall", type=float, required=True)
    parser.add_argument("--adaptive-wall", type=float, required=True)
    parser.add_argument("--speed-wall", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = {name: read_frames(path) for name, path in {
        "Full attention": args.full, "STA 0.1": args.sta,
        "TC-STA": args.tc, "Adaptive guarded": args.adaptive,
        "Adaptive speed": args.speed}.items()}
    teacher = videos["Full attention"]
    qualities = {name: quality(frames, teacher) for name, frames in videos.items()
                 if name != "Full attention"}
    metrics = {
        "STA 0.1": json.loads(args.sta_attention.read_text(encoding="utf-8")),
        "TC-STA": json.loads(args.tc_attention.read_text(encoding="utf-8")),
        "Adaptive guarded": json.loads(
            args.adaptive_attention.read_text(encoding="utf-8")),
        "Adaptive speed": json.loads(args.speed_attention.read_text(encoding="utf-8")),
    }
    walls = {"STA 0.1": args.sta_wall, "TC-STA": args.tc_wall,
             "Adaptive guarded": args.adaptive_wall,
             "Adaptive speed": args.speed_wall}
    qk = {name: 100 * total(value, "final_qk_pairs") / FULL_QK
          for name, value in metrics.items()}
    report = {
        "quality_vs_full": qualities,
        "wall_seconds": walls,
        "denoise_seconds": {name: value["denoise_seconds"]
                            for name, value in metrics.items()},
        "qk_percent_of_full": qk,
        "route_cardinality": {
            name: cardinality(metrics[name])
            for name in ("Adaptive guarded", "Adaptive speed")},
        "candidate_tile_scores": {
            name: total(metrics[name], "candidate_tile_scores")
            for name in ("Adaptive guarded", "Adaptive speed")},
    }
    (args.out_dir / "benchmark_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(videos, args.out_dir / "comparison_frames.png")
    make_chart(qualities, walls, qk, args.out_dir / "quality_speed_chart.png")
    write_attention_table(metrics["Adaptive guarded"], args.out_dir)
    speed_dir = args.out_dir / "speed_attention"
    speed_dir.mkdir(exist_ok=True)
    write_attention_table(metrics["Adaptive speed"], speed_dir)


if __name__ == "__main__":
    main()
