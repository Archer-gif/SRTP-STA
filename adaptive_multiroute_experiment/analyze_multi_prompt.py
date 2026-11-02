"""Analyze independent Full/Adaptive production prompt pairs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

experiment_root = Path(__file__).resolve().parents[1] / "correspondence_sta_experiment"
sys.path.insert(0, str(experiment_root / "code" if
                       (experiment_root / "code").is_dir() else experiment_root))
from analyze_benchmark import make_contact_sheet, quality, read_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = ("sports_car", "station_walk", "ocean_lighthouse")
    rows = []
    videos = {}
    for name in names:
        directory = args.benchmark_dir / name
        full = read_frames(directory / "full.mp4")
        sta = read_frames(directory / "sta_production.mp4")
        adaptive = read_frames(directory / "adaptive_production.mp4")
        sta_metrics = quality(sta, full)
        adaptive_metrics = quality(adaptive, full)
        attention = json.loads((directory / "adaptive_production.attention.json")
                               .read_text(encoding="utf-8"))
        sta_attention = json.loads((directory / "sta_production.attention.json")
                                   .read_text(encoding="utf-8"))
        row = {
            "name": name,
            "prompt": (directory / "prompt.txt").read_text(encoding="utf-8").strip(),
            **{f"sta_{key}": value for key, value in sta_metrics.items()},
            **{f"adaptive_{key}": value for key, value in adaptive_metrics.items()},
            "full_wall_seconds": float(
                (directory / "full.wall_seconds.txt").read_text()),
            "sta_wall_seconds": float(
                (directory / "sta_production.wall_seconds.txt").read_text()),
            "adaptive_wall_seconds": float(
                (directory / "adaptive_production.wall_seconds.txt").read_text()),
            "sta_denoise_seconds": float(sta_attention["denoise_seconds"]),
            "adaptive_denoise_seconds": float(attention["denoise_seconds"]),
        }
        row["wall_speedup_percent"] = 100 * (
            1 - row["adaptive_wall_seconds"] / row["full_wall_seconds"])
        row["adaptive_wall_vs_sta_percent"] = 100 * (
            row["adaptive_wall_seconds"] / row["sta_wall_seconds"] - 1)
        row["adaptive_ssim_minus_sta"] = (
            row["adaptive_ssim_to_full"] - row["sta_ssim_to_full"])
        row["adaptive_psnr_minus_sta"] = (
            row["adaptive_psnr_db"] - row["sta_psnr_db"])
        rows.append(row)
        videos[f"{name} - Full"] = full
        videos[f"{name} - STA"] = sta
        videos[f"{name} - Adaptive"] = adaptive

    metric_names = (
        "sta_rmse_8bit", "sta_psnr_db", "sta_ssim_to_full",
        "sta_temporal_ssim", "adaptive_rmse_8bit", "adaptive_psnr_db",
        "adaptive_ssim_to_full", "adaptive_temporal_ssim",
        "full_wall_seconds", "sta_wall_seconds", "adaptive_wall_seconds",
        "sta_denoise_seconds", "adaptive_denoise_seconds",
        "wall_speedup_percent", "adaptive_wall_vs_sta_percent",
        "adaptive_ssim_minus_sta", "adaptive_psnr_minus_sta")
    aggregate = {
        metric: {
            "mean": statistics.mean(row[metric] for row in rows),
            "min": min(row[metric] for row in rows),
            "max": max(row[metric] for row in rows),
        }
        for metric in metric_names
    }
    report = {"samples": rows, "aggregate": aggregate}
    (args.out_dir / "multi_prompt_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "multi_prompt_metrics.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(videos, args.out_dir / "multi_prompt_comparison.png")


if __name__ == "__main__":
    main()
