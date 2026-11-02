"""Build paired video metrics, attention tables, and visual comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


LENGTH = 32760
HEADS = 12
LAYERS = 30
STEPS = 50
FULL_QK = LENGTH * LENGTH * HEADS * LAYERS * STEPS * 2


def read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != 81:
        raise RuntimeError(f"expected 81 frames in {path}, got {len(frames)}")
    return frames


def mean_ssim(left: list[np.ndarray], right: list[np.ndarray]) -> float:
    def ssim(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float64), b.astype(np.float64)
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
        sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
        sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2) /
                 ((mu_a * mu_a + mu_b * mu_b + c1) *
                  (sigma_a + sigma_b + c2)))
        return float(score.mean())
    return float(np.mean([ssim(a, b) for a, b in zip(left, right)]))


def quality(candidate: list[np.ndarray], teacher: list[np.ndarray]) -> dict:
    mse = float(np.mean([
        np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        for a, b in zip(candidate, teacher)]))
    return {
        "rmse_8bit": float(np.sqrt(mse)),
        "psnr_db": float(10 * np.log10(255.0**2 / mse)),
        "ssim_to_full": mean_ssim(candidate, teacher),
        "temporal_ssim": mean_ssim(candidate[:-1], candidate[1:]),
    }


def total(metrics: dict, key: str) -> int:
    return sum(int(step[key]) for step in metrics["steps"].values())


def make_contact_sheet(videos: dict[str, list[np.ndarray]], out: Path) -> None:
    indices = (0, 20, 40, 60, 80)
    tile_w, tile_h, label_h = 416, 240, 34
    canvas = np.full((len(videos) * (tile_h + label_h), len(indices) * tile_w, 3),
                     245, dtype=np.uint8)
    for row, (label, frames) in enumerate(videos.items()):
        y = row * (tile_h + label_h)
        cv2.putText(canvas, label, (12, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (25, 25, 25), 2, cv2.LINE_AA)
        for column, index in enumerate(indices):
            frame = cv2.resize(frames[index], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            x = column * tile_w
            canvas[y + label_h:y + label_h + tile_h, x:x + tile_w] = frame
            cv2.putText(canvas, f"frame {index}", (x + 10, y + label_h + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def write_attention_table(metrics: dict, out_dir: Path) -> None:
    accum = [[0] * HEADS for _ in range(LAYERS)]
    calls = [0] * LAYERS
    modes: list[list[str] | None] = [None] * LAYERS
    routes = [[[0, 0, 0] for _ in range(HEADS)] for _ in range(LAYERS)]
    for step in metrics["steps"].values():
        for layer_text, layer in step["layers"].items():
            index = int(layer_text)
            calls[index] += int(layer["calls"])
            modes[index] = layer["head_modes"]
            for head in range(HEADS):
                accum[index][head] += int(layer["head_qk_pairs"][head])
                routes[index][head] = [
                    old + int(new) for old, new in
                    zip(routes[index][head], layer["route_histograms"][head])]

    rows = []
    for layer in range(LAYERS):
        for head in range(HEADS):
            ratio = accum[layer][head] / (calls[layer] * LENGTH * LENGTH)
            rows.append({
                "layer": layer, "head": head, "sta_mode": modes[layer][head],
                "attention_percent_of_full": 100 * ratio,
                "route_temporal": routes[layer][head][0],
                "route_height": routes[layer][head][1],
                "route_width": routes[layer][head][2],
            })
    with (out_dir / "layer_head_attention.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| Layer | " + " | ".join(f"H{head}" for head in range(HEADS)) + " |",
             "|---:" + "|---:" * HEADS + "|"]
    for layer in range(LAYERS):
        values = [100 * accum[layer][head] / (calls[layer] * LENGTH * LENGTH)
                  for head in range(HEADS)]
        lines.append(f"| {layer} | " + " | ".join(f"{value:.2f}%" for value in values) + " |")
    (out_dir / "layer_head_attention.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--sta", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--tc-sta", type=Path, required=True)
    parser.add_argument("--sta-attention", type=Path, required=True)
    parser.add_argument("--tc-attention", type=Path, required=True)
    parser.add_argument("--sta-wall", type=float, required=True)
    parser.add_argument("--tc-wall", type=float, required=True)
    parser.add_argument("--hybrid-wall", type=float, default=300.92772483825684)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    decoded = {name: read_frames(path) for name, path in {
        "Full attention": args.full, "STA 0.1": args.sta,
        "Hybrid 0.1": args.hybrid, "TC-STA": args.tc_sta}.items()}
    teacher = decoded["Full attention"]
    qualities = {name: quality(frames, teacher) for name, frames in decoded.items()
                 if name != "Full attention"}
    sta_metrics = json.loads(args.sta_attention.read_text(encoding="utf-8"))
    tc_metrics = json.loads(args.tc_attention.read_text(encoding="utf-8"))
    sta_qk = total(sta_metrics, "final_qk_pairs")
    tc_qk = total(tc_metrics, "final_qk_pairs")
    accepted = total(tc_metrics, "accepted_fixed_anchors")
    eligible = total(tc_metrics, "eligible_fixed_anchor_slots")
    report = {
        "quality_vs_full": qualities,
        "wall_seconds": {"STA 0.1 Flex": args.sta_wall,
                         "Hybrid 0.1": args.hybrid_wall, "TC-STA": args.tc_wall},
        "qk_percent_of_full": {"STA 0.1 Flex": 100 * sta_qk / FULL_QK,
                               "TC-STA": 100 * tc_qk / FULL_QK},
        "tc_sta_candidate_tile_scores": total(tc_metrics, "candidate_tile_scores"),
        "tc_sta_anchor_acceptance": {
            "accepted": accepted, "eligible_topk_slots": eligible,
            "percent": 100 * accepted / eligible if eligible else 0.0},
        "tc_sta_denoise_seconds": tc_metrics["denoise_seconds"],
    }
    (args.out_dir / "benchmark_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(decoded, args.out_dir / "comparison_frames.png")
    write_attention_table(tc_metrics, args.out_dir)

    labels = ["STA 0.1", "Hybrid 0.1", "TC-STA"]
    colors = [(128, 114, 107), (246, 130, 59), (110, 118, 15)]
    panels = [
        ("PSNR to full (dB)", [qualities[label]["psnr_db"] for label in labels]),
        ("SSIM to full", [qualities[label]["ssim_to_full"] for label in labels]),
        ("Wall time (s)", [args.sta_wall, args.hybrid_wall, args.tc_wall]),
    ]
    chart = np.full((520, 1500, 3), 250, dtype=np.uint8)
    cv2.putText(chart, "Wan2.1-1.3B paired benchmark, 50 denoising steps",
                (330, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2,
                cv2.LINE_AA)
    for panel, (title, values) in enumerate(panels):
        x0, y0, width, height = 30 + panel * 490, 90, 450, 320
        cv2.putText(chart, title, (x0 + 80, y0 - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (30, 30, 30), 2, cv2.LINE_AA)
        low, high = min(values), max(values)
        margin = max((high - low) * 0.8, high * 0.01)
        floor, ceiling = max(0, low - margin), high + margin
        cv2.line(chart, (x0, y0 + height), (x0 + width, y0 + height),
                 (170, 170, 170), 1)
        for index, (label, value, color) in enumerate(zip(labels, values, colors)):
            bar_x = x0 + 35 + index * 140
            bar_h = int((value - floor) / max(ceiling - floor, 1e-9) * height)
            cv2.rectangle(chart, (bar_x, y0 + height - bar_h),
                          (bar_x + 80, y0 + height), color, -1)
            cv2.putText(chart, f"{value:.3g}", (bar_x + 4, y0 + height - bar_h - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1,
                        cv2.LINE_AA)
            cv2.putText(chart, label, (bar_x - 5, y0 + height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (35, 35, 35), 1,
                        cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "quality_speed_chart.png"), chart)


if __name__ == "__main__":
    main()
