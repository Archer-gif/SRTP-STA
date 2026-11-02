"""Compare four paired Wan videos against the full-attention output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "correspondence_sta_experiment" / "code"))
from analyze_benchmark import quality, read_frames  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    full = read_frames(args.result_dir / "full.mp4")
    result = {}
    for name in ("sta", "tc_sta", "adaptive"):
        result[name] = quality(read_frames(args.result_dir / f"{name}.mp4"), full)
        wall = args.result_dir / f"{name}.wall_seconds.txt"
        if wall.is_file():
            result[name]["wall_seconds"] = float(wall.read_text().strip())
        metrics = args.result_dir / f"{name}.attention.json"
        if metrics.is_file():
            result[name]["denoise_seconds"] = json.loads(
                metrics.read_text(encoding="utf-8"))["denoise_seconds"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
