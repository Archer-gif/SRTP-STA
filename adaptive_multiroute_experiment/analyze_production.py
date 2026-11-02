"""Validate production output against the diagnostic run and Full teacher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

experiment_root = Path(__file__).resolve().parents[1] / "correspondence_sta_experiment"
sys.path.insert(0, str(experiment_root / "code" if
                       (experiment_root / "code").is_dir() else experiment_root))
from analyze_benchmark import make_contact_sheet, quality, read_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--sta", type=Path, required=True)
    parser.add_argument("--tc", type=Path, required=True)
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = {name: read_frames(path) for name, path in {
        "Full attention": args.full,
        "STA 0.1": args.sta,
        "TC-STA": args.tc,
        "Adaptive debug": args.debug,
        "Adaptive production": args.production,
    }.items()}
    teacher = videos["Full attention"]
    report = {
        "quality_vs_full": {
            name: quality(frames, teacher) for name, frames in videos.items()
            if name != "Full attention"
        },
        "production_vs_debug": quality(
            videos["Adaptive production"], videos["Adaptive debug"]),
    }
    (args.out_dir / "production_quality.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(videos, args.out_dir / "production_comparison.png")


if __name__ == "__main__":
    main()
