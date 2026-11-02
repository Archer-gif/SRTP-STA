"""Run reproducible Wan 1.3B STA mask search on separate calibration prompts.

Dry-run by default. Full run uses the unchanged Wan global attention output
as the teacher and saves sampled-query candidate losses per prompt. It does
not launch final evaluation or VBench scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from wan_sta_calibration import select_strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, default=Path("/root/wan_srtp/Wan2.1"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/model/HuggingFace/Wan-AI/Wan2.1-T2V-1.3B"))
    parser.add_argument("--python", type=Path,
                        default=Path("/usr/local/miniconda3/envs/py312/bin/python"))
    parser.add_argument("--bridge", type=Path,
                        default=Path("/root/wan_srtp/run_wan13b_saswa.py"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--keep-full-steps", type=int, default=12)
    parser.add_argument("--max-relative-mse", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.prompts = args.prompts.resolve()
    args.out_dir = args.out_dir.resolve()
    args.wan_repo = args.wan_repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.python = args.python.resolve()
    args.bridge = args.bridge.resolve()
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not prompts or args.steps <= 0 or not 0 <= args.keep_full_steps < args.steps:
        parser.error("invalid prompts or step counts")
    if args.limit:
        prompts = prompts[:args.limit]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.bridge.parent), str(args.wan_repo),
         environment.get("PYTHONPATH", "")])
    search_files = []
    for index, prompt in enumerate(prompts):
        job = f"prompt_{index:02d}"
        csv_path = args.out_dir / f"{job}.csv"
        log_path = args.out_dir / f"{job}.log"
        video_path = args.out_dir / f"{job}.mp4"
        metadata_path = args.out_dir / f"{job}.json"
        expected = {"prompt": prompt, "seed": 1234 + index,
                    "steps": args.steps, "keep_full_steps": args.keep_full_steps,
                    "shift": 8, "guide": 6, "checkpoint": str(args.checkpoint)}
        search_files.append(csv_path)
        command = [str(args.python), str(args.bridge), "--saswa-mode", "sta_search",
                   "--sta-search-out", str(csv_path), "--sta-search-skip-steps",
                   str(args.keep_full_steps), "--wan-repo", str(args.wan_repo),
                   "--task", "t2v-1.3B", "--size", "832*480", "--frame_num", "81",
                   "--sample_steps", str(args.steps), "--sample_guide_scale", "6",
                   "--sample_shift", "8", "--base_seed", str(1234 + index),
                   "--offload_model", "True", "--t5_cpu", "--ckpt_dir",
                   str(args.checkpoint), "--prompt", prompt,
                   "--save_file", str(video_path)]
        print(job, hashlib.sha256(prompt.encode()).hexdigest()[:12],
              "execute=" + str(args.execute), flush=True)
        if not args.execute:
            if index == 0:
                print("first command:", subprocess.list2cmdline(command))
            continue
        args.out_dir.mkdir(parents=True, exist_ok=True)
        if metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (previous == expected and csv_path.is_file()
                    and sum(1 for _ in csv_path.open(encoding="utf-8")) ==
                    1 + (args.steps - args.keep_full_steps) * 30 * 12 * 4):
                print("existing complete calibration sample; keeping", csv_path, flush=True)
                continue
        with log_path.open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=args.wan_repo, env=environment,
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        if (code or not csv_path.is_file() or
                sum(1 for _ in csv_path.open(encoding="utf-8")) !=
                1 + (args.steps - args.keep_full_steps) * 30 * 12 * 4):
            raise RuntimeError(f"STA calibration sample failed: {log_path}")
        metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    if not args.execute:
        return
    strategy_path = args.out_dir / "sta_wan13b_strategy.json"
    strategy = select_strategy(
        search_files, total_steps=args.steps,
        keep_full_steps=args.keep_full_steps,
        max_relative_mse=args.max_relative_mse, out=strategy_path)
    summary = {"prompt_count": len(prompts), "steps": args.steps,
               "full_first_steps": args.keep_full_steps,
               "max_relative_mse": args.max_relative_mse,
               "local_head_settings": sum(mask != "full" for step in strategy["masks"]
                                          for layer in step for mask in layer),
               "strategy": str(strategy_path)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
