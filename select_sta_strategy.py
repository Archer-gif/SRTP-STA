"""Select a Wan 1.3B STA mask table from completed search CSV files."""

import argparse
from pathlib import Path

from wan_sta_calibration import select_strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("search_csv", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--keep-full-steps", type=int, default=12)
    parser.add_argument("--max-relative-mse", type=float, default=0.1)
    args = parser.parse_args()
    strategy = select_strategy(
        args.search_csv, total_steps=args.steps,
        keep_full_steps=args.keep_full_steps,
        max_relative_mse=args.max_relative_mse, out=args.out)
    local = sum(mask != "full" for step in strategy["masks"]
                for layer in step for mask in layer)
    print(f"saved={args.out}, local_head_settings={local}")


if __name__ == "__main__":
    main()
