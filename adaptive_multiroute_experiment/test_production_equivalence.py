"""Check the production path against the diagnostic reference path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from correspondence_hybrid_flex import correspondence_hybrid_flex_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--step", type=int, default=23)
    parser.add_argument("--layer", type=int, default=28)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda")
    grid = (21, 30, 52)
    shape = (1, 32760, 12, 128)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)
    features = torch.randn((1, *grid, 1536), device=device, dtype=torch.bfloat16)
    strategy = json.loads(args.strategy.read_text(encoding="utf-8"))

    common = dict(
        step=args.step, layer=args.layer, strategy=strategy, token_grid=grid,
        fixed_anchor_count=8, fixed_temporal_lags=3, fixed_radius=2,
        fixed_gate_sigma=0.5, route_policy="adaptive",
        route_margin_high=0.35, route_margin_low=0.15,
        collect_diagnostics=False)

    def run(route_union: bool, block_mask: bool):
        result, _ = correspondence_hybrid_flex_attention(
            q, k, v, features, optimize_route_union=route_union,
            optimize_block_mask=block_mask, **common)
        torch.cuda.synchronize()
        return result

    reference = run(False, False)
    for name, route_union, block_mask in (
            ("route_union_only", True, False),
            ("block_mask_only", False, True),
            ("production", True, True)):
        candidate = run(route_union, block_mask)
        difference = (reference.float() - candidate.float()).abs()
        print(json.dumps({
            "variant": name,
            "equal": bool(torch.equal(reference, candidate)),
            "different_values": int(torch.count_nonzero(difference).item()),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
        }))

    for optimized_flag in (False, True):
        run(optimized_flag, optimized_flag)
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            run(optimized_flag, optimized_flag)
            samples.append(1000 * (time.perf_counter() - started))
        print(json.dumps({
            "production_optimized": optimized_flag,
            "milliseconds": samples,
            "mean_milliseconds": sum(samples) / len(samples),
        }))


if __name__ == "__main__":
    main()
