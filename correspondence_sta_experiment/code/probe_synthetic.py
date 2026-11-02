"""Compile and validate one full-geometry correspondence layer on CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from correspondence_hybrid_flex import (
    correspondence_hybrid_flex_attention, sta_temporal_anchor_flex_attention,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--step", type=int, default=25)
    parser.add_argument("--layer", type=int, default=5)
    args = parser.parse_args()
    strategy = json.loads(args.strategy.read_text(encoding="utf-8"))
    torch.manual_seed(17)
    shape = (1, 32760, 12, 128)
    q, k, v = [torch.randn(shape, device="cuda", dtype=torch.bfloat16)
               for _ in range(3)]
    features = torch.randn(1, 21, 30, 52, 1536, device="cuda", dtype=torch.bfloat16)
    output, info = correspondence_hybrid_flex_attention(
        q, k, v, features, step=args.step, layer=args.layer,
        strategy=strategy, token_grid=(21, 30, 52), rank_mode="qk",
        fixed_anchor_count=8, fixed_gate_sigma=0.5)
    if not torch.isfinite(output).all():
        raise RuntimeError("output contains NaN/Inf")
    adaptive_output, adaptive_info = correspondence_hybrid_flex_attention(
        q, k, v, features, step=args.step, layer=args.layer,
        strategy=strategy, token_grid=(21, 30, 52), rank_mode="qk",
        fixed_anchor_count=8, fixed_gate_sigma=0.5,
        route_policy="adaptive", route_margin_high=0.35,
        route_margin_low=0.15)
    if not torch.isfinite(adaptive_output).all():
        raise RuntimeError("adaptive output contains NaN/Inf")
    sta_output, sta_info = correspondence_hybrid_flex_attention(
        q, k, v, features, step=args.step, layer=args.layer,
        strategy=strategy, token_grid=(21, 30, 52), dynamic_settings=(),
        fixed_anchor_count=0)
    if not torch.isfinite(sta_output).all():
        raise RuntimeError("stable STA output contains NaN/Inf")
    anchor_output, anchor_info = sta_temporal_anchor_flex_attention(
        q, k, v, features, step=args.step, layer=args.layer,
        strategy=strategy, token_grid=(21, 30, 52), anchor_count=4)
    if not torch.isfinite(anchor_output).all():
        raise RuntimeError("anchor output contains NaN/Inf")
    print(json.dumps({key: value for key, value in info.items()
                      if key not in ("head_qk_pairs", "route_histograms")}, indent=2))
    print(json.dumps({key: value for key, value in anchor_info.items()
                      if key != "head_qk_pairs"}, indent=2))
    print(json.dumps({key: value for key, value in sta_info.items()
                      if key not in ("head_qk_pairs", "route_histograms")}, indent=2))
    print(json.dumps({key: value for key, value in adaptive_info.items()
                      if key not in ("head_qk_pairs", "route_histograms",
                                     "route_cardinality_histograms")}, indent=2))


if __name__ == "__main__":
    main()
