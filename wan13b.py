"""Wan2.1-T2V-1.3B geometry and adapter for the SASWA reference kernel.

The configuration below is an experimental starting point, not a measured
optimum or a reproduction of STA's HunyuanVideo window search.
"""

from __future__ import annotations

from math import prod
from typing import Literal

from torch import Tensor

from saswa import SASWAConfig, StageConfig, _window_tile_ids, saswa_attention


WAN13B_VAE_STRIDE = (4, 8, 8)
WAN13B_PATCH_SIZE = (1, 2, 2)
WAN13B_HEADS = 12
WAN13B_HEAD_DIM = 128  # model dim 1536 / 12 heads


def wan13b_token_grid(
    frames: int = 81, height: int = 480, width: int = 832
) -> tuple[int, int, int]:
    """Return (time, height, width) after Wan VAE and DiT patch embedding.

    Wan's causal VAE keeps the first temporal position; later frames are
    compressed by four. Spatial VAE stride is 8 and DiT patch stride is 2.
    """
    if frames < 1 or (frames - 1) % 4 or height % 16 or width % 16:
        raise ValueError("Expected 4n+1 frames and spatial dimensions divisible by 16")
    return ((frames - 1) // 4 + 1, height // 16, width // 16)


def make_wan13b_config(routing_mode: Literal["pooled", "exact"] = "pooled") -> SASWAConfig:
    """Three-stage pilot schedule for Wan's default 81-frame 480x832 case.

    A tile is 3x5x4 DiT tokens (60 query tokens), giving a 7x6x13 tile grid.
    Each directional extension adds the same number of key tiles in its stage.
    Every eligible tile keeps its best directional expansion by default.
    Detail weights are inert when rho=1 and threshold=0; retain them for
    comparisons with a restricted budget or positive threshold.
    """
    return SASWAConfig(
        tile_size=(3, 5, 4),
        stages=(
            StageConfig(
                base=(3, 3, 3), time=(5, 3, 3),
                height=(3, 5, 3), width=(3, 3, 5),
                expand_fraction=1.0, detail_lambda=0.0,
            ),
            StageConfig(
                base=(2, 2, 2), time=(4, 2, 2),
                height=(2, 4, 2), width=(2, 2, 4),
                expand_fraction=1.0, detail_lambda=0.5,
            ),
            StageConfig(
                base=(1, 2, 2), time=(2, 2, 2),
                height=(1, 4, 2), width=(1, 2, 4),
                expand_fraction=1.0, detail_lambda=1.0,
            ),
        ),
        # Placeholder values: calibrate from actual hidden-feature differences.
        spatial_scale=1.0,
        temporal_scale=1.0,
        spatial_weight=0.5,
        routing_mode=routing_mode,
    )


def wan13b_geometry_report(config: SASWAConfig | None = None) -> dict:
    """Summarize token and tile counts without allocating Q/K/V tensors."""
    config = config or make_wan13b_config()
    grid = wan13b_token_grid()
    if any(n % t for n, t in zip(grid, config.tile_size)):
        raise ValueError("The chosen tile_size does not divide Wan's token grid")
    tiles = tuple(n // t for n, t in zip(grid, config.tile_size))
    if any(any(w > g for w, g in zip(candidate, tiles))
           for stage in config.stages for candidate in stage.windows):
        raise ValueError("A candidate window exceeds the Wan tile grid")
    stages = []
    tile_centers = [
        (t, h, w)
        for t in range(tiles[0]) for h in range(tiles[1]) for w in range(tiles[2])
    ]
    tokens_per_tile = prod(config.tile_size)
    for stage in config.stages:
        base_keys = prod(stage.base) * tokens_per_tile
        expanded_keys = prod(stage.time) * tokens_per_tile
        union_keys = (sum(prod(w) for w in stage.windows)
                      - 3 * prod(stage.base)) * tokens_per_tile
        base_tile_counts = []
        union_tile_counts = []
        max_extra_tile_counts = []
        for center in tile_centers:
            candidates = [set(_window_tile_ids(center, window, tiles)) for window in stage.windows]
            base_tile_counts.append(len(candidates[0]))
            union_tile_counts.append(len(set().union(*candidates)))
            max_extra_tile_counts.append(max(len(ids) - len(candidates[0]) for ids in candidates[1:]))
        mean_base_keys = sum(base_tile_counts) / len(tile_centers) * tokens_per_tile
        mean_union_keys = sum(union_tile_counts) / len(tile_centers) * tokens_per_tile
        routing_pairs_per_query = (
            mean_union_keys / (tokens_per_tile ** 2)
            if config.routing_mode == "pooled" else mean_union_keys
        )
        max_extended = int(stage.expand_fraction * prod(tiles))
        mean_final_keys = mean_base_keys + (
            sum(sorted(max_extra_tile_counts, reverse=True)[:max_extended])
            / len(tile_centers) * tokens_per_tile
        )
        stages.append({
            "base_tokens": tuple(w * t for w, t in zip(stage.base, config.tile_size)),
            "base_key_count": base_keys,
            "extension_key_count": expanded_keys,
            "candidate_union_key_count": union_keys,
            "mean_base_key_count_after_clipping": mean_base_keys,
            "mean_candidate_union_key_count_after_clipping": mean_union_keys,
            "routing_qk_pairs_per_query_token_upper": routing_pairs_per_query,
            "routing_mode": config.routing_mode,
            "max_extended_tiles": max_extended,
            "mean_final_keys_at_full_budget": mean_final_keys,
            "score_plus_final_qk_ratio_to_fixed_base":
                (routing_pairs_per_query + mean_final_keys) / mean_base_keys,
        })
    return {
        "video_token_grid": grid,
        "video_token_count": prod(grid),
        "tile_size_tokens": config.tile_size,
        "tile_grid": tiles,
        "tile_count": prod(tiles),
        "stages": stages,
    }


def saswa_wan13b_flat_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    features: Tensor,
    *,
    step: int,
    total_steps: int,
    config: SASWAConfig | None = None,
    token_grid: tuple[int, int, int] | None = None,
    backend: Literal["reference", "grouped", "flex"] = "reference",
) -> tuple[Tensor, dict]:
    """Adapter for Wan self-attention tensors [batch, seq, heads, head_dim].

    Q and K must already have Wan's QK normalization and 3D RoPE applied.
    ``features`` must be the self-attention input [batch, seq, 1536]. Do not
    pass text cross-attention tensors. Padded sequences are not supported.
    Returns [batch, seq, heads, head_dim], before Wan's output projection.
    """
    config = config or make_wan13b_config()
    token_grid = token_grid or wan13b_token_grid()
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Wan q/k/v must share [batch, seq, heads, head_dim]")
    batch, sequence, heads, head_dim = q.shape
    if sequence != prod(token_grid) or heads != WAN13B_HEADS or head_dim != WAN13B_HEAD_DIM:
        raise ValueError("Q/K/V do not match the supplied Wan 1.3B token grid and heads")
    if features.shape != (batch, sequence, heads * head_dim):
        raise ValueError("features must be [batch, seq, 1536]")
    q_video = q.permute(0, 2, 1, 3).reshape(batch, heads, *token_grid, head_dim)
    k_video = k.permute(0, 2, 1, 3).reshape_as(q_video)
    v_video = v.permute(0, 2, 1, 3).reshape_as(q_video)
    x_video = features.reshape(batch, *token_grid, heads * head_dim)
    if backend not in ("reference", "grouped", "flex"):
        raise ValueError("unknown SASWA backend")
    if backend == "flex":
        from saswa_flex import saswa_flex_attention
        attention = saswa_flex_attention
    elif backend == "grouped":
        from saswa_grouped import saswa_grouped_attention
        attention = saswa_grouped_attention
    else:
        attention = saswa_attention
    output, diagnostics = attention(
        q_video, k_video, v_video, x_video, config, step, total_steps
    )
    return output.reshape(batch, heads, sequence, head_dim).permute(0, 2, 1, 3), diagnostics
