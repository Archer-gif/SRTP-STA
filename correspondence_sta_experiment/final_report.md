# TC-STA final experiment report

## Verdict

The recommended direction is TC-STA (confidence-gated temporal
correspondence STA), not further tuning of the original progressive schedule.
On the paired Wan2.1-T2V-1.3B sample, TC-STA is 17.47% faster than full
attention and materially closer to the full-attention output than both STA
0.1 and Hybrid 0.1. It costs 5.36% more wall time than the stable STA-Flex
baseline and 3.40% more than the earlier Hybrid 0.1 run.

This is a successful engineering result, not yet a publication-scale quality
claim: there is one complete paired video benchmark, supported by real-QKV
probes on two additional prompts and multiple denoising steps/layers.

## Method

The base calibration is the existing STA MSE-0.1 strategy. Its per-step,
per-layer, per-head `full` decisions are never weakened. Its local windows are
used as the minimum context.

Wan geometry in this experiment:

- 50 denoising steps, 30 layers, 12 heads, two CFG model calls per step.
- Token grid: 21 x 30 x 52 = 32,760 tokens.
- Tile: 3 x 5 x 4 = 60 tokens; tile grid: 7 x 6 x 13 = 546 tiles.
- Head dimension: 128; Q/K are RMS-normalized and RoPE-applied before pooling.

### Small-head online routing

Heads assigned by STA to `(1,2,2)` or `(2,2,2)` are routed independently for
every query tile. Three layouts compete:

1. `(1,2,2)` base plus at most four temporal anchors from the previous two
   temporal tiles and a spatial radius of one tile.
2. Height expansion `(1,4,2)`.
3. Width expansion `(1,2,4)`.

Pooled QK attention mass chooses the route. This is input-, CFG-, step-,
layer-, head-, and tile-dependent; it is not a fixed late-step expansion.

### Confidence-gated anchors

STA-local heads with `(3,3,3)` or `(5,3,5)` retain that complete base window.
For each query tile, candidates come from the previous three temporal tiles
within spatial radius two. Tiles already present in the STA base are removed.

Candidate score:

`s(q,c) = <mean_tile(Q_q), mean_tile(K_c)> / sqrt(128)`

The top eight candidates are considered, but candidate `c` is accepted only
when:

`s(q,c) >= mean(base_scores) + 0.5 * std(base_scores)`

Consequently, each head/tile receives zero to eight anchors. In the complete
run, only 12,060,714 of 43,159,544 eligible top-k slots were accepted (27.94%).
Final attention is still token-level FlexAttention; pooling is used only to
select blocks.

### Late-detail guard

At steps 35-49 and layers 25-29, appended fixed-head anchors are disabled.
Small-head routing remains active. This guard was added after unconditional
top-k anchors improved most probes but regressed deep layers 25 and 29 at
step 40 on the sports-car prompt.

### Kernel organization

Full heads use one dense SDPA call. All local heads use one stable-shape
FlexAttention call, padded to 12 head slots to prevent Dynamo recompilation.
The original root STA-Flex backend changed BlockMask/head metadata shapes and
OOMed at step 12 after falling back toward dense execution; the isolated
stable-shape baseline fixes this without modifying root code.

## Final parameters

| Parameter | Value |
|---|---:|
| STA strategy | MSE threshold 0.1 |
| Dynamic-small STA modes | `(1,2,2)`, `(2,2,2)` |
| Small-head temporal anchors | 4 |
| Small-head temporal lags / radius | 2 / 1 |
| Fixed-head maximum anchors | 8 |
| Fixed-head temporal lags / radius | 3 / 2 |
| Rank score | pooled RoPE QK |
| Confidence sigma | 0.5 |
| Late guard | step >= 35 and layer >= 25 |
| Flex block / valid tokens | 64 / 60 |

## Paired benchmark

Prompt: `A 3D model of a 1800s victorian house.` Seed 1234, 81 frames,
832 x 480, 50 steps, guidance 6, shift 8, one A800 80 GB.

| Method | Wall (s) | QK vs full | RMSE to full | PSNR | SSIM | Temporal SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Full attention | 376.996 | 100.000% | 0 | inf | 1 | - |
| STA 0.1 stable Flex | 295.310 | 51.274% | 51.822 | 13.840 | 0.5298 | 0.9497 |
| Hybrid 0.1 | 300.928 | 51.340% | 47.726 | 14.556 | 0.5619 | 0.9620 |
| TC-STA | 311.150 | 51.462% | 43.743 | 15.313 | 0.5874 | 0.9621 |

Relative to STA, TC-STA reduces RMSE by 15.59%, gains 1.47 dB PSNR, and
improves SSIM by 10.87% relative while adding 0.188 percentage points of
logical QK and 5.36% wall time. Relative to Hybrid 0.1, it reduces RMSE by
8.35% for 3.40% more wall time.

The kernel executes 52.575% of dense QK after head padding, versus 52.303%
for stable STA. Candidate selection evaluates 345,336,480 pooled tile scores,
only 0.000894% of the dense token-pair count. The measured overhead is mainly
pooling, indexing, mask construction, and extra sparse blocks, not the pooled
dot products themselves.

## Ablations and rejected paths

- Directional motion-aligned expansion changed QK and quality too little to
  justify continuing it.
- Feature-only correspondence ranking was consistently worse than pooled QK.
- Unconditional top-8 anchors improved many layers but caused prompt-dependent
  regressions in late deep layers.
- Similarity gating alone did not eliminate every regression; the late-detail
  guard was required.
- Sigma 1.0 was cheaper but discarded too much of the measured quality gain;
  sigma 0.5 was the selected balance.

## Difference from STA and Hybrid

STA uses an offline teacher calibration to freeze one mode for every
step/layer/head; the mask does not depend on the inference sample. TC-STA keeps
STA's full-head protection but changes local connectivity online for every
tile using the current Q/K correspondence. The accepted anchor count and
locations therefore vary with content.

Hybrid 0.1 routes only STA-small heads among three fixed geometric windows.
TC-STA adds explicit cross-time object-neighborhood anchors outside the base
window, confidence-gates them, and protects late deep-layer refinement. This
online correspondence path is the substantive mechanism beyond STA/Hybrid.

## Artifacts

- `benchmark_victorian/tc_sta.mp4`: final generated video.
- `benchmark_victorian/sta_flex.mp4`: stable-shape STA baseline.
- `benchmark_victorian/analysis/comparison_frames.png`: five-frame comparison.
- `benchmark_victorian/analysis/quality_speed_chart.png`: metric chart.
- `benchmark_victorian/analysis/layer_head_attention.md`: 30 x 12 average QK ratios.
- `benchmark_victorian/analysis/step_layer_head_attention.csv`: every one of
  the 50 x 30 x 12 ratios and route counts.
- `benchmark_victorian/tc_sta.attention.json`: raw per-step/layer/head metrics.

## Limitations

The full video benchmark is one prompt and should be expanded before making a
paper-level generalization claim. RMSE/PSNR/SSIM measure closeness to the full
teacher, not semantic preference. A next evaluation should use several motion
classes and VBench/FVD-style metrics. The current Python/PyTorch routing is
not fused; a fused pooling, top-k, and mask-builder kernel should recover part
of the 12.57-second denoising overhead.
