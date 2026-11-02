# TC-STA: confidence-gated temporal correspondence attention

This directory is isolated from the frozen STA, Hybrid, and Progressive
implementations in the repository root.

The final candidate keeps STA's calibrated full heads as a safety prior, then
adds input-dependent temporal correspondence in two places:

- STA-small heads route each tile among temporal-correspondence, height, and
  width layouts.
- Other local heads may append up to eight previous-time anchors, but only
  when their pooled QK score exceeds the local-window mean by 0.5 standard
  deviations.
- During the final 15 denoising steps, the deepest five layers disable the
  appended anchors so historical texture does not interfere with refinement.

See `final_report_zh.md` for the Chinese final report and `final_report.md` for
the English version. `benchmark_victorian/analysis/step_layer_head_attention.csv`
contains all 18,000 step/layer/head attention ratios from the final run.
