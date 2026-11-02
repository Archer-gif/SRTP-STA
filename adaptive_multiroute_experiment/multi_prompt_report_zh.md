# 独立多 Prompt 小规模验证

## 设置

新增 3 条未参与 STA/head 校准的 prompt，覆盖快速运动、人物遮挡和高频水面纹理。
每条使用相同 seed 1234、832x480、81 帧、50 步。对每条分别生成 Full teacher、
优化后的 STA production 和 Adaptive production。

## 结果

| Prompt | STA SSIM | Adaptive SSIM | SSIM 增量 | STA PSNR | Adaptive PSNR | Adaptive/STA 墙钟 |
|:---|---:|---:|---:|---:|---:|---:|
| sports car | 0.54672 | 0.59210 | +0.04539 | 12.396 | 13.519 | +2.03% |
| station walk | 0.48984 | 0.51760 | +0.02776 | 12.483 | 13.229 | +3.38% |
| ocean lighthouse | 0.62055 | 0.65940 | +0.03885 | 17.329 | 18.157 | +1.92% |
| **均值** | **0.55237** | **0.58970** | **+0.03733** | **14.069** | **14.968** | **+2.44%** |

其他均值：

- RMSE：STA `52.155`，Adaptive `46.968`，降低 `9.95%`。
- Temporal SSIM：STA `0.75612`，Adaptive `0.75617`，基本不变。
- Full 墙钟：`311.05s`；STA production：`285.73s`；Adaptive：`292.70s`。
- STA 去噪：`151.16s`；Adaptive 去噪：`160.41s`，慢 `6.13%`。
- Adaptive 相对 Full 墙钟平均快 `5.90%`。

## 视觉判断

Adaptive 在三条 prompt 上均比 STA 更接近 Full，块状背景、颜色断裂和纹理污染
有所减轻，因此指标提升不是单一样本偶然结果。

但 sports-car 和 station-walk 仍有明显 tile/block 伪影，station-walk 是当前最差
样本，SSIM 只有 `0.51760`。Ocean 场景相对稳定，但后段天空和水面仍有局部异常。
因此当前方法改善了 STA 的质量-速度折中，但不能宣称已接近 Full 的视觉质量。

## 结论

- 相对公平的优化 STA，Adaptive 的质量收益在 3/3 条 prompt 上一致成立。
- 端到端墙钟只慢 1.92%-3.38%，可以称为接近 STA。
- 纯去噪慢约 6%，不能称为与 STA 完全同速。
- 下一阶段首要目标应是定位和抑制 tile 边界伪影，而不是继续扩大窗口比例。
- SSIM/PSNR 必须与逐帧视觉图一起报告；仅看平均指标会低估 station 场景的问题。

## 产物

- `benchmark_multi_prompt/analysis_sta/multi_prompt_comparison.png`
- `benchmark_multi_prompt/analysis_sta/multi_prompt_metrics.json`
- `benchmark_multi_prompt/analysis_sta/multi_prompt_metrics.csv`
- 每个 prompt 目录中的 `full.mp4`、`sta_production.mp4` 和
  `adaptive_production.mp4`
