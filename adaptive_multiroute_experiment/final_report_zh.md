# Adaptive Multi-route TC-STA 实验报告

## 1. 结论

本轮建议保留 **Adaptive speed** 作为下一阶段候选：它相对原 TC-STA 在同一
prompt、seed、分辨率和 50 步采样下，SSIM、PSNR、RMSE 和时序 SSIM 均改善，
最终有效 QK 从 Full 的 51.462% 降至 50.563%。纯去噪耗时基本持平
（+0.157 秒，+0.09%），墙钟快 1.28 秒（-0.41%）。

这不能宣称为显著加速，但已经满足“速度基本不变、质量比 TC-STA 好”，同时将
实际多路扩窗率提高到 74.20%。更保守的 Adaptive guarded 质量最好，但纯去噪
慢 0.76%，不作为首选速度版本。

## 2. 方法设计

### 2.1 保留 STA 的部分

- 模型：Wan2.1-T2V-1.3B，30 层、每层 12 个 self-attention heads。
- tile：`(3, 5, 4)`，每 tile 60 tokens。
- STA 基础策略由 16 个 calibration prompts 以 Full attention 为 teacher 得到。
- STA 允许的静态模式仍为 `full`、`(1,2,2)`、`(2,2,2)`、`(3,3,3)`、
  `(5,3,5)`；前 12 个去噪步保持原始 full 保护。

### 2.2 在线多路扩窗

只对 STA 的 `(1,2,2)` 和 `(2,2,2)` 小窗口 heads 在线决策。每个 query tile
将池化后的 Q/K 与前 2 个时间 tile、空间半径 1 内的候选 tile 比较，并选 Top-4
时间锚点。每个 query tile 有三条候选路线：

1. `T`：`(1,2,2)` 基础窗口与 Top-4 时间锚点的并集。
2. `H`：高度扩展窗口 `(1,4,2)`。
3. `W`：宽度扩展窗口 `(1,2,4)`。

先用池化 QK softmax 概率分别计算三条路线覆盖的概率质量 `g1 >= g2 >= g3`，
再定义置信度：

```text
c = (g1 - g2) / max(abs(g1), 1e-6)
```

- `c >= 0.35`：只用 Top-1 路线。
- `0.15 <= c < 0.35`：使用 Top-2 路线并集。
- `c < 0.15`：使用三条路线并集。

因此扩窗数量由当前 prompt、当前去噪步、当前层、当前 head 和当前 tile 的真实
QK 决定，不是固定时间表，也不是把三个窗口无条件求并集。

### 2.3 固定小窗口的时间锚点

对 `(3,3,3)` 和 `(5,3,5)` heads，检查前 3 个时间 tile、空间半径 2 的候选，
最多加入 8 个锚点。只有候选分数满足
`score >= mean(base scores) + 0.5 * std(base scores)` 才接受。为保护后期细节，
`step >= 35 && layer >= 25` 时禁用这类固定锚点。

### 2.4 全注意力 head 削减

对原 STA 中的每个 full head，用 `(5,3,5)` 输出与 Full teacher 比较。对每个
calibration prompt 计算：

```text
head contribution = candidate head SSE / 该 step-layer 的 12-head teacher energy
```

仅当 16 个 prompts 上 `mean <= 0.002` 且 `P90 <= 0.005` 时，将该 full head
改成 `(5,3,5)`。共削减 200 个 step-layer-head 点：

| Layer | Head | Steps | 数量 |
|---:|---:|:---|---:|
| 4 | 6 | 49 | 1 |
| 19 | 8 | 47-48 | 2 |
| 26 | 0 | 14-23 | 10 |
| 26 | 7 | 12-20 | 9 |
| 27 | 4 | 25-48 | 24 |
| 28 | 4 | 12-48 | 37 |
| 28 | 7 | 12-49 | 38 |
| 29 | 3 | 12-47 | 36 |
| 29 | 7 | 17-48 | 32 |
| 29 | 10 | 39-49 | 11 |

原策略的 18,000 个 step-layer-head 点中有 8,845 个 full；新策略为 8,645 个
full，`(5,3,5)` 从 2,269 增至 2,469，其他模式完全不变。

## 3. 实际扩窗与计算量

Adaptive speed 完整生成统计：

| 指标 | 结果 |
|:---|---:|
| 使用 1 路的动态 tiles | 809,630（25.80%） |
| 使用 2 路的动态 tiles | 748,435（23.85%） |
| 使用 3 路的动态 tiles | 1,580,343（50.35%） |
| 实际扩窗率（2/3 路） | **74.20%** |
| 平均路线数 | **2.246** |
| 首选 T / H / W 方向 | 61.67% / 14.21% / 24.12% |
| 固定锚点通过率 | 12,563,846 / 44,002,064 = 28.55% |
| 有效 QK / Full | **50.563%** |
| padding 后实际执行 QK / Full | **51.672%** |

最后一项很重要：有效 QK 虽下降，但 FlexAttention 为稳定 shape 做的 padding 使
实际执行量高 1.110 个百分点，这也是 QK 降低没有直接变成明显 GPU 加速的原因。

## 4. 成对视频结果

统一设置：`832x480`、81 帧、50 denoising steps、guidance 6、shift 8、
seed 1234，prompt 为 `A 3D model of a 1800s victorian house.`。

| 方法 | RMSE↓ | PSNR↑ | SSIM↑ | Temporal SSIM↑ | 去噪秒数↓ | 墙钟秒数↓ | QK/Full↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| STA 0.1 | 51.822 | 13.840 | 0.52979 | 0.94967 | 159.472 | 295.31 | 51.274% |
| TC-STA | 43.743 | 15.313 | 0.58737 | 0.96211 | 172.039 | 311.15 | 51.462% |
| Adaptive guarded | **42.959** | **15.470** | **0.59409** | **0.96346** | 173.341 | **308.40** | 50.890% |
| Adaptive speed | 43.322 | 15.397 | 0.59321 | 0.96319 | **172.196** | 309.87 | **50.563%** |

Adaptive speed 相对 TC-STA：RMSE 降 0.96%，SSIM 升 0.99%，PSNR 升
0.084 dB，QK 相对减少 1.75%；去噪耗时 +0.09%，属于测量噪声量级。

## 5. 验证过程

- 全尺寸合成张量检查：验证 adaptive Top-1/2/3 union 的 shape、计数和执行。
- 真实 QKV 探针：sports-car 运动 prompt，在 step 14/23/33/40 和深层
  26/27/28/29 比较 Full teacher、TC Top-1、adaptive 和 head 削减候选。
- 完整生成两轮：135 点 guarded 策略与 200 点 energy-safe 策略均完成 81 帧
  paired generation，并逐帧计算指标。
- 原 progressive/TC 默认行为保持：新增参数默认仍为 `route_policy=top1`；只有
  新 runner 显式启用 adaptive。

## 6. 证据边界与下一步

目前 16 prompts 只用于 head 筛选；完整端到端视频质量比较只有一个 prompt。
RMSE/PSNR/SSIM 衡量的是对 Full teacher 的像素接近度，不等同于通用感知质量或
VBench。下一步应对至少 8-16 个包含大运动、人物、纹理和镜头运动的 prompts 做
paired benchmark，并报告均值、最差样本与置信区间。

若继续优化速度，优先方向不是继续盲目削 full heads，而是减少 adaptive union
构造和 padding 的 1.110 个百分点执行浪费；当前数据已经显示，QK 账面下降但
调度成本会抵消收益。

## 7. 产物

- `strategy_energy_safe.json`：推荐的 200 点策略。
- `strategy_guarded_adaptive.json`：质量优先的 135 点策略。
- `benchmark_victorian/adaptive_speed.mp4`：推荐版本视频。
- `benchmark_victorian/adaptive_guarded.mp4`：质量优先版本视频。
- `benchmark_victorian/analysis_final/comparison_frames.png`：逐帧对比图。
- `benchmark_victorian/analysis_final/quality_speed_chart.png`：质量、速度、QK 图。
- `benchmark_victorian/analysis_final/speed_attention/layer_head_attention.md`：最终
  30 层 x 12 heads 的实际 attention 比例。
- `benchmark_victorian/analysis_final/benchmark_metrics.json`：完整原始汇总指标。
