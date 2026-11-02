# TC-STA 最终实验报告

## 结论

建议停止继续优化原始 progressive 固定扩窗，转向本目录的 TC-STA
（Confidence-Gated Temporal Correspondence STA，置信度门控时序对应
STA）。在同 prompt、seed、采样参数的 Wan2.1-T2V-1.3B 配对实验中：

- 相对全注意力快 17.47%。
- 相对 STA 0.1，RMSE 下降 15.59%，PSNR 提升 1.47 dB，SSIM 从
  0.5298 提升到 0.5874。
- 相对已有 Hybrid 0.1，RMSE 继续下降 8.35%。
- 代价是比 STA-Flex 慢 5.36%，比 Hybrid 0.1 慢 3.40%。
- 逻辑 QK 比例仅由 STA 的 51.274% 增至 51.462%。

这已经满足“速度基本不变而质量明显优于 STA”的工程目标，但目前只有一个
完整配对视频，不应把它表述为已经完成论文规模的泛化验证。

## 完整方法

### 1. STA 作为安全底座

使用现有 `mse=0.1` STA 策略。STA 对每个去噪步、层、head 标记的
`full` 决策全部保留，不把敏感 head 稀疏化；局部窗口也作为最小上下文。

实验几何参数：

- 50 个去噪步，30 层，12 heads，CFG 每步调用模型两次，共 3000 次
  self-attention。
- token 网格为 `21 x 30 x 52 = 32760`。
- tile 为 `3 x 5 x 4 = 60 tokens`，tile 网格为 `7 x 6 x 13 = 546`。
- head dimension 为 128；Q/K 在 RMSNorm 与 RoPE 后进行 tile 池化。

### 2. 小窗口 head 的在线路由

STA 分配为 `(1,2,2)` 或 `(2,2,2)` 的 head，对每个 query tile 在线比较：

1. `(1,2,2)` 基础窗口，加最多 4 个时序对应锚点；候选来自前 2 个时间
   tile，空间半径为 1。
2. 高度扩窗 `(1,4,2)`。
3. 宽度扩窗 `(1,2,4)`。

根据池化 QK 在各候选区域上的注意力质量选择路线。因此选择会随输入内容、
CFG 分支、去噪步、层、head 和 tile 改变，不是固定在后 25 步扩窗。

### 3. 中大窗口 head 的置信度门控锚点

STA 分配为 `(3,3,3)` 或 `(5,3,5)` 的局部 head 完整保留原窗口，再从
前 3 个时间 tile、空间半径 2 内搜索候选，并排除已经位于 STA 窗口内的
tile。

候选分数为：

`s(q,c) = <mean_tile(Q_q), mean_tile(K_c)> / sqrt(128)`

先取 top-8，但只有满足下式的锚点才进入最终注意力：

`s(q,c) >= mean(base_scores) + 0.5 * std(base_scores)`

所以每个 tile/head 实际加入 0 到 8 个锚点。完整生成中，43,159,544 个
可用 top-k 槽位只接纳了 12,060,714 个，接纳率为 27.94%。池化只负责选
block，最终 block 内仍执行 token-level FlexAttention。

### 4. 后期细节保护

第 35–49 步且第 25–29 层关闭上述中大窗口附加锚点，但小窗口在线路由仍然
保留。原因是无条件 top-k 在多数探针层有效，却在 sports-car prompt 的
第 40 步第 25/29 层出现回退；相似度高不代表深层细化阶段一定需要历史信息。

### 5. 内核组织

- `full` heads 合并为一次 dense SDPA。
- 所有局部 heads 合并为一次 FlexAttention。
- 局部 head 维固定填充到 12，BlockMask 元数据形状保持稳定，避免 Dynamo
  因不同层反复重编译。

原根目录 `wan_sta_flex.py` 在完整 STA-Flex 运行到第 12 步时因形状变化触发
重编译上限，退化后额外申请 18.2 GiB 并 OOM。新目录提供了相同 STA mask、
稳定形状的纯 STA 基线；没有修改根目录实现。

## 最终参数

| 参数 | 值 |
|---|---:|
| STA 标定阈值 | MSE 0.1 |
| 动态小窗口 | `(1,2,2)`、`(2,2,2)` |
| 小窗口最大锚点数 | 4 |
| 小窗口时间跨度 / 空间半径 | 2 / 1 |
| 中大窗口最大锚点数 | 8 |
| 中大窗口时间跨度 / 空间半径 | 3 / 2 |
| 排序分数 | 池化后的 RoPE QK |
| 置信度阈值 sigma | 0.5 |
| late-detail guard | `step >= 35 && layer >= 25` |
| Flex block / 有效 token | 64 / 60 |

## 配对结果

Prompt 为 `A 3D model of a 1800s victorian house.`，seed 1234，81 帧，
832x480，50 steps，guidance 6，shift 8，单张 A800 80 GB。

| 方法 | 总耗时 | QK/Full | RMSE | PSNR | SSIM | Temporal SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Full | 376.996 s | 100.000% | 0 | inf | 1 | - |
| STA 0.1 Stable Flex | 295.310 s | 51.274% | 51.822 | 13.840 | 0.5298 | 0.9497 |
| Hybrid 0.1 | 300.928 s | 51.340% | 47.726 | 14.556 | 0.5619 | 0.9620 |
| TC-STA | 311.150 s | 51.462% | 43.743 | 15.313 | 0.5874 | 0.9621 |

TC-STA 的去噪耗时为 172.04 秒，STA 为 159.47 秒。考虑 Flex 的 head
填充后，实际执行 QK 比例分别为 52.575% 与 52.303%。候选筛选共计算
345,336,480 个 tile 级 QK 分数，只相当于全注意力 token-pair 数的
0.000894%；真实额外耗时主要来自池化、索引、mask 构造和新增稀疏块，而非
候选点积本身。这也说明不能只看理论 QK 数宣称加速。

## 消融与失败方案

- 运动方向扩窗的 QK 与质量变化都太小，已放弃。
- feature-only 对应排序持续弱于池化 QK。
- 无门控 top-8 在多数层更接近 full，但存在 prompt-dependent 的后期深层
  回退。
- 只有相似度门控仍不能消除所有回退，因此加入 late-detail guard。
- `sigma=1.0` 更省，但损失了较多质量收益，最终采用 `sigma=0.5`。
- 原 progressive 仅使用约 22.63% QK，但 RMSE 约 73.60，结构破坏严重，
  不适合作为后续主方向。

## 与 STA、Hybrid 的实质区别

STA 通过离线 full-attention teacher 标定，为每个 step/layer/head 固定一个
模式；正式推理时 mask 不随样本改变。TC-STA 只把 STA 的 full-head 结果当
安全先验，局部连接则由当前样本的 Q/K 对每个 tile 在线决定，锚点位置和数量
都会变化。

已有 Hybrid 只让 STA-small heads 在三个固定几何窗口间选择。TC-STA 新增了
原窗口外的显式跨时间对象邻域对应、置信度拒绝机制和深层后期保护。这条在线
时序对应路径才是相对 STA/Hybrid 有区分意义的部分。

## 文件与局限

- `benchmark_victorian/tc_sta.mp4`：最终生成视频。
- `benchmark_victorian/sta_flex.mp4`：公平 STA-Flex 基线。
- `benchmark_victorian/analysis/comparison_frames.png`：五帧视觉对比。
- `benchmark_victorian/analysis/quality_speed_chart.png`：质量速度图。
- `benchmark_victorian/analysis/layer_head_attention.md`：30x12 平均注意力比例。
- `benchmark_victorian/analysis/step_layer_head_attention.csv`：完整
  50x30x12 比例、模式与路由次数，共 18,000 行。
- `benchmark_victorian/tc_sta.attention.json`：逐步原始计数。

当前完整配对只有一个 prompt；下一步应扩展不同运动类型，并加入 VBench/FVD
等语义指标。当前池化、top-k 与 mask-builder 仍是 PyTorch 组合算子，若融合
成 CUDA/Triton kernel，有机会回收 12.57 秒去噪额外开销中的一部分。
