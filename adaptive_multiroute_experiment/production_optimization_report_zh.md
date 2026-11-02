# Adaptive TC-STA 生产路径优化

## 改动

1. 生产路径设置 `collect_diagnostics=False`，不再逐层执行 `.item()` 和
   `.tolist()`；debug 路径完整保留。
2. Top-1/2/3 路线并集由 `546 x 546` dense scatter 改成短有序列表的
   sort/unique/compact。
3. FlexAttention BlockMask 直接从 packed KV block IDs 构造，不再生成最终
   dense adjacency。
4. token grid、静态路线几何和常用索引 range 按分辨率/device 缓存。

默认行为未改变。只有 runner 显式传入 `--production-optimized` 才启用生产路径。

## 等价验证

在完整 `32760 tokens x 12 heads x 128 dim` 张量上：

- 新 route union 与原 dense union 输出逐元素完全一致。
- packed BlockMask 的可见 KV block 集合不变。
- attention 输出约 5.03 亿个元素中 8,089 个存在差异。
- 最大绝对差为 `0.001953125`（一个 BF16 量化级）。
- 平均绝对差为 `7.40e-9`。
- 单层热延迟由 `19.25ms` 降至 `12.83ms`。

差异来自数学等价 BlockMask 触发的不同 FlexAttention kernel lowering，不是路线、
锚点或可见 tile 改变。因此算法设计不变，但不能声称逐 bit 输出相同。

## 完整生成结果

统一设置：Victorian prompt、seed 1234、832x480、81 帧、50 步。

| 方法 | 去噪时间 | 墙钟时间 | RMSE↓ | PSNR↑ | SSIM↑ | Temporal SSIM↑ |
|:---|---:|---:|---:|---:|---:|---:|
| STA 0.1 | 159.472s | 295.31s | 51.822 | 13.840 | 0.52979 | 0.94967 |
| TC-STA | 172.039s | 311.15s | 43.743 | 15.313 | 0.58737 | 0.96211 |
| Adaptive debug | 172.196s | 309.87s | 43.322 | 15.397 | 0.59321 | 0.96319 |
| **Adaptive production** | **161.591s** | **298.19s** | **43.169** | **15.427** | **0.59378** | **0.96281** |

生产版相对 debug 版：

- 去噪节省 `10.605s`，加速 `6.16%`。
- 墙钟节省 `11.68s`，加速 `3.77%`。
- 与 STA 去噪只差 `2.120s`，慢 `1.33%`。
- 与 STA 墙钟只差 `2.88s`，慢 `0.98%`。
- 本次基准中 RMSE、PSNR、SSIM 没有下降；Temporal SSIM 下降 `0.00038`，
  但仍高于 TC-STA。

## 结论与边界

目标“保持原策略与质量，并把耗时拉回 STA 附近”在该 paired benchmark 上已经
达到。生产版与 STA 的时间差约 1%，同时保留了 74.2% 的真实多路扩窗率和此前的
质量增益。

当前完整端到端验证仍只有一个 prompt。上线或写入最终结论前，应在多 prompt
benchmark 上确认 BF16 轨迹差异不会产生个别质量回退。debug 模式应继续保留，
用于扩窗率、QK 和 head 行为分析；正式计时必须使用 production 模式。
