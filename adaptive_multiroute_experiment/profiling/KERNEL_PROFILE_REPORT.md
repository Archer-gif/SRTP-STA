# TC-STA / Adaptive Multi-route Kernel Profiling

## 结论先行

如果当前只写一个算子，优先写：

> **原生布局、可变 head 数的 block-sparse attention kernel**，直接读取
> `Q/K/V [B,L,H,D]` 和 packed KV block IDs，融合现有的
> `pack QKV -> FlexAttention -> unpack output`。

原因是这条链在代表层上的 CUDA 时间为：

| step 23 | Adaptive layer 6 | Adaptive layer 23 | Adaptive layer 28 |
|---|---:|---:|---:|
| pack + sparse attention + unpack | 11.615 ms | 6.398 ms | 7.398 ms |

而且当前 FlexAttention 为保持固定编译形状，会把局部 head 补到 12 个：

- layer 6：7 个真实 local head，补 5 个，占 sparse head 计算量的 41.7%
- layer 23：6 个真实 local head，补 6 个，占 50.0%
- layer 28：12 个真实 local head，无补齐

第二优先级是 **fused route builder**，把 tile pooling、候选 QK、gate、
Top-K、三路置信度和有序去重合并成一个或少数 kernel。不要只写
candidate-QK kernel：动态 QK 本身最坏只有约 0.26 ms/call，单独优化收益很小。

`BlockMask` 构造仅约 0.04-0.05 ms CPU/call，不是热点，不建议为它写算子。

## 测试条件

- GPU：NVIDIA A800-SXM4-80GB
- PyTorch：2.13.0 + CUDA 13.2
- 模型：Wan2.1-T2V-1.3B，BF16
- 生成：832x480，81 帧，50 个去噪步，CFG=6，shift=8
- prompt：红色跑车横向运动，seed=1234
- 抽样位置：真实生成的 step 23，layer 6/23/28，conditional CFG pass
- 每项：3 次 warm-up，10 次 profiler 测量
- 工具：`torch.profiler` CPU + CUDA activity，保留 Chrome trace
- QKV：真实模型激活，不是随机张量

输入几何：

- Q/K/V：`[1, 32760, 12, 128]`
- token grid：`[21, 30, 52]`
- tile：`[3, 5, 4]`，每 tile 60 token
- tile grid：`[7, 6, 13]`，共 546 tiles
- Flex block：64 token，packed sequence 长度 34944

下表的 GPU 时间取 profiler 中带 CPU range 的 inclusive CUDA time，并除以
10 次调用。顶层阶段可以用于定位热点；嵌套阶段不能与父阶段重复相加。
profiler wall time 含 trace/shape/memory 插桩开销，不作为真实推理延迟。

## 顶层阶段耗时

单位：ms / attention call。

| 方法/层 | Dense SDPA | Sparse Flex | Fixed route | Dynamic route | Pack | Unpack | 顶层阶段和 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive L6 | 15.612 | 8.872 | 2.266 | 1.017 | 2.169 | 0.574 | 30.510 |
| STA L6 | 16.107 | 8.746 | - | - | 2.213 | 0.585 | 27.651 |
| Adaptive L23 | 17.350 | 3.983 | 1.716 | 0.927 | 1.963 | 0.452 | 26.391 |
| STA L23 | 19.806 | 4.051 | - | - | 2.204 | 0.505 | 26.566 |
| Adaptive L28 | - | 4.481 | 2.079 | 2.206 | 2.032 | 0.885 | 11.683 |
| STA L28 | 7.309 | 2.767 | - | - | 2.405 | 0.881 | 13.362 |

解释：

- Dense SDPA 是现成 FlashAttention，虽是混合层最大项，但不应自研替代。
- 自研分支最大的一组是 sparse attention + 两侧数据重排。
- layer 28 中 Adaptive 去掉了 STA 的 2 个 full head，所以抽样层反而比 STA
  的顶层阶段和低约 12.6%；整个 50-step 生成仍比优化 STA 慢约 6.1%，说明
  路由开销在大量“没有省掉足够 dense 计算”的层上累积。
- Adaptive L28 的 sparse Flex 比 STA L28 高，是扩窗后实际 KV block 更多，
  属于算法计算量，不是单纯框架开销。

## 路由内部拆分

### Dynamic route，Adaptive layer 28

父阶段约 2.206 ms：

| 子阶段 | CUDA ms | 占父阶段 |
|---|---:|---:|
| Q/K tile pooling | 1.246 | 56.5% |
| 三路并集排序/去重 | 0.351 | 15.9% |
| candidate QK + softmax | 0.260 | 11.8% |
| anchor Top-K | 0.060 | 2.7% |
| 其余 route score / gather / launch gap | 0.289 | 13.1% |

动态候选张量：

- pooled Q/K：`[1, H_dynamic, 546, 128]`，FP32 accumulation
- temporal candidates：`[546, 18]`
- static+temporal union：`[546, 30]`
- logits：`[1, H_dynamic, 546, 30]`
- route options：3 路，每路最大 75 个 key tiles
- 当前并集输入最大宽度：225，输出容量最大 237

### Fixed route，Adaptive layer 28

第二次细粒度复测中父阶段约 2.456 ms。细粒度 record range 会增加 launch
间隙，因此这里看内部占比，不与主表绝对时间混用：

| 子阶段 | CUDA ms | 父阶段占比 |
|---|---:|---:|
| Q/K tile pooling | 1.060 | 43.1% |
| base mean/std gate | 0.654 | 26.6% |
| candidate QK | 0.488 | 19.9% |
| base+anchor union | 0.107 | 4.4% |
| Top-8 | 0.059 | 2.4% |
| 其余 | 0.088 | 3.6% |

固定候选最大为 `[546, 75]`。当前 gate 会 materialize/gather
`base_k [B,H_fixed,546,75,128]`；这比最终只需要的均值、方差和阈值大得多，
很适合在 fused kernel 内流式归约，避免中间张量。

## CUDA 操作层证据

Adaptive layer 28 每次调用的主要 ATen CUDA 操作：

| ATen op | self CUDA ms | calls/call |
|---|---:|---:|
| `aten::index` | 2.488 | 13 |
| `aten::copy_` | 1.556 | 93 |
| `aten::gather` | 0.917 | 11 |
| `aten::mul` | 0.387 | 6 |
| `aten::_index_put_impl_` | 0.388 | 1 |
| `aten::sort` | 0.302 | 5 |
| `aten::index_copy_` | 0.286 | 1 |
| `aten::mean` | 0.277 | 4 |
| `aten::sum` | 0.259 | 10 |
| `aten::topk` | 0.068 | 2 |

STA layer 28 只有约 30 次 `copy_`，Adaptive 为 93 次。问题不只是 FLOPs，
还有大量小 kernel、索引和中间张量 materialization。

## 建议算子 1：Native Sparse Attention

### 目标

融合：

1. 按静态 tile-token map 读取原生 Q/K/V
2. 按 `selected_ids/selected_counts` 做 block-sparse attention
3. 直接写回原始 token/head 顺序

避免：

- 三次 `index_select + contiguous` pack
- `index_copy_` unpack
- 把 local heads 补齐到 12 个
- 34944 长 padded Q/K/V 的显式中间张量

### 建议接口

```text
input:
  q, k, v          bf16 [B, L, 12, 128]
  local_head_ids   int32 [H_local]
  selected_ids     int32 [B, H_local, 546, Kmax]
  selected_counts  int32 [B, H_local, 546]
  tile_token_map   int32 [546, 60]      # 静态，可常驻 GPU/constant cache

output:
  out              bf16 [B, L, H_local, 128]
```

语义要求：

- 每个 query tile 的 60 个 query token 只看 `selected_ids` 指定的 key tiles
- 每个 key tile 恰有 60 个有效 token，不应计算 64-block 中 4 个 padding token
- 支持 `H_local=1..12`，不做 dummy-head padding
- accumulation 与当前 FlexAttention 对齐，输出允许 BF16 正常舍入误差

## 建议算子 2：Fused Route Builder

如果先做路由算子，建议一次处理 fixed 和 dynamic heads，而不是只做 QK：

```text
input:
  q, k                   bf16 [B, L, 12, 128]
  fixed_head_ids         int32 [Hf]
  dynamic_head_ids       int32 [Hd]
  static geometry:
    tile_token_map       [546, 60]
    temporal_ids_dyn     [546, 18]
    union_ids_dyn        [546, 30]
    temporal_ids_fixed   [546, 75]
    base_window_ids/counts per head
  scalars:
    fixed_sigma=0.5, dynamic_topk=4, fixed_topk=8
    margin_high=0.35, margin_low=0.15

output:
  selected_ids           int32 [B, H_local, 546, Kmax]
  selected_counts        int32 [B, H_local, 546]
```

融合内容：

- BF16 Q/K tile mean，FP32 accumulate
- candidate dot products
- fixed base mean/std 和 `score >= mean + 0.5*std`
- Top-4 / Top-8
- dynamic 三路 mass、confidence 和 Top-1/2/3 cardinality
- 小列表有序去重，直接写 packed IDs/counts

不应输出完整 `base_k`、`probabilities`、`route_ids` 或 dense adjacency。
如果 softmax 只用于三路 probability mass，可使用 online max/sum 和三路累积，
无需 materialize `[B,Hd,546,30]` probabilities。

## 验收标准

正确性：

1. route kernel：`selected_counts` 必须完全相同；有效区间内
   `selected_ids` 必须集合相同，最好保持升序和 tie-break 一致。
2. attention kernel：同一 route 下比较输出，记录 max/mean absolute error；
   BF16 建议 max abs <= 2e-2，并同时检查相对误差。
3. 固定 seed 跑至少 3 条 prompt，确认 SSIM/PSNR 不低于当前实现的正常波动。

性能：

1. 性能测试关闭 profiler/diagnostics，用 CUDA Event，至少 20 warm-up + 100 runs。
2. 第一算子目标：在 L6/L23/L28 三种 head 数上均胜过
   `pack + FlexAttention + unpack`；建议以 L28 至少节省 2 ms/call 为继续投入门槛。
3. 第二算子目标：L28 fixed+dynamic route 从约 4.29 ms 降到 1.5 ms 以下。
4. 最终必须复测完整 50-step denoise；不能用带 profiler 的 wall time判断收益。

## 附件

- `profile_summary.json`：完整结构化 profiler 结果
- `phase_summary.csv`：去重后的阶段表
- `step23_layer*_*.table.txt`：PyTorch profiler 操作表
- `step23_layer*_*.trace.json.gz`：Chrome / Perfetto trace
- `profiling_fixed_detail/profile_summary.json`：fixed route 细分复测

原始 trace 可用 Perfetto 打开。主 profile 的所有标记默认关闭，不影响生产推理。
