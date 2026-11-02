# 复现说明

## 1. 环境

实验原机为 NVIDIA A800 80 GB、Python 3.12、PyTorch `2.13.0+cu132`、BF16；完整生成使用 Linux、单 GPU、Wan2.1-T2V-1.3B。历史 Wan2.1 仓库的精确 commit 没有保存，因此新克隆的官方代码可能需要适配本仓库的 monkey-patch。**A800 不是运行前提**；RTX PRO 6000 等新卡应选择与其架构、驱动兼容的 PyTorch/CUDA 和 `flash_attn` 构建，重新进行正确性与速度测试，不能直接对比旧 A800 秒数。

建议在 Linux 上执行：

```bash
git clone https://github.com/Archer-gif/SRTP-STA.git
cd SRTP-STA
git clone https://github.com/Wan-Video/Wan2.1.git Wan2.1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按当前 GPU/驱动安装兼容的 PyTorch 与 CUDA wheel，再安装 Wan 依赖。
python -m pip install -r Wan2.1/requirements.txt
python -m pip install huggingface_hub
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Wan2.1-T2V-1.3B
python scripts/check_environment.py
```

官方依赖和模型来源：[Wan2.1](https://github.com/Wan-Video/Wan2.1)、[Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)、[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。Wan 的 `requirements.txt` 包含 `flash_attn`，安装前要先确认 PyTorch/CUDA 配套；`torch.nn.attention.flex_attention` 和 `torch.compile` 必须可用。若仅分析已有结果，安装 `numpy`、`opencv-python` 即可，不需要 GPU 或模型。

`scripts/check_environment.py` 会显示 GPU 名称、CUDA 能力、BF16 和 FlexAttention 是否可用，并检查 Wan 源码及模型目录。真实推理还需在新机器上至少完成一次小样本运行，以检查第三方依赖及编译兼容性。

## 2. 运行推理

仓库提供了平台路径可配置的入口。默认读取 `./Wan2.1` 与 `./models/Wan2.1-T2V-1.3B`；其他路径可通过环境变量覆盖：

```bash
export WAN_REPO=/path/to/Wan2.1
export CKPT_DIR=/path/to/Wan2.1-T2V-1.3B
export PYTHON_BIN=python
export OUT_DIR=/path/to/results/victorian
export PROMPT='A 3D model of a 1800s victorian house.'
export SEED=1234

bash scripts/run_reproduction.sh full
bash scripts/run_reproduction.sh sta
bash scripts/run_reproduction.sh tc_sta
bash scripts/run_reproduction.sh adaptive
```

不要在 `export` 后直接复制示例 `/path/to/...` 而不替换；如使用仓库内默认目录，可省略前四个路径变量。四条命令会分别生成 `full.mp4`、`sta.mp4`、`tc_sta.mp4`、`adaptive.mp4`、各自的 `*.wall_seconds.txt` 和日志；后三种另有 `*.attention.json`。固定参数是 832×480、81 帧、50 步、CFG 6、shift 8。单个 50 步视频可能需要数分钟，并在首次 FlexAttention 调用时编译较久。全部四种模式必须在同一 prompt、seed、模型和机器上比较。

`sta` 与 `tc_sta` 使用 `sta_calibration_50/strategies/sta_strategy_mse_0p1.json`；`adaptive` 使用 `adaptive_multiroute_experiment/strategy_energy_safe.json`。`sta` 在默认脚本中开启 production 路径，仅用于公平计时；`tc_sta` 默认关闭诊断同步，算法参数仍为 Top-1、Top-4/8、`0.5σ` gate、后期保护；`adaptive` 启用 Top-1/2/3 阈值 `0.35/0.15` 和 production 路径。若需要逐层路线直方图，请直接调用 `run_correspondence_wan.py` 并移除 `--production-optimized`；这会影响耗时。

`correspondence_sta_experiment/run_victorian_benchmark.sh` 和 `adaptive_multiroute_experiment/run_*.sh` 是原服务器使用的历史脚本，硬编码了 `/root/wan_srtp`、旧 Python 路径及模型路径。新机器优先用 `scripts/run_reproduction.sh`。

## 3. 分析输出

四种视频生成后，下面的命令重新计算每条视频对 Full teacher 的 RMSE/PSNR/SSIM、视频自身相邻帧的 temporal SSIM：

```bash
python scripts/compare_outputs.py --result-dir "$OUT_DIR" --out "$OUT_DIR/comparison_metrics.json"
```

RMSE 越低、PSNR/SSIM 越高表示与该 Full 视频越接近。temporal SSIM 衡量相邻帧相似度，不能单独判断运动是否自然。人工观看要特别留意 tile 边界、运动物体、遮挡和水面纹理。历史单 prompt 与三 prompt 的统计和对比图位于各 `benchmark_*/analysis*` 目录；仓库仅保存代表性图和汇总数据，不包含旧视频。当前三 prompt 对比仅包含 Full、STA production、Adaptive production，**没有 TC-STA 的三 prompt 结果**。

## 4. 从标定数据重建策略

仓库保留了 16 条标定 prompt 的原始 CSV 压缩包（分成 `sta_search_csv.zip.part-000` 等 25 个文件）和策略 JSON。推理直接使用已保存策略，不需要解压 CSV。若要检查或重建策略，先按文件名顺序合并，再校验、解压：

```bash
cat sta_calibration_50/sta_search_csv.zip.part-* > sta_calibration_50/sta_search_csv.zip
echo '5a0d0597f5d7e45e261dc77dacc5a0bf6d1df073d43cc96623952d84515376bf  sta_calibration_50/sta_search_csv.zip' | sha256sum -c -
python -m zipfile -e sta_calibration_50/sta_search_csv.zip sta_calibration_50
python build_sta_strategy_sweep.py \
  --calibration-dir sta_calibration_50 \
  --out-dir results/rebuilt_sta_strategies
python adaptive_multiroute_experiment/build_energy_strategy.py \
  --base results/rebuilt_sta_strategies/sta_strategy_mse_0p1.json \
  --calibration-glob 'sta_calibration_50/prompt_*.csv' \
  --mean-layer-limit 0.002 --p90-layer-limit 0.005 \
  --out results/rebuilt_adaptive_energy_safe.json
```

`build_sta_strategy_sweep.py` 使用所有 16 条 CSV，`mse=0.1` 版本应含 8,845 个 full head 位置；Adaptive 的 energy-safe 策略再把其中 200 个改成 `(5,3,5)`。旧 JSON 的 `calibration_files` 字段保留了原服务器绝对路径，只是来源记录；策略加载器读取的是 `masks`，在新机器上使用不需要恢复旧路径。若要**重新生成**标定 CSV，可查看 `run_sta_calibration.py --help`，并显式传入 `--wan-repo`、`--checkpoint`、`--python`、`--bridge` 与 `--execute`。这需要再跑 16 次完整模型，不属于快速复现。

## 5. 代码与限制

- `correspondence_sta_experiment/code/correspondence_hybrid_flex.py`：主要算法，按策略拆成 full / 小窗口 / 中大窗口 head；池化 RoPE 后 QK、挑时序锚点、算路线概率质量并构造 BlockMask。
- `correspondence_sta_experiment/code/correspondence_bridge.py`：只替换 Wan 1.3B 的 self-attention，保留其 cross-attention 与其余模型路径；同时记录 step、layer 与诊断。
- `correspondence_sta_experiment/code/run_correspondence_wan.py`：解析参数、加载策略、调用官方 `generate.py`。
- `saswa.py` / `saswa_flex.py` / `saswa_grouped.py`：tile 定义、池化、打包和 FlexAttention 布局；`wan_hybrid_flex.py` 提供窗口常量。
- `wan_sta_calibration.py` / `run_sta_calibration.py` / `build_sta_strategy_sweep.py`：Full teacher 标定与静态策略构建。
- `adaptive_multiroute_experiment/build_energy_strategy.py`：对低贡献 full head 的离线筛选。
- `adaptive_multiroute_experiment/profile_attention_hotspots.py`：真实 QKV profiler；A800 结果在 `profiling/`，新 GPU 必须重新 profile。

当前小窗口动态路线中，STA 标为 `(2,2,2)` 的 head **不保证保留原 `(2,2,2)` 全部连接**；T 路线从 `(1,2,2)` 起步。这一点影响对“STA 安全下界”的表述。GPU 实现还有固定补齐到 12 个 local heads 的冗余，下一步可以在新卡上验证去补齐或 fused sparse kernel，但注意不同形状可能触发重新编译。详见 [热点报告](adaptive_multiroute_experiment/profiling/KERNEL_PROFILE_REPORT.md)。

历史结果只证明已测试 prompt 上的质量/速度折中；对不同 GPU、当前官方 Wan 代码、更多 prompt 的效果仍需重新验证。改变路由或 kernel 后，请固定 prompt/seed，对比 Full、STA、TC-STA、Adaptive 四条输出，并分别报告 denoise、wall、质量和视频伪影。
