# RealDenseFace 官方离线视频追踪

日期：2026-10-09。实验标识：`offline-video-20261009-dtb5fZ`。

## 目的与配置

使用官方示例视频运行 `track_video_offline.py`，完成全序列推理、共享身份参数优化、逐帧姿态和表达拟合，以及视频渲染。

- 代码位置：`/home/wangzhikong666/projects/RealDenseFace`。
- 代码版本：`ebf0063835f520f07fdf5a813b1af9cdc0ae46b5`；未修改官方源码。
- 环境：独立的 `realdenseface`，Python 3.11、PyTorch 2.9.1+cu128、RTX 5060 Laptop GPU。
- 输入：官方 `assets/demo_video.mp4`，289 帧，`1920 × 872`，约 `23.976 fps`、12.05 秒。
- 输入 SHA256：`59116f4dd30866b3543cbe0c1a2467c96572795a0b5da68a21cce2ca1dfcda2a`。
- 模型：默认 ViT-B，100 个表达维度，官方 FLAME 2023。
- 拟合配置：`configs/fitting/video_offline.yaml`，32 个关键帧、20 次注册迭代、20 次追踪迭代；日志确认完成初始追踪和两轮身份优化后的追踪。
- 保留默认 `torch.compile`，进程设置 `TORCHINDUCTOR_COMPILE_THREADS=1`、`OMP_NUM_THREADS=2`。

## 结果与验证

程序正常退出，退出码为 0。

- `identity` 形状为 `(300,)`，全序列共享；逐帧参数 `x` 形状为 `(289, 118)`，所有参数字段均无 NaN 或 Inf。
- 相机垂直视场角约 `21.8034°`。
- 输出视频全部 289 帧可解码，帧数与输入及逐帧参数一致，尺寸为 `3840 × 1744`，约 `23.976 fps`、12.05 秒。
- 已查看第 0、144、288 帧：四宫格渲染完整，网格跟随人物头部朝向和嘴部变化。此检查未使用真值数据量化精度或时序稳定性。
- 官方日志中的推理和拟合计时为 `30.749 s`，不包含模型初始化、结果写出和视频渲染。本次整个进程运行约 `86.19 s`，包含这些阶段。

## 结果位置

结果保存于官方仓库 `output/` 下的新目录，保留已有单图结果：

```text
/home/wangzhikong666/projects/RealDenseFace/output/offline-video-20261009-dtb5fZ/
├── offline_visualization.mp4       # 官方四宫格追踪视频
├── offline_tracking.npz            # 共享身份、逐帧表达/姿态及相机参数
├── preview_first_middle_last.jpg   # 首帧、中间帧和末帧预览
├── run.log                        # 原始运行日志
├── run-metadata.json              # 输入、配置、代码版本和进程计时
└── validation.json                # 参数、视频解码及文件校验记录
```

Windows 文件资源管理器地址栏：

```text
\\wsl.localhost\Ubuntu-22.04\home\wangzhikong666\projects\RealDenseFace\output\offline-video-20261009-dtb5fZ
```

双击 `offline_visualization.mp4` 查看视频。官方仓库的 `.gitignore` 排除整个 `output/`；本研究仓库只记录文字实验说明。

## 复跑命令

每次新建输出目录，避免覆盖已有结果：

```bash
set -e
set -o pipefail
source /home/wangzhikong666/miniconda3/etc/profile.d/conda.sh
conda activate realdenseface
cd /home/wangzhikong666/projects/RealDenseFace
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=2
mkdir -p output
RDF_RUN_DIR="$(mktemp -d "$PWD/output/offline-video-repro-XXXXXX")"
python -u track_video_offline.py \
  --input assets/demo_video.mp4 \
  --output_npz "$RDF_RUN_DIR/offline_tracking.npz" \
  --output_mp4 "$RDF_RUN_DIR/offline_visualization.mp4" \
  2>&1 | tee "$RDF_RUN_DIR/run.log"
```
