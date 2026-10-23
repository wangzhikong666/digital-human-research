# 同一短视频：GAGAvatar Track 与 RealDenseFace

日期：2026-10-10。实验标识：`track-vs-realdenseface-20261010-ormgxg35`。

## 输入与比较范围

素材来自 [Pexels 8135478：正面说话视频](https://www.pexels.com/video/man-with-facial-hair-talking-while-looking-at-camera-8135478/)，作者为 ANTONI SHKRABA production。
下载的原视频经本地检查为 11.12 秒、278 帧、1920×1080、25 fps。
取前 10 秒，统一缩放为 960×540、25 fps，保存为无音频 H.264 视频，恰好 250 帧。

两套程序实际读取同一个绝对路径，运行前后的文件校验值一致：

```text
/home/wangzhikong666/digital-human-research/outputs/comparisons/track-vs-realdenseface-20261010-ormgxg35/input_10s.mp4
SHA256: 25496801ef63be3f947ba4104c5f06d0898addcab45fc9322c1694a2c4d1790f
```

使用同一台 RTX 5060 Laptop GPU，顺序运行，未同时争用 GPU。
两套方法保留各自默认的拟合和预处理：Track 会裁剪、缩放为 512×512 并抠图；RealDenseFace 从原画面提取模型输入并在原画面坐标中拟合和渲染。
因此本次提供相同输入、完整帧覆盖和同步可视化的定性比较，未做统一相机、统一 FLAME 资源、真值精度或稳定态速度基准。

## 方法与环境

| 项目 | GAGAvatar Track | RealDenseFace |
| --- | --- | --- |
| 入口 | `track_video.py` | `track_video_offline.py` |
| 环境 | `GAGAvatar-5060` | `realdenseface` |
| 源码版本 | `76fdabf901cff2552dd0b9b1c7b6b36257309b7d` | `ebf0063835f520f07fdf5a813b1af9cdc0ae46b5` |
| 主要配置 | 默认 focal_length=12、verts_scale=5、共享身份、300 步优化、默认平滑 | 默认 ViT-B、100 维表达、`configs/fitting/video_offline.yaml`、默认编译 |
| FLAME 资源 | Track 自带 `FLAME_with_eye.pt` | 官方 `flame2023.pkl` |
| 参数格式 | `optim.pkl`、`smoothed.pkl` | `offline_tracking.npz` |

两套环境均设置 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`；RealDenseFace 另设置 `TORCHINDUCTOR_COMPILE_THREADS=1`。
本次未修改两套项目源码。模型资源和配置校验值保存在本地 `model-assets-manifest.json`。

## 执行结果与内存问题

Track 完成全部 250 帧的初始估计、优化及平滑，保存了完整的 `base.pkl`、`optim.pkl`、`smoothed.pkl`。
其官方可视化先收集所有帧再编码，在视频写出阶段被 WSL 内存不足终止，原始进程退出码为 `-9`；内核日志确认 OOM 杀死该 Python 进程。
原始运行日志和失败状态保留。

随后使用 [逐帧导出脚本](render_gagavatar_track_stream.py)，读取原 `smoothed.pkl` 和原 LMDB，使用官方 FLAME 模型、RenderMesh、默认相机常量及三栏布局，逐帧写入 FFmpeg。
成功生成 `tracked_streamed.mp4`，没有重新拟合参数。
参数文件导出前后 SHA256 均为 `4176fc0823bc93c8e7bdecd3dbb6b1aecff8191005b21972eb66b36a29b4dbeb`。

RealDenseFace 官方离线脚本正常退出，退出码为 0，生成完整参数和四宫格视频。

## 验证结果

- Track 三套参数记录均为 250 帧，键后缀完整覆盖 0–249；全部平滑后参数为有限值，共享形状参数一致。
- Track 每帧字段包括 `shapecode(300)`、`expcode(100)`、`posecode(6)`、`eyecode(6)`、`transform_matrix(3,4)` 和 `bbox(4)`。
- RealDenseFace 的 `identity` 为 `(300,)`，`x` 为 `(250,118)`；所有参数字段无 NaN 或 Inf。相机垂直视场角约 33.4752°。
- 两个结果视频都完整解码为 250 帧、25 fps；Track 为 1536×512，RealDenseFace 为 1920×1080。
- 同步对照视频为 1280×1210、250 帧、25 fps、10 秒，按相同帧号拼接，保留各自官方视图布局；顶部为 Track，底部为 RealDenseFace。
- 已查看第 0、125、249 帧的对照，结果图包含有效面部网格，体现输入中的嘴部变化；未据此量化精度或排名。

运行时间仅作记录：Track 原始进程约 143.13 秒，结束于导出 OOM；补充逐帧渲染约 13.02 秒。
RealDenseFace 整个进程约 189.97 秒；官方日志的推理与拟合计时为 43.573 秒，不包含初始化、编译和视频渲染。这些时间不能直接作为两套算法的速度排名。

## 集中查看结果

Windows 文件资源管理器地址栏：

```text
\\wsl.localhost\Ubuntu-22.04\home\wangzhikong666\projects\RealDenseFace\output\track-vs-realdenseface-20261010-ormgxg35
```

| 文件 | 用途 |
| --- | --- |
| `input_10s.mp4` | 两套程序使用的同一输入副本 |
| `comparison.mp4` | 同步播放两套官方视图的对照视频 |
| `comparison_preview.jpg` | 第 0、125、249 帧对照 |
| `track_visualization.mp4` | Track 逐帧导出的视频副本 |
| `offline_visualization.mp4` | RealDenseFace 官方输出视频 |
| `track_parameters_optim.pkl` | Track 优化后参数副本 |
| `track_parameters_smoothed.pkl` | Track 平滑后参数副本 |
| `offline_tracking.npz` | RealDenseFace 参数 |
| `validation.json` | 两套参数、视频帧数与文件校验记录 |

复制到集中查看目录的文件已逐一校验与原文件一致。
原始素材、运行日志、输入与模型清单、验证和复制记录保留在研究仓库被忽略的 `outputs/comparisons/track-vs-realdenseface-20261010-ormgxg35/`；两套项目的原输出目录也保留。
所有视频、参数和生成结果均由 `.gitignore` 排除，没有加入 Git。

## 复跑与低内存导出

复跑时新建输出标识。Track 官方 `build_video` 会删除同名输出目录，因此执行前必须确保本次 Track 目标目录不存在。
本机推荐使用 `--no_vis` 完成官方拟合，再逐帧导出，避免再次积累全部渲染帧。

```bash
set -e
set -o pipefail
source /home/wangzhikong666/miniconda3/etc/profile.d/conda.sh
INPUT=/home/wangzhikong666/digital-human-research/outputs/comparisons/track-vs-realdenseface-20261010-ormgxg35/input_10s.mp4
RDF_NEW_DIR="$(mktemp -d /home/wangzhikong666/projects/RealDenseFace/output/track-comparison-repro-XXXXXX)"
TAG="$(basename "$RDF_NEW_DIR")"
TRACK_ROOT=/home/wangzhikong666/projects/GAGAvatar/gagavatar/libs/GAGAvatar_track
TRACK_RESULT="$TRACK_ROOT/outputs/$TAG/input_10s"
test ! -e "$TRACK_RESULT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

conda activate GAGAvatar-5060
cd "$TRACK_ROOT"
python -u track_video.py -v "$INPUT" -d "$TAG" --no_vis
python /home/wangzhikong666/digital-human-research/experiments/render_gagavatar_track_stream.py \
  --tracker-root "$TRACK_ROOT" --run-dir "$TRACK_RESULT" \
  --output "$TRACK_RESULT/tracked_streamed.mp4" --fps 25

conda activate realdenseface
export TORCHINDUCTOR_COMPILE_THREADS=1
cd /home/wangzhikong666/projects/RealDenseFace
python -u track_video_offline.py --input "$INPUT" \
  --output_npz "$RDF_NEW_DIR/offline_tracking.npz" \
  --output_mp4 "$RDF_NEW_DIR/offline_visualization.mp4"
```
