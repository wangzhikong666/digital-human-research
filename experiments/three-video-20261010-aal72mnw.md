# 三个新增视频：GAGAvatar Track 与 RealDenseFace

日期：2026-10-10。实验标识：`three-video-20261010-aal72mnw`。

## 输入与结果

下载三个新的 Pexels 视频，每段取前 10 秒，统一为 960×540、25 fps、250 帧、无音频 H.264。
缩放保持画面比例，竖屏视频左右补黑边；不裁掉人脸、不拉伸画面、不循环或补帧延长时长。
每组两套程序读取同一个 `input_10s.mp4` 绝对路径，运行前后 SHA256 一致。

| 查看子目录 | 素材来源与作者 | 下载原视频 | Track / RealDenseFace / 对照视频帧数 |
| --- | --- | --- | --- |
| `01_8814493` | [woman-speaking](https://www.pexels.com/video/woman-talking-and-looking-at-camera-8814493/)；ANTONI SHKRABA production | 12.52 秒，1080×1920，25 fps | 250 / 250 / 250 |
| `02_8048254` | [woman-headset](https://www.pexels.com/video/woman-looking-at-the-camera-while-talking-8048254/)；Antoni Shkraba | 12.64 秒，1920×1080，25 fps | 250 / 250 / 250 |
| `03_8724502` | [man-smiling](https://www.pexels.com/video/close-up-video-of-a-man-smiling-8724502/)；Tima Miroshnichenko | 18.80 秒，2160×4096，25 fps | 250 / 250 / 250 |

原候选 `3761680` 的原画面边缘裁掉了半张脸，因此更换为 `8724502`。原下载与更换原因保存在实验归档，未删除文件；该候选未运行追踪。

输入校验值：

- `8814493`：`cec884acb2fb3958eae65d15d50ff98c46c6121a1c78e18de45f18272abc0b7b`。
- `8048254`：`dfb8141ca05ca7207bcd7364c9592b84dea6129fa93880028c9abbd7d5079adb`。
- `8724502`：`534bd6c1c5aada03f9709f220219ec77a105d8cb836b357b7694072ba548fb1e`。

## 比较条件

同一台 RTX 5060 Laptop GPU，严格顺序运行。两种方法保留各自默认模型、拟合和预处理。
Track 使用 `GAGAvatar-5060` 环境、源码 `76fdabf901cff2552dd0b9b1c7b6b36257309b7d`；RealDenseFace 使用独立 `realdenseface` 环境、源码 `ebf0063835f520f07fdf5a813b1af9cdc0ae46b5`。
Track 默认共享身份、300 步优化、默认平滑、focal_length=12、verts_scale=5；RealDenseFace 默认 ViT-B、FLAME 2023、100 维表达及 `configs/fitting/video_offline.yaml`，保留默认编译。
环境均设置 `OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`，RealDenseFace 另设置 `TORCHINDUCTOR_COMPILE_THREADS=1`。
11 个实际使用的模型资源及配置文件已重新校验，与上一组实验一致，清单保存在归档 `model-assets-manifest.json`。

Track 用 `--no_vis` 完成官方拟合，再调用 [逐帧导出脚本](render_gagavatar_track_stream.py)，读取原 `smoothed.pkl` 和 LMDB，使用官方 FLAME、RenderMesh、相机及三栏布局导出。
三次导出前后参数文件校验值均一致。这次没有发生官方缓冲视频导出 OOM，选择逐帧编码是为避免再次积累整段渲染帧。
两套项目源码未修改，已有实验结果保留。

这些对照保证相同输入、时间和帧覆盖；Track 会裁剪、抠图为 512×512，而 RealDenseFace 在原画面中拟合，FLAME 资源和相机约定也不同。
本次为默认流程的定性比较，未测量带真值的精度或统一条件下的速度排名。运行时间包含初始化和编译，保存在各组运行元数据中。

## 验证

三个视频的六次官方追踪进程均正常退出，退出码为 0。
每组 `base.pkl`、`optim.pkl`、`smoothed.pkl` 都有完整 0–249 帧，所有参数值有限；平滑后的共享形状一致。
Track 每帧包含 `bbox(4)`、`shapecode(300)`、`expcode(100)`、`posecode(6)`、`eyecode(6)`、`transform_matrix(3,4)`。
RealDenseFace 每组 `identity(300)`、`x(250,118)`，所有字段无 NaN 或 Inf。

六个方法视频均完整解码为 250 帧、25 fps；三个同步对照也均完整解码为 250 帧、25 fps、10 秒。
对照视频为 1280×1210，上方 Track，下方 RealDenseFace，字幕标记同一帧号和时间。
已人工查看各组第 0、125、249 帧；两种方法均生成有效面部网格，可观察嘴部和头部变化。
集中查看目录的每份副本与原始文件 SHA256 一致。各组 `validation.json` 及归档 `viewing-copies.json` 保留验证记录。

## 在 Windows 查看

把下面地址粘贴到文件资源管理器地址栏：

```text
\\wsl.localhost\Ubuntu-22.04\home\wangzhikong666\projects\RealDenseFace\output\three-video-20261010-aal72mnw
```

其中有 `01_8814493`、`02_8048254`、`03_8724502` 三个子目录，每个包含：

| 文件 | 用途 |
| --- | --- |
| `comparison.mp4` | 按同一帧号同步的两种方法对照 |
| `comparison_preview.jpg` | 首帧、中间帧和末帧预览 |
| `input_10s.mp4` | 两种方法使用的相同输入副本 |
| `track_visualization.mp4` | Track 官方参数逐帧导出 |
| `offline_visualization.mp4` | RealDenseFace 官方可视化 |
| `track_parameters_optim.pkl` / `track_parameters_smoothed.pkl` | Track 优化后 / 平滑后参数 |
| `offline_tracking.npz` | RealDenseFace 参数 |
| `input-metadata.json` / `validation.json` | 来源、预处理、校验和验证记录 |

## 归档与复跑

本地归档：`/home/wangzhikong666/digital-human-research/outputs/comparisons/three-video-20261010-aal72mnw`。
包含原视频、统一输入、来源、预处理完整命令、运行日志与元数据、模型清单、同步视频及验证脚本快照。
Track 原输出在 `GAGAvatar/gagavatar/libs/GAGAvatar_track/outputs/<实验标识>-v<编号>-<素材ID>/input_10s/`；RealDenseFace 原输出在 `RealDenseFace/output/<实验标识>-v<编号>-<素材ID>/`。

复跑的环境激活、官方命令和逐帧导出示例见 [上一组实验的复跑方法](track-vs-realdenseface-20261010-ormgxg35.md#复跑与低内存导出)。
替换 `INPUT` 为本次某个统一输入路径，并新建输出标识。执行 Track 前确认目标目录不存在，因为官方 `build_video` 会删除同名输出目录。
本次生成的视频、参数、图片、日志和下载素材都放在被忽略的输出目录中，没有加入 Git。
