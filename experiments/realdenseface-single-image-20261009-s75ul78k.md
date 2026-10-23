# RealDenseFace 官方单图示例

日期：2026-10-09。实验标识：`single-image-20261009-s75ul78k`。

## 目的与配置

跑通官方 `fit_single_image.py` 的完整单图流程：预训练模型推理、FLAME 拟合、参数保存和结果渲染。

- 代码位置：`/home/wangzhikong666/projects/RealDenseFace`。
- 代码版本：`ebf0063835f520f07fdf5a813b1af9cdc0ae46b5`；本次未修改官方源码。
- 环境：独立的 `realdenseface`，Python 3.11、PyTorch 2.9.1+cu128、RTX 5060 Laptop GPU。
- 输入：官方 `assets/demo_video.mp4` 的第 0 帧，保存为 PNG，尺寸 `1920 × 872`。README 的单图命令需要自行指定输入图片。
- 输入 SHA256：`b6b148053b4784e45eb54e7cfcbcc3674f6017a6aa15d21dc472108ae842aeb4`。
- 模型和参数：默认 ViT-B、100 个表达维度、`configs/fitting/single_image.yaml`、自动搜索相机视场角。
- 模型编译：保留官方默认的 `torch.compile`；`TORCHINDUCTOR_COMPILE_THREADS=1`、`OMP_NUM_THREADS=2` 限制本次进程并发。

环境及资源版本参见 [环境记录](../docs/realdenseface-environment.md)、[Python 包快照](../configs/realdenseface-requirements.lock.txt) 和 [模型资源清单](../configs/realdenseface-resources.json)。

## 结果与验证

官方脚本首次执行正常退出，退出码为 0。

- 参数文件：`identity` 形状为 `(300,)`，`x` 形状为 `(118,)`，表达维度为 100；所有字段数值均有限，无 NaN 或 Inf。
- 自动搜索的垂直视场角：约 `21.8034°`，处于默认搜索范围内。
- 可视化尺寸：`3840 × 1744`；四个视图均有有效内容。
- 已人工查看结果图：预测对齐与拟合网格位于输入人物的面部区域，网格叠加和纯网格渲染均完成。
- 日志中的推理与拟合计时为 `25.35 s`；包括结果写出和渲染的脚本计时为 `26.30 s`。这两项均不包含模型加载、首次编译和预热。

| 本地文件 | 用途 |
| --- | --- |
| [single_image.jpg](../outputs/realdenseface/single-image-20261009-s75ul78k/attempt-01-default/single_image.jpg) | 原图、预测对齐、网格叠加和纯网格的四宫格结果 |
| [single_image.npz](../outputs/realdenseface/single-image-20261009-s75ul78k/attempt-01-default/single_image.npz) | FLAME 身份、表达、姿态和相机参数 |
| [run.log](../outputs/realdenseface/single-image-20261009-s75ul78k/attempt-01-default/run.log) | 原始运行日志 |
| [validation.json](../outputs/realdenseface/single-image-20261009-s75ul78k/attempt-01-default/validation.json) | 数值验证、尺寸和输出文件校验值 |

输入和生成结果保留在本机 `outputs/realdenseface/single-image-20261009-s75ul78k/` 下，由 `.gitignore` 排除；本仓库只记录实验说明。

为方便按官方 README 查看，已将结果图、参数、日志和验证记录复制到 `/home/wangzhikong666/projects/RealDenseFace/output/`，逐文件校验与原结果一致，原结果保留。Windows 文件资源管理器可打开 `\\wsl.localhost\Ubuntu-22.04\home\wangzhikong666\projects\RealDenseFace\output`，查看 `single_image.jpg`。该目录也被官方仓库的 `.gitignore` 排除。

## 复跑命令

复跑时新建结果目录，并重新提取官方示例视频的首帧：

```bash
source /home/wangzhikong666/miniconda3/etc/profile.d/conda.sh
conda activate realdenseface
cd /home/wangzhikong666/projects/RealDenseFace
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=2
mkdir -p /home/wangzhikong666/digital-human-research/outputs/realdenseface
RDF_RUN_DIR="$(mktemp -d /home/wangzhikong666/digital-human-research/outputs/realdenseface/single-image-repro-XXXXXX)"

python - "$RDF_RUN_DIR/input.png" <<'PY'
import sys
import cv2
capture = cv2.VideoCapture('assets/demo_video.mp4')
ok, frame = capture.read()
capture.release()
assert ok and frame is not None
assert cv2.imwrite(sys.argv[1], frame)
PY

python -u fit_single_image.py \
  --input "$RDF_RUN_DIR/input.png" \
  --output_npz "$RDF_RUN_DIR/single_image.npz" \
  --output_image "$RDF_RUN_DIR/single_image.jpg"
```

## 结论

本机已完成官方默认 ViT-B 单图流程的端到端运行。该示例用于确认流程可运行，未使用真值数据量化重建精度。
