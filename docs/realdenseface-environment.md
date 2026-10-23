# RealDenseFace 独立环境

记录日期：2026-10-09。

项目位置：`/home/wangzhikong666/projects/RealDenseFace`。
安装依据：该项目的 `README.md` 和 `requirements.txt`，代码版本
`ebf0063835f520f07fdf5a813b1af9cdc0ae46b5`。

## 环境约定

使用独立的 Conda 环境 `realdenseface`，不复用 GAGAvatar 的环境或依赖。
环境位于 `/home/wangzhikong666/miniconda3/envs/realdenseface`。

| 组件 | 版本或设置 |
| --- | --- |
| Python | 3.11.17 |
| PyTorch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| CUDA 编译工具 | 12.8.93 |
| C++ 编译器 | Conda GCC/G++ 13.4.0 |
| nvdiffrast | 0.4.0，源码提交 `253ac4fcea7de5f396371124af597e6cc957bfae` |
| 目标 GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| CUDA 编译架构 | `TORCH_CUDA_ARCH_LIST=12.0` |
| 编译并发 | `MAX_JOBS=1`，适用于本机约 8 GB 的 WSL 内存 |

## 使用环境

```bash
conda activate realdenseface
cd /home/wangzhikong666/projects/RealDenseFace
```

## 重建环境

先在本研究仓库中安装 Python 和 CUDA 编译工具：

```bash
conda env create -f configs/realdenseface-environment.yml
conda activate realdenseface
conda env config vars set \
  CUDA_HOME="$CONDA_PREFIX" \
  CPATH="$CONDA_PREFIX/targets/x86_64-linux/include" \
  TORCH_CUDA_ARCH_LIST=12.0 \
  MAX_JOBS=1
conda deactivate
conda activate realdenseface
```

这些变量仅作用于 `realdenseface` 环境。`CPATH` 指向 Conda CUDA 包的实际头文件目录；GCC/G++ 由 Conda 自带的激活脚本配置。

然后进入 RealDenseFace 项目，按依赖顺序安装：

```bash
cd /home/wangzhikong666/projects/RealDenseFace
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install \
  -r /home/wangzhikong666/digital-human-research/configs/realdenseface-requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --no-build-isolation
python -m pip install -e ./cuda_extensions/flame_solver --no-build-isolation
```

PyTorch 和 torchvision 的版本组合来自 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/)。
`--no-build-isolation` 按 [nvdiffrast 官方安装说明](https://nvlabs.github.io/nvdiffrast/#installation) 添加，使 CUDA 扩展使用当前环境的 PyTorch 构建。

本次实际安装使用原项目的 `requirements.txt`。上面的重建步骤使用本仓库记录的已安装版本快照，固定 Python 包版本及 nvdiffrast 源码提交；`flame_solver` 仍从 RealDenseFace 项目源码单独编译。

## 验证结果

本机已通过以下检查：

- `python -m pip check`：没有依赖冲突。
- PyTorch 在 RTX 5060 上完成 CUDA 矩阵运算。
- nvdiffrast 完成 CUDA 三角形光栅化、属性插值及反向传播。
- `flame_solver` 完成真实 GPU 数值求解，结果与 PyTorch 求解同一正规方程一致。
- ViT-S 在随机参数下完成 `512 × 512` 输入的前向计算，四个输出均为有限数值。
- `fit_single_image.py`、`track_video_offline.py`、`track_video_online.py` 和 `preprocess_nersemble_v2.py` 的 `--help` 均可正常执行。

可在已激活的环境中快速复查依赖与入口：

```bash
python -m pip check
python -c 'import torch; import nvdiffrast.torch; import flame_solver; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python fit_single_image.py --help
```

以上是环境依赖验证，没有使用预训练权重或产生实验输出；后续模型资源的安装和验证记录见下一节链接。

## 模型资源

模型资源的安装状态、来源和验证记录见 [模型资源配置记录](realdenseface-model-resources.md)。

运行真实图片或视频前，需要按 RealDenseFace README 将以下资源放入其 `weights/` 目录：

```text
weights/
├── facebox/face_box.pth
├── flame/flame2023.pkl
├── flame/flame_assets.npz
├── realdenseface/vitb.pth
└── realdenseface/vits.pth
```

`flame2023.pkl` 需要通过 [FLAME 官方网站](https://flame.is.tue.mpg.de/) 获取并遵守其许可。
模型资源和生成结果不纳入本研究仓库的 Git 提交。
