# FaceAnything 本机独立环境

项目目录：`/home/wangzhikong666/projects/FaceAnything`

Conda 环境：`faceanything`，位于
`/home/wangzhikong666/miniconda3/envs/faceanything`。

## 激活

```bash
cd /home/wangzhikong666/projects/FaceAnything
conda activate faceanything
```

如果新终端没有初始化 Conda，先执行：

```bash
source /home/wangzhikong666/miniconda3/etc/profile.d/conda.sh
```

## 安装依据与重建

遵循本仓库 README 的 Python 3.11、PyTorch 2.9.0、
torchvision 0.24.0、CUDA 12.8，以及 requirements.txt 的固定版本。

```bash
conda create -n faceanything python=3.11 pip -y
conda activate faceanything
python -m pip install torch==2.9.0 torchvision==0.24.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-local.txt
python -m pip install -e .
python -m pip check
```

本机安装使用官方 CUDA 12.8 索引中的
`torch==2.9.0+cu128`、`torchvision==0.24.0+cu128`、
`xformers==0.0.33.post1` 原始安装包，并校验官方索引提供的 SHA-256。
下载中断时仅优化传输方式，没有替换版本。
其余依赖通过清华 PyPI 镜像获取，CUDA 依赖严格使用 PyTorch 2.9.0
元数据锁定的版本，并与官方 PyPI 文件 SHA-256 核对。

`requirements-local.txt` 保留原始依赖版本，并补充内置
Depth Anything 3 代码直接导入、但原依赖清单没有列出的
`addict`、`moviepy==1.0.3`、`evo`、`pycolmap`。
MoviePy 需要 1.x，因为源码使用 `moviepy.editor`。

未安装可选的 `gsplat` 和 Open3D；FaceAnything 自带 PyTorch 点云渲染器。
DA3 导入时可能提示缺少用于 3DGS 渲染的 `gsplat`，
这不影响 FaceAnything 的点云重建和 PyTorch 渲染流程。

## 模型文件

README 指定的 FaceAnything 权重来自作者公开仓库：
<https://huggingface.co/UmutKocasari/FaceAnything>。
目标位置为 `checkpoints/checkpoint.pt`。

文件大小：`15458082007` 字节。
SHA-256：`b563aec09dc8401d8270ed41c89e69e4c765f3806255607cae420af6d80efb1a`。
本机已于 2026-10-23 完整下载并通过上述大小和 SHA-256 校验。
下载期间保留 `.part` 前缀和 `checkpoints/.download-chunks/` 中的分段，
完整下载并校验后再保存为 `checkpoint.pt`。

需要继续下载或重新校验时运行：

```bash
bash scripts/download_checkpoint.sh
```

模型首次加载还会自动下载 `depth-anything/DA3-GIANT-1.1`
的配置和基础权重；自动去背景另需下载 Robust Video Matting。
这两项由原项目运行逻辑管理。

## 验证与运行

```bash
python -m pip check
python run_inference.py --help
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

模型文件就绪后，可先尝试单张图片：

```bash
python run_inference.py \
  --input /path/to/face.jpg \
  --output output/first_run \
  --process-mode one-by-one \
  --process-res 252 \
  --render-backend torch
```

本机是 RTX 5060 8GB；上述命令降低初次运行的分辨率和同时处理的帧数。
实际模型推理是否能在当前内存和显存限制下完成，需要用完整权重和输入图片验证。

## 已完成的环境验证（2026-10-23）

- Python 3.11.16；独立环境路径与上述一致。
- 原始 `requirements.txt` 的 19 个固定依赖逐项匹配。
- PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、xformers 0.0.33.post1。
- `pip check`、推理脚本 `--help`、DA3 API 导入通过。
- RTX 5060 Laptop GPU 上的 CUDA 矩阵运算、xformers SwiGLU、DINO attention 通过。
- CUDA 点云渲染和 MP4 编码通过。

`requirements-local.lock.txt` 记录本次实际安装的完整版本；重建同版本环境时使用：

```bash
conda create -n faceanything python=3.11.16 pip -y
conda activate faceanything
python -m pip install -r requirements-local.lock.txt
python -m pip install -e . --no-deps
```

以上验证没有加载完整模型进行人脸重建，不代表已完成端到端推理测试。
