# AvatarForcing 本机独立环境

本项目使用专用 Conda 环境 `avatarforcing`，位于
`/home/wangzhikong666/miniconda3/envs/avatarforcing`。

## 日常进入项目

```bash
cd /home/wangzhikong666/projects/AvatarForcing
conda activate avatarforcing
```

若新终端尚未初始化 Conda，先执行：

```bash
source /home/wangzhikong666/miniconda3/etc/profile.d/conda.sh
conda activate avatarforcing
```

## 与上游 README 的差异

- Python 保持 3.10。
- 本机为 RTX 5060 Laptop GPU。将上游 PyTorch 2.0.1 / CUDA 11.8
  换为 PyTorch 2.9.1 / CUDA 12.8，并搭配 torchvision 0.24.1、torchaudio 2.9.1。
  [PyTorch 官方 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)
  和[官方版本配套表](https://pytorch.org/get-started/previous-versions/)给出了依据。
- `requirements-5060.txt` 读取原始 `requirements.txt`，保留其中明确指定的版本。
- OpenCV 与 OpenCV headless 固定为 4.11.0.86，以兼容上游要求的 NumPy 1.26.0。
- 补充源码导入但原始依赖清单漏写的 face-alignment 1.4.1，以及视频处理需要的 FFmpeg。
- 安装 gdown、带 `hf` 命令的 huggingface-hub，供后续下载权重使用。

原始 `environment.sh` 安装的是旧版 PyTorch，本机请使用以下重建命令。

## 重建环境

在项目目录运行：

```bash
conda env create -f environment-5060.yml
conda activate avatarforcing
python -m pip install --no-deps torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install --index-url https://pypi.org/simple \
    torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    -r requirements-5060.txt
```

首步只安装 PyTorch 套件，第二步从 PyPI 补齐 CUDA 运行库和项目依赖，
避免 PyTorch 索引中的 NVIDIA 下载地址连接不稳定。

`requirements-5060.lock.txt` 保存了本次验证通过的 Python 包版本。
需要恢复相同版本时，将第二步的 `-r requirements-5060.txt`
换成 `-r requirements-5060.lock.txt`。

## 验证

```bash
python -m pip check
python check_environment.py
python inference.py --help
python preprocess_user_video.py --help
```

检查脚本会导入推理与预处理模块，读取仓库示例音视频，实际运行 CUDA 矩阵乘法、
模型卷积、BF16 attention、torchvision NMS 和 FFmpeg H.264 编码，
不会下载模型或执行完整视频推理。

### 本机验证结果（2026-10-23）

- Python 3.10.22，PyTorch 2.9.1+cu128，CUDA 12.8。
- RTX 5060 Laptop GPU 已识别，计算能力 12.0。
- `pip check` 通过，依赖关系完整。
- 推理、预处理和模型模块导入通过，两个入口的 `--help` 均可正常启动。
- 示例音频解码与重采样、示例视频解码通过。
- CUDA 矩阵乘法、模型卷积、BF16 attention、torchvision CUDA NMS 通过。
- FFmpeg H.264 编码通过，`hf`、`gdown` 命令可用。
- 模型权重已下载，通过完整性、参数尺寸和 CPU 加载检查；完整生成视频流程尚未验证。

## 模型权重

上游 README 指定的资源已下载到：

```text
pretrained_dir/flow_transformer.pth
pretrained_dir/motion_autoencoder.pth
pretrained_dir/wav2vec2-base-960h/
```

- `flow_transformer.pth`：613,993,979 字节，约 614 MB。
- `motion_autoencoder.pth`：181,034,955 字节，约 181 MB。
- Wav2Vec2：377,607,901 字节的 `model.safetensors`，以及配置、词表等文件。
- 总计 1,172,644,508 字节，约 1.17 GB。

Wav2Vec2 使用 `model.safetensors` 格式，已验证能由本项目的 PyTorch 推理代码加载。
其官方仓库为 `facebook/wav2vec2-base-960h`，固定提交为
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`。

需要重新下载或继续中断的下载时，在项目目录、已激活的环境中运行：

```bash
python download_pretrained.py
```

此脚本使用上游提供的公开下载地址，支持断点续传和网络重试。
Wav2Vec2 从固定版本下载，避免重复获取同一模型的不同格式。

### 权重验证结果

- 两份 `.pth` 均通过 `torch.load(weights_only=True)` 读取和张量有限值检查。
- Wav2Vec2 的 9 个文件全部与官方固定提交的大小、LFS SHA256 或 Git blob SHA1 匹配。
- 项目模型在 CPU 上初始化成功，327 项 checkpoint 参数尺寸匹配并已加载。
- 逐项核对 Wav2Vec2 的 210 项预训练参数，与官方文件完全一致。
- Wav2Vec2 特征提取器加载成功，一秒 16 kHz 音频前向计算通过，输出形状为 `(1, 25, 768)`。
- 文件来源、字节数、SHA256 及验证记录保存在 `pretrained_dir/weights_manifest.json`。

Transformers 4.30.2 可能对 `weight_g/weight_v` 和新版 PyTorch 的参数名称输出加载警告。
本机已通过参数逐项比对确认位置卷积权重正确加载；相关背景见
[Transformers 官方问题记录](https://github.com/huggingface/transformers/issues/26796)。
`masked_spec_embed` 是训练时使用的掩码参数，当前 `.eval()` 推理不使用。

注意：README 示例中的图片参数 `--input_image_path` 与当前源码不一致，
当前 `inference.py` 实际接受 `--avatar_ref_path`。使用时以 `python inference.py --help` 为准。
