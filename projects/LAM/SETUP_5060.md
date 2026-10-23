# RTX 5060 Laptop 环境

环境名称为 `lam`，使用 Python 3.10、PyTorch 2.9.1+cu128、torchvision
0.24.1+cu128、xformers 0.0.33.post2、CUDA Toolkit 12.8 和 GCC 13。
原始 `requirements.txt` 和 cu118/cu121 安装脚本保留；本机采用以下单独配置。

2026-10-22 已在 RTX 5060 Laptop GPU 上完成环境验证，检查进程退出码为 0。
应用导入、CUDA 矩阵运算、DINOv2 注意力数值比较、LAM Transformer、
PyTorch3D、simple-knn、高斯渲染、FaceBoxes NMS 和 nvdiffrast 渲染均通过。
MoviePy/FFmpeg 视频编码与解码也已通过。

2026-10-23 已续传完成 LAM-20K 权重、两个辅助资产包和 DINOv2 编码器权重，
完成完整性与文件读取检查，两个资产包也已解压。尚未执行完整单图重建。

## 使用

```bash
conda activate lam
cd /home/wangzhikong666/projects/LAM
python scripts/check_environment_5060.py
```

检查脚本会导入 LAM 应用，并实际执行 GPU 矩阵运算、xformers 注意力、
PyTorch3D、simple-knn、高斯渲染、FaceBoxes NMS 和 nvdiffrast CUDA 渲染。
首次 nvdiffrast 检查会编译其本地扩展。

## 重建环境

以下命令用于新机器重建；现有 `lam` 环境不需要重新执行。

```bash
conda env create -f environment-5060.yml
conda activate lam
conda env config vars set \
  CUDA_HOME="$CONDA_PREFIX" \
  CPATH="$CONDA_PREFIX/targets/x86_64-linux/include" \
  CUB_HOME="$CONDA_PREFIX/targets/x86_64-linux/include" \
  TORCH_CUDA_ARCH_LIST=12.0 \
  MAX_JOBS=2 \
  TORCH_EXTENSIONS_DIR="$CONDA_PREFIX/var/torch_extensions" \
  TORCH_COMPILE_DISABLE=1 \
  TF_FORCE_GPU_ALLOW_GROWTH=true
conda deactivate
conda activate lam

python -m pip install \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
  xformers==0.0.33.post2 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-5060.txt
python -m pip install chumpy==0.70 --no-build-isolation

python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba'
python -m pip install --no-build-isolation \
  'git+https://github.com/ashawkey/diff-gaussian-rasterization.git@8829d14f814fccdaf840b7b0f3021a616583c0a1'
mkdir -p exps/setup-5060/src
git clone --branch backface-culling https://github.com/ShenhanQian/nvdiffrast.git \
  exps/setup-5060/src/nvdiffrast
git -C exps/setup-5060/src/nvdiffrast checkout 22718580f24a313c429ba2c304794c264351f108
git -C exps/setup-5060/src/nvdiffrast apply "$PWD/scripts/install/nvdiffrast-pytorch29.patch"
python -m pip install --no-build-isolation ./exps/setup-5060/src/nvdiffrast
python -m pip install --no-build-isolation \
  'git+https://github.com/camenduru/simple-knn.git@60f461f4a56b7967e5d8045bf92f8c33f36976d0'

(cd external/landmark_detection/FaceBoxesV2/utils && python build.py build_ext --inplace --force)
python -m pip check
python scripts/check_environment_5060.py
```

`MAX_JOBS=2` 限制 C++/CUDA 编译的内存使用；编译架构固定为 5060 的
`sm_120`。本机 WSL 内存约 8 GB，默认设置 `TORCH_COMPILE_DISABLE=1`，
使用普通 PyTorch 推理以减少首次图编译开销。nvdiffrast 自身的本地 CUDA
扩展仍会编译。需要测试 PyTorch 图编译时，可在当前终端设置
`export TORCH_COMPILE_DISABLE=0`。

依赖中的 NumPy 采用 1.23.5，以兼容 TensorFlow 2.12 和旧版 FLAME/chumpy；
Hugging Face 与 Gradio 相关版本也单独固定。

## 5060 注意力兼容修改

xformers 0.0.33.post2 在本机可以执行半精度注意力，但没有适用于
`sm_120` 的单精度注意力算子。LAM 的 DINOv2 编码器会使用单精度计算，
因此 `lam/models/encoders/dinov2/layers/attention.py` 增加了回退：
当 xformers 报告没有可用算子、且没有特殊注意力偏置时，调用 PyTorch
自带的 `scaled_dot_product_attention`。网络参数结构保持一致，检查脚本
会将结果与原始注意力实现作数值比较。

## 安装缓存

扩展源码和编译日志保存在 `exps/setup-5060/`，编译后的安装包保存在
`wheels/`。这些目录已被项目原有的 `.gitignore` 忽略，但会保留在本机，
便于以后继续安装或排查问题。

完整验证日志：`exps/setup-5060/logs/environment-check.log`；已安装的包版本：
`exps/setup-5060/installed-packages.txt`；Conda 精确版本清单：
`exps/setup-5060/conda-linux-64.explicit.txt`。

nvdiffrast 0.3.3 的插件加载也需要适配 PyTorch 2.9：安装时应用
`scripts/install/nvdiffrast-pytorch29.patch`，直接保存
`torch.utils.cpp_extension.load()` 返回的模块对象。旧实现会在编译成功后
再次按模块名导入，在本机出现 `ModuleNotFoundError: nvdiffrast_plugin`。

## 模型和示例资产

本机已完成以下下载，无需再次执行 README 中的下载命令：

- LAM-20K：`model_zoo/lam_models/releases/lam/lam-20k/step_045500/`
  中的 `model.safetensors`、`config.json` 和模型说明。
- 跟踪与抠图模型：`model_zoo/flame_tracking_models/`。
- FLAME 参数模型和模板：`model_zoo/human_parametric_models/`。
- 示例图片和驱动动作：`assets/sample_input/`、`assets/sample_motion/`。
- DINOv2 ViT-L/14 reg4：`model_zoo/dinov2/dinov2_vitl14_reg4_pretrain.pth`。
  PyTorch 默认缓存中的同名文件已通过符号链接指向这里，避免启动时再次下载。

LAM 权重和两个资产包与 Hugging Face / ModelScope 官方公布的 SHA-256 一致。
DINOv2 来自 Meta 官方下载地址，已核对官方 multipart ETag，并通过 ZIP CRC
与 `torch.load(weights_only=True, mmap=True)` 读取检查。LAM safetensors
头部和张量形状读取检查、辅助模型 ZIP CRC 检查也均通过。

下载清单与 SHA-256：`exps/setup-5060/downloads/verified-assets.json`；
读取检查结果：`exps/setup-5060/downloads/asset-validation.json`；
下载与验证日志：`exps/setup-5060/logs/asset-download-resume-20261023.log`、
`exps/setup-5060/logs/asset-validation-20261023.log`。
原始两个资产包保留在 `exps/setup-5060/downloads/`。

辅助资产中的 `vgg_heads_l.trcd` 为 TorchScript 模型。PyTorch 2.9 的
`torch.load` 默认 `weights_only=True` 无法加载它，因此
`external/vgghead_detector/VGGDetector.py` 已改用 `torch.jit.load`。
通过实际 `VGGHeadDetector` 应用类在 CPU 上加载成功。

现在可以启动应用，再进行完整单图重建验证：

```bash
conda activate lam
cd /home/wangzhikong666/projects/LAM
python app_lam.py
```

Blender、FBX SDK 和 OpenAvatarChat 导出所需资产按
`tools/AVATAR_EXPORT_GUIDE.md` 另行安装。
