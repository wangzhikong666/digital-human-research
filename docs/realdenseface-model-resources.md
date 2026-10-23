# RealDenseFace 模型资源

记录日期：2026-10-09。资源位置：`/home/wangzhikong666/projects/RealDenseFace/weights/`。

## 安装状态

| 资源 | 项目内路径 | 状态 |
| --- | --- | --- |
| FaceBox 人脸检测权重 | `weights/facebox/face_box.pth` | 已安装并验证 |
| FLAME 拓扑与 UV 附件 | `weights/flame/flame_assets.npz` | 已安装并验证 |
| RealDenseFace ViT-S 权重 | `weights/realdenseface/vits.pth` | 已安装并验证 |
| RealDenseFace ViT-B 权重 | `weights/realdenseface/vitb.pth` | 已安装并验证 |
| 官方 FLAME 2023 模型 | `weights/flame/flame2023.pkl` | 已安装并验证 |

## 来源与版本记录

前四个文件来自 RealDenseFace README 指定的 [作者发布资源包](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view?usp=sharing)。
下载的 `realdenseface.zip` 已通过 ZIP CRC 完整性校验，大小为 1,093,003,008 字节。
压缩包及各资源文件的 SHA256、文件大小和验证状态保存在 [资源清单](../configs/realdenseface-resources.json)。

官方 FLAME 模型来自用户下载的 `C:\Users\Dorian\Downloads\FLAME2023.zip`（WSL 路径 `/mnt/c/Users/Dorian/Downloads/FLAME2023.zip`）。压缩包完整性校验通过，已安装其中的 `FLAME2023/flame2023.pkl`，文件大小为 53,023,716 字节，原压缩包保留。

资源按 README 的原始目录结构安装，保留已有文件。本研究仓库只记录文档和校验信息，资源本体不加入 Git。

## 已完成的验证

- `flame_assets.npz` 包含 UV、三角形面和可视化边等必要数组；顶点 UV 对应 5023 个顶点，索引范围正确。
- ViT-S 和 ViT-B 均通过严格权重加载，并使用预训练权重在 RTX 5060 上完成合成图像推理，返回 5023 个顶点的坐标与相对深度；输出均为有限数值。
- FaceBox 权重加载成功，并完成 GPU 推理。
- 官方 FLAME 2023 模型加载成功，包含 5023 个顶点、300 个形状维度和 100 个表达维度；模型三角形面与 `flame_assets.npz` 完全一致。
- CUDA 优化器初始化成功，规范姿态前向结果与 PyTorch 参考结果一致，零姿态前向输出通过有限值和数值一致性检查。

所有模型资源已齐全并通过加载及 GPU 验证。上述资源检查关闭了 `torch.compile`，没有生成实验输出；后续已使用默认编译设置完成 [官方单图端到端拟合](../experiments/realdenseface-single-image-20261009-s75ul78k.md) 和 [官方离线视频追踪](../experiments/realdenseface-offline-video-20261009-dtb5fZ.md)。

## 官方 FLAME 模型

通过 [FLAME 官方下载页面](https://flame.is.tue.mpg.de/download.php) 登录并接受模型许可，下载 README 指定的 `flame2023.pkl`。
官网还提供 `flame2023_Open.pkl`，版本间的表达参数需要 [转换](https://github.com/Suraj-Bhor/flame_to_flame_open_converter)；本项目当前按 README 使用原版 `flame2023.pkl`。

模型已安装到：

```text
/home/wangzhikong666/projects/RealDenseFace/weights/flame/flame2023.pkl
```

安装和验证均使用独立的 `realdenseface` 环境，校验值和验证状态已更新到资源清单。
