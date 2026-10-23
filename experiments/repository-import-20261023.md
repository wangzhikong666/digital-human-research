# 数字人项目统一仓库整理

记录日期：2026-10-23。

## 目的与结构

将已有数字人项目的完整源码、环境配置和研究记录放入同一个 Git 仓库，便于通过 GitHub PR review。
原研究目录只有 `README.md` 被初始提交，后续配置和实验记录尚未纳入提交。

本次在 `/home/wangzhikong666/projects/digital-human-research` 建立独立工作分支，
保留 `/home/wangzhikong666/digital-human-research` 的原始未提交内容和各项目的运行目录。
已有 RealDenseFace、GAGAvatar 实验记录一并整理到统一仓库。

各项目源码位于 `projects/`，包括 GAGAvatar、RealDenseFace、LAM、AvatarForcing 和 FaceAnything。
GAGAvatar_track 的源码也直接纳入目录，因此没有嵌套 `.git` 或 Git 子模块指针。
各项目的上游地址、源码提交、已导入的本机安装文件及排除项见
[项目清单](../configs/projects.json)。

模型权重、数据集、音视频、动画 GIF、编译产物、缓存和本机环境不纳入 Git。
上游随源码提供的说明、许可证、静态文档图片和必要的源码依赖予以保留。
这些目录是源码快照；模型资源的现有位置仍按各项目安装记录使用。

## LAM 本次成果

- 增加 RTX 5060 Laptop 的 Python 3.10 / PyTorch 2.9.1+cu128 / CUDA 12.8 环境配置。
- 保留 DINOv2 的 SDPA 回退、VGGHead 的 TorchScript 加载修复及 nvdiffrast 补丁。
- 记录已完成的 CUDA 扩展检查，以及 LAM-20K、DINOv2、FLAME 和辅助资源下载校验。
- [环境说明](../docs/lam-environment.md) 与 [资源清单](../configs/lam-resources.json) 可直接 review。

## 验证范围

- 导入的 1,131 个项目源码/安装文件共 12,484,113 字节，逐个通过 SHA-256 对比，与原运行目录中的对应文件一致。
- 研究仓库中的 JSON、YAML 和 LAM 修改文件的 Python 语法检查通过。
- 纳入 Git 的文件清单检查通过：没有模型权重、被排除的媒体、数据文件、编译产物或嵌套 Git 子模块。
- 本次新增/整理的研究文档、配置与 LAM 安装文件通过 Git 空白字符检查。

本次导入保留已有源码及其本机修改，没有重新执行五个项目的完整运行流程。
LAM 的环境和模型读取检查此前已通过，完整单图重建尚未执行。
