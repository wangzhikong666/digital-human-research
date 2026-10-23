# 数字人研究项目

用于长期整理数字人系统的研究资料、代码、配置与实验记录。统一保存 GAGAvatar、RealDenseFace、LAM、AvatarForcing 和 FaceAnything 的源码、环境配置与研究记录。已完成 RealDenseFace 的图片与视频复现，以及 LAM 的 RTX 5060 环境和模型资源配置。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `projects/` | 各项目的完整源码快照与本机配置修改，保留各项目的许可证。 |
| `docs/` | 研究笔记、设计文档和决策记录。 |
| `src/` | 后续确定技术路线后编写的源代码。 |
| `experiments/` | 可复现的实验方案、脚本与文字记录；实验输出存放在被 Git 忽略的目录中。 |
| `configs/` | 可公开、可复现的配置模板；不放密钥或私有环境配置。 |
| `tests/` | 与代码和实验流程对应的验证内容。 |
| `README.md` | 项目入口、结构和使用说明。 |
| `AGENTS.md` | 仓库内协作约定。 |
| `.gitignore` | 排除数据、权重、媒体、生成结果、环境及密钥文件。 |

## 已纳入的项目

| 项目 | 源码 | 环境与记录 |
| --- | --- | --- |
| GAGAvatar | [projects/GAGAvatar](projects/GAGAvatar) | 包含本机源码修改、原环境文件和 GAGAvatar_track 源码；对照实验见下方记录。 |
| RealDenseFace | [projects/RealDenseFace](projects/RealDenseFace) | [环境记录](docs/realdenseface-environment.md)、[模型资源](docs/realdenseface-model-resources.md)。 |
| LAM | [projects/LAM](projects/LAM) | [RTX 5060 环境与模型资源](docs/lam-environment.md)。 |
| AvatarForcing | [projects/AvatarForcing](projects/AvatarForcing) | [本机安装记录](projects/AvatarForcing/SETUP_5060.md)。 |
| FaceAnything | [projects/FaceAnything](projects/FaceAnything) | [本机安装记录](projects/FaceAnything/SETUP_LOCAL.md)。 |

上游地址、源码版本、已导入的本机配置及排除文件见 [项目清单](configs/projects.json)。
这些目录是可直接 review 的源码快照，不包含嵌套 Git 仓库。
原运行目录 `/home/wangzhikong666/projects/<项目名>` 保留；各项目继续使用自己的环境。
模型权重、数据集、音视频、动画 GIF、缓存和编译产物保留在本地，不提交到本仓库。
各项目的上游 README 中，部分演示媒体链接因此不会随源码快照提供。

后续需要纳入 Git 的源码修改应在本仓库的 `projects/<项目名>/` 中进行；
外部运行目录中的新修改不会自动同步到这里。
本次整理及验证范围见 [统一仓库导入记录](experiments/repository-import-20261023.md)。

## 研究目标

待明确：目标场景、研究问题、预期成果和评价指标。

## 技术路线

待明确：数字人系统的组成模块、候选方法、技术栈和算力条件。

## 运行方法

RealDenseFace 使用独立的 `realdenseface` Conda 环境，安装、激活和验证方法见 [RealDenseFace 环境记录](docs/realdenseface-environment.md)。LAM 使用独立的 `lam` 环境，见 [LAM 配置记录](docs/lam-environment.md)。其他项目的安装方式见上方链接。不要将 API Key 或 `.env` 提交到仓库。

## 实验记录

已完成 [RealDenseFace 官方单图示例](experiments/realdenseface-single-image-20261009-s75ul78k.md) 和 [官方离线视频追踪](experiments/realdenseface-offline-video-20261009-dtb5fZ.md)。后续在 `experiments/` 中记录实验目的、所用代码与配置版本、数据来源、评价方法和结论。数据集、模型权重及生成结果保留在本地或专用存储中，不提交到 Git。

已完成 [GAGAvatar Track 与 RealDenseFace 的同一短视频对照](experiments/track-vs-realdenseface-20261010-ormgxg35.md)，记录同一输入校验值、完整帧验证、结果位置及 Track 的低内存视频导出方法。

已完成 [三个新增视频的 Track 与 RealDenseFace 对照](experiments/three-video-20261010-aal72mnw.md)，每组均为同一输入、10 秒、250 帧，结果集中在 RealDenseFace 的 `output` 目录。

已完成 LAM 的 RTX 5060 环境、CUDA 扩展与预训练资源校验，见 [LAM 配置记录](docs/lam-environment.md) 和 [资源清单](configs/lam-resources.json)。完整单图重建尚未执行。

## 待确认的问题

1. 首个研究任务及其成功标准是什么？
2. 计划使用哪种语言、运行环境和硬件？
3. 数据、模型和实验输出准备存放在哪里？
