# Imagenette Classification

第二次学习规划的 Imagenette 图像分类实验项目。

## 项目目标

在相同的数据划分和共同训练条件下完成三组实验：MLP 基线、CNN 基线和 CNN 水平翻转对照；按内部验证集选择模型，再在官方留出测试集上完成一次最终评价。

## 当前进度

- ✅ 建立项目目录
- ✅ 记录运行环境
- ✅ 下载并检查 Imagenette 数据
- ✅ 初始化本地 Git 仓库并推送至 GitHub
- ✅ 固定数据划分与类别映射
- ✅ 生成 10 类样本图与各划分的类别数量统计
- ✅ 实现共享的数据读取与预处理，并检查真实 batch
- ✅ 定义 MLP 与 CNN，并检查结构、参数量和层输出形状
- ✅ 建立三组实验配置与训练循环，并完成单 batch 冒烟测试
- ✅ 完成三组训练，并按内部验证集选择最佳权重
- ✅ 完成三组最终留出测试、混淆矩阵和错误样例分析
- ✅ 整理曲线、对比图表、实验报告和学习过程记录

## 最终结果

三组实验使用相同的数据划分与共同训练设置。模型只根据内部验证集选择最佳 epoch；官方 `val/` 的 3,925 张图片仅用于最后一次独立测试。

| 实验 | 最佳 epoch | 测试 Accuracy | 测试 Macro-F1 |
| --- | ---: | ---: | ---: |
| MLP | 15 | 0.3651 | 0.3558 |
| CNN | 18 | 0.7108 | 0.7110 |
| CNN + horizontal flip | 20 | **0.7266** | **0.7265** |

在本次固定 seed=42 的对照中，CNN 明显优于直接展平图片的 MLP；仅在训练集加入随机水平翻转的 CNN-Flip 获得最佳测试 Accuracy（72.66%）。完整的实验设计、曲线解读与错误分析见：[实验报告](report/experiment_report.md)。

![最终留出测试指标对比](outputs/summary/test-comparison.png)

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `src/` | 数据、模型、训练、评价和绘图代码 |
| `configs/` | 实验配置、类别映射和固定数据划分 |
| `data/` | Imagenette 原始数据（不提交到 Git） |
| `outputs/` | 训练记录、指标、曲线、混淆矩阵和预测案例 |
| `checkpoints/` | 按最佳验证集准确率保存的模型权重（不提交到 Git） |
| `report/` | 实验报告与结论说明 |

## 环境

2026-11-19 检测结果：

- 项目解释器：Python 3.13.7（`.venv`）
- PyTorch：2.7.1+cu118
- TorchVision：0.22.1+cu118
- 训练设备：NVIDIA GeForce RTX 3070 Ti Laptop GPU（CUDA 可用）

每次开始项目工作前，在 PowerShell 中执行：

```powershell
cd C:\Users\Gezihao\Desktop\学习记录\imagenette-classification
.\.venv\Scripts\Activate.ps1
```

## 数据位置

数据已下载并解压至：`data/imagenette2-160/`。

- 官方 `train/`：9,469 张 JPG 图片，已按类别分层划分为训练集和内部验证集。
- 官方 `val/`：3,925 张 JPG 图片，只用于最终留出测试。
- `noisy_imagenette.csv`：本项目不使用。

## 固定数据划分

运行下面的命令可重新生成固定划分与数据检查结果：

```powershell
python src\prepare_data.py
```

脚本使用 `seed=42`，不会移动或复制原始图片，只在 CSV 中记录每张图片的归属：

- `configs/class_mapping.json`：类别文件夹名、数字标签和可读类别名的对应关系。
- `configs/train.csv`：7,575 张训练图片。
- `configs/val.csv`：从官方 `train/` 分层划出的 1,894 张内部验证图片。
- `configs/test.csv`：官方 `val/` 的 3,925 张最终留出测试图片。
- `outputs/data/sample-grid.png`：覆盖 10 类、带标签的真实训练样本图。
- `outputs/data/class_counts.csv`：三个划分中各类别的图片数量。

后续 MLP、CNN 和 CNN 水平翻转对照必须复用这些 CSV，不重新随机划分数据。

## 数据读取与预处理

三个实验都通过 `src/data.py` 读取固定划分，并统一执行：RGB 转换、短边缩放至 160、中心裁剪为 `160×160`、`ToTensor()` 像素缩放到 `0~1`。

只有 `cnn-flip` 实验的**训练集**额外在中心裁剪之后、转张量之前执行 `RandomHorizontalFlip(p=0.5)`；验证集和测试集不翻转。

训练前可运行以下命令检查真实 batch：

```powershell
python src\inspect_batch.py
python src\inspect_batch.py --horizontal-flip
python src\inspect_batch.py --split val
```

检查结果应包含：训练集 7,575 张、验证集 1,894 张；图像批次形状为 `(64, 3, 160, 160)`，数据类型为 `float32`，像素范围为 `0~1`。

## 模型结构检查

模型定义位于 `src/models.py`，可使用真实训练 batch 检查模型：

```powershell
python src\inspect_models.py
```

- MLP：`Flatten(76800) → Linear(76800, 256) → ReLU → Linear(256, 10)`，可训练参数总量为 19,663,626。
- CNN：四组 `3×3 Conv → ReLU → 2×2 MaxPool`（输出通道为 32、64、128、128），接 `AdaptiveAvgPool2d(4×4)`、`Linear(2048, 256)`、ReLU 和 `Linear(256, 10)`；可训练参数总量为 767,946。
- 两种模型都输出形状 `(64, 10)` 的原始类别得分（logits）。训练时交叉熵损失直接接收 logits，不在模型末尾加入 softmax。

## 训练配置与运行

三组实验配置分别位于 `configs/mlp.yaml`、`configs/cnn.yaml` 和 `configs/cnn-flip.yaml`。三组共同使用：seed=42、交叉熵损失、Adam、学习率 0.001、batch size 64、weight decay 0、20 个 epoch、无预训练权重。

`cnn` 与 `cnn-flip` 的模型结构、数据划分、初始随机种子、优化器和训练参数相同；唯一差异是 `cnn-flip` 的训练预处理中开启 `horizontal_flip_train: true`。

训练前可做无输出文件的单 batch 冒烟测试：

```powershell
python src\train.py --config configs\mlp.yaml --smoke-test
python src\train.py --config configs\cnn.yaml --smoke-test
python src\train.py --config configs\cnn-flip.yaml --smoke-test
```

正式训练命令如下。每次正式运行都会重新创建模型和优化器，完整训练 20 个 epoch：

```powershell
python src\train.py --config configs\mlp.yaml
python src\train.py --config configs\cnn.yaml
python src\train.py --config configs\cnn-flip.yaml
```

每个 epoch 会保存训练/验证 loss 与 accuracy 至 `outputs/<实验名>/history.csv`。只有验证 accuracy 严格更高时才保存 `checkpoints/<实验名>/best.pt`；若并列，保留较早轮次。训练过程不读取最终测试集。

## 最终测试与结果绘图

训练结束后，重新加载每组由内部验证集选出的 `best.pt`，在 3,925 张最终测试图上生成 Accuracy、Macro-F1、逐类召回率、预测 CSV、混淆矩阵和多类别样例图：

```powershell
python src\evaluate.py --config configs\mlp.yaml
python src\evaluate.py --config configs\cnn.yaml
python src\evaluate.py --config configs\cnn-flip.yaml
```

下列命令只读取已保存的训练历史和测试指标，生成训练/验证曲线、测试指标柱状图与总对比表，不会重新训练模型：

```powershell
python src\plot_results.py
```

- 训练/验证曲线：`outputs/summary/training-curves.png`
- 最终测试对比图：`outputs/summary/test-comparison.png`
- 最终结果表：`outputs/summary/comparison.csv`
- 完整分析报告：[report/experiment_report.md](report/experiment_report.md)
- 正式中文报告：[PDF](report/experiment_report.pdf) / [LaTeX 源文件](report/experiment_report.tex)

重新编译报告：进入 `report/`，使用 XeLaTeX 连续编译两次。需保留 `outputs/` 中被引用的图表；编译不需要 Python 虚拟环境。

```powershell
cd report
xelatex -interaction=nonstopmode -halt-on-error experiment_report.tex
xelatex -interaction=nonstopmode -halt-on-error experiment_report.tex
cd ..
```

## 项目级 Skill

技能入口：[pytorch-image-classification](.agents/skills/pytorch-image-classification/SKILL.md)。它整理了本项目的环境检查、固定实验条件、训练/评估/绘图入口、常见问题与记录方式。

**主要作用：检查项目状态、复用最佳模型、重绘结果、开展新实验、维护学习记录。**

| 想完成的工作 | 调用后会做什么 |
| --- | --- |
| 检查状态或排错 | 检查解释器、依赖、GPU、数据、配置和结果文件。 |
| 使用已有模型 | 加载 20 轮中按内部验证集选出的最佳参数，运行评估并保存结果；不一定使用第 20 轮参数。 |
| 重画曲线和对比图 | 读取已有记录执行绘图，无需重新训练或加载权重。 |
| 从头开展新实验 | 根据新目标修改配置或补充代码，在独立输出位置运行实验并保留旧结果。 |
| 分析与记录 | 解释真实指标和图表，按要求更新学习记录、README 或报告。 |

Skill 是给 AI 的操作指导，由 AI 执行相应代码；它会根据当前要求选择流程。已有入口可以直接运行，尚未实现的功能需要先开发。当前训练脚本重新运行会从头训练，不会自动续训。

在本项目目录下开启新会话，发送：

```text
使用 $pytorch-image-classification 读取当前项目，复用现有结果重新生成三组曲线和测试对比图，不重新训练。检查输出并将本次 Skill 调用结果写入学习记录。
```

若技能未被发现，指定 `.agents/skills/pytorch-image-classification/SKILL.md` 让助手读取。技能文件建立与新会话实际调用是两个阶段；后者须根据真实执行结果记录。

已完成新会话 Skill 调用和中文 LaTeX/PDF 报告。Obsidian 与打包入口见下节；教师访问私有仓库的方式仍需本人确认。

## Obsidian 导航与提交资料包

本机知识库“葛子豪知识库”中新增 `ai学习笔记/课程笔记/计算机视觉/Imagenette实验入口.md`，连接项目 README、学习记录、报告及 CNN、Skill 等课程笔记。课程笔记也添加了返回入口的链接。入口使用本机外部文件路径，移动项目后需要更新；不复制多份过程记录。

打包命令（在项目根目录执行）：

```powershell
.\.venv\Scripts\python.exe src\package_submission.py
```

ZIP 生成到项目外的 `../提交资料/`，包含代码、配置、结果、报告、Skill 和三组最佳权重；排除运行环境、原始数据、缓存和 `.git`。包内有文件 SHA256 清单，包旁有 ZIP 校验文件，脚本会校验压缩包内容。时间戳命名保留旧包；后续修改项目后需要重新打包。

具体内容、复查方法和私有仓库访问事项见 [提交说明](提交说明.md)。教师账号及提交平台尚未提供，本次未授权他人访问或发送资料。

## 版本控制

- 本地默认分支：`main`
- GitHub 私有仓库：[smallmoon661/imagenette-classification](https://github.com/smallmoon661/imagenette-classification)
- 原始数据、虚拟环境和模型权重通过 `.gitignore` 排除，不进入普通 Git 提交。
