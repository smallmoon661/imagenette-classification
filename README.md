# Imagenette Classification

第二次学习规划的 Imagenette 图像分类实验项目。

## 项目目标

在相同的数据划分和共同训练条件下完成三组实验：MLP 基线、CNN 基线和 CNN 水平翻转对照；按内部验证集选择模型，再在官方留出测试集上完成一次最终评价。

## 当前进度

- [x] 建立项目目录
- [x] 记录运行环境
- [x] 下载并检查 Imagenette 数据
- [x] 初始化本地 Git 仓库并推送至 GitHub
- [x] 固定数据划分与类别映射
- [x] 生成 10 类样本图与各划分的类别数量统计
- [x] 实现共享的数据读取与预处理，并检查真实 batch
- [x] 定义 MLP 与 CNN，并检查结构、参数量和层输出形状
- [ ] 完成三组训练与最终测试
- [ ] 整理图表、报告和学习过程记录

## 目录说明

| 目录 | 用途 |
| --- | --- |
| `src/` | 数据、模型、训练、评价和绘图代码 |
| `configs/` | 实验配置、类别映射和固定数据划分 |
| `data/` | Imagenette 原始数据（不提交到 Git） |
| `outputs/` | 训练记录、指标、曲线、混淆矩阵和预测案例 |
| `checkpoints/` | 按最佳验证集准确率保存的模型权重（不提交到 Git） |
| `report/` | 中文 LaTeX 实验报告源文件与 PDF |

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

- 官方 `train/`：9,469 张 JPG 图片，后续将按类别分层划分为训练集和内部验证集。
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

## 版本控制

- 本地默认分支：`main`
- GitHub 私有仓库：[smallmoon661/imagenette-classification](https://github.com/smallmoon661/imagenette-classification)
- 原始数据、虚拟环境和模型权重通过 `.gitignore` 排除，不进入普通 Git 提交。
