# Imagenette Classification

第二次学习规划的 Imagenette 图像分类实验项目。

## 项目目标

在相同的数据划分和共同训练条件下完成三组实验：MLP 基线、CNN 基线和 CNN 水平翻转对照；按内部验证集选择模型，再在官方留出测试集上完成一次最终评价。

## 当前进度

- [x] 建立项目目录
- [x] 记录运行环境
- [x] 下载并检查 Imagenette 数据
- [x] 初始化本地 Git 仓库并推送至 GitHub
- [ ] 固定数据划分与类别映射
- [ ] 生成 10 类样本图与各划分的类别数量统计
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

## 版本控制

- 本地默认分支：`main`
- GitHub 私有仓库：[smallmoon661/imagenette-classification](https://github.com/smallmoon661/imagenette-classification)
- 原始数据、虚拟环境和模型权重通过 `.gitignore` 排除，不进入普通 Git 提交。
