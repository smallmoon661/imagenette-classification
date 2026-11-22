---
name: pytorch-image-classification
description: 在本 Imagenette 项目中复用已验证的 PyTorch 图像分类流程，检查环境和配置、训练 MLP/CNN、加载最佳权重评估、重绘结果以及维护实验记录。用于继续该项目或迁移其流程；不把 Imagenette 的固定设置直接套用到其他数据集。
---

# PyTorch 图像分类实验

## 主要作用：下次可以用它做什么

这个 Skill 是提供给 AI 的项目操作说明。调用时说明你的目标，AI 根据需要读取文件、执行对应代码、核对并保存结果，不会每次都重做全部流程。

1. **检查本地项目状态**：检查 Python/依赖/GPU、数据、配置、已有权重、结果文件与 Git 状态，定位环境或路径问题。
2. **复用已训练的最佳参数**：加载各实验在 20 轮中按内部验证 accuracy 选出的 `best.pt`，运行预测或评估并保存指标、预测表和图片。最佳参数不一定来自最后一轮，也不按测试成绩挑选。
3. **直接重绘已有结果**：读取训练历史和最终指标生成曲线、对比图与汇总表；这项工作不需要重新训练或加载模型。
4. **设计并运行新实验**：根据用户的新目标调整或新增配置，必要时补充代码，完成训练、验证选模和最终评估。新实验保留原结果并使用独立输出位置；迁移任务时重新确定实验条件。
5. **解释结果并维护记录**：说明曲线、指标和错误样例，按任务更新学习记录、README 或报告，保留实际命令与输出位置。

已有脚本支持的功能可以直接执行；未实现的入口（例如任意单张新图片预测）需先补充代码并验证。所需解释器、数据或权重缺失时应说明缺失项；当前训练脚本从头训练，不支持自动断点续训。

## 开始工作

本 Skill 位于项目的 `.agents/skills/pytorch-image-classification/`。从该目录向上三级是项目根目录；用 `README.md`、`src/train.py`、`configs/` 确认路径，不依赖某台电脑的用户名。

先阅读根目录 README 与学习记录的最新进度，并查看 `git status --short`。按本次请求选择下面的入口；解释代码、重绘图表、评估和正式训练是不同任务。读取相关代码、实际配置和产物确认现状，不把历史记录中的版本和分数当作当前检测结果。

所有示例命令在项目根目录的 PowerShell 中执行，直接调用项目解释器：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

如解释器不存在，先检查用户已有环境；缺包先核对解释器和 `python -m pip --version`。训练或评估前再检查 torch/torchvision 版本、CUDA 可用性与 GPU 名称。仅汇总现有结果不需要启动 GPU。

## 按任务选择入口

### 重绘已有结果（新会话复用测试的优先任务）

读取 `src/plot_results.py`，确认 `outputs/mlp/`、`outputs/cnn/`、`outputs/cnn-flip/` 各自包含 `history.csv`、`run_metadata.json`、`metrics.json`。执行：

```powershell
.\.venv\Scripts\python.exe src\plot_results.py
```

生成 `outputs/summary/comparison.csv`、`training-curves.png`、`test-comparison.png`。检查表格与源指标相符、两张图可打开且标签清楚。绘图缺少输入时报告具体缺失文件，不自动启动训练补齐。无需数据图片或 checkpoint。

### 加载模型做最终测试

读取 `src/evaluate.py`、`src/data.py`、目标配置和对应运行时配置副本，确认模型结构、预处理、类别映射与训练一致。以下以 cnn-flip 为例；mlp/cnn 使用各自同名配置。

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --config configs\cnn-flip.yaml
```

需要原始测试图片、固定 `configs/test.csv`、类别映射和 `checkpoints/cnn-flip/best.pt`。脚本新建模型后加载已有参数，使用 eval/no_grad，对 3,925 张图片输出指标、逐图预测、混淆矩阵和样例图。检查预测数量、标签映射、选中 epoch 与权重来源。现有脚本会覆盖该实验的评估产物；需保留旧版本时先保存旧产物。

当前没有独立 `predict.py` 或单图预测命令，不要虚构入口。若用户要求预测任意新图片，复用现有模型构建、类别映射、测试预处理及权重加载逻辑，按请求实现并验证。

仅更换样例选取或布局时优先复用 `predictions.csv`；不要为换图而重新训练。现有样例函数按类别固定取样，不是随机取样，展示原始 RGB 图而非裁剪后的模型输入；分析裁剪影响需查看实际输入。

### 正式训练或新实验

读取 `src/train.py`、`src/models.py`、`src/data.py` 和指定 YAML。只有任务需要训练时才运行训练命令。当前脚本无 resume：每次重新初始化模型与优化器，同实验名会覆盖历史和权重；需要重训时先保留既有结果，新实验采用独立名称/输出目录并检查配置校验是否支持。

训练前可检查真实 batch、模型形状和单 batch 冒烟测试：

```powershell
.\.venv\Scripts\python.exe src\inspect_batch.py
.\.venv\Scripts\python.exe src\inspect_models.py
.\.venv\Scripts\python.exe src\train.py --config configs\cnn.yaml --smoke-test
```

正式训练入口为 `src/train.py --config configs/<实验名>.yaml`。冒烟测试只验证流程，不保留训练权重；正式运行需从头初始化。保存日志、配置副本和元数据，核实 20 轮记录及按最高验证 accuracy（并列保留较早轮次）保存的最佳权重，再进行最终测试。

## 本轮固定实验条件

- 三组为 mlp、cnn、cnn-flip，不使用预训练权重。架构以 `src/models.py` 和原配置为准。
- 官方 train 的 9,469 张图按 seed=42 分层划分为训练 7,575、内部验证 1,894；官方 val 的 3,925 张图是最终测试集。复用 `configs/train.csv`、`val.csv`、`test.csv` 和 `class_mapping.json`，不为绘图或评估重新划分；不使用噪声标签 CSV。
- RGB → 短边缩放 160 → 中心裁剪 160×160 → 像素除以 255。仅 cnn-flip 训练集在裁剪后、转张量前以 p=0.5 水平翻转；验证和测试不翻转。
- seed=42、Adam、lr=0.001、batch size=64、weight decay=0、交叉熵、20 epoch。末批允许不足 64。
- CNN 与 CNN-Flip 独立训练，初始权重哈希应相同；不从 CNN 训练后的参数继续训练 flip。相同 seed 不代表所有随机增强与数据打乱过程逐步一致。
- 按内部验证集选模，不能用测试反馈反复调整本轮模型。重新加载相同权重复核输出不构成新独立实验。
- 迁移到其他任务时重新确认类别、划分、预处理、配置与评价协议；上述数字只约束本轮 Imagenette 实验。

## 解释结果与保存记录

- 从实际 metrics/history/metadata 提取数值，不写死历史最佳分数。比较参数量、选中 epoch、最佳验证 accuracy、测试 accuracy 和 macro-F1。
- 区分 loss 与 accuracy、百分点与相对提升；混淆矩阵是数量，跨类别比较需看召回率等比例。Accuracy 与 macro-F1 接近不证明逐类均衡。
- 样例背景、遮挡等只能支持可能原因；单 seed 改善不证明统计显著性。训练曲线指标来自逐 batch 更新过程，增强也会改变训练样本难度。
- 展平保留像素和固定顺序；卷积的优势应解释为局部连接与权重共享。backward 计算梯度，optimizer.step 更新参数。
- 完成实质工作后，在 `第二次学习过程记录.md` 写实际命令、产物、验证结果、问题和下一步；必要时更新 README。区分“Skill 已创建”“本会话验证”“新会话显式调用已成功”，未实际发生的步骤不能打勾。
- 报告引用真实产物。当前 Markdown 报告不等于学习规划要求的中文 LaTeX 源文件与 PDF。检查交付要求时单独列出 Skill 新会话验证、Obsidian 连接、LaTeX/PDF、权重资料包与仓库访问方式的完成证据。
- 提交前查看 status/diff；仅在用户要求提交或推送时执行对应 Git 操作。Skill 调用不自动授权上传。权重、原始数据和 `.venv` 被忽略，不能声称 GitHub 已备份全部项目。

## 新会话验收

用户可在此项目的新会话中显式发送：

> 使用 $pytorch-image-classification 读取当前项目，复用现有结果重新生成三组曲线和测试对比图，不重新训练。检查输出并将本次 Skill 调用结果写入学习记录。

执行重绘入口，核对输出与源指标，查看图片，将实际结果写入记录。若新会话未发现技能，直接指定本 Skill 的项目相对路径并让它读取；记录发现问题，不把文件存在当作自动发现已验证。仅重新生成结果图已满足本轮规划的复用选项，无需为验收重复训练。
