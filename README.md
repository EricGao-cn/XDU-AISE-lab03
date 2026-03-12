# VGG Filter Pruning (CIFAR)

本项目基于 VGG11，在 CIFAR-10/100 上实现了 **Filter 级别的 L1-norm 剪枝**，流程为：

1. 训练 Baseline 模型
2. 按卷积核 L1 范数执行结构化剪枝
3. 对剪枝模型进行微调
4. 对 Baseline / Pruned / Finetuned 三个模型做精度、参数量、FLOPs 对比

## 论文对应关系

- `ref/PRUNING FILTERS FOR EFFICIENT CONVNETS.pdf`：按卷积核 L1-norm 进行通道剪枝（本仓库 `vggprune.py` 的核心逻辑）
- `ref/VGG.pdf`：VGG 网络结构来源

## 环境准备

按照一下步骤创建 conda 环境

```bash
conda create -n vgg_prune python=3.8
conda activate vgg_prune
pip install -r requirements.txt
```

如果不行，可直接：

```bash
conda create -n vgg_prune python=3.8
conda activate vgg_prune
pip install torch==1.6.0 torchvision==0.7.0
pip install thop
```
如果还有其他的缺少的库，直接 pip install 即可

## 目录说明（含本次修改点）

```text
.
├── main.py
├── vggprune.py
├── main_finetune.py
├── test.py
├── models/vgg.py
├── ref/
├── requirements.txt
├── vgg_prune.yml
└── outputs/
    ├── baseline/
    ├── pruned/
    └── finetune/
```

| 路径 | 作用 | 本次是否修改 | 关键细节 |
|---|---|---|---|
| `main.py` | 训练 baseline（原始 VGG11） | 否 | 产出 `outputs/baseline/model_best.pth.tar` |
| `vggprune.py` | 按 Filter L1-norm 执行结构化剪枝 | 是 | 已补全 L1 计算、保留通道索引、Conv/BN/FC 权重迁移 |
| `main_finetune.py` | 微调剪枝后的模型 | 否 | 输入 `--refine outputs/pruned/pruned.pth.tar` |
| `test.py` | 对比 baseline / pruned / finetuned 的 Acc/Params/FLOPs | 是 | 已补全 checkpoint 加载、评测流程，并新增末尾汇总统计 |
| `models/vgg.py` | VGG 网络结构定义 | 是 | 已补全 `cfg` 解析（`'M'` 池化与卷积通道配置） |
| `ref/` | 参考论文 | 否 | 含剪枝论文与 VGG 论文 PDF |
| `requirements.txt` / `vgg_prune.yml` | 环境依赖 | 否 | 用于创建实验环境 |
| `outputs/baseline/` | baseline 模型文件目录 | 否 | 放置现成 `model_best.pth.tar` 或由 `main.py` 生成 |
| `outputs/pruned/` | 剪枝结果目录 | 否 | 产出 `pruned.pth.tar` 与 `prune.txt` |
| `outputs/finetune/` | 微调结果目录 | 否 | 产出 `finetune_model_best.pth.tar` |

## 快速开始

### 0) 先放好 Baseline 权重

将已有的 `VGG11` baseline 权重文件 `model_best.pth.tar`，放到：

```text
./outputs/baseline/model_best.pth.tar
```

可先创建目录：

```bash
mkdir -p ./outputs/baseline ./outputs/pruned ./outputs/finetune
```

检查文件是否存在：

```bash
ls -lh ./outputs/baseline/model_best.pth.tar
```

如果这一步完成，就不需要重新运行 `main.py`。

### 1) 执行剪枝

```bash
python vggprune.py \
  --dataset cifar10 \
  --depth 11 \
  --dir_data ./data \
  --model ./outputs/baseline/model_best.pth.tar \
  --save ./outputs/pruned
```

产物：`./outputs/pruned/pruned.pth.tar`

### 2) 微调剪枝模型

```bash
python main_finetune.py \
  --dataset cifar10 \
  --dir_data ./data \
  --depth 11 \
  --refine ./outputs/pruned/pruned.pth.tar \
  --epochs 40 \
  --batch-size 64 \
  --test-batch-size 64 \
  --lr 0.001 \
  --save ./outputs/finetune
```

产物：`./outputs/finetune/finetune_model_best.pth.tar`

### 3) 统一评测

```bash
python test.py \
  --dataset cifar10 \
  --dir_data ./data \
  --depth 11 \
  --baseline ./outputs/baseline/model_best.pth.tar \
  --pruned ./outputs/pruned/pruned.pth.tar \
  --finetune ./outputs/finetune/finetune_model_best.pth.tar
```

输出内容：

- 三个模型的测试集精度
- 每个模型的参数量（Params）
- 每个模型的 FLOPs

## 从头训练 Baseline

如果没有现成的 `model_best.pth.tar`，先运行这一步：

```bash
python main.py \
  --dataset cifar10 \
  --dir_data ./data \
  --depth 11 \
  --epochs 100 \
  --batch-size 64 \
  --test-batch-size 64 \
  --lr 0.1 \
  --save ./outputs/baseline
```

产物：`./outputs/baseline/model_best.pth.tar`

## 常用参数

- `--dataset`: `cifar10` 或 `cifar100`
- `--dir_data`: CIFAR 数据目录（首次会自动下载）
- `--no-cuda`: 强制 CPU 运行
- `--depth`: VGG 深度（默认 11）

## CPU/GPU 说明

- 默认自动使用 CUDA（若机器支持）。
- 你的电脑如果只有 CPU，不需要额外改代码，脚本会自动走 CPU。
- 若要强制 CPU，在任意命令后加 `--no-cuda`，例如：

```bash
python test.py --no-cuda --dataset cifar10 --dir_data ./data --depth 11 --baseline ./outputs/baseline/model_best.pth.tar --pruned ./outputs/pruned/pruned.pth.tar --finetune ./outputs/finetune/finetune_model_best.pth.tar
```

## 已补全的关键实现

- `models/vgg.py`: VGG `cfg` 解析与层构建（`M` 池化、卷积通道配置）
- `vggprune.py`: L1-norm 计算、保留通道索引选择、Conv/BN/FC 权重迁移
- `test.py`: Baseline/Pruned/Finetuned 模型加载、精度评估、FLOPs 统计

## 注意事项

- 本项目依赖 `thop` 统计 FLOPs。
- 若你在 CPU 或不同 CUDA 版本环境下加载模型，脚本已使用 `map_location='cpu'`，可更稳健地读取 checkpoint。
- `test.py` 中如果看到 `volatile was removed` 警告，可忽略，不影响结果。
- 如果终端提示 `python: command not found`，把文档中的 `python` 改为 `python3` 即可。
- 使用已有 `model_best.pth.tar` 时，命令行里的 `--dataset` 和 `--depth` 要与该权重训练时一致（你们当前是 `cifar10 + depth=11`）。

## 相关指标

- 参数压缩率：`(1 - pruned_params / baseline_params) * 100%`
- FLOPs 降低率：`(1 - pruned_flops / baseline_flops) * 100%`
- 精度损失（pp）：`baseline_acc - current_acc`
- 理论加速比：`baseline_flops / pruned_flops`
