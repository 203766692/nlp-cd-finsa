# 中文金融文本情感分析

## 任务简介

本项目面向 MapFinBen FinSA 与 CroFinSA 金融文本情感分析任务，使用预训练中文 Transformer 编码器进行三分类，类别为消极、中性、积极。代码支持中文、英文、日文、葡萄牙语、印尼语和希腊语等常见标签别名，并可将不同字段结构统一为训练所需格式。

模型在验证集上选择最佳 checkpoint，在测试集上计算以下指标：

- ACC：整体分类准确率。
- F1：按类别样本数加权的 F1。
- Macro-F1：对各类别 F1 取等权平均，便于观察小类别表现。
- MCC：多分类 Matthews 相关系数，兼顾各类别的真阳性、假阳性和假阴性。

项目默认使用 `hfl/chinese-roberta-wwm-ext`，可在 `configs/default.yaml` 或命令行中替换为其他中文预训练模型。

## 项目结构

```text
.
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── README.md
│   └── CroFinSA/
├── configs/
│   └── default.yaml
├── scripts/
│   ├── download_data.sh
│   ├── run_train.sh
│   └── run_eval.sh
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── utils.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── results/
    ├── .gitkeep
    ├── metrics.json
    ├── predictions.csv
    └── checkpoints/
```

各文件作用如下：

| 路径 | 作用 |
| --- | --- |
| `requirements.txt` | 人工维护的 Python 依赖及宽松版本范围。 |
| `data/README.md` | 数据集来源、放置路径、字段含义和 train/valid/test 用途。 |
| `data/CroFinSA/` | 本地数据目录，原始 JSON 被 `.gitignore` 排除。 |
| `configs/default.yaml` | 唯一的训练配置，包含模型名、批大小、学习率、轮数和设备等。 |
| `scripts/download_data.sh` | 根据显式传入的文件 URL 下载数据，不内置未经确认的数据地址。 |
| `scripts/run_train.sh` | 从仓库根目录启动训练，并将全部命令行参数转发给训练模块。 |
| `scripts/run_eval.sh` | 从仓库根目录启动测试集评估，并将全部命令行参数转发给评估模块。 |
| `src/data_loader.py` | 读取 JSON、提取正文、归一化标签、处理 tokenizer 并构建 Dataset。 |
| `src/model.py` | 加载预训练编码器、增加分类头、恢复 checkpoint。 |
| `src/utils.py` | 读取 YAML、解析设备、设置随机种子。 |
| `src/metrics.py` | 统一计算 ACC、加权 F1、Macro-F1 和 MCC。 |
| `src/train.py` | 训练循环、验证、混合精度、梯度累积、最佳模型保存。 |
| `src/evaluate.py` | 在 test 上计算指标并写入 `results/metrics.json`。 |
| `src/predict.py` | 加载 checkpoint，对单条或批量文本推理并写入 `results/predictions.csv`。 |
| `results/.gitkeep` | 保留结果目录；实际指标和预测文件在运行后生成。 |
| `results/checkpoints/` | 保存 `best_model.pt` 和 `last_model.pt`，目录内容不提交。 |

## 环境安装

建议使用 Python 3.10 或 3.11。先创建并激活 conda 环境：

```bash
conda create -n finsa python=3.10 -y
conda activate finsa
```

训练机 B 应先根据显卡驱动和 CUDA 版本安装 GPU 版 PyTorch，再安装其余依赖。下面以 CUDA 12.1 为例，实际 CUDA 版本请按 PyTorch 官方安装页调整：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

本地无 GPU 的 CPU 调试环境使用 CPU 版 PyTorch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

依赖文件中的 `torch>=2.0` 会接受已经安装好的 CPU 或 GPU 版本，不会主动替换 CUDA 构建。首次训练或推理时，`transformers` 还需要下载配置中的预训练模型；训练机应确保能够访问 Hugging Face 或已提前准备本地模型目录。

## 准备数据

从课程平台获取数据后，将文件放到：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

如果已经拿到下载 URL，可使用脚本：

```bash
TRAIN_URL="训练集地址" \
VALID_URL="验证集地址" \
TEST_URL="测试集地址" \
bash scripts/download_data.sh
```

也可分别传入：

```bash
bash scripts/download_data.sh \
  --train-url "训练集地址" \
  --valid-url "验证集地址" \
  --test-url "测试集地址"
```

若数据由教师直接提供，无需使用下载脚本，按 `data/README.md` 的说明放入本地目录即可。仓库根目录已有的 `MapFinSA_train.json` 和 `MapFinSA_valid.json` 不会被提交，可通过命令行参数直接指定。

## 使用方法

默认配置在 `configs/default.yaml` 中，面向训练机 GPU，使用 `cuda`、`batch_size=16` 和混合精度。训练机在仓库根目录运行：

```bash
bash scripts/run_train.sh
```

也可以直接调用 Python 模块：

```bash
python -m src.train --config configs/default.yaml
```

训练结束后，最佳模型保存到 `results/checkpoints/best_model.pt`，最后一轮模型保存到 `results/checkpoints/last_model.pt`。

本地 A 使用 CPU 调试时，不需要额外配置文件，直接在命令行覆盖关键参数：

```bash
python -m src.train \
  --device cpu \
  --batch_size 2 \
  --eval_batch_size 2 \
  --epochs 1 \
  --num_workers 0
```

CPU 模式下即使配置中的 `fp16` 为 `true`，训练脚本也会自动关闭混合精度。若当前数据仍位于仓库根目录，可显式传入路径：

```bash
python -m src.train \
  --device cpu \
  --batch_size 2 \
  --epochs 1 \
  --train_file MapFinSA_train.json \
  --valid_file MapFinSA_valid.json
```

测试集评估使用：

```bash
bash scripts/run_eval.sh
```

CPU 调试可运行：

```bash
python -m src.evaluate \
  --device cpu \
  --batch_size 4 \
  --checkpoint results/checkpoints/best_model.pt \
  --test_file data/CroFinSA/test.json
```

指标默认写入 `results/metrics.json`，包含总体指标、分类别 precision/recall/F1、支持样本数和混淆矩阵。

单条文本推理：

```bash
python -m src.predict \
  --device cpu \
  --checkpoint results/checkpoints/best_model.pt \
  --text "公司本季度净利润同比增长20%，经营活动现金流明显改善。"
```

批量推理支持 JSON、JSONL、CSV 和 TXT：

```bash
python -m src.predict \
  --device cpu \
  --checkpoint results/checkpoints/best_model.pt \
  --input_file data/CroFinSA/test.json \
  --output_file results/predictions.csv
```

CSV 输入需包含 `text`、`sentence`、`content` 或 `query` 之一作为文本列，可选包含 `id`。TXT 输入按非空行逐条处理。预测 CSV 包含原始文本、英文标签、中文标签、置信度和三个类别的概率。

## 注意事项

- 数据路径不写死在代码中，默认值集中在 `configs/default.yaml`，可通过 `--train_file`、`--valid_file`、`--test_file` 覆盖。
- 推理和评估优先读取 checkpoint 内记录的模型名称、标签顺序和最大长度，避免训练与部署错位。
- `data/CroFinSA/*.json`、仓库根目录的 `MapFinSA_*.json`、`results/checkpoints/`、缓存和环境变量文件均不会提交。
- 训练机 B 可根据显存调整 `batch_size`、`eval_batch_size` 和 `gradient_accumulation_steps`，不需要新建配置文件。
