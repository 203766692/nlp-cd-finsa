# 中文金融情感分析（CCL 2026 MapFinBen FinSA）

## 任务简介

本项目对应 CCL 2026 MapFinBen FinSA 中文金融文本情感分析任务。任务目标是对给定的中文金融文本进行情感分类，属于金融领域文本分类任务。模型需要判断文本的情感倾向，分为“消极”、“中性”、“积极”三类。

项目主页：[HgITSE/MapFinBen](https://github.com/HgITSE/MapFinBen)

数据地址：[MapFinBen data](https://github.com/HgITSE/MapFinBen/tree/main/data)

项目基于预训练中文 Transformer 编码器进行微调，在验证集上选择最佳 checkpoint，在测试集上计算以下评价指标：

- ACC：整体分类准确率。
- F1：按类别样本数加权的 F1。
- Macro-F1：对各类别 F1 取等权平均，便于观察小类别表现。
- MCC：多分类 Matthews 相关系数，兼顾各类别的真阳性、假阳性和假阴性。

项目默认使用 `hfl/chinese-roberta-wwm-ext`，可在 `configs/default.yaml` 或命令行中替换为其他中文预训练模型。

情感标签仅接受中文“积极”“中性”“消极”，以及内部标准键 positive/neutral/negative；其他标签会直接报错。

## 项目结构

```text
.
├── requirements.txt              记录程序运行所需的主要Python库及宽松版本范围。
├── README.md                     项目主说明文档。
├── .gitignore                    配置 Git 忽略规则，排除数据、模型权重等。
├── data/
│   ├── README.md                 说明数据集名称、来源或下载地址及数据使用方式。
│   └── CroFinSA/                 本地数据目录，原始和清洗后的 JSON 文件被 Git 忽略。
├── configs/
│   └── default.yaml              唯一的训练配置，包含模型名、批大小、学习率、轮数和设备等。
├── scripts/
│   ├── clean_chinese_data.py     通过假名过滤和中文标签校验清洗样本，并切分本地测试集。
│   ├── download_data.sh          根据显式传入的文件 URL 下载数据，不内置未经确认的数据地址。
│   ├── run_train.sh              从仓库根目录启动训练，并将全部命令行参数转发给训练模块。
│   └── run_eval.sh               从仓库根目录启动测试集评估，并将全部命令行参数转发给评估模块。
├── src/
│   ├── __init__.py               标记 src 为 Python 模块。
│   ├── data_loader.py            读取 JSON、提取正文、归一化中文标签、处理 tokenizer 并构建 Dataset。
│   ├── model.py                  加载预训练编码器、增加分类头、恢复 checkpoint。
│   ├── utils.py                  读取 YAML、解析设备、设置随机种子等公共工具函数。
│   ├── metrics.py                统一计算 ACC、加权 F1、Macro-F1 和 MCC。
│   ├── train.py                  训练循环、验证、混合精度、梯度累积、最佳模型保存。
│   ├── evaluate.py               在 test 上计算指标并写入 results/metrics.json。
│   └── predict.py                加载 checkpoint，对单条或批量文本推理并写入 results/predictions.csv。
└── results/
    ├── .gitkeep                  占位文件，用于保留空目录。
    ├── metrics.json              评估生成的各项指标结果，运行后产生。
    ├── predictions.csv           推理生成的预测结果，运行后产生。
    └── checkpoints/              保存 best_model.pt 和 last_model.pt，目录内容不提交。
```

## 环境安装

建议使用 Python 3.10 或 3.11。先创建并激活 conda 环境：

```bash
conda create -n finsa python=3.10 -y
conda activate finsa
```

训练机（拥有 GPU 的机器）应先根据显卡驱动和 CUDA 版本安装 GPU 版 PyTorch，再安装其余依赖。下面以 CUDA 12.1 为例，实际 CUDA 版本请按 PyTorch 官方安装页调整：

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

本任务的官方数据托管在 GitHub 上，数据地址为：`https://github.com/HgITSE/MapFinBen/tree/main/data`（其中包含 `MapFinSA` 文件夹）。课程提供的原始 MapFinSA 文件通常只有训练集和验证集，没有独立测试集。

请先将原始文件下载并放到 `data/CroFinSA/` 目录下：

```text
data/CroFinSA/MapFinSA_train.json
data/CroFinSA/MapFinSA_valid.json
```

我们已经在 `scripts/download_data.sh` 中支持自动下载。你可以直接传入 GitHub Raw 链接执行下载：

```bash
TRAIN_URL="https://raw.githubusercontent.com/HgITSE/MapFinBen/main/data/MapFinSA/MapFinSA_train.json" \
VALID_URL="https://raw.githubusercontent.com/HgITSE/MapFinBen/main/data/MapFinSA/MapFinSA_valid.json" \
bash scripts/download_data.sh
```

如果你是通过浏览器手动下载的，直接确保文件放在上述路径即可。

由于原始数据集包含多语言（如日语），而本任务仅针对中文，我们需要运行中文清洗脚本：

```bash
python scripts/clean_chinese_data.py
```

脚本采用“剔除假名 + 校验中文标签”的联合清洗策略。`text` 中平假名和片假名占非空白字符的比例超过 0.01 时直接过滤；同时要求 `answer`、`label` 或 `choices` 中出现“积极”“消极”或“中性”标签。训练集切分使用固定随机种子 42，原始文件不会被删除或修改。执行后生成：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

可以通过参数调整输入、输出目录和测试集比例：

```bash
python scripts/clean_chinese_data.py \
  --input data/CroFinSA/MapFinSA_train.json data/CroFinSA/MapFinSA_valid.json \
  --output_dir data/CroFinSA \
  --test_ratio 0.1
```

`--threshold` 参数仍保留用于兼容旧版命令，但当前不再参与过滤。当前 `test.json` 是为了跑通 `evaluate.py` 并生成评测指标而从训练集切分出的本地测试集，不代表官方独立测试结果。如果后续获得官方独立测试集，请直接替换 `data/CroFinSA/test.json`。

## 使用方法

默认配置在 `configs/default.yaml` 中，面向 GPU 训练机，使用 `cuda`、`batch_size=16` 和混合精度。在仓库根目录运行：

```bash
bash scripts/run_train.sh
```

也可以直接调用 Python 模块：

```bash
python -m src.train --config configs/default.yaml
```

训练结束后，最佳模型保存到 `results/checkpoints/best_model.pt`，最后一轮模型保存到 `results/checkpoints/last_model.pt`。

本地无 GPU 使用 CPU 调试时，不需要额外配置文件，直接在命令行覆盖关键参数：

```bash
python -m src.train \
  --device cpu \
  --batch_size 2 \
  --eval_batch_size 2 \
  --epochs 1 \
  --num_workers 0
```

CPU 模式下即使配置中的 `fp16` 为 `true`，训练脚本也会自动关闭混合精度。若当前数据仍位于根目录，可显式传入路径：

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

CSV 输入需包含 `text`、`sentence`、`content` 或 `query` 之一作为文本列，可选包含 `id`。TXT 输入按非空行逐条处理。预测 CSV 包含原始文本、中文标签、置信度和三个类别的概率。

## 注意事项

- 数据路径不写死在代码中，默认值集中在 `configs/default.yaml`，可通过 `--train_file`、`--valid_file`、`--test_file` 覆盖。
- 推理和评估优先读取 checkpoint 内记录的模型名称、标签顺序和最大长度，避免训练与部署错位。
- `data/CroFinSA/*.json`、`results/checkpoints/`、缓存和环境变量文件均不会提交。
- 训练机可根据显存调整 `batch_size`、`eval_batch_size` 和 `gradient_accumulation_steps`，不需要新建配置文件。
