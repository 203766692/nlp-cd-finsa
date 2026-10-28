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

项目支持 BERT 与 RoBERTa 两种预训练模型，并通过类别权重形成两种方法配置。所有对比实验都由 `configs/default.yaml` 驱动，训练、评估和推理会自动使用 `experiment_name` 对应的独立结果目录。

情感标签仅接受中文“积极”“中性”“消极”，以及内部标准键 positive/neutral/negative；其他标签会直接报错。

## 项目结构

```text
.
├── requirements.txt              记录程序运行所需的主要Python库及宽松版本范围。
├── README.md                     项目主说明文档。
├── .gitignore                    配置 Git 忽略规则，排除缓存、模型权重和环境文件。
├── data/
│   ├── README.md                 说明数据集名称、来源或下载地址及数据使用方式。
│   └── CroFinSA/                 原始数据及已提交的 train/valid/test 清洗数据。
├── configs/
│   ├── default.yaml              当前正在使用的实验配置。
│   ├── config_bert_base.yaml     BERT 基础实验预设。
│   ├── config_bert_improved.yaml BERT 类别权重改进实验预设。
│   ├── config_roberta_base.yaml  RoBERTa 基础实验预设。
│   └── config_roberta_improved.yaml RoBERTa 类别权重改进实验预设。
├── src/
│   ├── __init__.py               标记 src 为 Python 模块。
│   ├── baseline.py               TF-IDF 与逻辑回归基线实验。
│   ├── clean_chinese_data.py     通过假名过滤和中文标签校验清洗样本，并切分本地测试集。
│   ├── data_loader.py            读取 JSON、提取正文、归一化中文标签、处理 tokenizer 并构建 Dataset。
│   ├── model.py                  加载预训练编码器、增加分类头、恢复 checkpoint。
│   ├── utils.py                  读取 YAML、解析设备、设置随机种子等公共工具函数。
│   ├── metrics.py                统一计算 ACC、加权 F1、Macro-F1 和 MCC。
│   ├── train.py                  训练、类权重、早停、历史记录、曲线与 checkpoint 保存。
│   ├── evaluate.py               测试集指标与混淆矩阵输出。
│   └── predict.py                加载实验 checkpoint 执行单条或批量推理。
└── results/
    ├── bert_base/                每个实验独立保存模型、指标和图片。
    ├── bert_improved/
    ├── roberta_base/
    ├── roberta_improved/
    └── baseline/                 TF-IDF 基线指标与混淆矩阵。
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

本任务的官方数据托管在 GitHub 上，数据地址为：`https://github.com/HgITSE/MapFinBen/tree/main/data`（其中包含 `MapFinSA` 文件夹）。清洗后的数据已经提交到仓库，新成员克隆后可以直接使用：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

原始 MapFinSA 文件通常只有训练集和验证集，当前也保存在：

```text
data/CroFinSA/MapFinSA_train.json
data/CroFinSA/MapFinSA_valid.json
```

一般无需重新下载或清洗。只有原始数据更新或清洗逻辑调整时，才需要重新运行清洗脚本。

由于原始数据集包含多语言（如日语），而本任务仅针对中文，我们需要运行中文清洗脚本：

```bash
python src/clean_chinese_data.py
```

脚本采用“剔除假名 + 校验中文标签”的联合清洗策略。`text` 中平假名和片假名占非空白字符的比例超过 0.01 时直接过滤；同时要求 `answer`、`label` 或 `choices` 中出现“积极”“消极”或“中性”标签。测试集从训练集按标签分层抽取 10%，`random_state=42`，保证训练集与测试集的类别比例一致。原始文件不会被删除或修改。执行后生成：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

可以通过参数调整输入、输出目录和测试集比例：

```bash
python src/clean_chinese_data.py \
  --input data/CroFinSA/MapFinSA_train.json data/CroFinSA/MapFinSA_valid.json \
  --output_dir data/CroFinSA \
  --test_ratio 0.1
```

`--threshold` 参数仍保留用于兼容旧版命令，但当前不再参与过滤。当前 `test.json` 是为了跑通 `evaluate.py` 并生成评测指标而从训练集切分出的本地测试集，不代表官方独立测试结果。如果后续获得官方独立测试集，请直接替换 `data/CroFinSA/test.json`。

## 使用方法

默认配置在 `configs/default.yaml` 中，面向 GPU 训练机，使用 `cuda`、`batch_size=16` 和混合精度。训练、评估、推理和清洗入口都会直接读取固定路径的默认配置，不需要传入 `--config`。

### 配置切换实验

四个实验预设是完整配置，复制覆盖 `configs/default.yaml` 即可切换。以 BERT 基础实验为例：

```bash
cp configs/config_bert_base.yaml configs/default.yaml
python src/train.py
python src/evaluate.py
```

PowerShell 用户可运行：

```powershell
Copy-Item configs/config_bert_base.yaml configs/default.yaml
python src/train.py
python src/evaluate.py
```

五种实验配置如下：

| 实验 | 配置或入口 | 模型 | 改善方法 |
| --- | --- | --- | --- |
| BERT 基础 | `config_bert_base.yaml` | `bert-base-chinese` | 无 |
| BERT 改进 | `config_bert_improved.yaml` | `bert-base-chinese` | 类别权重 |
| RoBERTa 基础 | `config_roberta_base.yaml` | `hfl/chinese-roberta-wwm-ext` | 无 |
| RoBERTa 改进 | `config_roberta_improved.yaml` | `hfl/chinese-roberta-wwm-ext` | 类别权重 |
| 机器学习基线 | `python src/baseline.py` | TF-IDF + 逻辑回归 | 无 |

`use_class_weights: true` 时，训练集会根据三类样本频次计算平衡权重并传入交叉熵损失。`early_stopping_patience` 控制验证指标连续未提升时提前结束训练。

### 深度模型训练

复制所需预设配置后运行：

```bash
python src/train.py
```

训练输出位于 `results/<experiment_name>/`：

```text
best_model.pt
last_model.pt
training_history.json
training_curves.png
```

其中曲线图包含训练与验证 Loss 以及验证 Macro-F1。

本地无 GPU 使用 CPU 调试时，不需要额外配置文件，直接在命令行覆盖关键参数：

```bash
python src/train.py \
  --device cpu \
  --batch_size 2 \
  --eval_batch_size 2 \
  --epochs 1 \
  --num_workers 0
```

CPU 模式下即使配置中的 `fp16` 为 `true`，训练脚本也会自动关闭混合精度。

测试集评估使用：

```bash
python src/evaluate.py
```

CPU 调试可运行：

```bash
python src/evaluate.py \
  --device cpu \
  --batch_size 4 \
  --checkpoint results/bert_base/best_model.pt \
  --test_file data/CroFinSA/test.json
```

评估结果写入 `results/<experiment_name>/`：

```text
metrics.json
confusion_matrix.png
```

`metrics.json` 包含 ACC、加权 F1、Macro-F1、MCC、分类别指标和混淆矩阵。

### 机器学习基线

运行：

```bash
python src/baseline.py
```

基线使用中文字符 1 至 3 gram 的 TF-IDF 特征和逻辑回归，在同一个测试集上计算指标。结果保存在：

```text
results/baseline/metrics.json
results/baseline/confusion_matrix.png
```

深度模型四组实验加上机器学习基线，最终形成五组可对比的实验结果。

### 单条与批量推理

单条文本推理：

```bash
python src/predict.py \
  --device cpu \
  --checkpoint results/bert_base/best_model.pt \
  --text "公司本季度净利润同比增长20%，经营活动现金流明显改善。"
```

批量推理支持 JSON、JSONL、CSV 和 TXT：

```bash
python src/predict.py \
  --device cpu \
  --checkpoint results/bert_base/best_model.pt \
  --input_file data/CroFinSA/test.json \
  --output_file results/bert_base/predictions.csv
```

CSV 输入需包含 `text`、`sentence`、`content` 或 `query` 之一作为文本列，可选包含 `id`。TXT 输入按非空行逐条处理。预测 CSV 包含原始文本、中文标签、置信度和三个类别的概率。

## 注意事项

- 数据路径不写死在代码中，默认值集中在 `configs/default.yaml`，可通过 `--train_file`、`--valid_file`、`--test_file` 覆盖。
- 推理和评估优先读取 checkpoint 内记录的模型名称、标签顺序和最大长度，避免训练与部署错位。
- 每次训练、评估或推理前确认 `experiment_name`，不同实验的产物会分别写入 `results/<experiment_name>/`。
- `data/CroFinSA/` 下清洗后的 JSON、`results/` 下的模型、PNG 图和 JSON 指标都可以提交，方便团队共享实验产物。
- 训练机可根据显存调整 `batch_size`、`eval_batch_size` 和 `gradient_accumulation_steps`，不需要新建配置文件。
