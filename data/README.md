# 数据说明

## 数据集与来源

本项目对应 CCL 2026 MapFinBen FinSA 中文金融文本情感分析任务。数据来源于 MapFinBen 基准，官方项目主页为：`https://github.com/HgITSE/MapFinBen`，数据目录为：`https://github.com/HgITSE/MapFinBen/tree/main/data`。

原始数据集（MapFinSA）包含中文、英文、日文等多种语言的金融文本，但本课程设计任务**仅针对中文金融文本**进行情感分类，因此在使用前需要通过清洗脚本过滤出中文样本。

清洗后的 `train.json`、`valid.json` 和 `test.json` 已提交到仓库，新成员克隆后无需重新运行清洗脚本，可以直接开始训练和评估。

## 数据下载

课程提供的原始 MapFinSA 文件通常只有训练集和验证集，没有独立测试集。当前仓库中的原始文件位置为：

```text
data/CroFinSA/MapFinSA_train.json
data/CroFinSA/MapFinSA_valid.json
```

一般不需要重新下载。只有官方原始数据更新或需要重建中文数据集时，才从上述 GitHub 数据目录获取文件并覆盖本地原始文件。

## 中文数据清洗

数据清洗脚本位于 `src/clean_chinese_data.py`。仓库已经包含清洗结果；只有原始数据更新或清洗逻辑调整时，才需要重新执行。

运行以下命令：

```bash
python src/clean_chinese_data.py
```

脚本采用“剔除假名 + 校验中文标签”的联合清洗策略：
- 统计 `text` 中平假名和片假名占全部非空白字符的比例，超过 0.01 时直接过滤。这是因为日文常包含大量汉字，仅靠汉字占比无法区分中文和日文。
- 检查 `answer` 或 `label` 字段；其主标签必须是“积极”“消极”或“中性”。如果提供 `choices`，候选列表中还必须包含至少一个中文标签。
- 两个条件都满足后才会保留，并记录 `clean_chinese_ratio`（非空白字符中汉字的占比），以便人工抽查。

由于官方数据没有测试集，脚本会从清洗后的训练集中随机切分 10%（固定随机种子 42）作为本地测试集。你可以通过参数调整输入路径或切分比例：

```bash
python src/clean_chinese_data.py \
  --input data/CroFinSA/MapFinSA_train.json data/CroFinSA/MapFinSA_valid.json \
  --output_dir data/CroFinSA \
  --test_ratio 0.1
```

清洗完成后，会生成以下三个文件：
```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

原始数据不会被删除或修改。如果后续获得官方独立的测试集，直接替换 `data/CroFinSA/test.json` 即可。

## JSON 字段说明

清洗后的文件均为 JSON 数组，包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本编号。 |
| `text` | 待分析的金融文本。 |
| `label` | 训练集情感标签（如果存在）。 |
| `query` | 带任务说明的提示语，如果缺少 `text` 会从中提取。 |
| `answer` | 验证集或测试集的情感答案。 |
| `choices` | 候选类别列表。 |
| `gold` | 正确选项在 `choices` 中的下标，从 0 开始。 |
| `conversations` | 对话式样本，包含人类提示和模型回答。 |
| `clean_chinese_ratio` | 清洗时记录的中文字符占比，仅供抽查参考。 |

情感标签仅接受中文“积极”“中性”“消极”及其同义词，并统一映射为英文标准键 `positive`、`neutral`、`negative`，对应编码 2、1、0。当记录同时提供 `choices` 和 `gold` 时，程序优先根据 `choices[gold]` 解析标签。数据加载器不再支持日语、希腊语、印尼语、葡萄牙语等其他语言标签。

## 数据划分用途

- `train.json`：用于模型训练和参数更新。
- `valid.json`：用于每个 epoch 后的验证，并据此选择最佳 checkpoint。
- `test.json`：仅在最终评估时使用，计算 ACC、F1、Macro-F1 和 MCC。目前由训练集切分而来。

如果文件缺失，训练或评估程序会给出明确的中文错误提示。
