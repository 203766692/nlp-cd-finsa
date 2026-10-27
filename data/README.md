# 数据说明

## 数据集与来源

本项目使用 MapFinBen 基准中的 FinSA 金融文本情感分析子任务，本地数据目录沿用课程约定的 `CroFinSA` 名称。数据覆盖中文、英文、日文、葡萄牙语、印尼语和希腊语等金融文本，类别统一映射为消极、中性、积极三类。

课程数据由指导教师或课程平台提供。请先获取数据文件，再放到 `data/CroFinSA/` 下；仓库不提交任何原始 JSON 文件。公开数据可通过 MapFinBen 项目或课程提供的下载地址获取，下载和使用时应遵守数据集的许可与引用要求。

如果已从课程平台取得三个文件的下载地址，可运行：

```bash
TRAIN_URL="训练集地址" \
VALID_URL="验证集地址" \
TEST_URL="测试集地址" \
bash scripts/download_data.sh
```

脚本也支持 `--train-url`、`--valid-url`、`--test-url` 和 `--data-dir` 参数。教师直接提供的本地文件无需下载，手动放入下述目录即可。

## 放置路径

默认配置读取以下文件：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

文件名也可使用 `CroFinSA_train.json`、`CroFinSA_valid.json`、`CroFinSA_test.json` 等常见形式，训练、评估和推理脚本也支持通过 `--train_file`、`--valid_file`、`--test_file` 显式指定路径。

## JSON 字段

文件应为 JSON 数组，或包含对应 split 键的对象。不同来源的记录可能使用以下字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 样本编号。 |
| `text` | 待分析的金融文本；缺省时程序会从 `query` 或 `conversations` 中提取。 |
| `label` | 训练集情感标签，可写中文、英文、日文等类别别名。 |
| `query` | 带任务说明和文本的提示语，加载时会提取实际金融文本。 |
| `answer` | 验证集或测试集的情感答案。 |
| `choices` | 候选类别列表。 |
| `gold` | 候选类别中的正确选项下标，从 0 开始。 |
| `conversations` | 对话式样本，包含 `human` 提示和 `agent` 回答。 |

标签会统一为 `negative`、`neutral`、`positive`，并固定按该顺序编码为 0、1、2。当记录同时给出 `choices` 和 `gold` 时，优先根据 `choices[gold]` 解析标签。

## 数据划分用途

- `train.json`：用于参数更新和模型训练。
- `valid.json`：用于每个 epoch 后的验证、最佳 checkpoint 选择。
- `test.json`：仅在最终评估时使用，用于计算 ACC、F1、Macro-F1 和 MCC。

如果数据目录或文件缺失，程序会给出明确的中文错误，不会静默跳过。
