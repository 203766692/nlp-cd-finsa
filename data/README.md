# 数据说明

## 数据集与来源

本项目使用 MapFinBen 基准中的 FinSA 金融文本情感分析子任务，本地数据目录沿用课程约定的 `CroFinSA` 名称。数据覆盖中文、英文、日文、葡萄牙语、印尼语和希腊语等金融文本，类别统一映射为消极、中性、积极三类。

课程数据由指导教师或课程平台提供。请先获取数据文件，再放到 `data/CroFinSA/` 下；仓库不提交任何原始 JSON 文件。公开数据可通过 MapFinBen 项目或课程提供的下载地址获取，下载和使用时应遵守数据集的许可与引用要求。

如果已从课程平台取得训练集和验证集的下载地址，可运行：

```bash
TRAIN_URL="训练集地址" \
VALID_URL="验证集地址" \
bash scripts/download_data.sh
```

如果课程还提供独立测试集，可额外传入 `TEST_URL`。脚本也支持 `--train-url`、`--valid-url`、`--test-url` 和 `--data-dir` 参数。教师直接提供的本地文件无需下载，手动放入下述目录即可。

课程提供的原始 MapFinSA 文件通常只有训练集和验证集，没有独立测试集。默认原始文件位置为：

```text
data/CroFinSA/MapFinSA_train.json
data/CroFinSA/MapFinSA_valid.json
```

运行中文清洗脚本后，会从清洗后的训练集中随机切分 10% 作为本地测试集：

```bash
python scripts/clean_chinese_data.py
```

脚本采用“剔除假名 + 校验中文标签”的联合清洗策略：

- 统计 `text` 中平假名和片假名占全部非空白字符的比例，超过 0.01 时直接过滤。日文常包含大量汉字，仅靠 CJK 汉字占比无法可靠区分中文和日文。
- 检查 `answer` 或 `label` 字段；字段存在时其主标签必须是“积极”“消极”或“中性”。如果提供 `choices`，候选列表中还必须包含至少一个中文标签。
- 两个条件都满足后才计算并记录 `clean_chinese_ratio`，该比例不再参与过滤。

训练集与测试集切分使用固定随机种子 42。可以通过参数调整输入、输出目录和测试集比例：

```bash
python scripts/clean_chinese_data.py \
  --input data/CroFinSA/MapFinSA_train.json data/CroFinSA/MapFinSA_valid.json \
  --output_dir data/CroFinSA \
  --test_ratio 0.1
```

`--threshold` 参数仍可传入以兼容旧命令，但已经不影响样本保留结果。

清洗脚本不会删除或修改原始文件，输出如下：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

其中 `train.json` 和 `test.json` 来自清洗后的原始训练集，`valid.json` 来自清洗后的原始验证集且不参与切分。原始数据没有官方测试集，因此当前 `test.json` 是为了跑通 `evaluate.py` 而生成的本地测试集。如果后续获得官方独立测试集，请直接替换 `data/CroFinSA/test.json`。

## 放置路径

默认配置读取以下文件：

```text
data/CroFinSA/train.json
data/CroFinSA/valid.json
data/CroFinSA/test.json
```

文件名也可使用 `CroFinSA_train.json`、`CroFinSA_valid.json`、`CroFinSA_test.json` 等常见形式，训练、评估和推理脚本也支持通过 `--train_file`、`--valid_file`、`--test_file` 显式指定路径。

清洗脚本会保留原始样本的全部字段，并增加 `clean_chinese_ratio` 字段记录通过联合条件后的中文字符占比，便于人工抽查。该字段只是记录信息，不是过滤条件。

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
| `clean_chinese_ratio` | 清洗脚本增加的记录字段，表示保留样本的非空白字符中 CJK 统一表意文字所占比例。 |

标签会统一为 `negative`、`neutral`、`positive`，并固定按该顺序编码为 0、1、2。当记录同时给出 `choices` 和 `gold` 时，优先根据 `choices[gold]` 解析标签。

## 数据划分用途

- `train.json`：用于参数更新和模型训练。
- `valid.json`：用于每个 epoch 后的验证、最佳 checkpoint 选择。
- `test.json`：仅在最终评估时使用，用于计算 ACC、F1、Macro-F1 和 MCC。原始数据没有测试集时，默认由清洗后的训练集切分 10% 生成本地测试集；若获得官方测试集，应直接替换该文件。

如果数据目录或文件缺失，程序会给出明确的中文错误，不会静默跳过。
