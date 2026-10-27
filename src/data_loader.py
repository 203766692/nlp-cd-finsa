"""MapFinBen FinSA 与 CroFinSA 数据读取、标签归一化和 Dataset 构建。"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

LABEL_NAMES: tuple[str, str, str] = ("negative", "neutral", "positive")
DEFAULT_INDEX_TO_LABEL: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

_LABEL_ALIASES: dict[str, str] = {
    "negative": "negative",
    "neg": "negative",
    "消极": "negative",
    "负面": "negative",
    "负向": "negative",
    "ネガティブ": "negative",
    "negatif": "negative",
    "negativo": "negative",
    "negativa": "negative",
    "αρνητικη": "negative",
    "neutral": "neutral",
    "neut": "neutral",
    "中性": "neutral",
    "中立": "neutral",
    "ちゅうりつてき": "neutral",
    "ニュートラル": "neutral",
    "netral": "neutral",
    "neutro": "neutral",
    "neutra": "neutral",
    "ουδετερη": "neutral",
    "positive": "positive",
    "pos": "positive",
    "积极": "positive",
    "正面": "positive",
    "正向": "positive",
    "ポジティブ": "positive",
    "positif": "positive",
    "positivo": "positive",
    "positiva": "positive",
    "θετικη": "positive",
}

_PROMPT_SEPARATOR = re.compile(
    r"(?:文本|句子|文|sentence|text)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "valid": ("valid", "validation", "dev"),
    "test": ("test", "testing"),
}


def _normalize_label_text(value: str) -> str:
    """规范化文本标签。

    输入原始标签字符串，返回去空格、统一大小写和重音符号后的文本，便于匹配多语言别名。
    """
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    decomposed = unicodedata.normalize("NFKD", normalized)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", "", without_marks)


def normalize_label(value: Any, choices: Sequence[Any] | None = None) -> str:
    """将数据标签转换为统一类别。

    输入原始标签和可选候选列表，返回 `negative`、`neutral` 或 `positive`；
    数值标签优先通过 `choices` 映射，否则按 0、1、2 的默认顺序解释。
    """
    if isinstance(value, bool):
        raise ValueError(f"无法识别布尔类型标签：{value}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        index = int(value)
        if index != value:
            raise ValueError(f"标签下标必须为整数：{value}")
        if choices:
            if index < 0 or index >= len(choices):
                raise ValueError(f"标签下标 {index} 超出候选类别范围 0-{len(choices) - 1}")
            return normalize_label(choices[index])
        if index not in DEFAULT_INDEX_TO_LABEL:
            raise ValueError(f"数值标签 {index} 不在默认范围 0、1、2 内")
        return DEFAULT_INDEX_TO_LABEL[index]

    label_text = _normalize_label_text(str(value))
    canonical = _LABEL_ALIASES.get(label_text)
    if canonical is None:
        raise ValueError(f"无法识别的标签：{value!r}")
    return canonical


def _extract_prompt_text(prompt: str) -> str:
    """从任务提示语中提取金融文本。

    输入可能包含类别说明和“文:”前缀的提示，返回最后一个分隔符之后的正文；没有分隔符时返回原文。
    """
    pieces = [piece.strip() for piece in _PROMPT_SEPARATOR.split(prompt) if piece.strip()]
    return pieces[-1] if pieces else prompt.strip()


def extract_text(record: dict[str, Any]) -> str:
    """提取单条记录中的金融文本。

    输入原始记录，按 `text`、常见同义字段、`query`、`conversations` 的顺序查找并返回正文。
    """
    for field in ("text", "sentence", "content", "headline"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    query = record.get("query")
    if isinstance(query, str) and query.strip():
        return _extract_prompt_text(query)

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            value = turn.get("value")
            if role in {"human", "user"} and isinstance(value, str) and value.strip():
                return _extract_prompt_text(value)

    raise ValueError("记录中缺少可用文本字段 text、sentence、content、query 或 conversations")


def _extract_label(record: dict[str, Any]) -> str:
    """提取单条记录中的情感标签。

    输入原始记录，返回统一标签；同时支持 `label`、`answer`、`sentiment` 和 `gold + choices`。
    """
    choices = record.get("choices")
    if not isinstance(choices, list):
        choices = None

    gold = record.get("gold")
    if gold is not None and choices:
        return normalize_label(gold, choices)

    for field in ("label", "answer", "sentiment", "polarity", "gold"):
        value = record.get(field)
        if value is not None and value != "":
            return normalize_label(value, choices)
    raise ValueError("记录中缺少可用标签字段 label、answer、sentiment、polarity 或 gold")


def _read_json(path: Path) -> Any:
    """读取 JSON 文件。

    输入文件路径，返回解析后的对象；文件缺失或 JSON 格式错误时抛出带路径的中文错误。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"数据文件不存在：{path.resolve()}。"
            "请按 data/README.md 的说明准备数据，或通过文件参数指定实际路径。"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 文件格式错误：{path.resolve()}，"
            f"第 {error.lineno} 行第 {error.colno} 列，{error.msg}"
        ) from error


def _select_split_payload(payload: Any, split: str, path: Path) -> list[dict[str, Any]]:
    """从 JSON 根节点中选出指定数据划分。

    输入解析后的 JSON、split 名称和文件路径，返回记录列表；结构不符合要求时抛出中文错误。
    """
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        candidates = _SPLIT_ALIASES.get(split, (split,))
        records = next((payload[key] for key in candidates if key in payload), None)
        if records is None and "data" in payload and isinstance(payload["data"], list):
            records = payload["data"]
        if records is None:
            raise ValueError(
                f"数据文件 {path.resolve()} 中找不到 {split} 划分。"
                f"可用键：{', '.join(map(str, payload.keys())) or '无'}"
            )
    else:
        raise ValueError(f"数据文件 {path.resolve()} 的根节点必须是数组或对象")

    if not isinstance(records, list) or not records:
        raise ValueError(f"数据文件 {path.resolve()} 的 {split} 划分为空或不是数组")

    invalid_index = next(
        (index for index, record in enumerate(records) if not isinstance(record, dict)),
        None,
    )
    if invalid_index is not None:
        raise ValueError(
            f"数据文件 {path.resolve()} 的 {split} 划分第 {invalid_index + 1} 条记录不是对象"
        )
    return records


def resolve_data_path(path: str | Path, split: str) -> Path:
    """解析数据文件路径。

    输入文件或目录路径及 split 名称，返回实际 JSON 路径；支持目录下的 train/valid/test 常见命名。
    """
    candidate_path = Path(path)
    if candidate_path.is_file():
        return candidate_path
    if not candidate_path.is_dir():
        raise FileNotFoundError(
            f"{split} 数据路径不存在：{candidate_path.resolve()}。"
            "请检查 data/README.md 中的放置说明。"
        )

    names = {
        "train": ("train.json", "CroFinSA_train.json", "MapFinSA_train.json"),
        "valid": (
            "valid.json",
            "validation.json",
            "dev.json",
            "CroFinSA_valid.json",
            "MapFinSA_valid.json",
        ),
        "test": ("test.json", "CroFinSA_test.json", "MapFinSA_test.json"),
    }.get(split, (f"{split}.json",))

    for name in names:
        candidate = candidate_path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"在目录 {candidate_path.resolve()} 中找不到 {split} 数据文件，"
        f"已尝试：{', '.join(names)}"
    )


def load_records(path: str | Path, split: str, require_labels: bool) -> list[dict[str, Any]]:
    """读取并校验一个数据划分。

    输入数据路径、split 名称和是否要求标签，返回统一字段后的记录列表；非法记录会报告序号。
    """
    resolved_path = resolve_data_path(path, split)
    records = _select_split_payload(_read_json(resolved_path), split, resolved_path)
    normalized_records: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        try:
            text = extract_text(record)
            label = _extract_label(record) if require_labels else None
        except ValueError as error:
            raise ValueError(
                f"数据文件 {resolved_path.resolve()} 的 {split} 第 {index} 条记录无效：{error}"
            ) from error

        normalized_records.append(
            {
                "id": record.get("id", f"{split}-{index}"),
                "text": text,
                "label": label,
            }
        )
    return normalized_records


class FinancialSentimentDataset(Dataset):
    """供 PyTorch DataLoader 使用的金融情感文本数据集。"""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str | None],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        label_names: Sequence[str] = LABEL_NAMES,
    ) -> None:
        """保存文本、编码标签并检查长度。

        输入文本列表、标签列表、tokenizer、最大长度和标签顺序；
        标签为 `None` 时用于无标注推理。
        """
        if len(texts) != len(labels):
            raise ValueError("文本数量和标签数量不一致")
        if max_length <= 0:
            raise ValueError("max_length 必须为正整数")
        if tuple(label_names) != LABEL_NAMES:
            raise ValueError(f"标签顺序必须为 {LABEL_NAMES}")

        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = tuple(label_names)
        self.label_to_id = {name: index for index, name in enumerate(self.label_names)}

        for label in self.labels:
            if label is not None and label not in self.label_to_id:
                raise ValueError(f"标签 {label!r} 不在标签顺序 {self.label_names} 中")

    def __len__(self) -> int:
        """返回样本总数。"""
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """返回单条样本的张量字典。

        输入样本下标，输出 tokenizer 编码和可选 `labels`；固定长度填充便于批量训练。
        """
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        item = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in encoded.items()
        }
        label = self.labels[index]
        if label is not None:
            item["labels"] = torch.tensor(self.label_to_id[label], dtype=torch.long)
        return item

    def class_counts(self) -> dict[str, int]:
        """统计有标注样本的类别数量。

        无输入，返回标签到样本数的字典；无标注样本不计入。
        """
        counts = Counter(label for label in self.labels if label is not None)
        return {name: counts.get(name, 0) for name in self.label_names}


def build_dataset(
    path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    split: str,
    require_labels: bool,
    label_names: Sequence[str] = LABEL_NAMES,
) -> FinancialSentimentDataset:
    """构建指定数据划分的 Dataset。

    输入 JSON 路径、tokenizer、最大长度、split、是否要求标签及标签顺序，
    返回编码后的 `FinancialSentimentDataset`。
    """
    records = load_records(path, split=split, require_labels=require_labels)
    texts = [str(record["text"]) for record in records]
    labels = [record["label"] for record in records] if require_labels else [None] * len(records)
    return FinancialSentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        label_names=label_names,
    )


def dataset_summary(dataset: FinancialSentimentDataset) -> str:
    """生成数据集类别分布摘要。

    输入数据集，返回适合终端输出的字符串。
    """
    counts = dataset.class_counts()
    details = "，".join(f"{label}={count}" for label, count in counts.items())
    return f"样本数={len(dataset)}，{details}"
