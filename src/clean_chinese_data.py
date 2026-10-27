"""通过假名过滤和中文标签校验清洗 MapFinSA，并切分本地测试集。"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data/CroFinSA/MapFinSA_train.json",
    PROJECT_ROOT / "data/CroFinSA/MapFinSA_valid.json",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/CroFinSA"
DEFAULT_THRESHOLD = 0.5
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42
KANA_RATIO_THRESHOLD = 0.01
CHINESE_LABELS: tuple[str, str, str] = ("积极", "消极", "中性")

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train", "training"),
    "valid": ("valid", "validation", "dev"),
    "test": ("test", "testing"),
}


@dataclass
class CleaningSummary:
    """保存单个输入文件的清洗统计。"""

    source: Path
    role: str
    original_count: int
    kept_count: int
    kana_filtered_count: int
    label_filtered_count: int
    empty_text_count: int
    invalid_record_count: int
    ratios: list[float]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    无输入，返回参数对象；未指定 `--input` 时读取数据目录下默认的训练集和验证集。
    """
    parser = argparse.ArgumentParser(description="清洗 MapFinSA 中文样本并切分本地测试集")
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="一个或多个原始 JSON 文件，文件名需包含 train、valid 或 test",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="清洗结果输出目录，默认 data/CroFinSA",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="兼容旧版命令；中文占比当前仅记录，不参与过滤",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="从清洗后训练集切分测试集的比例，默认 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="训练集与测试集切分随机种子，默认 42",
    )
    return parser.parse_args()


def infer_split_role(path: Path) -> str:
    """识别输入文件所属的数据划分。

    输入 JSON 文件路径，根据文件名中的 train、valid 或 test 标记返回对应角色；
    无法识别时抛出中文错误。
    """
    tokens = set(re.split(r"[_\-\s.]+", path.stem.casefold()))
    for role, aliases in ROLE_ALIASES.items():
        if tokens.intersection(aliases):
            return role
    raise ValueError(
        f"无法从文件名识别数据划分：{path.resolve()}。"
        "文件名需包含 train、valid 或 test。"
    )


def extract_payload_records(payload: Any, role: str, source: Path) -> list[Any]:
    """从 JSON 根节点提取记录列表。

    输入解析后的 JSON、数据划分角色和文件路径，返回记录列表；
    支持 JSON 数组及包含 `data` 或对应 split 键的对象。
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        candidate_keys = (role, *ROLE_ALIASES.get(role, ()), "data", "records")
        for key in candidate_keys:
            records = payload.get(key)
            if isinstance(records, list):
                return records
    raise ValueError(
        f"JSON 文件 {source.resolve()} 的根节点必须是数组，"
        "或包含 data、records 或对应 split 键的对象。"
    )


def load_records(path: Path, role: str) -> list[Any]:
    """读取 JSON 记录。

    输入文件路径和数据划分角色，返回原始记录列表；文件缺失或 JSON 格式错误时给出中文错误。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"输入文件不存在：{path.resolve()}。"
            "请确认原始 MapFinSA 文件路径，或通过 --input 显式指定。"
        )
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 文件格式错误：{path.resolve()}，"
            f"第 {error.lineno} 行第 {error.colno} 列，{error.msg}"
        ) from error
    return extract_payload_records(payload, role, path)


def is_cjk_character(character: str) -> bool:
    """判断字符是否属于 CJK 统一表意文字。

    输入单个字符，若位于 Unicode 范围 `U+4E00` 至 `U+9FFF` 则返回 True。
    """
    return "\u4e00" <= character <= "\u9fff"


def calculate_chinese_ratio(text: str) -> float:
    """计算文本的中文字符占比。

    输入文本，返回 CJK 统一表意文字数量除以全部非空白字符数量的结果；
    没有非空白字符时返回 0.0。
    """
    non_whitespace = [character for character in text if not character.isspace()]
    if not non_whitespace:
        return 0.0
    cjk_count = sum(1 for character in non_whitespace if is_cjk_character(character))
    return cjk_count / len(non_whitespace)


def is_kana_character(character: str) -> bool:
    """判断字符是否属于日文平假名或片假名。

    输入单个字符，若位于 `U+3040` 至 `U+309F` 或 `U+30A0` 至 `U+30FF` 则返回 True。
    """
    return "\u3040" <= character <= "\u309f" or "\u30a0" <= character <= "\u30ff"


def calculate_kana_ratio(text: str) -> float:
    """计算文本中的日文假名占比。

    输入文本，返回平假名和片假名数量除以全部非空白字符数量的结果；
    没有非空白字符时返回 0.0。
    """
    non_whitespace = [character for character in text if not character.isspace()]
    if not non_whitespace:
        return 0.0
    kana_count = sum(1 for character in non_whitespace if is_kana_character(character))
    return kana_count / len(non_whitespace)


def extract_label_values(value: Any) -> list[str]:
    """将标签字段转换为待校验字符串列表。

    输入字符串或字符串列表，返回去除首尾空白后的非空字符串列表；
    其他类型返回空列表。
    """
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
    return []


def has_chinese_label(record: dict[str, Any]) -> bool:
    """校验样本是否包含中文情感标签。

    输入原始记录，检查存在的 `answer`、`label` 字段及 `choices` 列表；
    存在的主标签必须是“积极”“消极”或“中性”，若提供 choices 也必须包含中文标签。
    """
    found_label = False
    for field in ("answer", "label"):
        if field not in record or record[field] in (None, ""):
            continue
        values = extract_label_values(record[field])
        if not values or not any(value in CHINESE_LABELS for value in values):
            return False
        found_label = True

    if "choices" in record and record["choices"] is not None:
        choices = extract_label_values(record["choices"])
        if not choices or not any(choice in CHINESE_LABELS for choice in choices):
            return False
        found_label = True

    return found_label


def clean_records(
    records: Sequence[Any],
    role: str,
    source: Path,
) -> tuple[list[dict[str, Any]], CleaningSummary]:
    """按假名和中文标签联合条件过滤记录。

    输入原始记录、数据划分角色和来源路径，返回保留记录及统计；
    假名占比超过 1% 或标签不是中文的记录不会进入输出。
    """
    kept_records: list[dict[str, Any]] = []
    ratios: list[float] = []
    kana_filtered_count = 0
    label_filtered_count = 0
    empty_text_count = 0
    invalid_record_count = 0

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            invalid_record_count += 1
            logging.warning("%s 第 %d 条记录不是 JSON 对象，已过滤", source, index)
            continue

        text = record.get("text")
        if not isinstance(text, str):
            empty_text_count += 1
            logging.warning("%s 第 %d 条记录的 text 为空或不是字符串，已过滤", source, index)
            continue

        non_whitespace_count = sum(1 for character in text if not character.isspace())
        if non_whitespace_count == 0:
            empty_text_count += 1
            logging.warning("%s 第 %d 条记录没有非空白字符，已过滤", source, index)
            continue

        kana_ratio = calculate_kana_ratio(text)
        if kana_ratio > KANA_RATIO_THRESHOLD:
            kana_filtered_count += 1
            continue

        if not has_chinese_label(record):
            label_filtered_count += 1
            continue

        ratio = calculate_chinese_ratio(text)
        ratios.append(ratio)
        cleaned_record = dict(record)
        cleaned_record["clean_chinese_ratio"] = round(ratio, 6)
        kept_records.append(cleaned_record)

    summary = CleaningSummary(
        source=source,
        role=role,
        original_count=len(records),
        kept_count=len(kept_records),
        kana_filtered_count=kana_filtered_count,
        label_filtered_count=label_filtered_count,
        empty_text_count=empty_text_count,
        invalid_record_count=invalid_record_count,
        ratios=ratios,
    )
    return kept_records, summary


def split_train_test(
    records: Sequence[dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按固定随机种子切分训练集和测试集。

    输入清洗后的训练记录、测试集比例和随机种子，
    返回新的训练集与测试集列表；测试集至少包含一条记录。
    """
    if len(records) < 2:
        raise ValueError("清洗后的训练集少于 2 条，无法切分训练集和测试集")

    test_count = int(round(len(records) * test_ratio))
    test_count = min(max(test_count, 1), len(records) - 1)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    test_indices = set(indices[:test_count])

    train_records = [
        record for index, record in enumerate(records) if index not in test_indices
    ]
    test_records = [
        record for index, record in enumerate(records) if index in test_indices
    ]
    return train_records, test_records


def write_json(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """以 UTF-8 和缩进格式写入 JSON。

    输入输出路径和记录列表，无返回值；先写临时文件再替换，避免留下半写入结果。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def format_ratio_distribution(ratios: Sequence[float]) -> str:
    """格式化中文占比分布。

    输入参与判定的占比列表，返回包含最小值、中位数和最大值的字符串；空列表返回暂无数据。
    """
    if not ratios:
        return "最小值=暂无，中位数=暂无，最大值=暂无"
    return (
        f"最小值={min(ratios):.4f}，"
        f"中位数={statistics.median(ratios):.4f}，"
        f"最大值={max(ratios):.4f}"
    )


def print_summary(summary: CleaningSummary, output_paths: Sequence[Path]) -> None:
    """打印单个数据划分的清洗统计。

    输入清洗统计和输出路径列表，无返回值；统计只输出到终端，不写入 JSON。
    """
    print(
        f"{summary.role}：原始样本数={summary.original_count}，"
        f"清洗后中文样本数={summary.kept_count}，"
        f"假名过滤={summary.kana_filtered_count}，"
        f"标签非中文过滤={summary.label_filtered_count}，"
        f"空文本过滤={summary.empty_text_count}，"
        f"非法记录过滤={summary.invalid_record_count}"
    )
    print(f"{summary.role} 中文占比分布（保留样本）：{format_ratio_distribution(summary.ratios)}")
    for output_path in output_paths:
        print(f"{summary.role} 输出：{output_path.resolve()}")


def run_cleaning(
    input_paths: Sequence[Path],
    output_dir: Path,
    threshold: float,
    test_ratio: float,
    seed: int,
) -> None:
    """执行清洗、训练集切分和 JSON 输出。

    输入原始文件列表、输出目录、兼容阈值、测试集比例和随机种子，
    完成联合条件过滤并打印统计；原始文件只读不修改。
    """
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold 必须在 0 到 1 之间")
    if not 0 < test_ratio < 1:
        raise ValueError("--test_ratio 必须在 0 到 1 之间且不能等于 0 或 1")

    role_paths: dict[str, Path] = {}
    for raw_path in input_paths:
        path = Path(raw_path)
        role = infer_split_role(path)
        if role in role_paths:
            raise ValueError(
                f"检测到多个 {role} 输入文件：{role_paths[role].resolve()} 和 {path.resolve()}"
            )
        role_paths[role] = path

    if "train" not in role_paths:
        raise ValueError("未提供训练集文件，--input 中必须包含文件名带 train 的 JSON")

    output_dir.mkdir(parents=True, exist_ok=True)
    planned_outputs = {"train": output_dir / "train.json"}
    if "test" not in role_paths:
        planned_outputs["test"] = output_dir / "test.json"
    for role in role_paths:
        planned_outputs[role] = output_dir / f"{role}.json"

    for role, input_path in role_paths.items():
        if input_path.resolve() == planned_outputs[role].resolve():
            raise ValueError(
                f"输入文件与输出文件相同，拒绝覆盖原始数据：{input_path.resolve()}"
            )

    print(
        "清洗策略：假名占比超过 0.01 即过滤；"
        "answer、label 或 choices 必须包含中文标签。"
    )
    print(
        f"兼容参数 threshold={threshold} 仅保留用于旧命令，不参与过滤；"
        f"测试集比例={test_ratio}，随机种子={seed}"
    )

    train_records, train_summary = clean_records(
        load_records(role_paths["train"], "train"),
        role="train",
        source=role_paths["train"],
    )
    if not train_records:
        raise ValueError("训练集清洗后没有中文样本，请检查输入文件和联合清洗规则")

    cleaned_by_role: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, CleaningSummary] = {"train": train_summary}
    if "test" in role_paths:
        test_records, test_summary = clean_records(
            load_records(role_paths["test"], "test"),
            role="test",
            source=role_paths["test"],
        )
        if not test_records:
            raise ValueError("独立测试集清洗后没有中文样本")
        cleaned_by_role["train"] = train_records
        cleaned_by_role["test"] = test_records
        summaries["test"] = test_summary
        print("检测到独立 test 文件，训练集不再切分，直接使用清洗后的独立测试集。")
    else:
        cleaned_by_role["train"], cleaned_by_role["test"] = split_train_test(
            train_records,
            test_ratio=test_ratio,
            seed=seed,
        )

    if "valid" in role_paths:
        valid_records, valid_summary = clean_records(
            load_records(role_paths["valid"], "valid"),
            role="valid",
            source=role_paths["valid"],
        )
        if not valid_records:
            raise ValueError("验证集清洗后没有中文样本，请检查输入文件和联合清洗规则")
        cleaned_by_role["valid"] = valid_records
        summaries["valid"] = valid_summary

    for role, records in cleaned_by_role.items():
        write_json(planned_outputs[role], records)

    train_outputs = [planned_outputs["train"]]
    if "test" not in role_paths:
        train_outputs.append(planned_outputs["test"])
    print_summary(train_summary, train_outputs)
    if "test" not in role_paths:
        print(
            f"切分结果：训练集样本数={len(cleaned_by_role['train'])}，"
            f"测试集样本数={len(cleaned_by_role['test'])}"
        )

    if "valid" in role_paths:
        print_summary(valid_summary, [planned_outputs["valid"]])
    if "test" in role_paths:
        print_summary(test_summary, [planned_outputs["test"]])


def main() -> None:
    """解析参数并执行中文数据清洗。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    input_paths = (
        [Path(path) for path in args.input]
        if args.input is not None
        else list(DEFAULT_INPUTS)
    )
    run_cleaning(
        input_paths=input_paths,
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
