"""加载 checkpoint，对单条或批量金融文本执行情感推理。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    LABEL_NAMES,
    FinancialSentimentDataset,
    extract_text,
    load_records,
)
from src.model import load_checkpoint
from src.utils import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_device,
    resolve_experiment_dir,
)

LABEL_ZH = {
    "negative": "消极",
    "neutral": "中性",
    "positive": "积极",
}


def parse_args() -> argparse.Namespace:
    """解析推理命令行参数。

    无输入，返回参数对象；单条文本和批量输入文件必须二选一，其他字段沿用默认配置。
    """
    parser = argparse.ArgumentParser(description="对金融文本执行情感推理")
    parser.add_argument("--checkpoint", default=None, help="推理 checkpoint 路径")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", default=None, help="单条待推理文本")
    input_group.add_argument(
        "--input_file",
        default=None,
        help="批量输入文件，支持 JSON、JSONL、CSV 或 TXT",
    )
    parser.add_argument("--output_file", default=None, help="预测 CSV 输出路径")
    parser.add_argument("--batch_size", type=int, default=None, help="推理批大小")
    parser.add_argument("--max_length", type=int, default=None, help="最大序列长度")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader 工作进程数")
    parser.add_argument("--device", default=None, help="cpu、cuda 或 auto")
    return parser.parse_args()


def merge_predict_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """合并推理配置和命令行覆盖值。

    输入配置字典与参数对象，返回合并后的新字典。
    """
    merged = dict(config)
    overrides = {
        "checkpoint": args.checkpoint,
        "output_file": args.output_file,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    if args.max_length is not None:
        merged["max_length_override"] = args.max_length
    experiment_dir = resolve_experiment_dir(str(merged["experiment_name"]))
    if not merged.get("checkpoint"):
        merged["checkpoint"] = str(experiment_dir / "best_model.pt")
    if not merged.get("output_file"):
        merged["output_file"] = str(experiment_dir / "predictions.csv")
    return merged


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 记录。

    输入文件路径，返回记录字典列表；空行会跳过，非法行会报告行号。
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 文件 {path.resolve()} 第 {line_number} 行格式错误：{error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL 文件 {path.resolve()} 第 {line_number} 行必须是对象"
                )
            records.append(record)
    return records


def load_prediction_inputs(
    text: str | None,
    input_file: str | None,
) -> list[dict[str, Any]]:
    """读取单条或批量推理输入。

    输入单条文本或文件路径，返回含 `id` 与 `text` 的记录列表；
    支持 JSON、JSONL、CSV 和逐行 TXT。
    """
    if text is not None:
        if not text.strip():
            raise ValueError("--text 不能为空")
        return [{"id": "single-1", "text": text.strip()}]

    if input_file is None:
        raise ValueError("必须提供 --text 或 --input_file")
    path = Path(input_file)
    if not path.is_file():
        raise FileNotFoundError(f"推理输入文件不存在：{path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        records = load_records(path, split="test", require_labels=False)
    elif suffix == ".jsonl":
        records = [
            {
                "id": record.get("id", f"jsonl-{index}"),
                "text": extract_text(record),
            }
            for index, record in enumerate(_read_jsonl(path), start=1)
        ]
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        text_column = next(
            (column for column in ("text", "sentence", "content", "query") if column in frame),
            None,
        )
        if text_column is None:
            raise ValueError(
                f"CSV 文件 {path.resolve()} 缺少文本列，可用列名：text、sentence、content、query"
            )
        id_column = "id" if "id" in frame.columns else None
        records = []
        for index, row in frame.iterrows():
            raw_value = str(row[text_column]).strip()
            if not raw_value or raw_value.lower() == "nan":
                continue
            record = {"text": extract_text({text_column: raw_value})}
            record["id"] = str(row[id_column]) if id_column else f"csv-{index + 1}"
            records.append(record)
    elif suffix == ".txt":
        records = [
            {"id": f"txt-{index}", "text": line.strip()}
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if line.strip()
        ]
    else:
        raise ValueError(
            f"不支持的输入文件格式：{suffix or '无扩展名'}，请使用 .json、.jsonl、.csv 或 .txt"
        )

    if not records:
        raise ValueError(f"推理输入文件没有有效文本：{path.resolve()}")
    return records


def run_prediction(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """批量执行模型推理。

    输入模型、DataLoader、设备和标签顺序，按批次返回预测类别、置信度和各类概率。
    """
    model.eval()
    outputs: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, prediction_ids = torch.max(probabilities, dim=-1)
            for row_index in range(prediction_ids.size(0)):
                label_id = int(prediction_ids[row_index].item())
                output = {
                    "label": label_names[label_id],
                    "label_zh": LABEL_ZH[label_names[label_id]],
                    "label_id": label_id,
                    "confidence": float(confidence[row_index].item()),
                }
                for probability_index, label_name in enumerate(label_names):
                    output[f"prob_{label_name}"] = float(
                        probabilities[row_index, probability_index].item()
                    )
                outputs.append(output)
    return outputs


def predict(
    config: dict[str, Any],
    text: str | None,
    input_file: str | None,
) -> Path:
    """执行推理并保存预测 CSV。

    输入配置、单条文本或批量文件，返回预测结果路径。
    """
    device = resolve_device(str(config["device"]))
    experiment_name = str(config["experiment_name"]).strip()
    checkpoint_path = Path(str(config["checkpoint"]))
    model, tokenizer, metadata = load_checkpoint(checkpoint_path, device)
    checkpoint_experiment = metadata.get("train_config", {}).get("experiment_name")
    if checkpoint_experiment and str(checkpoint_experiment) != experiment_name:
        raise ValueError(
            f"checkpoint 属于实验 {checkpoint_experiment}，"
            f"但当前配置为 {experiment_name}"
        )
    label_names = tuple(metadata["label_names"])
    if label_names != LABEL_NAMES:
        raise ValueError(
            f"checkpoint 标签顺序 {label_names} 与程序标签顺序 {LABEL_NAMES} 不一致"
        )

    records = load_prediction_inputs(text, input_file)
    if config.get("max_length_override") is not None:
        max_length = int(config["max_length_override"])
    else:
        max_length = int(metadata.get("max_length", config.get("max_length", 256)))
    dataset = FinancialSentimentDataset(
        texts=[str(record["text"]) for record in records],
        labels=[None] * len(records),
        tokenizer=tokenizer,
        max_length=max_length,
        label_names=label_names,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    outputs = run_prediction(model, data_loader, device, label_names)

    rows = []
    for record, output in zip(records, outputs, strict=True):
        rows.append(
            {
                "id": record["id"],
                "text": record["text"],
                **output,
            }
        )
    columns = [
        "id",
        "text",
        "label",
        "label_zh",
        "label_id",
        "confidence",
        "prob_negative",
        "prob_neutral",
        "prob_positive",
    ]
    output_path = Path(str(config["output_file"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已完成 {len(rows)} 条预测，结果写入：{output_path.resolve()}")
    return output_path


def main() -> None:
    """读取项目默认配置、合并可选参数并启动单条或批量推理。"""
    args = parse_args()
    config = merge_predict_config(load_config(DEFAULT_CONFIG_PATH), args)
    predict(config, args.text, args.input_file)


if __name__ == "__main__":
    main()
