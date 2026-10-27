"""在测试集上评估金融情感分类模型并写入指标文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from src.data_loader import LABEL_NAMES, build_dataset, dataset_summary
from src.metrics import calculate_metrics
from src.model import load_checkpoint
from src.utils import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    """解析评估命令行参数。

    无输入，返回参数对象；未指定的路径与批大小会从配置文件读取。
    """
    parser = argparse.ArgumentParser(description="评估中文金融情感分类模型")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", default=None, help="待评估 checkpoint 路径")
    parser.add_argument("--test_file", default=None, help="测试集 JSON 路径")
    parser.add_argument("--output_file", default=None, help="指标 JSON 输出路径")
    parser.add_argument("--batch_size", type=int, default=None, help="评估批大小")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader 工作进程数")
    parser.add_argument("--device", default=None, help="cpu、cuda 或 auto")
    return parser.parse_args()


def merge_eval_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """合并评估配置和命令行覆盖值。

    输入配置字典与参数对象，返回合并后的新字典；命令行值为 None 时沿用配置。
    """
    merged = dict(config)
    overrides = {
        "checkpoint": args.checkpoint,
        "test_file": args.test_file,
        "output_file": args.output_file,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value

    if not merged.get("checkpoint"):
        merged["checkpoint"] = merged["best_checkpoint"]
    if not merged.get("output_file"):
        merged["output_file"] = "results/metrics.json"
    return merged


def run_evaluation(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """遍历测试集并收集真实标签和预测标签。

    输入模型、测试 DataLoader 和目标设备，返回两个整数列表。
    """
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    with torch.inference_mode():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"]
            logits = model(**batch)
            predictions = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    return all_labels, all_predictions


def build_metric_document(
    labels: list[int],
    predictions: list[int],
    checkpoint_path: Path,
    test_file: str,
) -> dict[str, Any]:
    """构建完整评估结果。

    输入真实标签、预测标签、checkpoint 路径和测试集路径，
    返回包含总体指标、分类别指标和混淆矩阵的字典。
    """
    metrics = calculate_metrics(labels, predictions)
    class_report = classification_report(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=list(LABEL_NAMES),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label_name: {
            "precision": float(class_report[label_name]["precision"]),
            "recall": float(class_report[label_name]["recall"]),
            "f1": float(class_report[label_name]["f1-score"]),
            "support": int(class_report[label_name]["support"]),
        }
        for label_name in LABEL_NAMES
    }
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "test_file": str(Path(test_file).resolve()),
        "num_samples": len(labels),
        "label_names": list(LABEL_NAMES),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(config: dict[str, Any]) -> Path:
    """执行测试集评估并保存 JSON。

    输入合并后的配置，加载 checkpoint 和测试集，返回指标文件路径。
    """
    device = resolve_device(str(config["device"]))
    checkpoint_path = Path(str(config["checkpoint"]))
    model, tokenizer, metadata = load_checkpoint(checkpoint_path, device)

    label_names = tuple(metadata["label_names"])
    if label_names != LABEL_NAMES:
        raise ValueError(
            f"checkpoint 标签顺序 {label_names} 与程序标签顺序 {LABEL_NAMES} 不一致"
        )

    max_length = int(metadata.get("max_length", config["max_length"]))
    test_file = str(config["test_file"])
    test_dataset = build_dataset(
        path=test_file,
        tokenizer=tokenizer,
        max_length=max_length,
        split="test",
        require_labels=True,
        label_names=label_names,
    )
    print(f"测试集：{dataset_summary(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    labels, predictions = run_evaluation(model, test_loader, device)
    document = build_metric_document(labels, predictions, checkpoint_path, test_file)

    output_path = Path(str(config["output_file"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")

    metrics = document["metrics"]
    print(
        f"ACC={metrics['accuracy']:.4f}，F1={metrics['f1']:.4f}，"
        f"Macro-F1={metrics['macro_f1']:.4f}，MCC={metrics['mcc']:.4f}"
    )
    print(f"指标已写入：{output_path.resolve()}")
    return output_path


def main() -> None:
    """读取配置、合并参数并执行测试集评估。"""
    args = parse_args()
    config = merge_eval_config(load_config(args.config), args)
    evaluate(config)


if __name__ == "__main__":
    main()
