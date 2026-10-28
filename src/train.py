"""金融情感分类模型训练、验证与 checkpoint 保存。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    LABEL_NAMES,
    FinancialSentimentDataset,
    build_dataset,
    dataset_summary,
)
from src.metrics import calculate_metrics
from src.model import create_model, load_tokenizer
from src.utils import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_device,
    resolve_experiment_dir,
    set_seed,
)


def _parse_bool(value: str | bool) -> bool:
    """解析命令行布尔值。

    输入 true/false、1/0、yes/no 等文本，返回对应布尔值；无法识别时抛出参数错误。
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"无法识别布尔值：{value}")


def parse_args() -> argparse.Namespace:
    """解析训练命令行参数。

    无输入，返回参数对象；仅保留常用覆盖项，未传入时完全沿用默认配置。
    """
    parser = argparse.ArgumentParser(description="训练中文金融情感分类模型")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--model_name", default=None, help="预训练模型名称或本地路径")
    parser.add_argument("--dropout", type=float, default=None, help="分类头前的 dropout")
    parser.add_argument("--train_file", default=None, help="训练集 JSON 路径")
    parser.add_argument("--valid_file", default=None, help="验证集 JSON 路径")
    parser.add_argument("--batch_size", type=int, default=None, help="训练批大小")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="验证批大小")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=None, help="权重衰减")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=None, help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="学习率预热比例")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="梯度累积步数",
    )
    parser.add_argument("--max_grad_norm", type=float, default=None, help="梯度裁剪阈值")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader 工作进程数")
    parser.add_argument("--device", default=None, help="cpu、cuda 或 auto")
    parser.add_argument("--fp16", type=_parse_bool, default=None, help="是否使用 CUDA 混合精度")
    parser.add_argument(
        "--save_best_metric",
        choices=("macro_f1", "f1", "accuracy", "mcc"),
        default=None,
        help="最佳 checkpoint 选择指标",
    )
    return parser.parse_args()


def merge_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """合并 YAML 配置与命令行覆盖值。

    输入配置字典和参数对象，返回合并后的新字典；命令行中为 None 的值不会覆盖配置。
    """
    merged = dict(config)
    override_fields = (
        "seed",
        "model_name",
        "dropout",
        "train_file",
        "valid_file",
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "weight_decay",
        "epochs",
        "max_length",
        "warmup_ratio",
        "gradient_accumulation_steps",
        "max_grad_norm",
        "num_workers",
        "device",
        "fp16",
        "save_best_metric",
    )
    for field in override_fields:
        value = getattr(args, field)
        if value is not None:
            merged[field] = value
    return merged


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """将批次张量移动到指定设备。

    输入批次字典和目标设备，返回移动后的新字典。
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def create_grad_scaler(use_fp16: bool) -> torch.amp.GradScaler:
    """创建混合精度梯度缩放器。

    输入是否启用混合精度，返回现代 `torch.amp.GradScaler`；
    旧版 PyTorch 不支持时回退到 CUDA 命名空间实现。
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_fp16)
    return torch.cuda.amp.GradScaler(enabled=use_fp16)


def create_loss_function(
    config: dict[str, Any],
    train_dataset: FinancialSentimentDataset,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """创建训练损失函数。

    输入配置、训练集和目标设备；启用类权重时，根据训练集类别频次计算平衡权重。
    """
    if not bool(config.get("use_class_weights", False)):
        return nn.CrossEntropyLoss()

    counts = train_dataset.class_counts()
    zero_classes = [label for label, count in counts.items() if count == 0]
    if zero_classes:
        raise ValueError(f"无法计算类别权重，以下类别没有训练样本：{', '.join(zero_classes)}")

    total_samples = sum(counts.values())
    class_weights = [
        total_samples / (len(LABEL_NAMES) * counts[label])
        for label in LABEL_NAMES
    ]
    print(
        "启用类别权重："
        + "，".join(
            f"{label}={weight:.4f}"
            for label, weight in zip(LABEL_NAMES, class_weights, strict=True)
        )
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def save_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    """保存逐轮训练历史。

    输入历史字典和输出路径，将训练与验证 Loss、验证 ACC、F1、Macro-F1、MCC 按 epoch 写入 JSON。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
        file.write("\n")


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
    experiment_name: str,
) -> None:
    """绘制 Loss 和多项验证指标曲线。

    输入历史字典、输出图片路径和实验名称；左图绘制训练与验证 Loss，
    右图绘制验证 Accuracy、F1、Macro-F1、MCC，无返回值。
    """
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["valid_loss"], marker="o", label="Valid loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    metric_series = (
        ("valid_accuracy", "Accuracy", "#2a6f97", "o"),
        ("valid_f1", "Weighted F1", "#e76f51", "s"),
        ("valid_macro_f1", "Macro-F1", "#2a9d8f", "^"),
        ("valid_mcc", "MCC", "#8d5a97", "D"),
    )
    for history_key, label, color, marker in metric_series:
        axes[1].plot(
            epochs,
            history[history_key],
            marker=marker,
            color=color,
            label=label,
        )
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle(f"Experiment: {experiment_name}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool,
) -> tuple[dict[str, float], float]:
    """在验证集上执行一次评估。

    输入模型、DataLoader、损失函数、设备和是否启用混合精度，
    返回指标字典和按样本加权的平均损失。
    """
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in data_loader:
            batch = _move_batch(batch, device)
            labels = batch["labels"]
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_fp16
                else nullcontext()
            )
            with autocast_context:
                logits = model(**batch)
                loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=-1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    metrics = calculate_metrics(all_labels, all_predictions)
    return metrics, total_loss / max(total_samples, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    tokenizer_name: str,
    dropout: float,
    max_length: int,
    epoch: int,
    metrics: dict[str, float],
    train_config: dict[str, Any],
) -> None:
    """保存可被评估和推理脚本恢复的 checkpoint。

    输入保存路径、模型、模型元数据、当前 epoch、指标和训练配置，无返回值。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "num_labels": len(LABEL_NAMES),
        "label_names": list(LABEL_NAMES),
        "dropout": dropout,
        "max_length": max_length,
        "epoch": epoch,
        "metrics": metrics,
        "train_config": train_config,
    }
    torch.save(checkpoint, path)


def train(config: dict[str, Any]) -> Path:
    """执行完整训练流程。

    输入合并后的配置字典，完成数据构建、模型训练、逐轮验证和最佳模型保存，
    返回最佳 checkpoint 路径。
    """
    if int(config.get("num_labels", 3)) != len(LABEL_NAMES):
        raise ValueError(f"当前实现要求 num_labels={len(LABEL_NAMES)}")

    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(str(config["device"]))
    use_fp16 = bool(config.get("fp16", False)) and device.type == "cuda"
    if bool(config.get("fp16", False)) and device.type == "cpu":
        print("提示：CPU 调试不使用 fp16，已自动关闭混合精度。")

    model_name = str(config["model_name"])
    tokenizer = load_tokenizer(model_name)
    max_length = int(config["max_length"])
    train_dataset = build_dataset(
        path=str(config["train_file"]),
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        require_labels=True,
    )
    valid_dataset = build_dataset(
        path=str(config["valid_file"]),
        tokenizer=tokenizer,
        max_length=max_length,
        split="valid",
        require_labels=True,
    )
    print(f"训练集：{dataset_summary(train_dataset)}")
    print(f"验证集：{dataset_summary(valid_dataset)}")

    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        model_name=model_name,
        num_labels=len(LABEL_NAMES),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    epochs = int(config["epochs"])
    if epochs <= 0:
        raise ValueError("epochs 必须为正整数")
    early_stopping_patience = int(config.get("early_stopping_patience", 2))
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience 不能为负数")
    accumulation_steps = max(int(config.get("gradient_accumulation_steps", 1)), 1)
    update_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_training_steps = max(update_steps_per_epoch * epochs, 1)
    warmup_steps = int(total_training_steps * float(config.get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    scaler = create_grad_scaler(use_fp16)
    loss_function = create_loss_function(config, train_dataset, device)

    experiment_name = str(config["experiment_name"]).strip()
    experiment_dir = resolve_experiment_dir(experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    best_metric_name = str(config.get("save_best_metric", "macro_f1"))
    best_metric_value = float("-inf")
    best_epoch = 0
    best_checkpoint = experiment_dir / "best_model.pt"
    last_checkpoint = experiment_dir / "last_model.pt"
    history_path = experiment_dir / "training_history.json"
    curves_path = experiment_dir / "training_curves.png"
    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_f1": [],
        "valid_macro_f1": [],
        "valid_mcc": [],
    }
    no_improvement_epochs = 0
    print(f"实验名称：{experiment_name}，输出目录：{experiment_dir.resolve()}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        processed_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for step, batch in enumerate(progress, start=1):
            batch = _move_batch(batch, device)
            labels = batch["labels"]
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_fp16
                else nullcontext()
            )
            with autocast_context:
                logits = model(**batch)
                loss = loss_function(logits, labels)

            scaler.scale(loss / accumulation_steps).backward()
            batch_size = labels.size(0)
            running_loss += float(loss.item()) * batch_size
            processed_samples += batch_size

            is_update_step = step % accumulation_steps == 0 or step == len(train_loader)
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.get("max_grad_norm", 1.0)),
                )
                scale_before_update = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before_update:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{running_loss / max(processed_samples, 1):.4f}")

        validation_metrics, validation_loss = evaluate_epoch(
            model=model,
            data_loader=valid_loader,
            criterion=loss_function,
            device=device,
            use_fp16=use_fp16,
        )
        train_loss = running_loss / max(processed_samples, 1)
        print(
            f"第 {epoch} 轮：train_loss={train_loss:.4f}，"
            f"valid_loss={validation_loss:.4f}，"
            f"ACC={validation_metrics['accuracy']:.4f}，"
            f"F1={validation_metrics['f1']:.4f}，"
            f"Macro-F1={validation_metrics['macro_f1']:.4f}，"
            f"MCC={validation_metrics['mcc']:.4f}"
        )
        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(validation_loss)
        history["valid_accuracy"].append(validation_metrics["accuracy"])
        history["valid_f1"].append(validation_metrics["f1"])
        history["valid_macro_f1"].append(validation_metrics["macro_f1"])
        history["valid_mcc"].append(validation_metrics["mcc"])

        save_checkpoint(
            path=last_checkpoint,
            model=model,
            model_name=model_name,
            tokenizer_name=model_name,
            dropout=float(config.get("dropout", 0.1)),
            max_length=max_length,
            epoch=epoch,
            metrics=validation_metrics,
            train_config=config,
        )

        current_metric = float(validation_metrics[best_metric_name])
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch
            no_improvement_epochs = 0
            save_checkpoint(
                path=best_checkpoint,
                model=model,
                model_name=model_name,
                tokenizer_name=model_name,
                dropout=float(config.get("dropout", 0.1)),
                max_length=max_length,
                epoch=epoch,
                metrics=validation_metrics,
                train_config=config,
            )
            print(f"已保存最佳模型：{best_checkpoint.resolve()}")
        else:
            no_improvement_epochs += 1
            print(f"验证指标未提升，连续 {no_improvement_epochs} 轮")
            if (
                early_stopping_patience > 0
                and no_improvement_epochs >= early_stopping_patience
            ):
                print(f"触发早停：连续 {early_stopping_patience} 轮未提升")
                break

    save_training_history(history, history_path)
    plot_training_curves(history, curves_path, experiment_name)
    print(
        f"训练完成，最佳验证 {best_metric_name}={best_metric_value:.4f}，"
        f"最佳 epoch={best_epoch}"
    )
    print(f"训练历史：{history_path.resolve()}")
    print(f"训练曲线：{curves_path.resolve()}")
    return best_checkpoint


def main() -> None:
    """读取项目默认配置、合并可选参数并启动训练。"""
    args = parse_args()
    config = merge_config(load_config(DEFAULT_CONFIG_PATH), args)
    train(config)


if __name__ == "__main__":
    main()
