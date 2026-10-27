"""金融情感分类模型训练、验证与 checkpoint 保存。"""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data_loader import (
    LABEL_NAMES,
    FinancialSentimentDataset,
    build_dataset,
    dataset_summary,
)
from src.metrics import calculate_metrics
from src.model import create_model, load_tokenizer
from src.utils import load_config, resolve_device, set_seed


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

    无输入，返回参数对象；未显式传入的字段会沿用 `configs/default.yaml`。
    """
    parser = argparse.ArgumentParser(description="训练中文金融情感分类模型")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--model_name", default=None, help="预训练模型名称或本地路径")
    parser.add_argument("--dropout", type=float, default=None, help="分类头前的 dropout")
    parser.add_argument("--train_file", default=None, help="训练集 JSON 路径")
    parser.add_argument("--valid_file", default=None, help="验证集 JSON 路径")
    parser.add_argument("--checkpoint_dir", default=None, help="checkpoint 输出目录")
    parser.add_argument("--best_checkpoint", default=None, help="最佳模型保存路径")
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

    if args.checkpoint_dir is not None:
        merged["checkpoint_dir"] = args.checkpoint_dir
        if args.best_checkpoint is None:
            merged["best_checkpoint"] = str(Path(args.checkpoint_dir) / "best_model.pt")
    if args.best_checkpoint is not None:
        merged["best_checkpoint"] = args.best_checkpoint
    return merged


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """将批次张量移动到指定设备。

    输入批次字典和目标设备，返回移动后的新字典。
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


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
    accumulation_steps = max(int(config.get("gradient_accumulation_steps", 1)), 1)
    update_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_training_steps = max(update_steps_per_epoch * epochs, 1)
    warmup_steps = int(total_training_steps * float(config.get("warmup_ratio", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    loss_function = nn.CrossEntropyLoss()

    best_metric_name = str(config.get("save_best_metric", "macro_f1"))
    best_metric_value = float("-inf")
    best_checkpoint = Path(config["best_checkpoint"])
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last_model.pt"

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
                scaler.step(optimizer)
                scaler.update()
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
        print(
            f"第 {epoch} 轮：train_loss={running_loss / max(processed_samples, 1):.4f}，"
            f"valid_loss={validation_loss:.4f}，"
            f"ACC={validation_metrics['accuracy']:.4f}，"
            f"F1={validation_metrics['f1']:.4f}，"
            f"Macro-F1={validation_metrics['macro_f1']:.4f}，"
            f"MCC={validation_metrics['mcc']:.4f}"
        )

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

    print(f"训练完成，最佳验证 {best_metric_name}={best_metric_value:.4f}")
    return best_checkpoint


def main() -> None:
    """读取配置、合并命令行参数并启动训练。"""
    args = parse_args()
    config = merge_config(load_config(args.config), args)
    train(config)


if __name__ == "__main__":
    main()
