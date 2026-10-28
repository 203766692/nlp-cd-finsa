"""预训练中文模型、分类头和 checkpoint 加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


class FinancialSentimentModel(nn.Module):
    """在预训练编码器上增加线性情感分类头的模型。"""

    def __init__(self, model_name: str, num_labels: int, dropout: float) -> None:
        """加载预训练编码器并初始化分类头。

        输入模型名称或本地路径、类别数和 dropout 比例，初始化编码器、dropout 与线性层。
        """
        super().__init__()
        if num_labels <= 0:
            raise ValueError("num_labels 必须为正整数")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 范围内")

        self.model_name = model_name
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
        )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行前向计算并返回分类 logits。

        输入 token id、注意力掩码、可选 token type id 和可选标签，
        输出形状为 `[batch_size, num_labels]` 的张量；损失由训练循环单独计算。
        """
        del labels
        encoder_inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """加载预训练 tokenizer。

    输入模型名称或本地路径，返回 tokenizer；加载失败时补充中文排查提示。
    """
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=True,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"无法从本地缓存加载 tokenizer：{model_name}。"
            "请先下载模型并确认 Hugging Face 缓存或本地路径可用。"
        ) from error


def create_model(model_name: str, num_labels: int = 3, dropout: float = 0.1) -> FinancialSentimentModel:
    """创建情感分类模型。

    输入模型名称、类别数和 dropout，返回已加载预训练权重的分类模型。
    """
    try:
        return FinancialSentimentModel(model_name, num_labels, dropout)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"无法从本地缓存加载预训练模型：{model_name}。"
            "请先下载模型并确认 Hugging Face 缓存或本地路径可用。"
        ) from error


def _torch_load(path: Path) -> dict[str, Any]:
    """安全读取本地 checkpoint。

    输入 checkpoint 路径，返回字典；优先使用 weights_only 模式并兼容旧版 PyTorch。
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    except RuntimeError as error:
        raise RuntimeError(f"checkpoint 内容无法读取：{path.resolve()}") from error

    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint 根节点必须是字典：{path.resolve()}")
    return checkpoint


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[FinancialSentimentModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """从 checkpoint 恢复模型和 tokenizer。

    输入 checkpoint 路径和目标设备，返回模型、tokenizer 和元数据；缺少必要字段时抛出中文错误。
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在：{path.resolve()}")

    checkpoint = _torch_load(path)
    required_fields = {"model_state_dict", "model_name", "num_labels", "label_names"}
    missing_fields = required_fields.difference(checkpoint)
    if missing_fields:
        raise ValueError(
            f"checkpoint 缺少字段：{', '.join(sorted(missing_fields))}，路径：{path.resolve()}"
        )

    model_name = str(checkpoint["model_name"])
    tokenizer = load_tokenizer(str(checkpoint.get("tokenizer_name", model_name)))
    model = create_model(
        model_name=model_name,
        num_labels=int(checkpoint["num_labels"]),
        dropout=float(checkpoint.get("dropout", 0.1)),
    )

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise RuntimeError(f"checkpoint 参数与模型结构不匹配：{path.resolve()}") from error

    model.to(device)
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    return model, tokenizer, metadata
