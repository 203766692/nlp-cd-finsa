"""配置、随机种子和设备等公共工具。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。

    输入为配置文件路径，返回字典；文件缺失、内容为空或根节点不是字典时抛出中文错误。
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path.resolve()}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not config:
        raise ValueError(f"配置文件内容为空或格式错误：{path.resolve()}")
    return config


def resolve_device(device_name: str) -> torch.device:
    """解析运行设备。

    输入 `cpu`、`cuda` 或 `auto`，返回 PyTorch 设备对象；请求 CUDA 但不可用时给出中文提示。
    """
    normalized = str(device_name).strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"

    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前设备无法使用 CUDA。请确认已安装 GPU 版 PyTorch，"
            "或使用 --device cpu 在本地调试。"
        )
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"不支持的设备：{device_name}，可选值为 cpu、cuda、auto")
    return torch.device(normalized)


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。

    输入整数种子，无返回值；同时控制 CUDA 随机性以保证训练尽量可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
