"""情感分类评估指标。"""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def calculate_metrics(
    labels: Iterable[int],
    predictions: Iterable[int],
) -> dict[str, float]:
    """计算分类指标。

    输入真实标签和预测标签，返回 ACC、加权 F1、Macro-F1 和 MCC 四个浮点指标。
    """
    label_list = list(labels)
    prediction_list = list(predictions)
    if not label_list:
        raise ValueError("无法在空标签集合上计算评估指标")
    if len(label_list) != len(prediction_list):
        raise ValueError("真实标签和预测标签数量不一致")
    return {
        "accuracy": float(accuracy_score(label_list, prediction_list)),
        "f1": float(f1_score(label_list, prediction_list, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(label_list, prediction_list, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(label_list, prediction_list)),
    }
