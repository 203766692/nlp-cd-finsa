"""使用 TF-IDF 和逻辑回归训练中文金融情感分类基线。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import LABEL_NAMES, load_records
from src.metrics import calculate_metrics
from src.utils import DEFAULT_CONFIG_PATH, load_config, resolve_experiment_dir


def load_training_data(
    train_file: str,
    test_file: str,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """读取基线实验所需的训练集和测试集。

    输入训练集与测试集路径，返回文本列表和整数标签列表；
    标签按 `negative`、`neutral`、`positive` 的顺序编码。
    """
    train_records = load_records(train_file, split="train", require_labels=True)
    test_records = load_records(test_file, split="test", require_labels=True)
    label_to_id = {label: index for index, label in enumerate(LABEL_NAMES)}
    train_texts = [str(record["text"]) for record in train_records]
    test_texts = [str(record["text"]) for record in test_records]
    train_labels = [label_to_id[str(record["label"])] for record in train_records]
    test_labels = [label_to_id[str(record["label"])] for record in test_records]
    return train_texts, train_labels, test_texts, test_labels


def train_baseline(
    train_texts: list[str],
    train_labels: list[int],
    seed: int,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """训练 TF-IDF 与逻辑回归基线。

    输入训练文本、训练标签和随机种子，返回拟合后的向量化器和分类器；
    使用中文字符 1 至 3 gram 降低分词依赖。
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(
        max_iter=2_000,
        random_state=seed,
    )
    classifier.fit(train_features, train_labels)
    return vectorizer, classifier


def plot_confusion_matrix(
    labels: list[int],
    predictions: list[int],
    output_path: Path,
) -> None:
    """绘制基线模型混淆矩阵。

    输入真实标签、预测标签和图片输出路径，保存热力图，无返回值。
    """
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
    )
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=False,
        ax=axis,
    )
    axis.set_title("Confusion Matrix: TF-IDF + Logistic Regression")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def build_metrics_document(
    labels: list[int],
    predictions: list[int],
    train_file: str,
    test_file: str,
    num_train_samples: int,
    seed: int,
) -> dict[str, Any]:
    """构建基线实验指标文档。

    输入真实标签、预测标签、训练集路径、测试集路径、训练样本数和随机种子，
    返回包含总体指标、分类别指标和混淆矩阵的字典。
    """
    metrics = calculate_metrics(labels, predictions)
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
        target_names=list(LABEL_NAMES),
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label_name: {
            "precision": float(report[label_name]["precision"]),
            "recall": float(report[label_name]["recall"]),
            "f1": float(report[label_name]["f1-score"]),
            "support": int(report[label_name]["support"]),
        }
        for label_name in LABEL_NAMES
    }
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(LABEL_NAMES))),
    )
    return {
        "experiment_name": "baseline",
        "model": "TF-IDF char n-gram + LogisticRegression",
        "seed": seed,
        "use_class_weights": False,
        "train_file": str(Path(train_file).resolve()),
        "test_file": str(Path(test_file).resolve()),
        "num_train_samples": num_train_samples,
        "num_test_samples": len(labels),
        "label_names": list(LABEL_NAMES),
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def run_baseline(config: dict[str, Any]) -> Path:
    """运行完整基线实验并保存结果。

    输入默认配置，读取训练集和测试集，训练基线并保存指标与混淆矩阵，
    返回指标文件路径。
    """
    train_texts, train_labels, test_texts, test_labels = load_training_data(
        str(config["train_file"]),
        str(config["test_file"]),
    )
    vectorizer, classifier = train_baseline(
        train_texts,
        train_labels,
        seed=int(config["seed"]),
    )
    test_features = vectorizer.transform(test_texts)
    predictions = classifier.predict(test_features).tolist()

    output_dir = resolve_experiment_dir("baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    document = build_metrics_document(
        test_labels,
        predictions,
        str(config["train_file"]),
        str(config["test_file"]),
        len(train_labels),
        int(config["seed"]),
    )
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2)
        file.write("\n")
    confusion_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(test_labels, predictions, confusion_path)

    metrics = document["metrics"]
    print(
        f"基线结果：ACC={metrics['accuracy']:.4f}，"
        f"Macro-F1={metrics['macro_f1']:.4f}"
    )
    print(f"指标已写入：{metrics_path.resolve()}")
    print(f"混淆矩阵已写入：{confusion_path.resolve()}")
    return metrics_path


def main() -> None:
    """读取默认配置并运行基线实验。"""
    config = load_config(DEFAULT_CONFIG_PATH)
    run_baseline(config)


if __name__ == "__main__":
    main()
