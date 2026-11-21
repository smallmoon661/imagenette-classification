"""Reload one selected checkpoint and evaluate it once on the held-out test set.

Examples from the project root:
    python src/evaluate.py --config configs/mlp.yaml
    python src/evaluate.py --config configs/cnn.yaml
    python src/evaluate.py --config configs/cnn-flip.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from data import DATA_ROOT, build_dataloader, load_class_mapping
from models import CNNClassifier, MLPClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    """Read the run's saved YAML via PyYAML, without changing its settings."""
    import yaml

    with config_path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "mlp":
        return MLPClassifier()
    if model_name == "cnn":
        return CNNClassifier()
    raise ValueError(f"Unsupported model: {model_name}")


def class_names_by_label() -> list[str]:
    mapping = load_class_mapping()
    by_label = {entry["label"]: entry["class_name"] for entry in mapping.values()}
    return [by_label[label] for label in range(len(by_label))]


def predict(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return true and predicted labels in the fixed test CSV order."""
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            true_labels.extend(labels.tolist())
            predicted_labels.extend(logits.argmax(dim=1).cpu().tolist())
    return np.asarray(true_labels), np.asarray(predicted_labels)


def save_confusion_matrix(matrix: np.ndarray, names: list[str], output_path: Path, experiment: str) -> None:
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=names,
        yticklabels=names,
        ax=axis,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{experiment}: held-out test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    plt.setp(axis.get_yticklabels(), rotation=0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_examples(predictions: pd.DataFrame, output_path: Path, experiment: str) -> None:
    """Save deterministic examples from multiple true classes, rather than CSV order."""
    correct_pool = predictions[predictions["is_correct"]]
    incorrect_pool = predictions[~predictions["is_correct"]]

    # One correct image from four spread-out classes, and one incorrect image
    # from eight different classes. This prevents the figure from showing only
    # the first class in the test CSV (previously all samples were tench).
    correct_labels = [0, 3, 6, 9]
    incorrect_labels = list(range(8))
    correct = pd.concat(
        [correct_pool[correct_pool["true_label"] == label].head(1) for label in correct_labels],
        ignore_index=True,
    )
    incorrect = pd.concat(
        [incorrect_pool[incorrect_pool["true_label"] == label].head(1) for label in incorrect_labels],
        ignore_index=True,
    )
    selected = pd.concat([correct, incorrect], ignore_index=True)
    figure, axes = plt.subplots(3, 4, figsize=(16, 12))
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        with Image.open(DATA_ROOT / row["relative_path"]) as image:
            axis.imshow(image.convert("RGB"))
        prefix = "correct" if row["is_correct"] else "wrong"
        color = "forestgreen" if row["is_correct"] else "firebrick"
        axis.set_title(
            f"{prefix}\ntrue: {row['true_class']}\npred: {row['pred_class']}",
            color=color,
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle(f"{experiment}: diverse correct and incorrect test predictions", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def evaluate(config: dict[str, Any]) -> None:
    experiment = config["experiment_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "outputs" / experiment
    checkpoint_path = PROJECT_ROOT / "checkpoints" / experiment / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint_path}")

    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = build_dataloader("test", batch_size=config["training"]["batch_size"])
    true_labels, predicted_labels = predict(model, test_loader, device)

    test_records = test_loader.dataset.records.copy().reset_index(drop=True)
    if len(test_records) != len(true_labels):
        raise RuntimeError("Prediction count does not match the fixed test manifest.")
    names = class_names_by_label()
    test_records["true_label"] = true_labels
    test_records["true_class"] = [names[label] for label in true_labels]
    test_records["pred_label"] = predicted_labels
    test_records["pred_class"] = [names[label] for label in predicted_labels]
    test_records["is_correct"] = true_labels == predicted_labels
    predictions = test_records[
        ["relative_path", "true_label", "true_class", "pred_label", "pred_class", "is_correct"]
    ]

    labels = list(range(len(names)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)
    recalls = recall_score(true_labels, predicted_labels, labels=labels, average=None, zero_division=0)
    metrics = {
        "experiment_name": experiment,
        "checkpoint": str(checkpoint_path),
        "selected_epoch": int(checkpoint["epoch"]),
        "selected_val_accuracy": float(checkpoint["val_accuracy"]),
        "test_image_count": int(len(true_labels)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_recall": {name: float(recalls[label]) for label, name in enumerate(names)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_confusion_matrix(matrix, names, output_dir / "confusion-matrix.png", experiment)
    save_examples(predictions, output_dir / "examples.png", experiment)

    print(f"Experiment: {experiment}")
    print(f"Reloaded checkpoint: {checkpoint_path}")
    print(f"Selected epoch: {metrics['selected_epoch']}")
    print(f"Test images: {metrics['test_image_count']}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Saved metrics and prediction artifacts to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one selected Imagenette checkpoint on the held-out test set.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML configuration.")
    args = parser.parse_args()
    evaluate(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()
