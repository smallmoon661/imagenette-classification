"""Create comparison artifacts from the three completed Imagenette experiments.

Run from the project root after train.py and evaluate.py have completed:
    python src/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "summary"
EXPERIMENTS = [
    ("mlp", "MLP"),
    ("cnn", "CNN"),
    ("cnn-flip", "CNN + horizontal flip"),
]
COLORS = {"mlp": "#7b68aa", "cnn": "#2878b5", "cnn-flip": "#2a9d62"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_experiment(experiment: str, display_name: str) -> tuple[pd.DataFrame, dict]:
    """Load the artifacts written by one completed train/evaluate run."""
    experiment_dir = PROJECT_ROOT / "outputs" / experiment
    required_paths = {
        "history": experiment_dir / "history.csv",
        "metadata": experiment_dir / "run_metadata.json",
        "metrics": experiment_dir / "metrics.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed experiment artifacts:\n" + "\n".join(missing))

    history = pd.read_csv(required_paths["history"])
    metadata = read_json(required_paths["metadata"])
    metrics = read_json(required_paths["metrics"])
    summary = {
        "experiment": experiment,
        "display_name": display_name,
        "model": metadata["model"],
        "horizontal_flip_train": metadata["horizontal_flip_train"],
        "trainable_parameters": metadata["trainable_parameters"],
        "best_epoch": metrics["selected_epoch"],
        "best_val_accuracy": metrics["selected_val_accuracy"],
        "test_images": metrics["test_image_count"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
    }
    return history, summary


def save_training_curves(histories: dict[str, pd.DataFrame], summaries: pd.DataFrame) -> None:
    """Save one loss/accuracy row per experiment, marking selected validation epoch."""
    figure, axes = plt.subplots(nrows=len(EXPERIMENTS), ncols=2, figsize=(13, 12), sharex=True)

    for row, (experiment, display_name) in enumerate(EXPERIMENTS):
        history = histories[experiment]
        best_epoch = int(summaries.loc[summaries["experiment"] == experiment, "best_epoch"].iloc[0])
        color = COLORS[experiment]
        loss_axis, accuracy_axis = axes[row]

        loss_axis.plot(history["epoch"], history["train_loss"], "--", color=color, label="train loss")
        loss_axis.plot(history["epoch"], history["val_loss"], "-", color=color, linewidth=2, label="validation loss")
        loss_axis.axvline(best_epoch, color="black", linestyle=":", label=f"selected epoch {best_epoch}")
        loss_axis.set_title(f"{display_name}: loss")
        loss_axis.set_ylabel("cross-entropy loss")
        loss_axis.grid(alpha=0.25)
        loss_axis.legend(fontsize=8)

        accuracy_axis.plot(history["epoch"], history["train_accuracy"], "--", color=color, label="train accuracy")
        accuracy_axis.plot(history["epoch"], history["val_accuracy"], "-", color=color, linewidth=2, label="validation accuracy")
        accuracy_axis.axvline(best_epoch, color="black", linestyle=":", label=f"selected epoch {best_epoch}")
        accuracy_axis.set_title(f"{display_name}: accuracy")
        accuracy_axis.set_ylabel("accuracy")
        accuracy_axis.set_ylim(0, 1)
        accuracy_axis.grid(alpha=0.25)
        accuracy_axis.legend(fontsize=8)

    for axis in axes[-1]:
        axis.set_xlabel("epoch")
    figure.suptitle("Imagenette training and internal-validation curves", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(OUTPUT_DIR / "training-curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_test_comparison(summaries: pd.DataFrame) -> None:
    """Save a bar chart for the two final held-out test metrics."""
    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = range(len(summaries))
    width = 0.36
    accuracy_bars = axis.bar(
        [position - width / 2 for position in positions],
        summaries["test_accuracy"],
        width,
        label="test accuracy",
        color="#2878b5",
    )
    f1_bars = axis.bar(
        [position + width / 2 for position in positions],
        summaries["test_macro_f1"],
        width,
        label="test macro-F1",
        color="#2a9d62",
    )
    axis.set_xticks(list(positions), summaries["display_name"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("score")
    axis.set_title("Final held-out test comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for bars in (accuracy_bars, f1_bars):
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in bars.datavalues], padding=3, fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for experiment, display_name in EXPERIMENTS:
        history, summary = load_experiment(experiment, display_name)
        histories[experiment] = history
        rows.append(summary)

    summaries = pd.DataFrame(rows)
    summaries.to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8")
    save_training_curves(histories, summaries)
    save_test_comparison(summaries)

    print("Saved comparison table:", OUTPUT_DIR / "comparison.csv")
    print("Saved training curves:", OUTPUT_DIR / "training-curves.png")
    print("Saved test chart:", OUTPUT_DIR / "test-comparison.png")
    print()
    print(summaries.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
