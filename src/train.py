"""Train one required Imagenette experiment from a YAML configuration.

Examples from the project root:
    python src/train.py --config configs/mlp.yaml --smoke-test
    python src/train.py --config configs/mlp.yaml
    python src/train.py --config configs/cnn.yaml
    python src/train.py --config configs/cnn-flip.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from data import build_dataloader
from models import CNNClassifier, MLPClassifier, count_trainable_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required_top_level = {"experiment_name", "model", "data", "training"}
    missing = required_top_level - set(config)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")
    if config["model"] not in {"mlp", "cnn"}:
        raise ValueError("model must be 'mlp' or 'cnn'")
    if config["experiment_name"] == "cnn-flip" and not config["data"]["horizontal_flip_train"]:
        raise ValueError("cnn-flip must enable horizontal_flip_train")
    if config["experiment_name"] != "cnn-flip" and config["data"]["horizontal_flip_train"]:
        raise ValueError("Only cnn-flip may enable horizontal_flip_train")
    return config


def set_seed(seed: int) -> None:
    """Set all random-number generators before model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name: str) -> nn.Module:
    if model_name == "mlp":
        return MLPClassifier()
    if model_name == "cnn":
        return CNNClassifier()
    raise ValueError(f"Unsupported model: {model_name}")


def state_dict_hash(model: nn.Module) -> str:
    """Create a reproducible fingerprint of initial weights for run metadata."""
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run either one training epoch (optimizer present) or validation epoch."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_images += batch_size

    return total_loss / total_images, total_correct / total_images


def make_loaders(config: dict[str, Any]) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    training = config["training"]
    return (
        build_dataloader(
            "train",
            batch_size=training["batch_size"],
            horizontal_flip=config["data"]["horizontal_flip_train"],
            num_workers=training["num_workers"],
        ),
        build_dataloader(
            "val",
            batch_size=training["batch_size"],
            num_workers=training["num_workers"],
        ),
    )


def smoke_test(config: dict[str, Any]) -> None:
    """Test one train batch and one val batch, then exit without saving artifacts."""
    training = config["training"]
    set_seed(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create the model after setting the seed. A future formal run starts fresh.
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = make_loaders(config)

    train_loss, train_accuracy = run_epoch(
        model, [next(iter(train_loader))], criterion, device, optimizer
    )
    val_loss, val_accuracy = run_epoch(model, [next(iter(val_loader))], criterion, device)
    print("Smoke test passed; no artifacts were saved.")
    print(f"device: {device}")
    print(f"one train batch - loss: {train_loss:.4f}, accuracy: {train_accuracy:.4f}")
    print(f"one val batch   - loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}")


def train(config: dict[str, Any], config_path: Path) -> None:
    experiment = config["experiment_name"]
    training = config["training"]
    set_seed(training["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # This happens immediately after set_seed. cnn and cnn-flip therefore begin
    # with identical CNN weights while remaining independent training runs.
    model = build_model(config["model"]).to(device)
    initial_hash = state_dict_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = make_loaders(config)

    output_dir = PROJECT_ROOT / "outputs" / experiment
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    metadata = {
        "experiment_name": experiment,
        "model": config["model"],
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "initial_weights_sha256": initial_hash,
        "seed": training["seed"],
        "batch_size": training["batch_size"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "horizontal_flip_train": config["data"]["horizontal_flip_train"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    history_path = output_dir / "history.csv"
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "is_best"]
    best_val_accuracy = float("-inf")
    best_epoch = 0

    print(f"Experiment: {experiment}")
    print(f"Device: {device}")
    print(f"Trainable parameters: {metadata['trainable_parameters']:,}")
    print(f"Initial weights SHA256: {initial_hash}")
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, training["epochs"] + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
            is_best = val_accuracy > best_val_accuracy
            if is_best:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": val_accuracy,
                        "config": config,
                    },
                    checkpoint_dir / "best.pt",
                )

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_accuracy:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_accuracy:.6f}",
                "is_best": is_best,
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"Epoch {epoch:02d}/{training['epochs']} | "
                f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}" + (" | best" if is_best else "")
            )

    metadata["best_epoch"] = best_epoch
    metadata["best_val_accuracy"] = best_val_accuracy
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best checkpoint: {checkpoint_dir / 'best.pt'}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Imagenette experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to one experiment YAML file.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one train and val batch without saving results.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.smoke_test:
        smoke_test(config)
    else:
        train(config, config_path)


if __name__ == "__main__":
    main()
