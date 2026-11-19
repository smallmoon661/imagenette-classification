"""Print model structures, parameter totals, and shapes from a real image batch.

Run from the project root:
    python src/inspect_models.py
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from data import build_dataloader
from models import CNNClassifier, MLPClassifier, count_trainable_parameters


INSPECTED_LAYERS = (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten)


def layer_shape_trace(model: nn.Module, images: torch.Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """Record output shapes after every relevant leaf layer during one forward pass."""
    trace: list[tuple[str, tuple[int, ...]]] = []
    hooks = []

    def make_hook(name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            trace.append((name, tuple(output.shape)))

        return hook

    for name, module in model.named_modules():
        if name and not any(module.children()) and isinstance(module, INSPECTED_LAYERS):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(images)
    finally:
        for hook in hooks:
            hook.remove()
    return trace


def inspect_model(name: str, model: nn.Module, images: torch.Tensor) -> None:
    print("=" * 72)
    print(name)
    print("=" * 72)
    print(model)
    print(f"\nTrainable parameters: {count_trainable_parameters(model):,}")
    print(f"Input batch shape: {tuple(images.shape)}")

    trace = layer_shape_trace(model, images)
    print("Layer output shapes:")
    for layer_name, shape in trace:
        print(f"  {layer_name:<22} -> {shape}")

    with torch.no_grad():
        logits = model(images)
    print(f"Model output (logits) shape: {tuple(logits.shape)}")
    print("Each row contains 10 raw class scores; the largest score is the prediction.\n")


def main() -> None:
    images, _labels = next(iter(build_dataloader("train", batch_size=64)))
    inspect_model("MLP baseline", MLPClassifier(), images)
    inspect_model("CNN baseline / CNN-flip architecture", CNNClassifier(), images)


if __name__ == "__main__":
    main()
