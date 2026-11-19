"""Model definitions for the three Imagenette experiments.

The models deliberately output raw class scores (logits), not probabilities.
CrossEntropyLoss receives those logits directly during training.
"""

from __future__ import annotations

import torch
from torch import nn


NUM_CLASSES = 10
IMAGE_CHANNELS = 3
IMAGE_SIZE = 160


class MLPClassifier(nn.Module):
    """Baseline: flatten RGB pixels, then 76800 -> 256 -> 10."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        input_features = IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE  # 3 * 160 * 160 = 76,800
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.flatten(images))


class CNNClassifier(nn.Module):
    """Four convolution-ReLU-pool blocks followed by a 256-unit classifier head."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled = self.adaptive_pool(features)
        return self.classifier(self.flatten(pooled))


def count_trainable_parameters(model: nn.Module) -> int:
    """Count every parameter that an optimizer may update."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
