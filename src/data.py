"""Shared Imagenette dataset and preprocessing utilities.

All three experiments reuse the CSV manifests created by prepare_data.py.
The original files are never moved; each CSV identifies the image path and label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "imagenette2-160"
CONFIG_DIR = PROJECT_ROOT / "configs"
VALID_SPLITS = {"train", "val", "test"}


class ImagenetteManifestDataset(Dataset):
    """Load Imagenette images using one fixed CSV manifest."""

    def __init__(self, split: str, transform: transforms.Compose, data_root: Path = DATA_ROOT) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
        self.records = pd.read_csv(CONFIG_DIR / f"{split}.csv")
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records.iloc[index]
        image_path = self.data_root / row["relative_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(row["label"])


def build_transform(*, train: bool, horizontal_flip: bool = False) -> transforms.Compose:
    """Build the required preprocessing pipeline.

    Resize(160) scales the *shorter* image edge to 160 while preserving the
    aspect ratio. CenterCrop then makes every image exactly 160 by 160.
    The optional flip is deliberately after the crop and before ToTensor.
    """
    if horizontal_flip and not train:
        raise ValueError("Horizontal flip is permitted only for training data.")

    operations: list[object] = [
        transforms.Resize(160),
        transforms.CenterCrop((160, 160)),
    ]
    if horizontal_flip:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.append(transforms.ToTensor())
    return transforms.Compose(operations)


def load_class_mapping() -> dict[str, dict[str, int | str]]:
    with (CONFIG_DIR / "class_mapping.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def build_dataloader(
    split: str,
    *,
    batch_size: int = 64,
    horizontal_flip: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Return a manifest-backed DataLoader for one experiment split."""
    is_train = split == "train"
    dataset = ImagenetteManifestDataset(
        split=split,
        transform=build_transform(train=is_train, horizontal_flip=horizontal_flip),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
