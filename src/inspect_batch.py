"""Inspect one real batch before model training.

Examples from the project root:
    python src/inspect_batch.py
    python src/inspect_batch.py --horizontal-flip
    python src/inspect_batch.py --split val
"""

from __future__ import annotations

import argparse

from data import build_dataloader, load_class_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Imagenette DataLoader batch.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--horizontal-flip",
        action="store_true",
        help="Enable p=0.5 horizontal flip; valid only for the training split.",
    )
    args = parser.parse_args()
    if args.horizontal_flip and args.split != "train":
        parser.error("--horizontal-flip can only be used with --split train")

    loader = build_dataloader(
        args.split,
        batch_size=args.batch_size,
        horizontal_flip=args.horizontal_flip,
    )
    images, labels = next(iter(loader))
    label_to_name = {entry["label"]: entry["class_name"] for entry in load_class_mapping().values()}

    print(f"split: {args.split}")
    print(f"dataset images: {len(loader.dataset)}")
    print(f"horizontal flip: {args.horizontal_flip} (p=0.5 when enabled)")
    print(f"image batch shape: {tuple(images.shape)}")
    print(f"image dtype: {images.dtype}")
    print(f"pixel range: [{images.min().item():.4f}, {images.max().item():.4f}]")
    print(f"label batch shape: {tuple(labels.shape)}")
    print(f"first 10 labels: {labels[:10].tolist()}")
    print(f"first 10 classes: {[label_to_name[label] for label in labels[:10].tolist()]}")


if __name__ == "__main__":
    main()
