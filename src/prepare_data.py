"""Prepare the fixed Imagenette split and exploration artifacts.

Run from the project root:
    python src/prepare_data.py

The script never moves or copies original images.  Instead, it writes CSV files
that state which original relative paths belong to train, internal validation,
and final test splits.  All later experiments must reuse these CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "imagenette2-160"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42
TRAIN_SIZE = 7_575
VAL_SIZE = 1_894
TEST_SIZE = 3_925

# Imagenette's official ten ImageNet synset folders.  Sorted folders define the
# numeric labels used by torchvision.datasets.ImageFolder and this project.
CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def image_paths(split_root: Path) -> list[Path]:
    """Return only supported image files, excluding files such as .DS_Store."""
    return sorted(
        path
        for path in split_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_frame(paths: list[Path], data_root: Path, split: str, label_by_folder: dict[str, int]) -> pd.DataFrame:
    rows = []
    for path in paths:
        folder = path.parent.name
        if folder not in label_by_folder:
            raise ValueError(f"Unknown class folder: {folder} ({path})")
        rows.append(
            {
                "relative_path": path.relative_to(data_root).as_posix(),
                "label": label_by_folder[folder],
                "class_folder": folder,
                "class_name": CLASS_NAMES[folder],
                "split": split,
            }
        )
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


def save_sample_grid(train_frame: pd.DataFrame, data_root: Path, output_path: Path) -> None:
    """Save one deterministic training example per class in a 2 by 5 grid."""
    figure, axes = plt.subplots(2, 5, figsize=(16, 7))
    for axis, (folder, group) in zip(axes.flat, train_frame.groupby("class_folder", sort=True)):
        # The first item after the deterministic sort makes this artifact reproducible.
        row = group.iloc[0]
        with Image.open(data_root / row["relative_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{row['label']}: {row['class_name']}\n({folder})", fontsize=10)
        axis.axis("off")

    figure.suptitle("Imagenette training samples: one image per class", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Imagenette split manifests and data exploration outputs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    train_root = data_root / "train"
    test_root = data_root / "val"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected train/ and val/ under {data_root}")

    folders = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    if folders != sorted(CLASS_NAMES):
        raise ValueError(f"Unexpected class folders. Found: {folders}")
    test_folders = sorted(path.name for path in test_root.iterdir() if path.is_dir())
    if test_folders != folders:
        raise ValueError("Official val/ classes do not match train/ classes.")

    label_by_folder = {folder: label for label, folder in enumerate(folders)}
    mapping = {
        folder: {"label": label_by_folder[folder], "class_name": CLASS_NAMES[folder]}
        for folder in folders
    }

    train_paths = image_paths(train_root)
    test_paths = image_paths(test_root)
    if len(train_paths) != TRAIN_SIZE + VAL_SIZE:
        raise ValueError(f"Expected {TRAIN_SIZE + VAL_SIZE} train images, found {len(train_paths)}")
    if len(test_paths) != TEST_SIZE:
        raise ValueError(f"Expected {TEST_SIZE} test images, found {len(test_paths)}")

    train_labels = [path.parent.name for path in train_paths]
    fit_paths, val_paths = train_test_split(
        train_paths,
        train_size=TRAIN_SIZE,
        test_size=VAL_SIZE,
        random_state=args.seed,
        shuffle=True,
        stratify=train_labels,
    )

    train_frame = build_frame(fit_paths, data_root, "train", label_by_folder)
    val_frame = build_frame(val_paths, data_root, "val", label_by_folder)
    test_frame = build_frame(test_paths, data_root, "test", label_by_folder)
    if (len(train_frame), len(val_frame), len(test_frame)) != (TRAIN_SIZE, VAL_SIZE, TEST_SIZE):
        raise AssertionError("Split sizes do not match the required experiment setup.")

    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_CONFIG_DIR / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for frame, filename in ((train_frame, "train.csv"), (val_frame, "val.csv"), (test_frame, "test.csv")):
        frame.to_csv(DEFAULT_CONFIG_DIR / filename, index=False, encoding="utf-8")

    counts = (
        pd.concat([train_frame, val_frame, test_frame], ignore_index=True)
        .groupby(["split", "label", "class_folder", "class_name"], sort=True)
        .size()
        .rename("image_count")
        .reset_index()
    )
    counts.to_csv(DEFAULT_OUTPUT_DIR / "class_counts.csv", index=False, encoding="utf-8")
    summary = {
        "seed": args.seed,
        "data_root": str(data_root),
        "split_sizes": {"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)},
        "class_count": len(folders),
        "image_extensions": sorted(IMAGE_EXTENSIONS),
    }
    (DEFAULT_OUTPUT_DIR / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_sample_grid(train_frame, data_root, DEFAULT_OUTPUT_DIR / "sample-grid.png")

    print("Prepared fixed Imagenette manifests.")
    print(f"  train: {len(train_frame)} images")
    print(f"  val:   {len(val_frame)} images")
    print(f"  test:  {len(test_frame)} images")
    print(f"  mapping: {DEFAULT_CONFIG_DIR / 'class_mapping.json'}")
    print(f"  outputs: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
