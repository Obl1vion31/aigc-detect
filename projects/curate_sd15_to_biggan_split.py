#!/usr/bin/env python3
"""Create a SD1.5-trained GenImage split for the second M2 experiment.

The original Milestone 1 split trains on BigGAN and tests generalization to
Stable Diffusion 1.5. This script creates the reverse protocol:

* train/val: split Stable Diffusion 1.5 train images with seed 488.
* test_seen: use Stable Diffusion 1.5 val images.
* test_unseen: use BigGAN val images.

The curated dataset uses hard links, so images are not duplicated on disk.
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter
from pathlib import Path


BASE_DIR = Path("/root/autodl-tmp/488FP")
GENIMAGE_DIR = BASE_DIR / "datasets" / "GenImage"
CURATED_DIR = BASE_DIR / "datasets" / "curated_genimage_sd15_train"
OUTPUT_DIR = BASE_DIR / "outputs" / "M2_sd15_train" / "curation"
SEED = 488

SD15_DIR = GENIMAGE_DIR / "stable_diffusion_v_1_5" / "imagenet_ai_0424_sdv5"
BIGGAN_DIR = GENIMAGE_DIR / "BigGAN" / "imagenet_ai_0419_biggan"

CLASS_TO_LABEL = {"nature": "real", "ai": "fake"}


def list_images(directory: Path) -> list[Path]:
    """Return image files in a deterministic order."""
    return sorted(path for path in directory.iterdir() if path.is_file())


def hardlink_file(src: Path, dst: Path) -> None:
    """Hard-link one image into the curated split without duplicating bytes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def add_rows(
    rows: list[dict[str, str]],
    files: list[Path],
    split: str,
    generator: str,
    class_name: str,
) -> None:
    """Link files into one curated split and append metadata rows."""
    label = CLASS_TO_LABEL[class_name]
    for src in files:
        dst = CURATED_DIR / split / generator / class_name / src.name
        hardlink_file(src, dst)
        rows.append(
            {
                "relative_path": str(dst.relative_to(BASE_DIR)),
                "source_path": str(src.relative_to(BASE_DIR)),
                "split": split,
                "generator": generator,
                "class_name": class_name,
                "label": label,
                "is_fake": "1" if label == "fake" else "0",
                "filename": src.name,
            }
        )


def split_sd15_train(rows: list[dict[str, str]]) -> None:
    """Split SD1.5 train into 90% train and 10% val for each class."""
    rng = random.Random(SEED)
    for class_name in ("ai", "nature"):
        files = list_images(SD15_DIR / "train" / class_name)
        rng.shuffle(files)
        train_count = int(len(files) * 0.9)
        train_files = sorted(files[:train_count])
        val_files = sorted(files[train_count:])
        add_rows(rows, train_files, "train", "stable_diffusion_v_1_5", class_name)
        add_rows(rows, val_files, "val", "stable_diffusion_v_1_5", class_name)


def add_test_splits(rows: list[dict[str, str]]) -> None:
    """Use SD1.5 val as seen test and BigGAN val as unseen test."""
    for class_name in ("ai", "nature"):
        add_rows(
            rows,
            list_images(SD15_DIR / "val" / class_name),
            "test_seen",
            "stable_diffusion_v_1_5",
            class_name,
        )
        add_rows(
            rows,
            list_images(BIGGAN_DIR / "val" / class_name),
            "test_unseen",
            "BigGAN",
            class_name,
        )


def write_metadata(rows: list[dict[str, str]]) -> None:
    """Write one row per curated image."""
    fieldnames = [
        "relative_path",
        "source_path",
        "split",
        "generator",
        "class_name",
        "label",
        "is_fake",
        "filename",
    ]
    path = CURATED_DIR / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))


def write_statistics(rows: list[dict[str, str]]) -> None:
    """Write real/fake counts by split and generator for reporting."""
    generator_counts = Counter((row["generator"], row["label"]) for row in rows)
    split_counts = Counter((row["split"], row["label"]) for row in rows)
    path = CURATED_DIR / "dataset_statistics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_type", "group_name", "real", "fake", "total"])
        for generator in sorted({row["generator"] for row in rows}):
            real = generator_counts[(generator, "real")]
            fake = generator_counts[(generator, "fake")]
            writer.writerow(["generator", generator, real, fake, real + fake])
        for split in ("train", "val", "test_seen", "test_unseen"):
            real = split_counts[(split, "real")]
            fake = split_counts[(split, "fake")]
            writer.writerow(["split", split, real, fake, real + fake])


def export_outputs() -> None:
    """Copy curation CSVs into outputs/M2_sd15_train for easy reporting."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.csv", "dataset_statistics.csv"):
        src = CURATED_DIR / name
        dst = OUTPUT_DIR / name
        if dst.exists():
            dst.unlink()
        os.link(src, dst)


def main() -> None:
    if CURATED_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {CURATED_DIR}")

    rows: list[dict[str, str]] = []
    split_sd15_train(rows)
    add_test_splits(rows)
    write_metadata(rows)
    write_statistics(rows)
    export_outputs()

    print(f"Curated dataset: {CURATED_DIR}")
    print(f"Curation outputs: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")
    print(f"Total images: {len(rows)}")


if __name__ == "__main__":
    main()
