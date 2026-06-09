#!/usr/bin/env python3
"""Curate GenImage splits for AIGC detection.

This script keeps the original GenImage directories unchanged and creates a
curated view with hard links plus metadata/statistics CSV files.
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter
from pathlib import Path


BASE_DIR = Path("/root/autodl-tmp/488FP")
GENIMAGE_DIR = BASE_DIR / "datasets" / "GenImage"
OUTPUT_DIR = BASE_DIR / "datasets" / "curated_genimage"
SEED = 488

BIGGAN_DIR = GENIMAGE_DIR / "BigGAN" / "imagenet_ai_0419_biggan"
SD15_DIR = GENIMAGE_DIR / "stable_diffusion_v_1_5" / "imagenet_ai_0424_sdv5"

CLASS_TO_LABEL = {
    "ai": "fake",
    "nature": "real",
}


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file())


def hardlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    os.link(src, dst)


def add_rows_for_split(
    rows: list[dict[str, str]],
    files: list[Path],
    split: str,
    generator: str,
    class_name: str,
) -> None:
    label = CLASS_TO_LABEL[class_name]
    for src in files:
        dst = OUTPUT_DIR / split / generator / class_name / src.name
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


def split_biggan_train(rows: list[dict[str, str]]) -> None:
    rng = random.Random(SEED)
    for class_name in ("ai", "nature"):
        files = list_images(BIGGAN_DIR / "train" / class_name)
        rng.shuffle(files)
        train_count = int(len(files) * 0.9)
        train_files = sorted(files[:train_count])
        val_files = sorted(files[train_count:])
        add_rows_for_split(rows, train_files, "train", "BigGAN", class_name)
        add_rows_for_split(rows, val_files, "val", "BigGAN", class_name)


def add_existing_val_as_test(rows: list[dict[str, str]]) -> None:
    for class_name in ("ai", "nature"):
        add_rows_for_split(
            rows,
            list_images(BIGGAN_DIR / "val" / class_name),
            "test_seen",
            "BigGAN",
            class_name,
        )
        add_rows_for_split(
            rows,
            list_images(SD15_DIR / "val" / class_name),
            "test_unseen",
            "stable_diffusion_v_1_5",
            class_name,
        )


def write_metadata(rows: list[dict[str, str]]) -> None:
    metadata_path = OUTPUT_DIR / "metadata.csv"
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
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))


def write_statistics(rows: list[dict[str, str]]) -> None:
    stats_path = OUTPUT_DIR / "dataset_statistics.csv"
    generator_counts = Counter((row["generator"], row["label"]) for row in rows)
    split_counts = Counter((row["split"], row["label"]) for row in rows)

    with stats_path.open("w", newline="", encoding="utf-8") as handle:
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


def main() -> None:
    if OUTPUT_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {OUTPUT_DIR}")

    rows: list[dict[str, str]] = []
    split_biggan_train(rows)
    add_existing_val_as_test(rows)
    write_metadata(rows)
    write_statistics(rows)

    print(f"Curated dataset written to: {OUTPUT_DIR}")
    print(f"Random seed: {SEED}")
    print(f"Total images: {len(rows)}")


if __name__ == "__main__":
    main()
