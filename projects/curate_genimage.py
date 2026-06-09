#!/usr/bin/env python3
"""Curate GenImage splits for AIGC detection.

This script keeps the original GenImage directories unchanged. It creates a
curated hard-link dataset, writes metadata/statistics CSV files, and exports
Milestone 1 outputs under outputs/M1.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path


BASE_DIR = Path("/root/autodl-tmp/488FP")
GENIMAGE_DIR = BASE_DIR / "datasets" / "GenImage"
CURATED_DIR = BASE_DIR / "datasets" / "curated_genimage"
M1_OUTPUT_DIR = BASE_DIR / "outputs" / "M1"
SEED = 488

BIGGAN_DIR = GENIMAGE_DIR / "BigGAN" / "imagenet_ai_0419_biggan"
SD15_DIR = GENIMAGE_DIR / "stable_diffusion_v_1_5" / "imagenet_ai_0424_sdv5"

CLASS_TO_LABEL = {
    "ai": "fake",
    "nature": "real",
}


def parse_args() -> argparse.Namespace:
    """Read command-line options for full curation or M1 export only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-m1-only",
        action="store_true",
        help="Reuse existing metadata.csv and only refresh outputs/M1 files.",
    )
    return parser.parse_args()


def list_images(directory: Path) -> list[Path]:
    """Return all image files in one class directory in deterministic order."""
    return sorted(path for path in directory.iterdir() if path.is_file())


def hardlink_file(src: Path, dst: Path) -> None:
    """Create a hard link so curated outputs do not duplicate image storage."""
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
    """Link files into one curated split and append their metadata rows."""
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


def split_biggan_train(rows: list[dict[str, str]]) -> None:
    """Split BigGAN train into 90% train and 10% val per class."""
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
    """Map original validation sets to seen and unseen test splits."""
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


def metadata_path() -> Path:
    """Return the curated metadata CSV path used by training and M1 export."""
    return CURATED_DIR / "metadata.csv"


def statistics_path() -> Path:
    """Return the curated statistics CSV path used by reporting."""
    return CURATED_DIR / "dataset_statistics.csv"


def write_metadata(rows: list[dict[str, str]]) -> None:
    """Write one metadata row per curated image."""
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
    with metadata_path().open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))


def write_statistics(rows: list[dict[str, str]]) -> None:
    """Summarize real/fake counts by generator and by split."""
    generator_counts = Counter((row["generator"], row["label"]) for row in rows)
    split_counts = Counter((row["split"], row["label"]) for row in rows)

    with statistics_path().open("w", newline="", encoding="utf-8") as handle:
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


def read_metadata() -> list[dict[str, str]]:
    """Load existing metadata rows for validation or output export."""
    with metadata_path().open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_csv_to_m1_outputs() -> None:
    """Place required M1 CSV deliverables under outputs/M1."""
    M1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in (metadata_path(), statistics_path()):
        dst = M1_OUTPUT_DIR / src.name
        if dst.exists():
            dst.unlink()
        os.link(src, dst)


def export_sample_images(rows: list[dict[str, str]]) -> None:
    """Export one real and one fake sample image for each split."""
    seen_keys: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: item["relative_path"]):
        key = (row["split"], row["label"])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        src = BASE_DIR / row["relative_path"]
        dst = M1_OUTPUT_DIR / "samples" / row["split"] / row["label"] / row["filename"]
        hardlink_file(src, dst)


def export_m1_outputs(rows: list[dict[str, str]]) -> None:
    """Refresh the M1 output folder with CSV files and image examples."""
    copy_csv_to_m1_outputs()
    export_sample_images(rows)


def curate_dataset() -> list[dict[str, str]]:
    """Create the curated split directory and return all metadata rows."""
    if CURATED_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {CURATED_DIR}")

    rows: list[dict[str, str]] = []
    split_biggan_train(rows)
    add_existing_val_as_test(rows)
    write_metadata(rows)
    write_statistics(rows)
    return rows


def main() -> None:
    """Run full curation, or refresh only the M1 output deliverables."""
    args = parse_args()
    rows = read_metadata() if args.export_m1_only else curate_dataset()
    export_m1_outputs(rows)

    print(f"Curated dataset: {CURATED_DIR}")
    print(f"M1 outputs: {M1_OUTPUT_DIR}")
    print(f"Random seed: {SEED}")
    print(f"Total images: {len(rows)}")


if __name__ == "__main__":
    main()
