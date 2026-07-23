#!/usr/bin/env python3
"""Curate MS COCOAI splits for the existing ResNet-18 detector.

This script converts the Hugging Face MS COCOAI dataset into the metadata CSV
format already consumed by train_resnet18_baseline.py. It keeps the model and
training pipeline unchanged while replacing the old GenImage sources with a
2026 dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


BASE_DIR = Path("/root/autodl-tmp/488FP")
DATASET_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
CURATED_DIR = BASE_DIR / "datasets" / "curated_mscocoai"
LOCAL_DATA_DIR = BASE_DIR / "datasets" / "Defactify_Image_Dataset" / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "M3_mscocoai" / "curation"
SEED = 488

LABEL_B_TO_GENERATOR = {
    0: "real",
    1: "stable_diffusion_2_1",
    2: "sdxl",
    3: "stable_diffusion_3",
    4: "dalle_3",
    5: "midjourney_v6",
}
GENERATOR_TO_LABEL_B = {value: key for key, value in LABEL_B_TO_GENERATOR.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--local-data-dir", type=Path, default=LOCAL_DATA_DIR)
    parser.add_argument("--curated-dir", type=Path, default=CURATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-generator", default="stable_diffusion_3")
    parser.add_argument("--seen-generator", default=None)
    parser.add_argument("--unseen-generator", default="midjourney_v6")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-per-label-per-split", type=int, default=None)
    parser.add_argument("--parquet-batch-size", type=int, default=256)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable Hugging Face streaming. Streaming is enabled by default to keep memory usage low.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow refreshing metadata in an existing curated directory.",
    )
    return parser.parse_args()


def import_datasets() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install Hugging Face datasets first, for example "
            "`pip install datasets`."
        ) from exc
    return load_dataset


def parse_generator_list(value: str) -> list[str]:
    generators = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in generators if item not in GENERATOR_TO_LABEL_B]
    if invalid:
        valid = ", ".join(sorted(GENERATOR_TO_LABEL_B))
        raise SystemExit(f"Unsupported generator(s): {', '.join(invalid)}. Valid choices: {valid}")
    return generators


def local_parquet_files(data_dir: Path) -> dict[str, list[str]]:
    """Return local parquet files grouped by Hugging Face split name."""
    return {
        "train": [str(path) for path in sorted(data_dir.glob("train-*.parquet"))],
        "validation": [str(path) for path in sorted(data_dir.glob("validation-*.parquet"))],
        "test": [str(path) for path in sorted(data_dir.glob("test-*.parquet"))],
    }


def load_source_dataset(load_dataset: Any, args: argparse.Namespace) -> Any:
    """Load MS COCOAI from local parquet files when available."""
    if args.local_data_dir.exists():
        data_files = local_parquet_files(args.local_data_dir)
        missing = [split for split, files in data_files.items() if not files]
        if missing:
            raise SystemExit(f"Missing local parquet files for split(s): {', '.join(missing)}")
        print(f"Loading local parquet files from: {args.local_data_dir}", flush=True)
        return load_dataset("parquet", data_files=data_files, streaming=not args.no_streaming)

    print(f"Loading remote dataset from Hugging Face: {args.dataset_id}", flush=True)
    return load_dataset(args.dataset_id, streaming=not args.no_streaming)


def normalize_generator_name(label_b: int) -> str:
    try:
        return LABEL_B_TO_GENERATOR[int(label_b)]
    except KeyError as exc:
        raise ValueError(f"Unsupported Label_B value: {label_b}") from exc


def label_from_label_a(label_a: int) -> str:
    return "fake" if int(label_a) == 1 else "real"


def image_digest(image: Image.Image) -> str:
    h = hashlib.sha1()
    h.update(image.tobytes())
    h.update(str(image.size).encode("utf-8"))
    return h.hexdigest()[:12]


def bytes_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    rgb = image.convert("RGB")
    rgb.save(path, format="JPEG", quality=95)


def save_image_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def iter_local_parquet_rows(data_dir: Path, split: str, batch_size: int) -> Iterable[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    patterns = {
        "train": "train-*.parquet",
        "validation": "validation-*.parquet",
        "test": "test-*.parquet",
    }
    row_index = 0
    for parquet_path in sorted(data_dir.glob(patterns[split])):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            columns = batch.to_pydict()
            for offset in range(len(columns["Label_A"])):
                yield row_index, {key: value[offset] for key, value in columns.items()}
                row_index += 1


def image_bytes_and_name(image_field: Any, fallback_index: int) -> tuple[bytes, str]:
    if isinstance(image_field, dict) and image_field.get("bytes") is not None:
        source_name = image_field.get("path") or f"image_{fallback_index}.jpg"
        return image_field["bytes"], Path(source_name).name

    if isinstance(image_field, Image.Image):
        buffer = io.BytesIO()
        image_field.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue(), f"image_{fallback_index}.jpg"

    raise TypeError(f"Unsupported Image field type: {type(image_field)!r}")


def add_rows_for_local_generator(
    metadata_rows: list[dict[str, str]],
    args: argparse.Namespace,
    source_split: str,
    split: str,
    generator: str,
) -> None:
    fake_label_b = GENERATOR_TO_LABEL_B[generator]
    saved_counts = {0: 0, fake_label_b: 0}

    for source_index, row in iter_local_parquet_rows(args.local_data_dir, source_split, args.parquet_batch_size):
        label_b = int(row["Label_B"])
        if label_b not in saved_counts:
            continue
        if args.max_per_label_per_split is not None and saved_counts[label_b] >= args.max_per_label_per_split:
            continue

        label = label_from_label_a(row["Label_A"])
        class_name = "ai" if label == "fake" else "nature"
        source_generator = normalize_generator_name(label_b)
        data, source_name = image_bytes_and_name(row["Image"], source_index)
        suffix = Path(source_name).suffix or ".jpg"
        filename = f"{source_index:06d}_{bytes_digest(data)}{suffix}"
        dst = args.curated_dir / split / generator / class_name / filename
        save_image_bytes(data, dst)
        saved_counts[label_b] += 1
        metadata_rows.append(
            {
                "relative_path": str(dst.relative_to(args.base_dir)),
                "source_path": f"{source_split}:{source_index}",
                "split": split,
                "generator": generator,
                "source_generator": source_generator,
                "class_name": class_name,
                "label": label,
                "is_fake": "1" if label == "fake" else "0",
                "label_a": str(row["Label_A"]),
                "label_b": str(row["Label_B"]),
                "caption": row.get("Caption", ""),
                "filename": filename,
            }
        )

    print(
        f"{split}/{generator}: saved real={saved_counts[0]} fake={saved_counts[fake_label_b]}",
        flush=True,
    )


def add_rows_for_generator(
    metadata_rows: list[dict[str, str]],
    hf_rows: Iterable[dict[str, Any]],
    base_dir: Path,
    curated_dir: Path,
    split: str,
    generator: str,
    max_per_label: int | None,
) -> None:
    fake_label_b = GENERATOR_TO_LABEL_B[generator]
    saved_counts = {0: 0, fake_label_b: 0}

    for source_index, row in enumerate(hf_rows):
        label_b = int(row["Label_B"])
        if label_b not in saved_counts:
            continue
        if max_per_label is not None and saved_counts[label_b] >= max_per_label:
            continue

        label = label_from_label_a(row["Label_A"])
        class_name = "ai" if label == "fake" else "nature"
        source_generator = normalize_generator_name(label_b)
        image = row["Image"]
        filename = f"{source_index:06d}_{image_digest(image)}.jpg"
        dst = curated_dir / split / generator / class_name / filename
        save_image(image, dst)
        saved_counts[label_b] += 1
        metadata_rows.append(
            {
                "relative_path": str(dst.relative_to(base_dir)),
                "source_path": f"{split}:{source_index}",
                "split": split,
                "generator": generator,
                "source_generator": source_generator,
                "class_name": class_name,
                "label": label,
                "is_fake": "1" if label == "fake" else "0",
                "label_a": str(row["Label_A"]),
                "label_b": str(row["Label_B"]),
                "caption": row.get("Caption", ""),
                "filename": filename,
            }
        )

    print(
        f"{split}/{generator}: saved real={saved_counts[0]} fake={saved_counts[fake_label_b]}",
        flush=True,
    )


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "relative_path",
        "source_path",
        "split",
        "generator",
        "source_generator",
        "class_name",
        "label",
        "is_fake",
        "label_a",
        "label_b",
        "caption",
        "filename",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))


def write_statistics(path: Path, rows: list[dict[str, str]]) -> None:
    generator_counts = Counter((row["generator"], row["label"]) for row in rows)
    split_counts = Counter((row["split"], row["label"]) for row in rows)
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


def main() -> None:
    args = parse_args()
    args.base_dir = args.base_dir.resolve()
    args.local_data_dir = args.local_data_dir.resolve()
    args.curated_dir = args.curated_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    train_generators = parse_generator_list(args.train_generator)
    seen_generators = parse_generator_list(args.seen_generator) if args.seen_generator else train_generators
    unseen_generators = parse_generator_list(args.unseen_generator)

    if args.curated_dir.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing directory: {args.curated_dir}")

    metadata_rows: list[dict[str, str]] = []
    protocols = (
        [("train", "train", generator) for generator in train_generators]
        + [("validation", "val", generator) for generator in train_generators]
        + [("test", "test_seen", generator) for generator in seen_generators]
        + [("test", "test_unseen", generator) for generator in unseen_generators]
    )
    if args.local_data_dir.exists():
        missing = [split for split, files in local_parquet_files(args.local_data_dir).items() if not files]
        if missing:
            raise SystemExit(f"Missing local parquet files for split(s): {', '.join(missing)}")
        print(f"Reading local parquet files from: {args.local_data_dir}", flush=True)
        for source_split, split, generator in protocols:
            add_rows_for_local_generator(metadata_rows, args, source_split, split, generator)
    else:
        load_dataset = import_datasets()
        dataset = load_source_dataset(load_dataset, args)
        for hf_split, split, generator in protocols:
            add_rows_for_generator(
                metadata_rows,
                dataset[hf_split],
                base_dir=args.base_dir,
                curated_dir=args.curated_dir,
                split=split,
                generator=generator,
                max_per_label=args.max_per_label_per_split,
            )

    args.curated_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.curated_dir / "metadata.csv", metadata_rows)
    write_statistics(args.curated_dir / "dataset_statistics.csv", metadata_rows)
    write_metadata(args.output_dir / "metadata.csv", metadata_rows)
    write_statistics(args.output_dir / "dataset_statistics.csv", metadata_rows)

    print(f"Curated dataset: {args.curated_dir}")
    print(f"Curation outputs: {args.output_dir}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Train/seen/unseen: {args.train_generator}/{','.join(seen_generators)}/{args.unseen_generator}")
    print(f"Total images: {len(metadata_rows)}")


if __name__ == "__main__":
    main()
