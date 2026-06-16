#!/usr/bin/env python3
"""Analyze image formats and resolutions for the M1 curated subset.

The script reads outputs/M1/metadata.csv, opens each selected image without
modifying it, records image dimensions, reports corrupted/empty files, and
exports CSV summaries plus plots under outputs/M1/resolution_analysis.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BASE_DIR = Path("/root/autodl-tmp/488FP")
METADATA_PATH = BASE_DIR / "outputs" / "M1" / "metadata.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "M1" / "resolution_analysis"


def read_metadata() -> list[dict[str, str]]:
    """Load the M1 metadata rows that define the selected image subset."""
    with METADATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def inspect_images(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read image headers and keep only fields needed for statistics/plots."""
    valid_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []

    for row in rows:
        image_path = BASE_DIR / row["relative_path"]
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format or ""
        except Exception as exc:
            invalid_rows.append(
                {
                    "relative_path": row["relative_path"],
                    "split": row["split"],
                    "generator": row["generator"],
                    "label": row["label"],
                    "file_size_bytes": str(image_path.stat().st_size if image_path.exists() else -1),
                    "error": type(exc).__name__,
                }
            )
            continue

        valid_rows.append(
            {
                "split": row["split"],
                "generator": row["generator"],
                "label": row["label"],
                "width": str(width),
                "height": str(height),
                "area": str(width * height),
                "image_format": image_format,
                "extension": image_path.suffix.lower(),
            }
        )

    return valid_rows, invalid_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV file with a stable header order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_resolutions(valid_rows: list[dict[str, str]], invalid_rows: list[dict[str, str]]) -> dict[str, str]:
    """Compute min, median, max, and count statistics for the report."""
    sorted_by_area = sorted(valid_rows, key=lambda row: (int(row["area"]), int(row["width"]), int(row["height"])))
    min_row = sorted_by_area[0]
    median_row = sorted_by_area[(len(sorted_by_area) - 1) // 2]
    max_row = sorted_by_area[-1]

    return {
        "total_rows": str(len(valid_rows) + len(invalid_rows)),
        "valid_images": str(len(valid_rows)),
        "invalid_images": str(len(invalid_rows)),
        "unique_resolutions": str(len({(row["width"], row["height"]) for row in valid_rows})),
        "minimum_resolution": f"{min_row['width']}x{min_row['height']}",
        "minimum_area": min_row["area"],
        "median_resolution": f"{median_row['width']}x{median_row['height']}",
        "median_area": median_row["area"],
        "maximum_resolution": f"{max_row['width']}x{max_row['height']}",
        "maximum_area": max_row["area"],
    }


def write_summary(summary: dict[str, str]) -> None:
    """Write one-row summary statistics for quick reporting."""
    write_csv(OUTPUT_DIR / "resolution_summary.csv", [summary], list(summary.keys()))


def write_count_tables(valid_rows: list[dict[str, str]]) -> None:
    """Write format and top-resolution count tables used by the plots."""
    format_counts = Counter((row["extension"], row["image_format"]) for row in valid_rows)
    format_rows = [
        {"extension": ext, "image_format": fmt, "count": str(count)}
        for (ext, fmt), count in sorted(format_counts.items())
    ]
    write_csv(OUTPUT_DIR / "format_counts.csv", format_rows, ["extension", "image_format", "count"])

    resolution_counts = Counter((row["width"], row["height"]) for row in valid_rows)
    resolution_rows = [
        {"resolution": f"{width}x{height}", "width": width, "height": height, "count": str(count)}
        for (width, height), count in resolution_counts.most_common(30)
    ]
    write_csv(OUTPUT_DIR / "top_resolution_counts.csv", resolution_rows, ["resolution", "width", "height", "count"])


def plot_format_counts(valid_rows: list[dict[str, str]]) -> None:
    """Create a bar chart showing how many selected images use each extension."""
    counts = Counter(row["extension"] for row in valid_rows)
    labels = list(sorted(counts))
    values = [counts[label] for label in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4C78A8", "#F58518"][: len(labels)])
    plt.title("M1 Selected Images by File Extension")
    plt.xlabel("Extension")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "format_counts.png", dpi=200)
    plt.close()


def plot_resolution_scatter(valid_rows: list[dict[str, str]]) -> None:
    """Create a width-vs-height scatter plot sampled for readability."""
    widths = np.array([int(row["width"]) for row in valid_rows])
    heights = np.array([int(row["height"]) for row in valid_rows])
    labels = np.array([row["label"] for row in valid_rows])

    rng = np.random.default_rng(488)
    sample_size = min(20000, len(valid_rows))
    sample_idx = rng.choice(len(valid_rows), size=sample_size, replace=False)

    plt.figure(figsize=(7, 5))
    for label, color in (("real", "#4C78A8"), ("fake", "#F58518")):
        mask = labels[sample_idx] == label
        plt.scatter(widths[sample_idx][mask], heights[sample_idx][mask], s=4, alpha=0.35, label=label, c=color)
    plt.title("M1 Resolution Scatter Sample")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resolution_scatter_sample.png", dpi=200)
    plt.close()


def plot_area_histogram(valid_rows: list[dict[str, str]]) -> None:
    """Create a histogram of image pixel areas on a log-scaled x-axis."""
    areas = np.array([int(row["area"]) for row in valid_rows])

    plt.figure(figsize=(7, 5))
    plt.hist(areas, bins=np.logspace(np.log10(areas.min()), np.log10(areas.max()), 60), color="#54A24B")
    plt.xscale("log")
    plt.title("M1 Image Area Distribution")
    plt.xlabel("Pixel Area (width x height, log scale)")
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "resolution_area_histogram.png", dpi=200)
    plt.close()


def make_plots(valid_rows: list[dict[str, str]]) -> None:
    """Generate all visualization images for the resolution analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_format_counts(valid_rows)
    plot_resolution_scatter(valid_rows)
    plot_area_histogram(valid_rows)


def main() -> None:
    """Run the full M1 resolution analysis and export all artifacts."""
    rows = read_metadata()
    valid_rows, invalid_rows = inspect_images(rows)
    summary = summarize_resolutions(valid_rows, invalid_rows)

    write_summary(summary)
    write_count_tables(valid_rows)
    write_csv(
        OUTPUT_DIR / "invalid_images.csv",
        invalid_rows,
        ["relative_path", "split", "generator", "label", "file_size_bytes", "error"],
    )
    make_plots(valid_rows)

    print("Resolution analysis written to:", OUTPUT_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
