#!/usr/bin/env python3
"""Train a frozen-CLIP linear detector for real/fake image detection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor


LABEL_TO_INDEX = {"real": 0, "fake": 1}
INDEX_TO_LABEL = {0: "real", 1: "fake"}


def parse_generator_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass
class SplitMetrics:
    split: str
    loss: float
    accuracy: float
    f1: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int


class MetadataImageDataset(Dataset):
    def __init__(
        self,
        base_dir: Path,
        metadata_path: Path,
        split: str,
        generator: str,
        max_samples: int | None,
        seed: int,
    ) -> None:
        self.base_dir = base_dir
        self.metadata_path = metadata_path
        self.split = split
        self.generators = parse_generator_list(generator)
        self.rows = self._read_rows()

        if max_samples is not None and max_samples < len(self.rows):
            rng = random.Random(seed)
            indices = list(range(len(self.rows)))
            rng.shuffle(indices)
            self.rows = [self.rows[index] for index in sorted(indices[:max_samples])]

    def _read_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with self.metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == self.split and row["generator"] in self.generators:
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"No rows found for split={self.split}, generator={','.join(sorted(self.generators))}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        row = self.rows[index]
        image_path = self.base_dir / row["relative_path"]
        label = LABEL_TO_INDEX[row["label"]]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        return image, label


def collate_images(batch: list[tuple[Image.Image, int]]) -> tuple[list[Image.Image], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("/root/autodl-tmp/488FP"))
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-generator", required=True)
    parser.add_argument("--seen-generator", required=True)
    parser.add_argument("--unseen-generator", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=488)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-endpoint", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_datasets(args: argparse.Namespace) -> dict[str, MetadataImageDataset]:
    return {
        "train": MetadataImageDataset(
            args.base_dir,
            args.metadata_path,
            "train",
            args.train_generator,
            args.max_train_samples,
            args.seed,
        ),
        "val": MetadataImageDataset(
            args.base_dir,
            args.metadata_path,
            "val",
            args.train_generator,
            args.max_eval_samples,
            args.seed,
        ),
        "test_seen": MetadataImageDataset(
            args.base_dir,
            args.metadata_path,
            "test_seen",
            args.seen_generator,
            args.max_eval_samples,
            args.seed,
        ),
        "test_unseen": MetadataImageDataset(
            args.base_dir,
            args.metadata_path,
            "test_unseen",
            args.unseen_generator,
            args.max_eval_samples,
            args.seed,
        ),
    }


@torch.no_grad()
def extract_features_for_split(
    split: str,
    dataset: MetadataImageDataset,
    model: CLIPModel,
    processor: CLIPProcessor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_images,
        pin_memory=torch.cuda.is_available(),
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for batch_index, (images, batch_labels) in enumerate(loader, start=1):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(args.device)
        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = image_features / image_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        features.append(image_features.cpu().numpy().astype(np.float32))
        labels.append(batch_labels.numpy().astype(np.int64))
        if batch_index % 50 == 0:
            print(f"features {split}: batch={batch_index}/{len(loader)}", flush=True)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def load_or_extract_features(args: argparse.Namespace) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    datasets = make_datasets(args)
    for split, dataset in datasets.items():
        print(f"{split}: {len(dataset)} images")

    model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    extracted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, dataset in datasets.items():
        cache_path = feature_dir / f"{split}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            extracted[split] = (data["features"], data["labels"])
            print(f"loaded cached features: {cache_path}")
            continue
        features, labels = extract_features_for_split(split, dataset, model, processor, args)
        np.savez_compressed(cache_path, features=features, labels=labels)
        extracted[split] = (features, labels)
        print(f"saved features: {cache_path} shape={features.shape}")
    return extracted


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(order, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    sum_pos_ranks = ranks[positives].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(split: str, loss: float, labels: np.ndarray, preds: np.ndarray, fake_scores: np.ndarray) -> SplitMetrics:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    accuracy = float((tp + tn) / max(1, len(labels)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    auc = binary_auc(labels, fake_scores)
    return SplitMetrics(split, loss, accuracy, f1, auc, tn, fp, fn, tp)


@torch.no_grad()
def evaluate(classifier: nn.Module, features: np.ndarray, labels: np.ndarray, criterion: nn.Module, device: str, split: str) -> SplitMetrics:
    dataset = FeatureDataset(features, labels)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    classifier.eval()
    total_loss = 0.0
    total_count = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        logits = classifier(batch_features)
        loss = criterion(logits, batch_labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += float(loss.item()) * batch_labels.size(0)
        total_count += batch_labels.size(0)
        all_labels.append(batch_labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_scores.append(probs[:, 1].cpu().numpy())
    return compute_metrics(
        split,
        total_loss / max(1, total_count),
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_scores),
    )


def train_classifier(features: dict[str, tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> tuple[nn.Module, list[dict[str, float | int]], int]:
    train_features, train_labels = features["train"]
    val_features, val_labels = features["val"]
    classifier = nn.Linear(train_features.shape[1], 2).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(FeatureDataset(train_features, train_labels), batch_size=1024, shuffle=True)

    best_loss = math.inf
    best_epoch = 0
    no_improve = 0
    rows: list[dict[str, float | int]] = []
    best_path = args.output_dir / "checkpoints" / "best_clip_linear.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        total_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(args.device)
            batch_labels = batch_labels.to(args.device)
            logits = classifier(batch_features)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_labels.size(0)
            total_count += batch_labels.size(0)
        train_loss = total_loss / max(1, total_count)
        val_metrics = evaluate(classifier, val_features, val_labels, criterion, args.device, "val")
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_auc": val_metrics.auc,
        }
        rows.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f} val_auc={val_metrics.auc:.4f}",
            flush=True,
        )
        if val_metrics.loss < best_loss - args.early_stopping_min_delta:
            best_loss = val_metrics.loss
            best_epoch = epoch
            no_improve = 0
            torch.save(classifier.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}", flush=True)
                break

    classifier.load_state_dict(torch.load(best_path, map_location=args.device))
    return classifier, rows, best_epoch


def write_training_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "val_auc"])
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: Path, metrics: list[SplitMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "loss", "accuracy", "f1", "auc", "tn", "fp", "fn", "tp"])
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.__dict__)


def plot_training_curves(path: Path, rows: list[dict[str, float | int]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(8, 5))
    for name in ("train_loss", "val_loss", "val_accuracy", "val_f1", "val_auc"):
        plt.plot(epochs, [float(row[name]) for row in rows], marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("CLIP Linear Detector Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion_matrix(path: Path, metric: SplitMetrics) -> None:
    matrix = np.array([[metric.tn, metric.fp], [metric.fn, metric.tp]])
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix: {metric.split}")
    plt.xticks([0, 1], [INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]])
    plt.yticks([0, 1], [INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_split_metrics(path: Path, metrics: list[SplitMetrics]) -> None:
    splits = [metric.split for metric in metrics]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.ravel(), ("loss", "accuracy", "f1", "auc")):
        values = [getattr(metric, name) for metric in metrics]
        ax.bar(splits, values, color=["#4C78A8", "#72B7B2", "#F58518"])
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.25)
        if name != "loss":
            ax.set_ylim(0, 1)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("CLIP Linear Split Metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_run_config(path: Path, args: argparse.Namespace, best_epoch: int) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["best_epoch"] = best_epoch
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = load_or_extract_features(args)
    classifier, rows, best_epoch = train_classifier(features, args)
    criterion = nn.CrossEntropyLoss()
    final_metrics = [
        evaluate(classifier, *features["val"], criterion, args.device, "val"),
        evaluate(classifier, *features["test_seen"], criterion, args.device, "test_seen"),
        evaluate(classifier, *features["test_unseen"], criterion, args.device, "test_unseen"),
    ]
    write_training_log(args.output_dir / "training_log.csv", rows)
    write_metrics(args.output_dir / "metrics.csv", final_metrics)
    plot_training_curves(args.output_dir / "training_curves.png", rows)
    plot_confusion_matrix(args.output_dir / "confusion_matrix_seen.png", final_metrics[1])
    plot_confusion_matrix(args.output_dir / "confusion_matrix_unseen.png", final_metrics[2])
    plot_split_metrics(args.output_dir / "split_metrics.png", final_metrics)
    save_run_config(args.output_dir / "run_config.json", args, best_epoch)

    for metric in final_metrics:
        print(
            f"{metric.split}: loss={metric.loss:.4f} acc={metric.accuracy:.4f} "
            f"f1={metric.f1:.4f} auc={metric.auc:.4f} "
            f"tn={metric.tn} fp={metric.fp} fn={metric.fn} tp={metric.tp}",
            flush=True,
        )


if __name__ == "__main__":
    main()
