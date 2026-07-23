#!/usr/bin/env python3
"""Train a ResNet-18 baseline for real/fake image detection.

The current final-project pipeline uses MS COCOAI metadata. The script can
train on one generator or a comma-separated generator list, then evaluate on
seen and unseen generator splits.
"""

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
from torchvision import models, transforms


LABEL_TO_INDEX = {"real": 0, "fake": 1}
INDEX_TO_LABEL = {0: "real", 1: "fake"}


def parse_generator_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass
class SplitMetrics:
    """Metrics for one validation/test split."""

    split: str
    loss: float
    accuracy: float
    f1: float
    auc: float
    tn: int
    fp: int
    fn: int
    tp: int


class GenImageMetadataDataset(Dataset):
    """Read images listed in the curated MS COCOAI metadata CSV.

    The metadata file stores paths relative to the project root. We filter rows
    by split and generator so different cross-generator protocols can reuse the
    same training script.
    """

    def __init__(
        self,
        base_dir: Path,
        split: str,
        generator: str,
        transform: transforms.Compose,
        metadata_path: Path | None = None,
        max_samples: int | None = None,
        seed: int = 488,
    ) -> None:
        self.base_dir = base_dir
        self.split = split
        self.generator = generator
        self.generators = parse_generator_list(generator)
        self.transform = transform
        self.metadata_path = metadata_path or self.base_dir / "datasets" / "curated_mscocoai" / "metadata.csv"
        self.rows = self._read_rows()

        # A small subset is useful for a smoke test before committing to the
        # full dataset. Sampling is deterministic so results are reproducible.
        if max_samples is not None and max_samples < len(self.rows):
            rng = random.Random(seed)
            indices = list(range(len(self.rows)))
            rng.shuffle(indices)
            chosen = sorted(indices[:max_samples])
            self.rows = [self.rows[index] for index in chosen]

    def _read_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with self.metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["split"] == self.split and row["generator"] in self.generators:
                    rows.append(row)

        if not rows:
            raise RuntimeError(f"No rows found for split={self.split}, generator={self.generator}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.base_dir / row["relative_path"]
        label = LABEL_TO_INDEX[row["label"]]

        # Some large web datasets contain a few broken images. We keep the
        # training/evaluation loop robust by replacing unreadable files with a
        # black RGB image, while preserving the original label.
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("/root/autodl-tmp/488FP"))
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Metadata CSV to read. Defaults to datasets/curated_genimage/metadata.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/488FP/outputs/M2"))
    parser.add_argument("--train-generator", default="BigGAN")
    parser.add_argument("--seen-generator", default="BigGAN")
    parser.add_argument("--unseen-generator", default="stable_diffusion_v_1_5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--augment-strength", choices=("light", "strong"), default="strong")
    parser.add_argument("--random-erasing-prob", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-metric", choices=("val_loss", "val_f1", "val_auc"), default="val_loss")
    parser.add_argument("--seed", type=int, default=488)
    parser.add_argument("--weights", choices=("auto", "imagenet", "none"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100, help="Print training progress every N batches.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Make sampling and PyTorch operations as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_transforms(args: argparse.Namespace) -> tuple[transforms.Compose, transforms.Compose]:
    """Create train/eval transforms compatible with ResNet ImageNet inputs.

    The strong training transform intentionally makes low-level generator
    artifacts less stable across batches. This is useful for this project
    because the first baseline quickly overfit BigGAN-specific cues: training
    loss kept decreasing while validation loss increased after epoch 1.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.augment_strength == "strong":
        train_steps = [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.15,
                        hue=0.03,
                    )
                ],
                p=0.5,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=args.random_erasing_prob, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    else:
        train_steps = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    """Build DataLoaders for the four Milestone 2 splits."""
    train_transform, eval_transform = make_transforms(args)
    datasets = {
        "train": GenImageMetadataDataset(
            args.base_dir,
            split="train",
            generator=args.train_generator,
            transform=train_transform,
            metadata_path=args.metadata_path,
            max_samples=args.max_train_samples,
            seed=args.seed,
        ),
        "val": GenImageMetadataDataset(
            args.base_dir,
            split="val",
            generator=args.train_generator,
            transform=eval_transform,
            metadata_path=args.metadata_path,
            max_samples=args.max_eval_samples,
            seed=args.seed,
        ),
        "test_seen": GenImageMetadataDataset(
            args.base_dir,
            split="test_seen",
            generator=args.seen_generator,
            transform=eval_transform,
            metadata_path=args.metadata_path,
            max_samples=args.max_eval_samples,
            seed=args.seed,
        ),
        "test_unseen": GenImageMetadataDataset(
            args.base_dir,
            split="test_unseen",
            generator=args.unseen_generator,
            transform=eval_transform,
            metadata_path=args.metadata_path,
            max_samples=args.max_eval_samples,
            seed=args.seed,
        ),
    }

    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )
        print(f"{split}: {len(dataset)} images")
    return loaders


def make_model(weights_mode: str, dropout: float) -> nn.Module:
    """Create a ResNet-18 with a two-class output head.

    In auto mode we try ImageNet weights first because they usually give a
    stronger baseline. If the server cannot download them, the script falls
    back to random initialization instead of failing the whole run.
    """
    weights = None
    if weights_mode in ("auto", "imagenet"):
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            print("Loaded ImageNet-pretrained ResNet-18 weights.")
        except Exception as exc:
            if weights_mode == "imagenet":
                raise
            print(f"Could not load ImageNet weights; using random init instead: {exc}")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
        print("Using randomly initialized ResNet-18.")

    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, 2))
    else:
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def metric_improved(
    metric_name: str,
    current: SplitMetrics,
    best_value: float,
    min_delta: float,
) -> tuple[bool, float]:
    """Return whether the selected validation metric improved enough.

    Validation loss is minimized, while F1 and AUC are maximized. Using
    validation loss by default makes checkpoint selection sensitive to the
    overfitting pattern where probabilities become overconfident even before
    accuracy collapses.
    """
    if metric_name == "val_loss":
        value = current.loss
        improved = value < best_value - min_delta
    elif metric_name == "val_f1":
        value = current.f1
        improved = value > best_value + min_delta
    elif metric_name == "val_auc":
        value = current.auc
        improved = value > best_value + min_delta
    else:
        raise ValueError(f"Unsupported checkpoint metric: {metric_name}")
    return improved, value


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC for binary labels without an sklearn dependency."""
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

    # Average ranks for ties, matching the Mann-Whitney U definition of AUC.
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
    """Compute accuracy, F1, AUC, and confusion matrix counts."""
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
    return SplitMetrics(split=split, loss=loss, accuracy=accuracy, f1=f1, auc=auc, tn=tn, fp=fp, fn=fn, tp=tp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    log_every: int,
) -> float:
    """Run one training epoch and return average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        if log_every > 0 and batch_index % log_every == 0:
            average_loss = total_loss / max(1, total_count)
            print(
                f"epoch={epoch} batch={batch_index}/{len(loader)} "
                f"train_loss_running={average_loss:.4f}",
                flush=True,
            )
    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, split: str) -> SplitMetrics:
    """Evaluate one split and return metrics.

    Softmax converts the two logits into class probabilities. The fake class
    probability is used as the continuous score for AUC.
    """
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_fake_scores: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_fake_scores.append(probabilities[:, 1].cpu().numpy())

    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    scores_np = np.concatenate(all_fake_scores)
    return compute_metrics(split, total_loss / max(1, total_count), labels_np, preds_np, scores_np)


def write_training_log(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "val_auc"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: Path, metrics: list[SplitMetrics]) -> None:
    fieldnames = ["split", "loss", "accuracy", "f1", "auc", "tn", "fp", "fn", "tp"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(item.__dict__)


def plot_training_curves(path: Path, rows: list[dict[str, float | int]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [float(row["train_loss"]) for row in rows], marker="o", label="train loss")
    plt.plot(epochs, [float(row["val_loss"]) for row in rows], marker="o", label="val loss")
    plt.plot(epochs, [float(row["val_accuracy"]) for row in rows], marker="o", label="val accuracy")
    plt.plot(epochs, [float(row["val_f1"]) for row in rows], marker="o", label="val F1")
    plt.plot(epochs, [float(row["val_auc"]) for row in rows], marker="o", label="val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("ResNet-18 Baseline Training Curves")
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


def plot_seen_unseen_comparison(path: Path, metrics: list[SplitMetrics], seen_generator: str, unseen_generator: str) -> None:
    selected = {metric.split: metric for metric in metrics if metric.split in {"test_seen", "test_unseen"}}
    names = ["accuracy", "f1", "auc"]
    x = np.arange(len(names))
    width = 0.35
    seen_values = [getattr(selected["test_seen"], name) for name in names]
    unseen_values = [getattr(selected["test_unseen"], name) for name in names]

    plt.figure(figsize=(7, 4.5))
    plt.bar(x - width / 2, seen_values, width, label=f"seen: {seen_generator}")
    plt.bar(x + width / 2, unseen_values, width, label=f"unseen: {unseen_generator}")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Seen vs Unseen Generator Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_run_config(path: Path, args: argparse.Namespace) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_run_config(args.output_dir / "run_config.json", args)

    loaders = make_loaders(args)
    model = make_model(args.weights, args.dropout).to(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_metric = math.inf if args.checkpoint_metric == "val_loss" else -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    training_rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            args.device,
            epoch=epoch,
            log_every=args.log_every,
        )
        val_metrics = evaluate(model, loaders["val"], criterion, args.device, "val")
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "val_accuracy": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_auc": val_metrics.auc,
        }
        training_rows.append(row)
        write_training_log(args.output_dir / "training_log.csv", training_rows)
        plot_training_curves(args.output_dir / "training_curves.png", training_rows)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f} val_auc={val_metrics.auc:.4f}"
        )

        improved, current_metric = metric_improved(
            args.checkpoint_metric,
            val_metrics,
            best_metric,
            args.early_stopping_min_delta,
        )
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "checkpoints" / "best_resnet18.pt")
            print(f"saved best checkpoint at epoch={epoch} using {args.checkpoint_metric}={best_metric:.6f}")
        else:
            epochs_without_improvement += 1
            print(
                f"no {args.checkpoint_metric} improvement for "
                f"{epochs_without_improvement}/{args.early_stopping_patience} epochs"
            )

        scheduler.step()
        if args.early_stopping_patience >= 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"early stopping at epoch={epoch}; "
                f"best_epoch={best_epoch}, best_{args.checkpoint_metric}={best_metric:.6f}"
            )
            break

    model.load_state_dict(torch.load(args.output_dir / "checkpoints" / "best_resnet18.pt", map_location=args.device))
    final_metrics = [
        evaluate(model, loaders["val"], criterion, args.device, "val"),
        evaluate(model, loaders["test_seen"], criterion, args.device, "test_seen"),
        evaluate(model, loaders["test_unseen"], criterion, args.device, "test_unseen"),
    ]
    write_metrics(args.output_dir / "metrics.csv", final_metrics)
    plot_confusion_matrix(args.output_dir / "confusion_matrix_seen.png", final_metrics[1])
    plot_confusion_matrix(args.output_dir / "confusion_matrix_unseen.png", final_metrics[2])
    plot_seen_unseen_comparison(
        args.output_dir / "seen_unseen_comparison.png",
        final_metrics,
        seen_generator=args.seen_generator,
        unseen_generator=args.unseen_generator,
    )

    for metric in final_metrics:
        print(
            f"{metric.split}: loss={metric.loss:.4f} acc={metric.accuracy:.4f} "
            f"f1={metric.f1:.4f} auc={metric.auc:.4f} "
            f"tn={metric.tn} fp={metric.fp} fn={metric.fn} tp={metric.tp}"
        )


if __name__ == "__main__":
    main()
