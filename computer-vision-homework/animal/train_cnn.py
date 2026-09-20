"""Train the from-scratch CNN and export reproducible evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ColorJitter, Compose, RandomAffine, Resize, ToTensor
from tqdm import tqdm

from dataset import AnimalDataset
from models import AdvancedCNN


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch animal CNN")
    parser.add_argument("-d", "--data-path", default="datasets")
    parser.add_argument("-i", "--image-size", type=int, default=224)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-t", "--tensorboard", default="animal_log")
    parser.add_argument("--checkpoint-dir", default=".")
    parser.add_argument("--results-dir", default="output-assets")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-class", type=int, default=0)
    parser.add_argument("--max-val-per-class", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_confusion_matrix(labels: list[int], predictions: list[int]) -> np.ndarray:
    matrix = confusion_matrix(labels, predictions)
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)


def save_confusion_matrix(
    writer: SummaryWriter,
    matrix: np.ndarray,
    class_names: list[str],
    epoch: int,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    axis.set(
        title=f"Normalized confusion matrix — epoch {epoch}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "#102a43",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    writer.add_figure("Validation/Confusion matrix", figure, epoch)
    plt.close(figure)


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    loss_axis.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    loss_axis.plot(epochs, [row["val_loss"] for row in history], marker="o", label="Validation loss")
    loss_axis.set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    metric_axis.plot(epochs, [row["accuracy"] for row in history], marker="o", label="Accuracy")
    metric_axis.plot(epochs, [row["macro_f1"] for row in history], marker="o", label="Macro F1")
    metric_axis.set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    metric_axis.grid(alpha=0.25)
    metric_axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    tensorboard_dir = Path(args.tensorboard)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if tensorboard_dir.exists() and not args.resume:
        shutil.rmtree(tensorboard_dir)

    train_transform = Compose(
        [
            Resize((args.image_size, args.image_size)),
            RandomAffine(degrees=5, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),
            ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05),
            ToTensor(),
        ]
    )
    val_transform = Compose([Resize((args.image_size, args.image_size)), ToTensor()])
    train_dataset = AnimalDataset(
        args.data_path,
        is_train=True,
        transform=train_transform,
        max_samples_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    val_dataset = AnimalDataset(
        args.data_path,
        is_train=False,
        transform=val_transform,
        max_samples_per_class=args.max_val_per_class,
        seed=args.seed,
    )
    if train_dataset.categories != val_dataset.categories:
        raise ValueError("Training and validation splits must contain the same class directories")

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_options)

    model = AdvancedCNN(num_classes=len(train_dataset.categories)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_accuracy = 0.0
    best_epoch = 0
    start_epoch = 0
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"

    if args.resume:
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"])
        if start_epoch >= args.epochs:
            raise ValueError(
                f"Checkpoint already completed {start_epoch} epochs; "
                f"set --epochs above {start_epoch} to resume."
            )

    print(f"[config] device={device} seed={args.seed} image_size={args.image_size} batch_size={args.batch_size}")
    print(f"[config] epochs={args.epochs} learning_rate={args.lr} momentum={args.momentum} workers={args.workers}")
    print(f"[dataset] train={len(train_dataset)} validation={len(val_dataset)} classes={len(train_dataset.categories)}")
    print(f"[classes] {', '.join(train_dataset.categories)}")
    print(f"[model] AdvancedCNN parameters={parameter_count:,} pretrained=False")

    writer = SummaryWriter(tensorboard_dir, purge_step=start_epoch if args.resume else None)
    history: list[dict[str, float]] = []
    started_at = time.perf_counter()
    final_labels: list[int] = []
    final_predictions: list[int] = []

    for epoch_index in range(start_epoch, args.epochs):
        epoch = epoch_index + 1
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        train_progress = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", colour="cyan", dynamic_ncols=True)
        for images, labels in train_progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            train_loss_total += loss.item() * batch_size
            train_examples += batch_size
            train_progress.set_postfix(loss=f"{train_loss_total / train_examples:.4f}")

        train_loss = train_loss_total / train_examples
        model.eval()
        val_loss_total = 0.0
        val_examples = 0
        labels_all: list[int] = []
        predictions_all: list[int] = []
        val_progress = tqdm(val_loader, desc=f"Valid {epoch}/{args.epochs}", colour="yellow", dynamic_ncols=True)
        with torch.inference_mode():
            for images, labels in val_progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)
                batch_size = images.size(0)
                val_loss_total += loss.item() * batch_size
                val_examples += batch_size
                labels_all.extend(labels.cpu().tolist())
                predictions_all.extend(predictions.cpu().tolist())
                val_progress.set_postfix(loss=f"{val_loss_total / val_examples:.4f}")

        val_loss = val_loss_total / val_examples
        accuracy = accuracy_score(labels_all, predictions_all)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels_all, predictions_all, average="macro", zero_division=0
        )
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
        }
        history.append(row)
        for name, value in row.items():
            if name != "epoch":
                writer.add_scalar(name.replace("_", "/").title(), value, epoch)

        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_epoch = epoch
            is_best = True
        else:
            is_best = False
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "categories": train_dataset.categories,
            "image_size": args.image_size,
        }
        torch.save(checkpoint, last_checkpoint)
        if is_best:
            torch.save(checkpoint, best_checkpoint)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} "
            f"| accuracy={accuracy:.4f} | macro_precision={macro_precision:.4f} "
            f"| macro_recall={macro_recall:.4f} | macro_f1={macro_f1:.4f}"
        )
        final_labels = labels_all
        final_predictions = predictions_all

    elapsed_seconds = time.perf_counter() - started_at
    confusion = normalized_confusion_matrix(final_labels, final_predictions)
    save_confusion_matrix(
        writer,
        confusion,
        train_dataset.categories,
        args.epochs,
        results_dir / "normalized-confusion-matrix.png",
    )
    save_training_curves(history, results_dir / "training-curves.png")
    writer.flush()
    writer.close()

    report = {
        "run": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "scikit_learn": sklearn.__version__,
            "seed": args.seed,
            "elapsed_seconds": elapsed_seconds,
        },
        "data": {
            "path": str(Path(args.data_path)),
            "classes": train_dataset.categories,
            "train_images": len(train_dataset),
            "validation_images": len(val_dataset),
            "max_train_per_class": args.max_train_per_class,
            "max_val_per_class": args.max_val_per_class,
        },
        "model": {"name": "AdvancedCNN", "parameters": parameter_count, "pretrained": False},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.lr,
            "momentum": args.momentum,
            "optimizer": "SGD",
            "loss": "CrossEntropyLoss",
            "workers": args.workers,
        },
        "best": {"epoch": best_epoch, "accuracy": best_accuracy},
        "history": history,
        "final_confusion_matrix_normalized": confusion.tolist(),
    }
    with (results_dir / "training-metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"[complete] best_epoch={best_epoch} best_accuracy={best_accuracy:.4f} elapsed={elapsed_seconds:.1f}s")
    print(f"[artifacts] metrics={results_dir / 'training-metrics.json'} tensorboard={tensorboard_dir}")
    print(f"[checkpoints] best={best_checkpoint} last={last_checkpoint} (ignored by Git)")
    return report


if __name__ == "__main__":
    train(get_args())
