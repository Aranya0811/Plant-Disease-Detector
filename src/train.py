"""
Train the plant disease classifier.

Usage:
    python -m src.train
    python -m src.train --epochs 15 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import create_dataloaders, save_class_map, set_seed
from src.models.classifier import build_model, set_backbone_trainable


def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease CNN")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def plot_training_history(history: dict, logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(logs_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, logs_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        xticklabels=[name[:20] for name in class_names],
        yticklabels=[name[:20] for name in class_names],
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(logs_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    config = load_config()
    set_seed(config["random_seed"])

    if args.epochs:
        config["num_epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, metadata = create_dataloaders(config)
    print(
        f"Samples -> train: {metadata['train_count']}, "
        f"val: {metadata['val_count']}, test: {metadata['test_count']}"
    )
    print(f"Classes: {metadata['num_classes']}")

    save_class_map(metadata["class_to_idx"], config["class_map_path"])

    model = build_model(config["model_name"], metadata["num_classes"], pretrained=True)
    if config.get("freeze_backbone", True):
        set_backbone_trainable(model, trainable=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    logs_dir = Path(config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_f1 = 0.0
    best_path = Path(config["model_save_path"])
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["num_epochs"] + 1):
        unfreeze_epoch = config.get("unfreeze_epoch", 0)
        if unfreeze_epoch and epoch == unfreeze_epoch:
            print(f"Epoch {epoch}: unfreezing backbone for full fine-tuning")
            set_backbone_trainable(model, trainable=True)
            optimizer = Adam(model.parameters(), lr=config["learning_rate"] * 0.1)

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{config['num_epochs']} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} f1 {train_f1:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config["model_name"],
                    "num_classes": metadata["num_classes"],
                    "image_size": config["image_size"],
                    "class_to_idx": metadata["class_to_idx"],
                    "best_val_f1": best_val_f1,
                },
                best_path,
            )
            print(f"  Saved best model to {best_path} (val F1={best_val_f1:.4f})")

    # ── Test evaluation ──────────────────────────────────────────────────────
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_acc, test_f1, test_preds, test_labels = run_epoch(
        model, test_loader, criterion, optimizer, device, train=False
    )

    idx_to_class = metadata["idx_to_class"]
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print("\n=== Test Results ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    with open(logs_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
                "classification_report": report,
            },
            f,
            indent=2,
        )

    plot_training_history(history, logs_dir)
    plot_confusion_matrix(test_labels, test_preds, class_names, logs_dir)
    print(f"Logs saved to {logs_dir}")


if __name__ == "__main__":
    main()
