"""Dataset loading, splitting, and PyTorch DataLoaders."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_samples(data_dir: Path, selected_classes: list[str] | None = None) -> list[tuple[str, str]]:
    """Return list of (filepath, class_name) from folder-per-class layout."""
    samples: list[tuple[str, str]] = []

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            "Download PlantVillage from Kaggle and extract to data/PlantVillage/color/"
        )

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if selected_classes:
        class_dirs = [p for p in class_dirs if p.name in selected_classes]

    if not class_dirs:
        raise ValueError("No class folders found. Check data_dir and selected_classes in config.yaml.")

    for class_dir in class_dirs:
        class_name = class_dir.name
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(image_path), class_name))

    if not samples:
        raise ValueError(f"No images found under {data_dir}")

    return samples


def build_class_map(class_names: list[str]) -> dict[str, int]:
    sorted_names = sorted(class_names)
    return {name: idx for idx, name in enumerate(sorted_names)}


def friendly_label(class_name: str) -> dict[str, str]:
    """Turn 'Tomato___Early_blight' into readable fields."""
    if "___" in class_name:
        plant, condition = class_name.split("___", 1)
    elif "__" in class_name:
        parts = class_name.split("__")
        plant = parts[0]
        condition = "__".join(parts[1:])
    else:
        plant, condition = class_name, "unknown"

    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ")
    return {
        "raw_class": class_name,
        "plant": plant,
        "condition": condition,
        "display_name": f"{plant} — {condition}",
    }


class PlantDiseaseDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, str]],
        class_to_idx: dict[str, int],
        transform=None,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, class_name = self.samples[index]
        image = Image.open(path).convert("RGB")
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: int, train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def split_samples(
    samples: list[tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    labels = [label for _, label in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(1.0 - train_ratio),
        stratify=labels,
        random_state=seed,
    )
    temp_labels = [label for _, label in temp_samples]
    relative_test = test_ratio / (val_ratio + test_ratio)

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=relative_test,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_samples, val_samples, test_samples


def create_dataloaders(config: dict):
    data_dir = Path(config["data_dir"])
    selected = config.get("selected_classes")
    samples = discover_samples(data_dir, selected)

    class_names = sorted({label for _, label in samples})
    class_to_idx = build_class_map(class_names)

    train_samples, val_samples, test_samples = split_samples(
        samples,
        config["train_ratio"],
        config["val_ratio"],
        config["test_ratio"],
        config["random_seed"],
    )

    train_ds = PlantDiseaseDataset(
        train_samples,
        class_to_idx,
        transform=get_transforms(config["image_size"], train=True),
    )
    val_ds = PlantDiseaseDataset(
        val_samples,
        class_to_idx,
        transform=get_transforms(config["image_size"], train=False),
    )
    test_ds = PlantDiseaseDataset(
        test_samples,
        class_to_idx,
        transform=get_transforms(config["image_size"], train=False),
    )

    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    metadata = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
        "num_classes": len(class_to_idx),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
    }
    return train_loader, val_loader, test_loader, metadata


def save_class_map(class_to_idx: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
        "friendly_labels": {k: friendly_label(k) for k in class_to_idx},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_class_map(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
