"""Quick EDA script — run after downloading dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import load_config
from src.data.dataset import discover_samples, friendly_label


def main():
    config = load_config()
    data_dir = Path(config["data_dir"])
    selected = config.get("selected_classes")

    samples = discover_samples(data_dir, selected)
    labels = [label for _, label in samples]
    counts = Counter(labels)

    print(f"Total images: {len(samples)}")
    print(f"Classes: {len(counts)}")
    print("\nClass distribution:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        info = friendly_label(name)
        print(f"  {info['display_name']}: {count}")

    logs_dir = Path(config["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    names = [friendly_label(k)["display_name"][:25] for k in counts.keys()]
    sns.barplot(x=names, y=list(counts.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Class Distribution")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(logs_dir / "class_distribution.png", dpi=150)
    print(f"\nSaved chart to {logs_dir / 'class_distribution.png'}")


if __name__ == "__main__":
    main()
