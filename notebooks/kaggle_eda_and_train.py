import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper__bell___healthy",
    "Pepper__bell___Bacterial_spot",
    "Apple___Apple_scab",
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
SEED = 42
FREEZE_BACKBONE = True
UNFREEZE_EPOCH = 5

DATA_DIR = Path("/kaggle/input/plantvillage-dataset/color")
if not DATA_DIR.exists():
    for p in Path("/kaggle/input").rglob("color"):
        if p.is_dir() and any(p.iterdir()):
            DATA_DIR = p
            break
print("DATA_DIR:", DATA_DIR)

EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def discover(data_dir, selected):
    samples = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name not in selected:
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in EXT:
                samples.append((str(f), d.name))
    return samples

samples = discover(DATA_DIR, SELECTED_CLASSES)
print("Total images:", len(samples))
labels = [l for _, l in samples]
for name, cnt in Counter(labels).most_common():
    print(f"  {name}: {cnt}")

plt.figure(figsize=(12, 5))
sns.barplot(x=list(Counter(labels).keys()), y=list(Counter(labels).values())
plt.xticks(rotation=45, ha="right")
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()

class_to_idx = {c: i for i, c in enumerate(sorted(SELECTED_CLASSES))}
idx_to_class = {v: k for k, v in class_to_idx.items()}

def get_tfms(train=True):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), norm,
    ])

class LeafDS(Dataset):
    def __init__(self, data, tfm):
        self.data, self.tfm = data, tfm
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        path, cls = self.data[i]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), class_to_idx[cls]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

y = [l for _, l in samples]
train_s, temp_s = train_test_split(samples, test_size=0.30, stratify=y, random_state=SEED)
temp_y = [l for _, l in temp_s]
val_s, test_s = train_test_split(temp_s, test_size=0.50, stratify=temp_y, random_state=SEED)
print(f"Train {len(train_s)} | Val {len(val_s)} | Test {len(test_s)}")

train_loader = DataLoader(LeafDS(train_s, get_tfms(True)), BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(LeafDS(val_s, get_tfms(False)), BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(LeafDS(test_s, get_tfms(False)), BATCH_SIZE, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(SELECTED_CLASSES))

if FREEZE_BACKBONE:
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("fc.")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    loss_sum, preds, tgts = 0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            tgts += y.cpu().tolist()
    n = len(loader.dataset)
    f1 = f1_score(tgts, preds, average="macro")
    acc = sum(p == t for p, t in zip(preds, tgts)) / n
    return loss_sum / n, acc, f1, preds, tgts

best_f1, history = 0, {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    if UNFREEZE_EPOCH and epoch == UNFREEZE_EPOCH:
        print("Unfreezing backbone")
        for p in model.parameters(): p.requires_grad = True
        optimizer = Adam(model.parameters(), lr=LR * 0.1)

    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, True)
    va_loss, va_acc, va_f1, _, _ = run_epoch(val_loader, False)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(va_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(va_acc)
    print(f"Ep {epoch:02d} | tr loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f} f1 {va_f1:.3f}")

    if va_f1 > best_f1:
        best_f1 = va_f1
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": "resnet18",
            "num_classes": len(SELECTED_CLASSES),
            "image_size": IMAGE_SIZE,
            "class_to_idx": class_to_idx,
            "best_val_f1": best_f1,
        }, "plant_disease_resnet18.pt")
        print("  ✓ saved best model")

ckpt = torch.load("plant_disease_resnet18.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
_, te_acc, te_f1, te_preds, te_tgts = run_epoch(test_loader, False)
names = [idx_to_class[i] for i in range(len(idx_to_class))]
print(f"\nTEST accuracy={te_acc:.4f}  macro_f1={te_f1:.4f}")
print(classification_report(te_tgts, te_preds, target_names=names))

cm = confusion_matrix(te_tgts, te_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, xticklabels=[n[:18] for n in names], yticklabels=[n[:18] for n in names], cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

class_map = {
    "class_to_idx": class_to_idx,
    "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
}
with open("class_map.json", "w") as f:
    json.dump(class_map, f, indent=2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val"); plt.legend(); plt.title("Loss")
plt.subplot(1, 2, 2); plt.plot(history["train_acc"], label="train"); plt.plot(history["val_acc"], label="val"); plt.legend(); plt.title("Acc")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

print("\nDownload from Output panel:")
print("  - plant_disease_resnet18.pt  → put in models/")
print("  - class_map.json             → put in models/")
print("  - *.png                      → put in logs/ for README")
