# Clean Git Commit Guide

Follow this **exact order**. One logical change = one commit.

---

## Before first commit

```powershell
cd C:\Users\ARANYA\Projects\plant-disease-detector
git status   # see untracked files
```

Create GitHub repo: https://github.com/new → name: `plant-disease-detector` → **no** README (you have one).

```powershell
git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
git branch -M main
```

---

## Commit sequence

### Commit 1 — Project scaffold
```powershell
git add .gitignore requirements.txt config.yaml Dockerfile
git add src/config.py src/__init__.py
git add data/.gitkeep models/.gitkeep
git commit -m "chore: initialize project with config and dependencies"
```

### Commit 2 — Data pipeline
```powershell
git add src/data/
git add scripts/eda.py
git commit -m "feat: add dataset loading, stratified splits, and OpenCV preprocessing"
```

### Commit 3 — Model + training
```powershell
git add src/models/
git add src/train.py
git commit -m "feat: add ResNet18 classifier and training script with metrics logging"
```

### Commit 4 — Inference
```powershell
git add src/inference.py
git commit -m "feat: add inference pipeline with OpenCV CLAHE and PyTorch prediction"
```

### Commit 5 — FastAPI
```powershell
git add api/
git commit -m "feat: add FastAPI REST endpoints for health, classes, and predict"
```

### Commit 6 — Kaggle notebook
```powershell
git add notebooks/
git commit -m "docs: add Kaggle training notebook script for GPU training"
```

### Commit 7 — Documentation
```powershell
git add README.md docs/
git commit -m "docs: add README, 4-day plan, and interview preparation guide"
```

### Commit 8 — Trained model (after Kaggle training)
```powershell
git add models/plant_disease_resnet18.pt models/class_map.json
git add logs/training_curves.png logs/confusion_matrix.png logs/test_metrics.json
git commit -m "feat: add trained ResNet18 model and evaluation artifacts"
```

> If `.pt` file > 100MB, use Git LFS:
> ```powershell
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> ```

### Commit 9 — Deployment
```powershell
# after adding render.yaml or updating README with live URL
git add README.md
git commit -m "docs: add deployment URL and production setup instructions"
```

### Push
```powershell
git push -u origin main
```

---

## Commit message rules

| Prefix | Use for |
|--------|---------|
| `feat:` | New feature (API, model, pipeline) |
| `docs:` | README, guides, comments |
| `chore:` | Config, gitignore, deps |
| `fix:` | Bug fixes |

**Never commit:** `venv/`, `data/PlantVillage/`, `__pycache__/`, `.env`

---

## If you already have uncommitted everything

Split with:
```powershell
git add -p   # stage hunks interactively (avoid on Windows if annoying)
```

Or reset and follow commits 1–9 in order.
