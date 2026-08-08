# 4-Day Execution Checklist

Use this as your hour-by-hour guide. Check boxes as you go.

---

## Day 1 — Data + Understand the Codebase

- [ ] Create Kaggle account
- [ ] Download PlantVillage dataset
- [ ] Extract to `data/PlantVillage/color/`
- [ ] Create venv + `pip install -r requirements.txt`
- [ ] Run `python scripts/eda.py` — verify ~10 classes, ~5000+ images
- [ ] Read `src/data/preprocessing.py` — understand CLAHE
- [ ] Read `src/models/classifier.py` — understand ResNet18 head replacement
- [ ] Start API without model: `uvicorn api.main:app --reload`
- [ ] Open http://localhost:8000/docs

**Learn today:** What is transfer learning? What is a CNN? (30 min YouTube: "PyTorch transfer learning tutorial")

---

## Day 2 — Train the Model

- [ ] Set `num_workers: 0` in config.yaml if on Windows
- [ ] Run `python -m src.train`
- [ ] Watch training output — loss should decrease
- [ ] Open `logs/training_curves.png`
- [ ] Open `logs/confusion_matrix.png`
- [ ] Read `logs/test_metrics.json` — note accuracy and macro F1
- [ ] If val F1 < 0.85: train longer (`--epochs 15`) or unfreeze earlier

**Learn today:** CrossEntropyLoss, Adam optimizer, train/val split, F1 score

**Target metrics (10 classes):** Accuracy > 90%, Macro F1 > 0.88

---

## Day 3 — API Integration + Documentation

- [ ] Restart API (model auto-loads on startup)
- [ ] Test `GET /api/v1/health`
- [ ] Test `GET /api/v1/classes`
- [ ] Test `POST /api/v1/predict` with 5 different leaf images
- [ ] Test `POST /api/v1/predict/annotated`
- [ ] Screenshot Swagger + sample predictions for README/submission
- [ ] Customize README with YOUR actual test metrics
- [ ] Add architecture diagram screenshot if needed

**Learn today:** FastAPI lifecycle, model loading at startup, REST design

---

## Day 4 — Deploy + Interview Prep

- [ ] Push to GitHub
- [ ] Deploy on Render / Railway / HF Spaces
- [ ] Test live URL with curl
- [ ] Add deployed URL to README
- [ ] Practice explaining:
  - [ ] Full pipeline in 2 minutes
  - [ ] Why ResNet18
  - [ ] Why OpenCV CLAHE
  - [ ] One failure case (blurry image, non-leaf object)
  - [ ] One improvement you'd make

---

## Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| `Dataset not found` | Check `data/PlantVillage/color/` path |
| DataLoader freeze (Windows) | `num_workers: 0` |
| CUDA out of memory | `batch_size: 16` or `8` |
| Low accuracy | Train more epochs, set `unfreeze_epoch: 3` |
| API 503 Model not loaded | Run `python -m src.train` first |

---

## Submission Package

Zip or GitHub repo with:

1. All source code
2. `models/plant_disease_resnet18.pt`
3. `models/class_map.json`
4. `requirements.txt`
5. README with live API URL
6. Training logs / screenshots
