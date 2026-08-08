# Interview Preparation — Plant Disease Project

## Elevator Pitch (memorize this)

> I built an end-to-end plant leaf disease detection system. Farmers upload a leaf photo to a FastAPI backend. OpenCV normalizes lighting with CLAHE and resizes the image. A fine-tuned ResNet18 CNN — trained on the PlantVillage dataset — classifies the disease. The API returns the plant type, disease name, confidence score, and top-3 predictions. I used transfer learning because our dataset is medium-sized, stratified splits to handle class balance, and macro F1 instead of raw accuracy because some diseases are rarer than others.

---

## Architecture Questions

### Q: Walk me through your pipeline.

1. **Data:** PlantVillage, 10 classes, folder-per-class labels
2. **EDA:** Class distribution analysis, stratified split 70/15/15
3. **Preprocessing (train):** Random crop, flip, color jitter + ImageNet normalize
4. **Preprocessing (inference):** OpenCV CLAHE + resize
5. **Model:** ResNet18 pretrained on ImageNet, replaced final FC layer for 10 classes
6. **Training:** Freeze backbone → train head → unfreeze all → fine-tune
7. **Evaluation:** Macro F1, confusion matrix, per-class precision/recall
8. **Deploy:** FastAPI loads model once at startup, `/predict` endpoint

### Q: Why CNN and not a regular neural network (MLP)?

Images have **2D spatial structure**. Neighboring pixels form patterns (spots, edges, textures). CNNs use convolution filters that slide across the image and learn hierarchical features: edges → textures → disease patterns. An MLP would flatten the image and lose spatial relationships.

### Q: What is transfer learning?

We start with ResNet18 **already trained on ImageNet** (1.2M general photos). Lower layers know generic features (edges, colors). We replace the last layer for our 10 classes and fine-tune on leaf images. This needs less data and trains faster than from scratch.

### Q: Why freeze the backbone first?

Early training updates only the classifier head, preventing large gradients from destroying useful pretrained weights. After the head converges, we unfreeze the full network for fine-tuning at a lower learning rate.

---

## OpenCV Questions

### Q: What does OpenCV do in your project?

At **inference time:**
1. Decode uploaded image bytes
2. Apply **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the L channel in LAB color space — fixes shadows and overexposure
3. Resize to 224×224
4. Draw prediction text on annotated output

At **train time:** torchvision handles augmentation; OpenCV is optional for inference consistency.

### Q: What is CLAHE and why not global histogram equalization?

CLAHE improves local contrast in small tiles instead of the whole image.global equalization can over-amplify noise. Field leaf photos often have one side in shadow — CLAHE helps.

### Q: Why not use OpenCV for the whole classification?

Hand-crafted rules cannot capture the variety of disease appearances (different stages, angles, backgrounds). CNNs learn these patterns from data. OpenCV is reliable for **geometry and photometry**; CNNs for **semantics**.

---

## Training & Metrics Questions

### Q: Why macro F1 over accuracy?

Macro F1 averages F1 across all classes equally. If "Tomato healthy" has 1000 images and a rare disease has 100, accuracy can look good while the model fails on rare diseases. Macro F1 exposes that.

### Q: How did you handle overfitting?

- Data augmentation (flip, crop, color jitter)
- Transfer learning (pretrained weights)
- Validation-based checkpointing (save best val F1, not last epoch)
- Two-phase training (frozen then unfrozen)

### Q: What loss function and why?

**CrossEntropyLoss** — standard for multi-class classification. It penalizes wrong class predictions probabilistically. Works with softmax output.

### Q: Train/val/test split strategy?

**Stratified split** — each split has the same proportion of each class. Random split without stratification could leave a class out of test set.

---

## FastAPI Questions

### Q: Why load the model at startup (lifespan) not per request?

Model loading is expensive (~1–2 seconds). Loading once and reusing in memory makes inference fast (~50ms on CPU). Production APIs always warm-load models.

### Q: How does `/predict` work?

1. Client sends multipart form with image file
2. FastAPI reads bytes asynchronously
3. OpenCV decodes → preprocess → tensor
4. Model forward pass → softmax → argmax
5. Return JSON with human-readable labels

---

## Failure Cases (shows maturity)

| Case | What happens | Fix |
|------|--------------|-----|
| Blurry photo | Low confidence | Return `is_confident: false`, ask for retake |
| Non-leaf image (hand, soil) | Wrong prediction | Add OOD detection or threshold |
| New plant not in training | Wrong class | Expand dataset / return "unknown" |
| Class imbalance | Lower recall on rare class | Weighted loss, oversampling |

---

## Improvements (if they ask "what next?")

1. **Grad-CAM** — visual heatmap showing which leaf region drove the prediction
2. **ONNX export** — run on mobile for farmers in the field
3. **More species** — expand beyond 10 classes
4. **RAG layer** — "What treatment for Early blight?" using agricultural PDFs
5. **Active learning** — collect low-confidence images to retrain

---

## Key Numbers to Know (fill after training)

After you run `python -m src.train`, fill these in:

- Dataset size: ___ images, ___ classes
- Test accuracy: ___%
- Test macro F1: ___
- Training time: ___ minutes
- Model size: ___ MB
- Inference time: ___ ms per image

---

## DL Concepts to Study (1 hour each)

| Topic | Resource |
|-------|----------|
| CNN basics | 3Blue1Brown neural networks + CNN chapter |
| Transfer learning | PyTorch official tutorial |
| ResNet / skip connections | Original paper summary blog |
| Softmax + CrossEntropy | StatQuest YouTube |
| Confusion matrix | sklearn docs |

You don't need to derive backprop — but explain **what** each component does and **why you chose it**.
