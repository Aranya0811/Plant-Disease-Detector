"""Model loading and prediction with OpenCV preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.dataset import friendly_label, load_class_map
from src.data.preprocessing import bgr_to_rgb, preprocess_image
from src.models.classifier import build_model


class PlantDiseasePredictor:
    def __init__(
        self,
        model_path: str | Path,
        class_map_path: str | Path,
        confidence_threshold: float = 0.5,
        apply_clahe: bool = True,
        apply_leaf_mask: bool = False,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.confidence_threshold = confidence_threshold
        self.apply_clahe = apply_clahe
        self.apply_leaf_mask = apply_leaf_mask

        checkpoint = torch.load(model_path, map_location=self.device)
        self.image_size = checkpoint["image_size"]
        self.model_name = checkpoint["model_name"]
        self.num_classes = checkpoint["num_classes"]
        self.class_map = load_class_map(class_map_path)

        self.idx_to_class = {int(k): v for k, v in self.class_map["idx_to_class"].items()}

        self.model = build_model(self.model_name, self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _prepare_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        processed = preprocess_image(
            image_bgr,
            image_size=self.image_size,
            use_clahe=self.apply_clahe,
            use_leaf_mask=self.apply_leaf_mask,
        )
        rgb = bgr_to_rgb(processed)
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_from_bgr(self, image_bgr: np.ndarray) -> dict:
        tensor = self._prepare_tensor(image_bgr)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(probs.argmax())
        top_conf = float(probs[top_idx])
        class_name = self.idx_to_class[top_idx]
        label_info = friendly_label(class_name)

        top3_indices = probs.argsort()[-3:][::-1]
        top3 = [
            {
                **friendly_label(self.idx_to_class[int(i)]),
                "confidence": float(probs[i]),
            }
            for i in top3_indices
        ]

        return {
            "prediction": label_info["display_name"],
            "plant": label_info["plant"],
            "condition": label_info["condition"],
            "class_name": class_name,
            "confidence": top_conf,
            "is_confident": top_conf >= self.confidence_threshold,
            "top3": top3,
        }

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image. Upload a valid JPG/PNG file.")
        return self.predict_from_bgr(image_bgr)

    def annotate_image(self, image_bgr: np.ndarray, result: dict) -> np.ndarray:
        """Draw prediction on image for API response visualization."""
        annotated = image_bgr.copy()
        text = f"{result['prediction']} ({result['confidence']:.0%})"
        color = (0, 180, 0) if result["is_confident"] else (0, 140, 255)

        cv2.rectangle(annotated, (10, 10), (min(len(text) * 12 + 20, annotated.shape[1] - 10), 45), color, -1)
        cv2.putText(
            annotated,
            text[:70],
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated
