"""Prediction routes."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.schemas import ClassListResponse, HealthResponse, PredictionResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        num_classes=predictor.num_classes if predictor else None,
        device=str(predictor.device) if predictor else "unknown",
    )


@router.get("/classes", response_model=ClassListResponse)
async def list_classes(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    classes = []
    for idx in range(predictor.num_classes):
        class_name = predictor.idx_to_class[idx]
        info = predictor.class_map["friendly_labels"][class_name]
        classes.append(
            {
                "index": idx,
                "class_name": class_name,
                **info,
            }
        )
    return ClassListResponse(classes=classes)


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first: python -m src.train",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a JPG or PNG image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = predictor.predict_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    message = None
    if not result["is_confident"]:
        message = (
            f"Low confidence ({result['confidence']:.0%}). "
            "Try a clearer leaf photo with plain background."
        )

    return PredictionResponse(**result, message=message)


@router.post("/predict/annotated")
async def predict_annotated(request: Request, file: UploadFile = File(...)):
    """Return JSON prediction + base64 annotated image (OpenCV overlay)."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    image_bytes = await file.read()
    result = predictor.predict_from_bytes(image_bytes)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    annotated = predictor.annotate_image(image_bgr, result)

    _, buffer = cv2.imencode(".jpg", annotated)
    encoded = base64.b64encode(buffer).decode("utf-8")

    return {
        **result,
        "annotated_image_base64": encoded,
        "message": "Prediction drawn on image using OpenCV",
    }
