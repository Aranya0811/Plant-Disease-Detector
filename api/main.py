"""
Plant Disease Detection API

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Swagger docs: http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import predict
from src.config import load_config
from src.inference import PlantDiseasePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    model_path = Path(config["model_save_path"])
    class_map_path = Path(config["class_map_path"])

    if model_path.exists() and class_map_path.exists():
        app.state.predictor = PlantDiseasePredictor(
            model_path=model_path,
            class_map_path=class_map_path,
            confidence_threshold=config.get("confidence_threshold", 0.5),
            apply_clahe=config.get("apply_clahe", True),
            apply_leaf_mask=config.get("apply_leaf_mask", False),
        )
        print(f"Model loaded from {model_path}")
    else:
        app.state.predictor = None
        print("WARNING: Model not found. Run: python -m src.train")

    yield
    app.state.predictor = None


app = FastAPI(
    title="Plant Disease Detection API",
    description=(
        "Deep Learning (ResNet18) + OpenCV pipeline for classifying plant leaf diseases. "
        "Built for ML Wing selection — Core ML track."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.mount("/static", StaticFiles(directory="api/static"), name="static")


@app.get("/")
async def root():
    return FileResponse("api/static/index.html")
