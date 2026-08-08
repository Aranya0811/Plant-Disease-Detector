"""Pydantic schemas for API request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    plant: str
    condition: str
    display_name: str = Field(alias="display_name")
    confidence: float

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction: str
    plant: str
    condition: str
    class_name: str
    confidence: float
    is_confident: bool
    top3: list[dict]
    message: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int | None = None
    device: str


class ClassListResponse(BaseModel):
    classes: list[dict]
