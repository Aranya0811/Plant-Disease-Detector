"""PyTorch model definitions."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}. Use resnet18 or mobilenet_v2.")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if name.startswith("fc.") or name.startswith("classifier."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable
