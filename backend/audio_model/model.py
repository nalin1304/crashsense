"""
Model definition for the CrashSense Audio_Detector (Requirement 3).

ResNet-18 with a 2-class head, initialized from ImageNet pretrained weights.
A `timm` EfficientNet-B0 backbone is provided as the fallback per criterion 13.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2


def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = NUM_CLASSES) -> nn.Module:
    import timm  # imported lazily so torchvision-only installs still work
    return timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=num_classes,
    )


def build_model(backbone: str = "resnet18", num_classes: int = NUM_CLASSES) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        return build_resnet18(num_classes)
    if backbone in ("effnet", "efficientnet", "efficientnet_b0"):
        return build_efficientnet_b0(num_classes)
    raise ValueError(f"Unknown backbone: {backbone}")
