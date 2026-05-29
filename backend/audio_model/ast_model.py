"""
Audio Spectrogram Transformer (AST) backbone for CrashSense.

We start from MIT/ast-finetuned-audioset-10-10-0.4593 — an 86M-parameter
ViT-style model pretrained on the full AudioSet (527 labels). The original
classification head is dropped and replaced with a 2-class head for
{crash, noise}.

Why AST instead of ResNet-18:
  - Pretrained on 2M+ real-world audio clips covering exactly the kinds of
    sounds we care about (vehicle, glass, impact, siren, etc.)
  - Operates on log-mel spectrograms internally (16 kHz, 128 mel bands,
    10 ms hop, 25 ms window, 1024 frames = 10 s context)
  - Closes the domain gap that ResNet-18 + ImageNet pretraining cannot

Public API:
  load_ast_model(num_classes=2) -> (model, feature_extractor, sample_rate)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000


def load_ast_model(num_classes: int = 2) -> Tuple[nn.Module, "AutoFeatureExtractor", int]:
    """Load AST and replace the classifier head with a fresh `num_classes` Linear."""
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    feat = AutoFeatureExtractor.from_pretrained(AST_MODEL_ID)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL_ID)

    # The AST classifier is a Sequential(LayerNorm, Linear); we replace just
    # the Linear so the LayerNorm stays around. Safer than rebuilding the
    # whole sequential.
    in_features = model.classifier.dense.in_features
    model.classifier.dense = nn.Linear(in_features, num_classes)

    # Pretrained head no longer matches `num_classes`; force-update config so
    # transformers stops complaining about size mismatches at load.
    model.config.num_labels = num_classes
    model.config.id2label = {i: f"class_{i}" for i in range(num_classes)}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    model.num_labels = num_classes

    return model, feat, AST_SAMPLE_RATE


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze every AST layer except the classification head."""
    for name, p in model.named_parameters():
        if "classifier" in name:
            p.requires_grad = True
        else:
            p.requires_grad = not freeze
