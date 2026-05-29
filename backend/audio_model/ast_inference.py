"""
Inference using AST as a frozen feature extractor + the trained linear head.

Uses crash_detector_ast_head.pth (saved by ast_head.py). Falls back to
ResNet inference if the AST head checkpoint is missing.

Activated by setting CRASHSENSE_USE_AST=1 in the environment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .ast_model import AST_SAMPLE_RATE, load_ast_model

LOG = logging.getLogger("ast_inference")

AST_HEAD_PATH = Path(__file__).resolve().parent / "crash_detector_ast_head.pth"
SAMPLE_RATE = AST_SAMPLE_RATE


class _LinearHead(nn.Module):
    """Mirror of backend.audio_model.ast_head._LinearHead so we can load weights."""

    def __init__(self, in_features: int, num_classes: int = 2,
                 hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ASTPredictor:
    _instance: "_ASTPredictor | None" = None

    def __init__(self) -> None:
        self.backbone = None
        self.feat = None
        self.head = None
        self.temperature = 1.0
        self.ready = False
        self._load()

    @classmethod
    def instance(cls) -> "_ASTPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if not AST_HEAD_PATH.exists():
            LOG.info("AST head checkpoint missing at %s; AST inference disabled", AST_HEAD_PATH)
            return
        try:
            full_model, feat, _ = load_ast_model(num_classes=2)
            self.backbone = full_model.audio_spectrogram_transformer
            self.backbone.eval()
            self.feat = feat
            payload = torch.load(str(AST_HEAD_PATH), map_location="cpu", weights_only=False)
            self.head = _LinearHead(
                payload["in_features"], num_classes=2,
                hidden=int(payload.get("hidden", 0)),
            )
            self.head.load_state_dict(payload["state_dict"])
            self.head.eval()
            self.temperature = float(payload.get("temperature", 1.0))
            self.ready = True
            LOG.info("AST inference loaded (T=%.3f)", self.temperature)
        except Exception as exc:
            LOG.error("could not load AST inference: %s", exc)

    @torch.no_grad()
    def predict(self, samples: np.ndarray) -> dict:
        if not self.ready:
            raise RuntimeError("AST predictor not ready; check CRASHSENSE_USE_AST and AST head")
        if samples.shape[0] < SAMPLE_RATE * 1:
            samples = np.pad(samples, (0, SAMPLE_RATE - samples.shape[0]), mode="constant")
        if samples.shape[0] > SAMPLE_RATE * 10:
            samples = samples[: SAMPLE_RATE * 10]
        feats = self.feat(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        outputs = self.backbone(input_values=feats["input_values"])
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.head(pooled).squeeze(0).numpy()
        z = logits / max(self.temperature, 1e-6)
        z = z - z.max()
        probs = np.exp(z) / np.exp(z).sum()
        crash_p = float(probs[0])
        if crash_p >= 0.5:
            return {"event": "CRASH", "confidence": crash_p}
        return {"event": "NORMAL", "confidence": float(probs[1])}


def is_available() -> bool:
    """True if the AST head checkpoint is present and CRASHSENSE_USE_AST=1."""
    return os.environ.get("CRASHSENSE_USE_AST") == "1" and AST_HEAD_PATH.exists()
