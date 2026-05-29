"""
Test-time augmentation for AST inference.

Runs inference on the original clip plus N perturbed versions (pitch shift,
time stretch, additive noise) and averages the softmax probabilities.

Cost: each TTA variant adds one full AST forward pass (~0.3 s on CPU per
clip). For latency-sensitive paths use 2-3 variants. For batch evaluation
or offline labelling use 5+.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

LOG = logging.getLogger("tta")


def _augmentations(samples: np.ndarray, sr: int) -> list[np.ndarray]:
    """Return [original, pitch+1, pitch-1, stretch×0.95, stretch×1.05]."""
    import librosa
    out = [samples]
    try:
        out.append(librosa.effects.pitch_shift(samples, sr=sr, n_steps=1))
        out.append(librosa.effects.pitch_shift(samples, sr=sr, n_steps=-1))
        out.append(librosa.effects.time_stretch(samples, rate=0.95))
        out.append(librosa.effects.time_stretch(samples, rate=1.05))
    except Exception as exc:
        LOG.warning("TTA augmentation failed (%s); falling back to original only", exc)
    # All variants must match the same target length the model expects;
    # the caller is responsible for clipping/padding.
    return out


def predict_with_tta(predict_one, samples: np.ndarray, sr: int, *,
                     n_variants: int = 5) -> dict:
    """Run `predict_one` (a function returning {"event","confidence"}) against
    several augmented variants and average the probabilities."""
    variants = _augmentations(samples, sr)[:max(1, n_variants)]
    crash_probs: list[float] = []
    for v in variants:
        r = predict_one(v)
        # If the model returned NORMAL with confidence c, crash prob is 1-c.
        crash_p = float(r["confidence"]) if r["event"] == "CRASH" else 1.0 - float(r["confidence"])
        crash_probs.append(crash_p)
    avg = float(np.mean(crash_probs))
    if avg >= 0.5:
        return {"event": "CRASH", "confidence": avg, "tta_variants": len(variants)}
    return {"event": "NORMAL", "confidence": 1.0 - avg, "tta_variants": len(variants)}
