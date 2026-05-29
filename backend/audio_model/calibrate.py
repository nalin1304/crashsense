"""
Post-hoc probability calibration for the CrashSense Audio_Detector.

Implements R5.1, R5.3, R5.4, R5.5, R5.6 of the crashsense-hardening spec.

Two methods are supported:

* ``temperature_scaling`` — single scalar T fitted by minimizing NLL on a
  validation set (Guo et al., 2017). Transform applies ``softmax(logits / T)``.
* ``isotonic_regression`` — piecewise-linear monotonic mapping fitted with
  ``sklearn.isotonic.IsotonicRegression`` on per-class softmax confidences.

Public surface:

* ``fit(method, validation_logits, validation_labels) -> Calibrator``
* ``Calibrator.transform(logits: np.ndarray) -> np.ndarray``  (calibrated probs)
* ``Calibrator.to_dict() / from_dict(d)``                      (round-trippable)
* ``save(calibrator, path)`` / ``load(path)``                  (JSON I/O)

Files are persisted as plain-text JSON keyed by ``method`` so a checkpoint can
be diffed and reviewed in PRs. Isotonic breakpoints are sorted by ``x`` on
serialization.

CLI (preserved for backward compatibility):

    python -m backend.audio_model.calibrate --method temperature_scaling \\
        --checkpoint backend/audio_model/crash_detector.pth \\
        --out backend/audio_model/checkpoints/resnet18_calibrator_<sha>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import numpy as np

LOG = logging.getLogger("calibrate")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECTROGRAM_ROOT = REPO_ROOT / "data" / "spectrograms"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "crash_detector.pth"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

CalibrationMethod = Literal["temperature_scaling", "isotonic_regression"]


# ---------------------------------------------------------------------------
# Calibrator interface
# ---------------------------------------------------------------------------


class Calibrator(ABC):
    """Abstract base for post-hoc calibrators.

    Concrete subclasses implement ``transform`` (logits -> probs) and the
    ``to_dict`` / ``from_dict`` round trip.
    """

    method: str = ""

    @abstractmethod
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Map raw logits (..., n_classes) to calibrated probabilities."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Plain-Python representation suitable for JSON serialization."""

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        """Reconstruct a calibrator from its serialized representation."""
        method = d.get("method")
        if method == "temperature_scaling":
            return TemperatureScalingCalibrator.from_dict(d)
        if method == "isotonic_regression":
            return IsotonicRegressionCalibrator.from_dict(d)
        raise ValueError(f"unknown calibration method: {method!r}")


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    np.exp(z, out=z)
    s = z.sum(axis=axis, keepdims=True)
    return z / s


class TemperatureScalingCalibrator(Calibrator):
    """Single-parameter calibration: ``probs = softmax(logits / T)``.

    T > 1 sharpens overconfident models toward the prior; T < 1 flattens
    underconfident models. Fit by minimizing NLL on a validation set.
    """

    method = "temperature_scaling"

    def __init__(self, temperature: float):
        T = float(temperature)
        if not np.isfinite(T):
            raise ValueError(f"temperature must be finite, got {temperature!r}")
        if T <= 0.0:
            raise ValueError(f"temperature must be > 0, got {T}")
        self.temperature = T

    def transform(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64)
        T = max(self.temperature, 1e-6)
        return _softmax(x / T, axis=-1)

    def to_dict(self) -> dict:
        return {"method": self.method, "temperature": float(self.temperature)}

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureScalingCalibrator":
        return cls(temperature=float(d["temperature"]))


class IsotonicRegressionCalibrator(Calibrator):
    """Piecewise-linear monotonic calibration on per-class confidences.

    The serialized form stores ``breakpoints`` as a list of
    ``{"x": float, "y": float}`` pairs sorted by ``x`` ascending. ``transform``
    first softmaxes the logits, then applies ``np.interp`` against the
    breakpoints; in the binary case the second class is set to
    ``1 - calibrated_first_class`` so the output remains a probability vector.
    """

    method = "isotonic_regression"

    def __init__(
        self,
        breakpoints_x: np.ndarray,
        breakpoints_y: np.ndarray,
        n_classes: int = 2,
    ):
        xs = np.asarray(breakpoints_x, dtype=np.float64).ravel()
        ys = np.asarray(breakpoints_y, dtype=np.float64).ravel()
        if xs.shape != ys.shape:
            raise ValueError(
                f"breakpoints_x and breakpoints_y must have identical shape, "
                f"got {xs.shape} vs {ys.shape}"
            )
        if xs.size < 2:
            raise ValueError("isotonic regression requires at least 2 breakpoints")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        # Sort by x ascending; this is also enforced on serialization.
        order = np.argsort(xs, kind="stable")
        self.breakpoints_x = xs[order]
        self.breakpoints_y = ys[order]
        self.n_classes = int(n_classes)

    def _apply_per_class(self, p: np.ndarray) -> np.ndarray:
        """Map a per-class probability through the monotone interpolator."""
        return np.clip(
            np.interp(p, self.breakpoints_x, self.breakpoints_y),
            0.0,
            1.0,
        )

    def transform(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64)
        probs = _softmax(x, axis=-1)
        if self.n_classes == 2:
            calibrated_class0 = self._apply_per_class(probs[..., 0])
            out = np.empty_like(probs)
            out[..., 0] = calibrated_class0
            out[..., 1] = 1.0 - calibrated_class0
            return out
        # General multi-class: apply per-class then renormalize so each row sums to 1.
        cal = np.empty_like(probs)
        for k in range(probs.shape[-1]):
            cal[..., k] = self._apply_per_class(probs[..., k])
        s = cal.sum(axis=-1, keepdims=True)
        s = np.where(s == 0.0, 1.0, s)
        return cal / s

    def to_dict(self) -> dict:
        # Sorted breakpoints (already sorted in __init__, sort again on serialization
        # for defensive determinism).
        bps = sorted(
            (
                {"x": float(x), "y": float(y)}
                for x, y in zip(self.breakpoints_x, self.breakpoints_y)
            ),
            key=lambda b: b["x"],
        )
        return {
            "method": self.method,
            "n_classes": int(self.n_classes),
            "breakpoints": bps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicRegressionCalibrator":
        bps = d["breakpoints"]
        if not bps:
            raise ValueError("isotonic calibrator missing breakpoints")
        xs = np.array([float(bp["x"]) for bp in bps], dtype=np.float64)
        ys = np.array([float(bp["y"]) for bp in bps], dtype=np.float64)
        n_classes = int(d.get("n_classes", 2))
        return cls(xs, ys, n_classes=n_classes)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit(
    method: CalibrationMethod,
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
) -> Calibrator:
    """Fit a calibrator on a held-out validation set.

    Parameters
    ----------
    method
        ``"temperature_scaling"`` or ``"isotonic_regression"``.
    validation_logits
        Float array of shape ``(N, n_classes)``.
    validation_labels
        Integer array of shape ``(N,)`` with class indices in ``[0, n_classes)``.
    """
    logits = np.asarray(validation_logits, dtype=np.float64)
    labels = np.asarray(validation_labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"validation_logits must be 2D, got shape {logits.shape}")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"validation_labels must be 1D with length {logits.shape[0]}, "
            f"got shape {labels.shape}"
        )
    n_classes = int(logits.shape[1])
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= n_classes:
        raise ValueError(
            f"validation_labels must be in [0, {n_classes}), got "
            f"min={labels.min()} max={labels.max()}"
        )

    if method == "temperature_scaling":
        return _fit_temperature_scaling(logits, labels)
    if method == "isotonic_regression":
        return _fit_isotonic_regression(logits, labels, n_classes)
    raise ValueError(f"unknown calibration method: {method!r}")


def _fit_temperature_scaling(
    logits: np.ndarray, labels: np.ndarray
) -> TemperatureScalingCalibrator:
    """Minimize per-sample mean NLL over T using bounded scalar optimization."""
    from scipy.optimize import minimize_scalar

    n = labels.shape[0]
    rows = np.arange(n)

    def nll(T: float) -> float:
        T_clamped = max(float(T), 1e-6)
        z = logits / T_clamped
        z = z - z.max(axis=-1, keepdims=True)
        log_partition = np.log(np.exp(z).sum(axis=-1))
        log_probs_correct = z[rows, labels] - log_partition
        return float(-log_probs_correct.mean())

    res = minimize_scalar(
        nll,
        bounds=(0.05, 10.0),
        method="bounded",
        options={"xatol": 1e-4},
    )
    T_opt = float(res.x)
    LOG.info("temperature_scaling fit: T=%.4f, NLL=%.4f", T_opt, float(res.fun))
    return TemperatureScalingCalibrator(temperature=T_opt)


def _fit_isotonic_regression(
    logits: np.ndarray, labels: np.ndarray, n_classes: int
) -> IsotonicRegressionCalibrator:
    """Fit isotonic regression on the maximum-class confidence."""
    from sklearn.isotonic import IsotonicRegression

    probs = _softmax(logits.copy(), axis=-1)

    if n_classes == 2:
        # Binary: calibrate class-0 probability against the indicator of class-0.
        x_in = probs[..., 0]
        y_target = (labels == 0).astype(np.float64)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(x_in, y_target)
        xs = np.asarray(ir.X_thresholds_, dtype=np.float64)
        ys = np.asarray(ir.y_thresholds_, dtype=np.float64)
    else:
        # Multi-class: pool one-vs-rest fits into a single monotone curve over the
        # max-class confidence. This is the standard "max-prob" calibration form.
        confidences = probs.max(axis=-1)
        predictions = probs.argmax(axis=-1)
        correct = (predictions == labels).astype(np.float64)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(confidences, correct)
        xs = np.asarray(ir.X_thresholds_, dtype=np.float64)
        ys = np.asarray(ir.y_thresholds_, dtype=np.float64)

    LOG.info(
        "isotonic_regression fit: %d breakpoints, x=[%.4f, %.4f]",
        xs.size,
        float(xs.min()),
        float(xs.max()),
    )
    return IsotonicRegressionCalibrator(xs, ys, n_classes=n_classes)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save(calibrator: Calibrator, path: Path | str) -> None:
    """Serialize a fitted calibrator to plain-text JSON.

    Isotonic breakpoints are sorted by ``x`` on disk so a checkpoint diff is
    semantic, not order-dependent.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = calibrator.to_dict()
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load(path: Path | str) -> Optional[Calibrator]:
    """Load a fitted calibrator from JSON.

    Returns ``None`` and logs a WARNING if the file is missing or cannot be
    parsed; the caller is expected to fall back to uncalibrated softmax in
    that case (see Audio_Detector integration in ``inference.py``).
    """
    p = Path(path)
    if not p.exists():
        LOG.warning(
            "calibrator file missing at %s; falling back to uncalibrated softmax",
            p,
        )
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning(
            "calibrator file at %s is corrupt (%s); falling back to uncalibrated softmax",
            p,
            exc,
        )
        return None
    try:
        return Calibrator.from_dict(d)
    except (KeyError, ValueError, TypeError) as exc:
        LOG.warning(
            "calibrator file at %s is corrupt (%s); falling back to uncalibrated softmax",
            p,
            exc,
        )
        return None


def find_active_calibrator(
    arch: str | None = None,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
) -> Path | None:
    """Locate the most recent calibrator JSON for an architecture.

    Filenames follow ``<arch>_calibrator_<git_sha>.json``. When ``arch`` is
    given, only matching files are considered; otherwise any
    ``*calibrator*.json`` is eligible. Returns the newest file by mtime, or
    ``None`` when none match.
    """
    if not checkpoints_dir.is_dir():
        return None
    if arch:
        candidates = list(checkpoints_dir.glob(f"{arch}_calibrator_*.json"))
    else:
        candidates = list(checkpoints_dir.glob("*calibrator*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# CLI (backward-compatible)
# ---------------------------------------------------------------------------


def _gather_validation_logits(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Run the trained model over the val split and return (logits, labels)."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from .data_split import load_split
    from .model import build_model

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    backbone = payload.get("backbone", "resnet18")
    model = build_model(backbone)
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    _train_idx, val_idx, _test_idx, _samples, _classes = load_split()
    base = ImageFolder(str(SPECTROGRAM_ROOT))

    class _ValDataset(torch.utils.data.Dataset):
        def __init__(self, indices: list[int]):
            self.indices = indices
            self.tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ])

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int):
            from PIL import Image
            path, label = base.samples[self.indices[idx]]
            with Image.open(path) as pil:
                pil = pil.convert("RGB")
                return self.tfm(pil), label

    loader = DataLoader(_ValDataset(val_idx), batch_size=64, shuffle=False, num_workers=0)
    logits_chunks, labels_chunks = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits_chunks.append(model(x).cpu().numpy())
            labels_chunks.append(y.numpy())
    return (
        np.concatenate(logits_chunks).astype(np.float64),
        np.concatenate(labels_chunks).astype(np.int64),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Fit a post-hoc calibrator")
    parser.add_argument(
        "--method",
        choices=["temperature_scaling", "isotonic_regression"],
        default="temperature_scaling",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <checkpoints>/<arch>_calibrator_<sha>.json)",
    )
    parser.add_argument("--git-sha", type=str, default="local")
    args = parser.parse_args(argv)

    logits, labels = _gather_validation_logits(args.checkpoint)
    LOG.info("gathered %d validation samples for calibration", len(labels))

    cal = fit(args.method, logits, labels)

    out_path = args.out
    if out_path is None:
        # Best-effort default naming based on checkpoint backbone.
        try:
            import torch
            payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
            arch = payload.get("backbone", "resnet18")
        except Exception:
            arch = "resnet18"
        out_path = CHECKPOINTS_DIR / f"{arch}_calibrator_{args.git_sha}.json"

    save(cal, out_path)
    print(f"wrote calibrator to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
