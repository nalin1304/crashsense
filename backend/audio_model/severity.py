"""
Severity estimation on top of the binary crash classifier.

Architecture: instead of training a separate severity head (we don't have
labels for it), we derive severity from acoustic signal characteristics
that correlate well with crash energy:

  * **peak SPL** (peak amplitude after high-pass filtering)
  * **total energy** in a 500ms window centered on the loudest sample
  * **spectral centroid** — broadband impacts (metal-on-metal) skew higher
  * **transient sharpness** — how fast the envelope rises, in dB/ms

These four features are mapped via a simple deterministic rubric to a
severity score in 0-1 and a category in {minor, moderate, major}.

Why not a learned head? Without ground truth this is a risky thing to
fake. The deterministic mapping is defensible because it's transparent,
auditable, and tunable — exactly what a safety-critical system needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOG = logging.getLogger("severity")

SeverityLabel = Literal["minor", "moderate", "major"]


@dataclass
class SeverityEstimate:
    severity: float                # 0..1, higher = more severe
    label: SeverityLabel
    peak_dbfs: float
    energy_jfs: float              # joules/full-scale (proxy)
    spectral_centroid_hz: float
    transient_db_per_ms: float

    def as_dict(self) -> dict:
        return {
            "severity": round(self.severity, 4),
            "label": self.label,
            "peak_dbfs": round(self.peak_dbfs, 2),
            "energy_jfs": round(self.energy_jfs, 6),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 1),
            "transient_db_per_ms": round(self.transient_db_per_ms, 4),
        }


def _highpass(samples: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    sos = butter(4, cutoff_hz / (0.5 * sr), btype="highpass", output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32)


def _peak_dbfs(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples)) + 1e-9)
    return 20.0 * float(np.log10(peak))


def _energy_around_peak(samples: np.ndarray, sr: int, window_s: float = 0.5) -> tuple[float, int]:
    if samples.size == 0:
        return 0.0, 0
    peak_idx = int(np.argmax(np.abs(samples)))
    half = int(window_s * sr / 2)
    lo = max(0, peak_idx - half)
    hi = min(samples.size, peak_idx + half)
    seg = samples[lo:hi].astype(np.float64)
    energy = float(np.sum(seg ** 2))
    return energy, peak_idx


def _spectral_centroid(samples: np.ndarray, sr: int) -> float:
    if samples.size < 256:
        return 0.0
    # FFT of the loudest 1-second window
    n = min(samples.size, sr)
    seg = samples[:n].astype(np.float64)
    spectrum = np.fft.rfft(seg)
    mag = np.abs(spectrum) + 1e-9
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    return centroid


def _transient_slope_db_per_ms(samples: np.ndarray, sr: int) -> float:
    """Steepest 10ms rise in the envelope (dB/ms). Captures attack sharpness."""
    if samples.size < int(0.05 * sr):
        return 0.0
    abs_samples = np.abs(samples) + 1e-6
    win = max(1, int(0.005 * sr))  # 5 ms RMS window
    env = np.convolve(abs_samples, np.ones(win) / win, mode="same")
    env_db = 20.0 * np.log10(env)
    # max rise over 10 ms windows
    step = max(1, int(0.001 * sr))
    look = max(1, int(0.010 * sr))
    diffs = env_db[look::step] - env_db[: -look:step]
    if diffs.size == 0:
        return 0.0
    return float(np.max(diffs)) / 10.0  # convert per-10ms to per-ms


def _label_from_score(score: float) -> SeverityLabel:
    if score >= 0.66:
        return "major"
    if score >= 0.33:
        return "moderate"
    return "minor"


def estimate_severity(samples: np.ndarray, sr: int = 22050) -> SeverityEstimate:
    """Compute severity from acoustic features. Pure-numpy, no model.

    The score is a weighted combination of normalized features with weights
    tuned against ESC-50/UrbanSound8K crash samples. Treat it as an
    indicator, not a clinical metric.
    """
    if samples.size == 0:
        return SeverityEstimate(0.0, "minor", -120.0, 0.0, 0.0, 0.0)

    samples = samples.astype(np.float32, copy=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    hp = _highpass(samples, sr)
    peak = _peak_dbfs(hp)
    energy, _ = _energy_around_peak(hp, sr)
    centroid = _spectral_centroid(hp, sr)
    slope = _transient_slope_db_per_ms(hp, sr)

    # Normalize each feature against typical ranges seen in our training set.
    f_peak = np.clip((peak + 30.0) / 30.0, 0.0, 1.0)             # -30 dBFS .. 0 dBFS
    f_energy = np.clip(np.log10(energy * 1000 + 1) / 3.0, 0.0, 1.0)
    f_centroid = np.clip(centroid / 6000.0, 0.0, 1.0)             # 0..6 kHz
    f_slope = np.clip(slope / 5.0, 0.0, 1.0)                      # 0..5 dB/ms

    score = float(
        0.40 * f_peak
        + 0.35 * f_energy
        + 0.10 * f_centroid
        + 0.15 * f_slope
    )
    return SeverityEstimate(
        severity=score,
        label=_label_from_score(score),
        peak_dbfs=peak,
        energy_jfs=energy,
        spectral_centroid_hz=centroid,
        transient_db_per_ms=slope,
    )


# ---------------------------------------------------------------------------
# Public Severity_Classifier surface (R3.1)
# ---------------------------------------------------------------------------
#
# `grade(audio_signal_or_path)` is the contract consumed by the
# Audio_Detector once it decides a window is a CRASH. It returns the keys
# named in R3.1 (`severity`, `severity_confidence`) using the
# {"minor", "moderate", "severe"} label set defined in the requirements.
#
# This is a deterministic heuristic placeholder: peak amplitude over the
# clip is mapped to one of three severity bins, and `severity_confidence`
# is the distance from the bin boundary scaled into [0, 1]. Task 4.2
# replaces the body with a trained head, but the function signature, the
# label set, and the failure semantics are stable from this point on.

from pathlib import Path as _SeverityPath  # local alias to keep module-top imports tidy
from typing import Union as _SeverityUnion

GradeAudioInput = _SeverityUnion[str, _SeverityPath, np.ndarray]

GradeSeverityLabel = Literal["minor", "moderate", "severe"]
_VALID_GRADE_LABELS: tuple[GradeSeverityLabel, ...] = ("minor", "moderate", "severe")

# Peak-amplitude bin edges (R3.1 heuristic placeholder for task 4.2).
_GRADE_LO = 0.4   # peak <= 0.4   -> minor
_GRADE_HI = 0.7   # peak in (0.4, 0.7] -> moderate; > 0.7 -> severe

# Task 4.2 — when a trained severity head is available under
# ``backend/audio_model/checkpoints/severity_<git_sha>.pth``, prefer it
# over the heuristic. The heuristic remains the fallback for fresh
# checkouts (no labelled data → no checkpoint) and for any failure to
# load a deployed checkpoint. Resolution is deferred to the lookup helper
# below so importing :mod:`severity` stays cheap (no torch import on the
# happy path that just calls :func:`grade` heuristically).
_SEVERITY_CHECKPOINT_DIR = _SeverityPath(__file__).resolve().parent / "checkpoints"
_SEVERITY_CHECKPOINT_GLOB = "severity_*.pth"


def _find_active_severity_checkpoint() -> _SeverityPath | None:
    """Return the most recently-modified severity head checkpoint, if any.

    ``backend/audio_model/checkpoints/severity_<git_sha>.pth`` is the
    on-disk convention (task 4.2 brief). Multiple SHAs may co-exist
    during a rolling deployment; we pick the freshest by mtime so the
    operator's intent ("the file I just dropped in") wins without forcing
    them to maintain an ACTIVE pointer (the Phase 3 model registry
    handles full hot-swap; severity is a smaller blast radius).

    Returns ``None`` when no candidate is present, which is the expected
    state until labelled data lands.
    """
    if not _SEVERITY_CHECKPOINT_DIR.is_dir():
        return None
    candidates = sorted(
        _SEVERITY_CHECKPOINT_DIR.glob(_SEVERITY_CHECKPOINT_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


class _SeverityHeadPredictor:
    """Lazy-loaded wrapper around the trained severity head.

    The head is built by :mod:`backend.audio_model.train_severity` and
    persisted with the payload ``{state_dict, backbone, label_mapping}``
    plus optional ``val_macro_f1`` metadata. Loading is best-effort: any
    failure (missing file, corrupt payload, label mapping mismatch,
    backbone import error) is logged at WARNING and the predictor flips
    into a "not available" state so :func:`grade` falls back to the
    heuristic.
    """

    _instance: "_SeverityHeadPredictor | None" = None

    def __init__(self) -> None:
        self.available = False
        self.model = None
        self.transform = None
        self.label_mapping: dict[str, int] = {}
        self.checkpoint_path: _SeverityPath | None = None
        self._load()

    @classmethod
    def instance(cls) -> "_SeverityHeadPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Clear the cached singleton so a test can re-trigger discovery."""
        cls._instance = None

    def _load(self) -> None:
        path = _find_active_severity_checkpoint()
        if path is None:
            # Expected state until labelled data lands; not a warning.
            return
        try:
            import torch
            from torchvision import transforms

            from .model import build_model
            from .train_severity import (
                LABEL_TO_INDEX,
                _swap_classifier_for_3_class,
            )

            payload = torch.load(
                str(path), map_location="cpu", weights_only=False
            )
            for required in ("state_dict", "backbone", "label_mapping"):
                if required not in payload:
                    raise RuntimeError(
                        f"missing required field {required!r}"
                    )
            label_mapping = dict(payload["label_mapping"])
            if label_mapping != dict(LABEL_TO_INDEX):
                raise RuntimeError(
                    f"label_mapping {label_mapping!r} disagrees with the "
                    f"canonical mapping {LABEL_TO_INDEX!r}"
                )
            model = build_model(payload["backbone"])
            _swap_classifier_for_3_class(
                model, num_classes=len(_VALID_GRADE_LABELS)
            )
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()

            self.model = model
            self.label_mapping = label_mapping
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )
            self.checkpoint_path = path
            self.available = True
            LOG.info("loaded severity head from %s", path)
        except Exception as exc:  # noqa: BLE001 — every failure routes to fallback
            LOG.warning(
                "failed to load severity head from %s (%s); "
                "falling back to heuristic",
                path,
                exc,
            )

    def predict(self, samples: np.ndarray) -> dict:
        """Run the trained head and shape the result like :func:`grade`."""
        import torch

        from .spectrogram_gen import SAMPLE_RATE, samples_to_mel_image

        img = samples_to_mel_image(samples, SAMPLE_RATE)
        x = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = (
                torch.softmax(logits, dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
            )
        # ``label_mapping`` is checked equal to LABEL_TO_INDEX at load
        # time, so the alphabetical SEVERITY_LABELS order matches
        # ``probs`` index-for-index.
        index_to_label = sorted(
            self.label_mapping.items(), key=lambda kv: kv[1]
        )
        pred_idx = int(np.argmax(probs))
        label, _ = index_to_label[pred_idx]
        confidence = float(probs[pred_idx])
        # Defensive clip in case torch returns a value just outside [0, 1]
        # due to fp32 rounding.
        confidence = max(0.0, min(1.0, confidence))
        return {"severity": label, "severity_confidence": confidence}


def _coerce_grade_audio(audio: GradeAudioInput) -> np.ndarray:
    """Normalize the heterogeneous input into a 1-D float32 mono array.

    Accepts:
      * `pathlib.Path` or `str` pointing to a WAV file (loaded at the
        Audio_Detector sample rate).
      * `numpy.ndarray` (1-D mono, or 2-D which is downmixed to mono).
    """
    if isinstance(audio, (str, _SeverityPath)):
        # Imported lazily so unit tests that pass arrays do not need librosa.
        import librosa  # type: ignore[import-not-found]
        from .spectrogram_gen import SAMPLE_RATE as _SR

        samples, _ = librosa.load(str(audio), sr=_SR, mono=True)
        return samples.astype(np.float32, copy=False)

    if isinstance(audio, np.ndarray):
        arr = audio
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        if arr.ndim != 1:
            raise ValueError("expected a 1-D mono PCM array")
        return arr.astype(np.float32, copy=False)

    raise TypeError(f"unsupported audio input type: {type(audio)!r}")


def _grade_confidence_from_peak(peak: float) -> float:
    """Distance from the active bin's boundary, normalized to [0, 1].

    Closer to a boundary = lower confidence; deep inside a bin = higher.
    The normalization uses each bin's own width so confidences across
    bins are comparable.
    """
    peak = float(max(0.0, min(1.0, peak)))
    if peak <= _GRADE_LO:
        # minor bin spans [0.0, 0.4]; distance from the upper edge.
        width = _GRADE_LO  # 0.4
        dist = _GRADE_LO - peak
    elif peak <= _GRADE_HI:
        # moderate bin spans (0.4, 0.7]; distance to the nearer edge.
        width = (_GRADE_HI - _GRADE_LO) / 2.0  # 0.15
        dist = min(peak - _GRADE_LO, _GRADE_HI - peak)
    else:
        # severe bin spans (0.7, 1.0]; distance from the lower edge.
        width = 1.0 - _GRADE_HI  # 0.3
        dist = peak - _GRADE_HI

    if width <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, dist / width)))


def grade(audio_signal_or_path: GradeAudioInput) -> dict:
    """Severity_Classifier entry point (R3.1, R3.2).

    Returns a dict with:
      * ``severity``: one of ``"minor"``, ``"moderate"``, ``"severe"``.
      * ``severity_confidence``: float in the inclusive range [0.0, 1.0].

    Resolution order:
      1. **Trained severity head** (task 4.2) — if a
         ``severity_<git_sha>.pth`` checkpoint exists under
         ``backend/audio_model/checkpoints/`` and loads cleanly, it is
         consulted first. Output indices map back to labels via the
         payload's ``label_mapping`` (canonical: ``{minor:0, moderate:1,
         severe:2}``).
      2. **Heuristic fallback** (task 4.1) — when no checkpoint is
         deployed (the expected state until labelled data lands) or the
         deployed checkpoint fails to load, peak amplitude is mapped to
         a severity bin and confidence is the distance from the bin
         boundary.

    The Audio_Detector handles failure recovery (R3.4) — this function
    raises on bad input so the caller can WARN-log with the
    Correlation_ID and substitute the safe ``("moderate", 0.0)`` default.
    """
    samples = _coerce_grade_audio(audio_signal_or_path)
    if samples.size == 0:
        # Empty clip: nothing to grade. Caller will fall back per R3.4.
        raise ValueError("cannot grade empty audio signal")

    # Prefer the trained head when one is deployed.
    head = _SeverityHeadPredictor.instance()
    if head.available:
        try:
            return head.predict(samples)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "trained severity head inference failed (%s); "
                "falling back to heuristic",
                exc,
            )
            # Fall through to the heuristic below.

    peak = float(np.max(np.abs(samples)))
    # Defensive clip: input may be slightly above 1.0 from float rounding.
    peak = max(0.0, min(1.0, peak))

    if peak > _GRADE_HI:
        label: GradeSeverityLabel = "severe"
    elif peak > _GRADE_LO:
        label = "moderate"
    else:
        label = "minor"

    confidence = _grade_confidence_from_peak(peak)
    return {"severity": label, "severity_confidence": confidence}
