"""
In-memory inference for the CrashSense Audio_Detector (Requirement 4).

Provides:
  predict(audio)       -> {"event": "CRASH"|"NORMAL", "confidence": float}
  predict_stream(audio)-> generator of per-window prediction dicts (sliding 3.0s, hop 0.5s)

`audio` may be:
  * a string filesystem path to a WAV file, or
  * a 1-D numpy float32 array of mono PCM at 22050 Hz

If the trained checkpoint is missing, we fall back to a deterministic
energy-based stub so the demo pipeline still runs end-to-end. The runtime
error path mandated by Requirement 4 criterion 5 is gated by the
CRASHSENSE_REQUIRE_CHECKPOINT environment variable.
"""

from __future__ import annotations

import io
import logging
import math
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Generator, Union

import numpy as np

from . import calibrate as _calibrate
from .model import build_model
from .onset import OnsetDetector
from .severity import grade as severity_grade
from .spectrogram_gen import (
    SAMPLE_RATE,
    TARGET_LEN,
    samples_to_mel_image,
)

LOG = logging.getLogger("inference")

CHECKPOINT_PATH = Path(__file__).resolve().parent / "crash_detector.pth"
AST_HEAD_PATH = Path(__file__).resolve().parent / "crash_detector_ast_head.pth"
CALIBRATOR_DIR = Path(__file__).resolve().parent / "checkpoints"
USE_AST = os.environ.get("CRASHSENSE_USE_AST", "0") == "1"
CONF_THRESHOLD = 0.5
PRINT_THRESHOLD = 0.85
WINDOW_S = 3.0
HOP_S = 0.5
WINDOW_LEN = TARGET_LEN
HOP_LEN = int(SAMPLE_RATE * HOP_S)

# Onset_Detector parameters (R8). The 3-of-4 sliding-window vote has been
# replaced by an OnsetDetector keyed off the per-window model score with a
# configurable refractory window. Defaults are 500 ms refractory and the
# existing PRINT_THRESHOLD (0.85) above which a window counts as an onset.
# Both knobs are env-overridable so deployments can tune without a code
# change.
ONSET_REFRACTORY_MS = int(os.environ.get("CRASHSENSE_ONSET_REFRACTORY_MS", "500"))
ONSET_THRESHOLD = float(os.environ.get("CRASHSENSE_ONSET_THRESHOLD", str(PRINT_THRESHOLD)))

# Legacy 3-of-4 consensus knobs are retained as module-level constants for
# backward compatibility with any external caller that imported them. They
# are no longer used by predict_stream; the OnsetDetector replaces them.
# Deprecated: removed from the streaming consensus path in task 4.14.
CONSENSUS_N = 4
CONSENSUS_K = 3

AudioInput = Union[str, Path, np.ndarray]


def _load_torch():
    import torch  # imported lazily; the stub fallback works without torch installed
    from torchvision import transforms
    return torch, transforms


def _half_up_one_decimal(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


class _Predictor:
    _instance: "_Predictor | None" = None

    def __init__(self) -> None:
        self.model = None
        self.transform = None
        self.device = None
        self.backbone = None
        self.loaded_from_checkpoint = False
        self.temperature = 1.0
        # R5.3 — calibration is loaded lazily on first construction; if the
        # active calibrator file is missing or corrupt the predictor falls
        # back to uncalibrated softmax with calibration_active=false (R5.5).
        self.calibrator: _calibrate.Calibrator | None = None
        self.calibration_active = False
        self._load()
        self._load_calibrator()

    @classmethod
    def instance(cls) -> "_Predictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_calibrator(self) -> None:
        """Locate and load the active calibrator (R5.3, R5.5).

        Filename convention: ``<arch>_calibrator_<git_sha>.json`` under
        ``backend/audio_model/checkpoints/``. We prefer a calibrator
        matching the loaded backbone; if none is present the search falls
        back to any ``*calibrator*.json`` so deployments can drop in a
        single file without naming gymnastics.

        Missing or corrupt files are non-fatal: ``calibrate.load`` already
        emits a WARNING and returns None (R5.5), and we surface that state
        through ``calibration_active=false`` on every prediction.
        """
        path = _calibrate.find_active_calibrator(
            arch=self.backbone,
            checkpoints_dir=CALIBRATOR_DIR,
        )
        if path is None:
            # No calibrator present is the expected state until R4.6 ships
            # a fitted file; do not log a WARNING here (the file is simply
            # not deployed yet). Predictions still return calibration_active=false.
            self.calibrator = None
            self.calibration_active = False
            return
        cal = _calibrate.load(path)
        if cal is None:
            # `calibrate.load` already logged a WARNING for missing/corrupt files.
            self.calibrator = None
            self.calibration_active = False
            return
        self.calibrator = cal
        self.calibration_active = True
        LOG.info("loaded calibrator (%s) from %s", cal.method, path)

    def _load(self) -> None:
        if not CHECKPOINT_PATH.exists():
            msg = (
                f"Crash detector checkpoint not found at {CHECKPOINT_PATH}. "
                "Run backend/audio_model/train.py to produce one."
            )
            if os.environ.get("CRASHSENSE_REQUIRE_CHECKPOINT") == "1":
                raise RuntimeError(msg)
            LOG.warning("%s Falling back to energy-based stub.", msg)
            return
        try:
            torch, transforms = _load_torch()
            payload = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
            backbone = payload.get("backbone", "resnet18")
            model = build_model(backbone)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            self.model = model
            self.backbone = backbone
            self.temperature = float(payload.get("temperature", 1.0))
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            self.loaded_from_checkpoint = True
            LOG.info("Loaded crash detector checkpoint (%s, T=%.3f)", backbone, self.temperature)
        except Exception as exc:
            if os.environ.get("CRASHSENSE_REQUIRE_CHECKPOINT") == "1":
                raise RuntimeError(
                    f"Crash detector checkpoint at {CHECKPOINT_PATH} failed to load: {exc}"
                ) from exc
            LOG.error("Checkpoint load failed (%s); using stub.", exc)

    def predict_samples(self, samples: np.ndarray) -> dict:
        # When CRASHSENSE_USE_AST=1 and the AST head is trained, route through it.
        if USE_AST:
            try:
                from . import ast_inference
                if ast_inference.is_available():
                    return ast_inference._ASTPredictor.instance().predict(samples)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("AST inference fallback (%s); using ResNet path", exc)
        if self.loaded_from_checkpoint and self.model is not None:
            return self._predict_with_model(samples)
        return self._predict_stub(samples)

    def _predict_with_model(self, samples: np.ndarray) -> dict:
        torch, _ = _load_torch()
        img = samples_to_mel_image(samples, SAMPLE_RATE)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_t = self.model(x)
            logits_np = logits_t.squeeze(0).cpu().numpy()
        # R5.4 — when an active calibrator is loaded, route raw logits
        # through it; otherwise fall back to the legacy temperature-scaled
        # softmax. The `calibration_active` flag (R5.5) tells downstream
        # consumers whether the returned confidence is calibrated.
        if self.calibrator is not None:
            probs = np.asarray(self.calibrator.transform(logits_np), dtype=np.float64)
            calibration_active = True
        else:
            z = logits_np / max(self.temperature, 1e-6)
            z = z - z.max()
            probs = np.exp(z) / np.exp(z).sum()
            calibration_active = False
        # ImageFolder sorts class names alphabetically: crash=0, noise=1
        crash_p = float(probs[0])
        if crash_p >= CONF_THRESHOLD:
            return {
                "event": "CRASH",
                "confidence": crash_p,
                "calibration_active": calibration_active,
            }
        return {
            "event": "NORMAL",
            "confidence": float(probs[1]),
            "calibration_active": calibration_active,
        }

    def _predict_stub(self, samples: np.ndarray) -> dict:
        """Heuristic fallback: detect crashes via short-term energy spikes."""
        if samples.size == 0:
            return {"event": "NORMAL", "confidence": 1.0, "calibration_active": False}
        x = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x)))
        peak = float(np.max(np.abs(x)))
        crest = peak / max(rms, 1e-6)
        # Map crest factor + peak into a 0-1 confidence
        score = max(0.0, min(1.0, (crest - 4.0) / 12.0)) * 0.6 + min(peak, 1.0) * 0.4
        if score >= CONF_THRESHOLD:
            return {
                "event": "CRASH",
                "confidence": float(score),
                "calibration_active": False,
            }
        return {
            "event": "NORMAL",
            "confidence": float(1.0 - score),
            "calibration_active": False,
        }


def _coerce_audio(audio: AudioInput) -> np.ndarray:
    """Normalize the heterogeneous input into a 1-D float32 mono array.

    Spatial audio features are deferred (see ``docs/spatial_audio_decision.md``),
    so the supported channel count is one. Multi-channel input is accepted
    by downmixing to mono via channel-wise mean (R4.4); a single-channel
    2-D array (shape ``(N, 1)``) is squeezed without logging since it
    carries no extra channels.
    """
    if isinstance(audio, (str, Path)):
        import librosa
        samples, _ = librosa.load(str(audio), sr=SAMPLE_RATE, mono=True)
        return samples.astype(np.float32)
    if isinstance(audio, np.ndarray):
        arr = audio
        if arr.ndim == 2:
            channels = arr.shape[1]
            if channels > 1:
                # R4.4 — log INFO once per downmix with the correlation id
                # so the operator can correlate spatial-input handling with
                # the surrounding CrashEvent.
                LOG.info(
                    "audio_downmixed_to_mono | correlation_id=%s | original_channels=%d",
                    _correlation_id_or_dash(),
                    channels,
                )
                arr = arr.mean(axis=1)
            else:
                # Single-channel 2-D input is effectively mono; squeeze
                # without logging.
                arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError("expected a 1-D mono PCM array")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr
    raise TypeError(f"unsupported audio input type: {type(audio)!r}")


def _fix_window(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < WINDOW_LEN:
        return np.pad(samples, (0, WINDOW_LEN - samples.shape[0]), mode="constant")
    return samples[:WINDOW_LEN]


# Severity_Classifier label set (R3.1). Kept here as the single source of
# truth for the failure-recovery branch in `_apply_severity` so a drift
# between this module and `severity.py` is caught at the next predict().
_VALID_SEVERITY_LABELS = ("minor", "moderate", "severe")


def _correlation_id_or_dash() -> str:
    """Best-effort lookup of the request-scoped correlation id.

    The Audio_Detector runs both inside the FastAPI request scope (where
    `backend.api.logging_config.request_id_ctx` is set) and from offline
    scripts (where it is not). Importing the context var lazily avoids a
    hard dependency from the audio package on the API package.
    """
    try:
        from backend.api.logging_config import request_id_ctx  # type: ignore
        return request_id_ctx.get()
    except Exception:  # noqa: BLE001 — never let logging break inference
        return "-"


def _apply_severity(result: dict, samples: np.ndarray) -> dict:
    """Merge severity fields into a CRASH prediction (R3.2).

    Non-CRASH predictions (NORMAL, DEADLINE, etc.) are returned unchanged
    so the response shape stays minimal in the common case.

    Failure recovery (R3.4): any exception from the classifier and any
    return value with a `severity` outside ``{"minor", "moderate",
    "severe"}`` is logged at WARNING with the correlation id and replaced
    with the safe default ``("moderate", 0.0)``. The Audio_Detector keeps
    serving requests even if the classifier is broken or missing.
    """
    if result.get("event") != "CRASH":
        return result

    try:
        graded = severity_grade(samples)
        label = graded.get("severity")
        confidence = graded.get("severity_confidence")
        if label not in _VALID_SEVERITY_LABELS:
            raise ValueError(
                f"severity classifier returned invalid label: {label!r}"
            )
        if not isinstance(confidence, (int, float)):
            raise ValueError(
                f"severity classifier returned non-numeric confidence: {confidence!r}"
            )
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"severity_confidence out of range: {confidence!r}"
            )
        result = dict(result)
        result["severity"] = label
        result["severity_confidence"] = confidence
        return result
    except Exception as exc:  # noqa: BLE001 — R3.4 mandates a global catch
        LOG.warning(
            "severity_grading_failed | correlation_id=%s | error=%s",
            _correlation_id_or_dash(),
            exc,
        )
        result = dict(result)
        result["severity"] = "moderate"
        result["severity_confidence"] = 0.0
        return result


def predict(audio: AudioInput) -> dict:
    """Return a single CRASH/NORMAL decision for the supplied audio."""
    samples = _coerce_audio(audio)
    window = _fix_window(samples)
    result = _Predictor.instance().predict_samples(window)
    return _apply_severity(result, window)


def predict_stream(audio: AudioInput, *, print_detections: bool = False) -> Generator[dict, None, None]:
    """
    Sliding-window predictions over a longer clip.

    Yields one dict per window in chronological order. The dict contains:
        event: "CRASH" | "NORMAL"     — single-window decision
        confidence: float             — softmax probability
        window_start_s: float         — window start in the input audio
        consensus: bool               — True on the first window of an onset
                                        (i.e. an above-threshold score outside
                                        the OnsetDetector's refractory window)

    When `print_detections` is True, prints
        CRASH DETECTED — confidence: <value>%
    only on the onset-emitting window. Subsequent above-threshold windows
    that fall within ``ONSET_REFRACTORY_MS`` of the last emission are
    suppressed (R8.2). The 3-of-4 sliding-window vote previously used here
    has been replaced by the OnsetDetector (task 4.14).
    """
    samples = _coerce_audio(audio)
    onset = OnsetDetector(
        refractory_ms=ONSET_REFRACTORY_MS,
        threshold=ONSET_THRESHOLD,
    )

    if samples.shape[0] <= WINDOW_LEN:
        result = predict(samples)
        result = dict(result)
        result["window_start_s"] = 0.0
        score = float(result["confidence"]) if result["event"] == "CRASH" else 0.0
        result["consensus"] = onset.process(score, t_ms=0)
        if print_detections and result["consensus"]:
            pct = _half_up_one_decimal(result["confidence"] * 100.0)
            print(f"CRASH DETECTED — confidence: {pct}%")
        yield result
        return

    n = samples.shape[0]

    for start in range(0, n - WINDOW_LEN + 1, HOP_LEN):
        window = samples[start:start + WINDOW_LEN]
        result = _Predictor.instance().predict_samples(window)
        result = _apply_severity(result, window)
        # The OnsetDetector treats only above-threshold CRASH scores as
        # candidates; non-CRASH windows feed score=0.0 so they never fire
        # but still advance the timeline.
        score = float(result["confidence"]) if result["event"] == "CRASH" else 0.0
        t_ms = int(round(start * 1000.0 / SAMPLE_RATE))
        emit = onset.process(score, t_ms=t_ms)

        out = dict(result)
        out["window_start_s"] = start / float(SAMPLE_RATE)
        out["consensus"] = emit

        if emit and print_detections:
            pct = _half_up_one_decimal(result["confidence"] * 100.0)
            print(f"CRASH DETECTED — confidence: {pct}%")

        yield out
