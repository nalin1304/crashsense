"""Tests for the audio inference module (Requirement 4).

These tests use real ESC-50-derived clips that the dataset_prep step has
placed under data/raw_audio/. They are skipped if those files aren't on
disk yet (e.g. CI before setup).
"""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CRASH_DIR = REPO_ROOT / "data" / "raw_audio" / "crash"
NOISE_DIR = REPO_ROOT / "data" / "raw_audio" / "noise"
CHECKPOINT = REPO_ROOT / "backend" / "audio_model" / "crash_detector.pth"


def _have_dataset() -> bool:
    return CRASH_DIR.is_dir() and NOISE_DIR.is_dir() and any(CRASH_DIR.glob("*.wav"))


pytestmark = pytest.mark.skipif(not _have_dataset(),
                                 reason="dataset not prepared")


@pytest.fixture(scope="module")
def predict_fn():
    from backend.audio_model.inference import predict
    return predict


@pytest.fixture(scope="module")
def predict_stream_fn():
    from backend.audio_model.inference import predict_stream
    return predict_stream


def _load_one(directory: Path) -> np.ndarray:
    import librosa
    candidates = [
        p for p in directory.glob("esc_*.wav")
        if not any(t in p.name for t in ("_pitch", "_stretch", "_noise"))
    ]
    assert candidates, f"no original ESC-50 clips in {directory}"
    samples, _ = librosa.load(str(candidates[0]), sr=22050, mono=True)
    return samples.astype(np.float32)


class TestPredictReturnContract:
    def test_returns_dict_with_event_and_confidence(self, predict_fn):
        samples = _load_one(CRASH_DIR)
        result = predict_fn(samples)
        assert isinstance(result, dict)
        assert result["event"] in {"CRASH", "NORMAL"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_accepts_path_input(self, predict_fn):
        candidates = [
            p for p in CRASH_DIR.glob("esc_*.wav")
            if not any(t in p.name for t in ("_pitch", "_stretch", "_noise"))
        ]
        result = predict_fn(str(candidates[0]))
        assert isinstance(result, dict)
        assert result["event"] in {"CRASH", "NORMAL"}


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint missing")
class TestPredictAccuracy:
    """When a real checkpoint is present, the model should classify the
    training-domain samples correctly with high confidence."""

    def test_crash_sample_classifies_as_crash(self, predict_fn):
        samples = _load_one(CRASH_DIR)
        result = predict_fn(samples)
        assert result["event"] == "CRASH"
        assert result["confidence"] > 0.8

    def test_noise_sample_classifies_as_normal(self, predict_fn):
        samples = _load_one(NOISE_DIR)
        result = predict_fn(samples)
        assert result["event"] == "NORMAL"
        assert result["confidence"] > 0.8


class TestPredictStream:
    def test_short_audio_yields_one_window(self, predict_stream_fn):
        # Less than the 3.0 s window length -> one prediction
        samples = np.zeros(22050 * 2, dtype=np.float32)
        results = list(predict_stream_fn(samples))
        assert len(results) == 1

    def test_long_audio_yields_multiple_windows(self, predict_stream_fn):
        samples = np.zeros(22050 * 6, dtype=np.float32)
        results = list(predict_stream_fn(samples))
        assert len(results) > 1
        for r in results:
            assert "event" in r and "confidence" in r and "window_start_s" in r
