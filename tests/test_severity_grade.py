"""
Tests for ``backend.audio_model.severity.grade`` (R3.1, R3.2, R3.4).

The Severity_Classifier surface is a small heuristic placeholder for task
4.1 and is exercised here in isolation. Integration with the
Audio_Detector (R3.2) is verified by the inference tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from backend.audio_model import severity as severity_module
from backend.audio_model.severity import grade


SR = 22050


def _signal_with_peak(peak: float, duration_s: float = 3.0) -> np.ndarray:
    """Build a deterministic 1-D float32 signal whose maximum |x| equals ``peak``."""
    n = int(duration_s * SR)
    samples = np.zeros(n, dtype=np.float32)
    samples[n // 2] = float(peak)
    return samples


class TestGradeReturnContract:
    """R3.1: return shape, label set, and confidence range."""

    @pytest.mark.parametrize("peak", [0.1, 0.5, 0.9])
    def test_returns_required_keys(self, peak):
        out = grade(_signal_with_peak(peak))
        assert set(out.keys()) == {"severity", "severity_confidence"}

    @pytest.mark.parametrize("peak", [0.1, 0.5, 0.9])
    def test_severity_in_permitted_label_set(self, peak):
        out = grade(_signal_with_peak(peak))
        assert out["severity"] in {"minor", "moderate", "severe"}

    @pytest.mark.parametrize("peak", [0.1, 0.5, 0.9])
    def test_confidence_in_unit_interval(self, peak):
        out = grade(_signal_with_peak(peak))
        c = out["severity_confidence"]
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0


class TestGradeBins:
    """Heuristic peak->severity bins per the task notes."""

    def test_loud_signal_is_severe(self):
        # Peak > 0.7 -> severe.
        assert grade(_signal_with_peak(0.95))["severity"] == "severe"

    def test_medium_signal_is_moderate(self):
        # 0.4 < peak <= 0.7 -> moderate.
        assert grade(_signal_with_peak(0.55))["severity"] == "moderate"

    def test_quiet_signal_is_minor(self):
        # Peak <= 0.4 -> minor.
        assert grade(_signal_with_peak(0.2))["severity"] == "minor"

    def test_boundary_at_lo_is_minor(self):
        # Just below the lower boundary -> minor. (Exact 0.4 is unsafe to
        # test because float32 storage of 0.4 is ~0.4000000059604645.)
        assert grade(_signal_with_peak(0.39))["severity"] == "minor"

    def test_just_above_lo_is_moderate(self):
        # Just above the lower boundary -> moderate.
        assert grade(_signal_with_peak(0.41))["severity"] == "moderate"

    def test_boundary_at_hi_is_moderate(self):
        # Boundary case: peak exactly at 0.7 stays in moderate.
        assert grade(_signal_with_peak(0.7))["severity"] == "moderate"

    def test_just_above_hi_is_severe(self):
        # Just above the upper boundary -> severe.
        assert grade(_signal_with_peak(0.71))["severity"] == "severe"


class TestGradeConfidence:
    """Confidence is distance from the active bin's boundary, normalized."""

    def test_deep_minor_has_higher_confidence_than_near_boundary(self):
        deep = grade(_signal_with_peak(0.05))["severity_confidence"]
        edge = grade(_signal_with_peak(0.39))["severity_confidence"]
        assert deep > edge

    def test_deep_severe_has_higher_confidence_than_near_boundary(self):
        deep = grade(_signal_with_peak(0.99))["severity_confidence"]
        edge = grade(_signal_with_peak(0.71))["severity_confidence"]
        assert deep > edge


class TestGradeInputs:
    """Accepts numpy arrays and rejects unsupported types cleanly."""

    def test_accepts_2d_input_via_downmix(self):
        # Stereo input should be downmixed; behaviour matches mono baseline.
        peak = 0.8
        stereo = np.stack(
            [_signal_with_peak(peak), _signal_with_peak(peak)], axis=1
        )
        out = grade(stereo)
        assert out["severity"] == "severe"

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            grade(12345)  # type: ignore[arg-type]

    def test_rejects_empty_signal(self):
        with pytest.raises(ValueError):
            grade(np.zeros(0, dtype=np.float32))


class TestAudioDetectorFallback:
    """R3.4: out-of-range labels and exceptions fall back to ('moderate', 0.0).

    Drives the Audio_Detector wrapper directly so the public grade() API
    is not coupled to the recovery semantics.
    """

    def test_fallback_on_invalid_label(self, monkeypatch, caplog):
        from backend.audio_model import inference as inference_module

        def _bad_label(_audio):
            return {"severity": "catastrophic", "severity_confidence": 0.9}

        monkeypatch.setattr(inference_module, "severity_grade", _bad_label)
        result = {"event": "CRASH", "confidence": 0.95}

        with caplog.at_level(logging.WARNING, logger="inference"):
            wrapped = inference_module._apply_severity(result, _signal_with_peak(0.9))

        assert wrapped["severity"] == "moderate"
        assert wrapped["severity_confidence"] == 0.0
        assert any("severity_grading_failed" in m for m in caplog.messages)

    def test_fallback_on_exception(self, monkeypatch, caplog):
        from backend.audio_model import inference as inference_module

        def _boom(_audio):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(inference_module, "severity_grade", _boom)
        result = {"event": "CRASH", "confidence": 0.95}

        with caplog.at_level(logging.WARNING, logger="inference"):
            wrapped = inference_module._apply_severity(result, _signal_with_peak(0.9))

        assert wrapped["severity"] == "moderate"
        assert wrapped["severity_confidence"] == 0.0
        assert any("severity_grading_failed" in m for m in caplog.messages)

    def test_non_crash_predictions_get_no_severity_fields(self):
        """R3.2 invariant: only CRASH windows are graded."""
        from backend.audio_model import inference as inference_module

        for event in ("NORMAL", "DEADLINE", "NOISE"):
            result = {"event": event, "confidence": 0.9}
            wrapped = inference_module._apply_severity(result, _signal_with_peak(0.9))
            assert "severity" not in wrapped
            assert "severity_confidence" not in wrapped

    def test_crash_predictions_get_severity_fields_on_happy_path(self):
        from backend.audio_model import inference as inference_module

        result = {"event": "CRASH", "confidence": 0.95}
        wrapped = inference_module._apply_severity(result, _signal_with_peak(0.9))
        assert wrapped["severity"] in {"minor", "moderate", "severe"}
        assert 0.0 <= wrapped["severity_confidence"] <= 1.0
        # The original prediction fields are preserved.
        assert wrapped["event"] == "CRASH"
        assert wrapped["confidence"] == 0.95
