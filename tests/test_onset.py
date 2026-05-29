"""Tests for backend.audio_model.onset.OnsetDetector (R8.1, R8.2, R8.3, R8.4).

Covers:
  * Below-threshold scores never emit (R8.2).
  * The first above-threshold score emits (R8.2).
  * Within-refractory above-threshold scores do not re-emit (R8.2, R8.4).
  * After-refractory above-threshold scores re-emit (R8.2, R8.4).
  * 200 ms-spaced impacts: default 500 ms refractory yields 1 emission;
    `refractory_ms=100` yields 2 (R8.4).
  * Parameter validation rejects out-of-range `refractory_ms` and
    `threshold` (R8.3).
  * Streaming integration via predict_stream replaces the 3-of-4 vote
    (task 4.14): a single above-threshold window fires consensus, and
    refractory suppresses re-fires.

The refractory-monotonicity property test (R8.5, P5) is task 4.16.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from backend.audio_model import inference
from backend.audio_model.onset import OnsetDetector


# ---------------------------------------------------------------------------
# OnsetDetector unit tests (R8.1, R8.2, R8.3, R8.4)
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """R8.3 — refractory_ms in [100, 2000], threshold in [0.0, 1.0]."""

    @pytest.mark.parametrize("bad", [-1, 0, 50, 99, 2001, 10_000])
    def test_refractory_ms_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError):
            OnsetDetector(refractory_ms=bad)

    @pytest.mark.parametrize("good", [100, 500, 1000, 2000])
    def test_refractory_ms_boundary_accepted(self, good):
        d = OnsetDetector(refractory_ms=good)
        assert d.refractory_ms == good

    @pytest.mark.parametrize("bad", [-0.1, -1.0, 1.0001, 2.0])
    def test_threshold_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError):
            OnsetDetector(threshold=bad)

    @pytest.mark.parametrize("good", [0.0, 0.25, 0.5, 0.85, 1.0])
    def test_threshold_boundary_accepted(self, good):
        d = OnsetDetector(threshold=good)
        assert d.threshold == pytest.approx(good)

    def test_refractory_ms_must_be_int(self):
        with pytest.raises(TypeError):
            OnsetDetector(refractory_ms=500.0)

    def test_threshold_accepts_int(self):
        # int is a valid float-shaped value
        d = OnsetDetector(threshold=1)
        assert d.threshold == 1.0


class TestBelowThresholdNeverEmits:
    """R8.2 — scores below threshold never emit."""

    def test_zero_score_no_emit(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        assert d.process(0.0, t_ms=0) is False
        assert d.last_emit_t_ms is None

    def test_just_below_threshold_no_emit(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        for t in range(0, 5000, 100):
            assert d.process(0.4999, t_ms=t) is False
        assert d.last_emit_t_ms is None

    def test_threshold_zero_lets_any_score_emit(self):
        # Boundary: threshold=0.0 makes every score >= 0.0 a candidate
        d = OnsetDetector(threshold=0.0, refractory_ms=100)
        assert d.process(0.0, t_ms=0) is True


class TestFirstAboveThresholdEmits:
    """R8.2 — the first above-threshold score emits."""

    def test_first_above_threshold_emits(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        assert d.process(0.9, t_ms=0) is True
        assert d.last_emit_t_ms == 0

    def test_score_equal_to_threshold_emits(self):
        # `score < threshold` is the suppression rule, so equality emits.
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        assert d.process(0.5, t_ms=0) is True


class TestRefractoryWindow:
    """R8.2, R8.4 — within-refractory suppressed; after-refractory re-emits."""

    def test_within_refractory_suppressed(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        assert d.process(0.9, t_ms=0) is True
        # Every subsequent above-threshold score before t=500ms is suppressed.
        for t in (100, 200, 300, 400, 499):
            assert d.process(0.95, t_ms=t) is False
        # last_emit_t_ms must remain at the original onset.
        assert d.last_emit_t_ms == 0

    def test_after_refractory_re_emits(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        assert d.process(0.9, t_ms=0) is True
        # t == refractory_ms is the boundary; >= rule lets it emit.
        assert d.process(0.9, t_ms=500) is True
        assert d.last_emit_t_ms == 500

    def test_refractory_advances_from_last_emission(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        d.process(0.9, t_ms=0)        # emit @ 0
        d.process(0.9, t_ms=200)      # suppressed
        emitted_at_700 = d.process(0.9, t_ms=700)  # 700 - 0 >= 500 -> emit
        assert emitted_at_700 is True
        # Now refractory clock restarts at 700; 800 should suppress.
        assert d.process(0.9, t_ms=800) is False

    def test_reset_clears_state(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        d.process(0.9, t_ms=0)
        d.reset()
        assert d.last_emit_t_ms is None
        # After reset the next above-threshold score emits immediately.
        assert d.process(0.9, t_ms=10) is True


class TestDoubleImpact200ms:
    """R8.4 — two impacts 200 ms apart.

    With default 500 ms refractory: exactly 1 emission.
    With refractory_ms=100:        exactly 2 emissions.
    """

    def test_default_refractory_yields_one_emission(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=500)
        first = d.process(0.95, t_ms=0)
        second = d.process(0.95, t_ms=200)
        assert (first, second) == (True, False)

    def test_short_refractory_yields_two_emissions(self):
        d = OnsetDetector(threshold=0.5, refractory_ms=100)
        first = d.process(0.95, t_ms=0)
        second = d.process(0.95, t_ms=200)
        assert (first, second) == (True, True)


# ---------------------------------------------------------------------------
# Streaming integration via predict_stream (task 4.14)
# ---------------------------------------------------------------------------


def _make_audio(n_seconds: float = 6.0) -> np.ndarray:
    return np.zeros(int(inference.SAMPLE_RATE * n_seconds), dtype=np.float32)


@pytest.fixture
def fake_predict():
    """Patch the per-window predictor with a scripted sequence of outcomes."""

    def _factory(script):
        it = iter(script)

        def _impl(self, samples):
            event, conf = next(it)
            return {"event": event, "confidence": conf}

        return _impl

    return _factory


def test_predict_stream_single_above_threshold_window_fires(fake_predict):
    """Onset-driven: a single above-threshold window emits a CRASH consensus.

    This is the semantic change from the 3-of-4 vote: a high-confidence
    onset no longer needs three corroborating windows to fire.
    """
    audio = _make_audio()
    script = [("NORMAL", 0.99)] * 7
    script[3] = ("CRASH", 0.97)
    with patch.object(inference._Predictor, "predict_samples", fake_predict(script)):
        results = list(inference.predict_stream(audio, print_detections=False))
    fires = [i for i, r in enumerate(results) if r["consensus"]]
    assert len(fires) == 1, f"expected exactly one onset emission, got {fires}"
    assert results[fires[0]]["event"] == "CRASH"


def test_predict_stream_refractory_suppresses_back_to_back_fires(fake_predict):
    """Several consecutive above-threshold windows (0.5 s hop) within the
    refractory window must collapse to a single emission."""
    audio = _make_audio()
    # 4 above-threshold windows back-to-back at 0, 0.5, 1.0, 1.5 s.
    # With refractory=2000 ms (> 1.5 s span), all four collapse to one.
    script = [
        ("CRASH", 0.95), ("CRASH", 0.95), ("CRASH", 0.95), ("CRASH", 0.95),
        ("NORMAL", 0.99), ("NORMAL", 0.99), ("NORMAL", 0.99),
    ]
    with patch.object(inference._Predictor, "predict_samples", fake_predict(script)), \
         patch.object(inference, "ONSET_REFRACTORY_MS", 2000):
        results = list(inference.predict_stream(audio, print_detections=False))
    fires = [i for i, r in enumerate(results) if r["consensus"]]
    assert len(fires) == 1, f"expected one onset emission within refractory, got {fires}"


def test_predict_stream_below_threshold_crash_does_not_fire(fake_predict):
    """Sub-threshold CRASH labels must not trigger an onset emission."""
    audio = _make_audio()
    # Confidences below the default ONSET_THRESHOLD (0.85).
    script = [("CRASH", 0.70)] * 12
    with patch.object(inference._Predictor, "predict_samples", fake_predict(script)):
        results = list(inference.predict_stream(audio, print_detections=False))
    assert not any(r["consensus"] for r in results), \
        "below-threshold scores must not emit"


def test_predict_stream_short_audio_single_window(fake_predict):
    """Audio shorter than WINDOW_LEN takes the single-window code path."""
    audio = np.zeros(int(inference.SAMPLE_RATE * 1.5), dtype=np.float32)
    script = [("CRASH", 0.95)]
    with patch.object(inference._Predictor, "predict_samples", fake_predict(script)):
        results = list(inference.predict_stream(audio, print_detections=False))
    assert len(results) == 1
    assert results[0]["consensus"] is True


def test_predict_stream_re_emits_after_refractory(fake_predict):
    """A second crash that lands outside the refractory window emits again."""
    audio = _make_audio(n_seconds=8.0)
    # Pad the script with NORMALs between two CRASH bursts so the gap
    # between them exceeds 500 ms.
    script = (
        [("CRASH", 0.95)]
        + [("NORMAL", 0.99)] * 4   # 4 hops * 500 ms = 2.0 s gap
        + [("CRASH", 0.95)]
        + [("NORMAL", 0.99)] * 6
    )
    with patch.object(inference._Predictor, "predict_samples", fake_predict(script)):
        results = list(inference.predict_stream(audio, print_detections=False))
    fires = [i for i, r in enumerate(results) if r["consensus"]]
    assert len(fires) == 2, f"expected two onset emissions across refractory gap, got {fires}"
