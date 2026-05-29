"""Tests for severity estimation."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from backend.audio_model.severity import (
    SeverityEstimate,
    estimate_severity,
)


def _make_silence(duration_s: float = 3.0, sr: int = 22050) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _make_burst(duration_s: float = 3.0, sr: int = 22050,
                impact_at_s: float = 1.5,
                impact_amp: float = 0.95,
                impact_len_s: float = 0.25) -> np.ndarray:
    n = int(duration_s * sr)
    samples = (np.random.default_rng(0).normal(0, 0.01, n)).astype(np.float32)
    impact_start = int(impact_at_s * sr)
    impact_n = int(impact_len_s * sr)
    t = np.linspace(0, impact_len_s, impact_n, dtype=np.float32)
    chirp = np.sin(2 * np.pi * (200 + 4000 * t) * t) * np.exp(-t * 6) * impact_amp
    samples[impact_start:impact_start + impact_n] += chirp.astype(np.float32)
    return np.clip(samples, -1.0, 1.0)


class TestSeverityEstimate:
    def test_silence_is_minor(self):
        est = estimate_severity(_make_silence())
        assert est.label == "minor"
        assert est.severity < 0.3

    def test_loud_burst_is_more_severe_than_silence(self):
        loud = estimate_severity(_make_burst(impact_amp=0.95))
        quiet = estimate_severity(_make_silence())
        assert loud.severity > quiet.severity

    def test_louder_means_higher_severity(self):
        soft = estimate_severity(_make_burst(impact_amp=0.2))
        hard = estimate_severity(_make_burst(impact_amp=0.95))
        assert hard.severity > soft.severity

    def test_severity_in_zero_to_one(self):
        for amp in (0.1, 0.5, 0.9):
            est = estimate_severity(_make_burst(impact_amp=amp))
            assert 0.0 <= est.severity <= 1.0

    def test_label_matches_severity_thresholds(self):
        for amp in (0.1, 0.5, 0.95):
            est = estimate_severity(_make_burst(impact_amp=amp))
            if est.severity >= 0.66:
                assert est.label == "major"
            elif est.severity >= 0.33:
                assert est.label == "moderate"
            else:
                assert est.label == "minor"

    def test_handles_empty_input(self):
        est = estimate_severity(np.zeros(0, dtype=np.float32))
        assert est.severity == 0.0
        assert est.label == "minor"

    def test_handles_stereo_input(self):
        stereo = np.stack([_make_burst(), _make_burst()], axis=1)
        est = estimate_severity(stereo)
        assert isinstance(est, SeverityEstimate)
        assert 0.0 <= est.severity <= 1.0

    def test_as_dict_returns_jsonable(self):
        est = estimate_severity(_make_burst())
        d = est.as_dict()
        for k in ("severity", "label", "peak_dbfs", "energy_jfs",
                  "spectral_centroid_hz", "transient_db_per_ms"):
            assert k in d


class TestSeverityProperties:
    @given(st.floats(min_value=0.05, max_value=1.0))
    @settings(max_examples=15, deadline=None)
    def test_severity_monotone_in_amplitude_for_strong_bursts(self, amp):
        """A burst with the same shape should always be more severe than silence."""
        burst = estimate_severity(_make_burst(impact_amp=amp))
        silence = estimate_severity(_make_silence())
        assert burst.severity >= silence.severity
