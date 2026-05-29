"""Unit tests for the post-hoc calibrator (R5.1, R5.3, R5.4, R5.5).

The PBT round-trip and ECE tests live in test_calibrator_roundtrip.py and
test_calibrate_ece.py (task 4.7). This file covers:

* temperature_scaling fits a sane T on validation logits/labels
* isotonic_regression fits and produces monotonic per-class outputs
* save + load returns a calibrator whose transform matches the original
  within 1e-6 elementwise
* load(missing_path) returns None and emits a WARNING
* load(corrupt_json) returns None and emits a WARNING
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from backend.audio_model.calibrate import (
    Calibrator,
    IsotonicRegressionCalibrator,
    TemperatureScalingCalibrator,
    fit,
    load,
    save,
)


# ---------------------------------------------------------------------------
# Synthetic validation set
# ---------------------------------------------------------------------------


def _make_validation_set(
    n: int = 400,
    n_classes: int = 2,
    overconfidence: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (logits, labels) where the model is overconfident.

    A classifier whose logits have been multiplied by ``overconfidence > 1``
    looks overconfident relative to its true accuracy; temperature scaling
    should recover a T close to ``overconfidence`` to flatten the softmax.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n).astype(np.int64)
    # Calibrated logits: each row has correct-class logit ~ N(2, 1), others ~ N(0, 1).
    logits = rng.standard_normal((n, n_classes)).astype(np.float64)
    rows = np.arange(n)
    logits[rows, labels] += 2.0
    # Inject a fixed fraction of label noise so the model isn't 100% accurate
    # (overconfidence relative to ~85% accuracy is what we are calibrating).
    flip_mask = rng.random(n) < 0.15
    flips = rng.integers(1, n_classes, size=n)
    labels = np.where(flip_mask, (labels + flips) % n_classes, labels).astype(np.int64)
    # Overscale to make the model overconfident.
    return logits * overconfidence, labels


# ---------------------------------------------------------------------------
# Fit behaviour
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    def test_fit_returns_temperature_scaling_calibrator(self):
        logits, labels = _make_validation_set()
        cal = fit("temperature_scaling", logits, labels)
        assert isinstance(cal, TemperatureScalingCalibrator)
        assert cal.method == "temperature_scaling"

    def test_fit_recovers_sane_temperature_on_overconfident_logits(self):
        # Logits scaled by 2.0; an ideal calibrator should land near T=2.0.
        logits, labels = _make_validation_set(overconfidence=2.0, seed=7)
        cal = fit("temperature_scaling", logits, labels)
        # Bounded check: the optimizer should land somewhere in [1.0, 4.0].
        # We don't assert tight equality because the validation set is finite
        # and the relationship between scale and optimal T is approximate.
        assert 1.0 < cal.temperature < 4.0

    def test_transform_returns_valid_probability_simplex(self):
        logits, labels = _make_validation_set()
        cal = fit("temperature_scaling", logits, labels)
        probs = cal.transform(logits)
        # Each row sums to 1, all entries in [0, 1].
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-9)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_transform_preserves_argmax(self):
        # Temperature scaling is a strictly-positive monotone transform on
        # softmax inputs and therefore must not change the predicted class.
        logits, labels = _make_validation_set()
        cal = fit("temperature_scaling", logits, labels)
        raw_probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        cal_probs = cal.transform(logits)
        np.testing.assert_array_equal(
            raw_probs.argmax(axis=-1),
            cal_probs.argmax(axis=-1),
        )


class TestIsotonicRegression:
    def test_fit_returns_isotonic_calibrator(self):
        logits, labels = _make_validation_set()
        cal = fit("isotonic_regression", logits, labels)
        assert isinstance(cal, IsotonicRegressionCalibrator)
        assert cal.method == "isotonic_regression"

    def test_isotonic_breakpoints_are_sorted(self):
        logits, labels = _make_validation_set()
        cal = fit("isotonic_regression", logits, labels)
        xs = cal.breakpoints_x
        assert (np.diff(xs) >= -1e-12).all(), "breakpoints must be sorted ascending"

    def test_isotonic_per_class_output_is_monotone(self):
        # The piecewise-linear interpolator is monotone non-decreasing in
        # the input probability by construction (sklearn's isotonic).
        logits, labels = _make_validation_set()
        cal = fit("isotonic_regression", logits, labels)
        # Sweep the per-class probability axis and confirm class-0 mapping
        # is non-decreasing.
        x_sweep = np.linspace(0.0, 1.0, 101)
        y_sweep = cal._apply_per_class(x_sweep)
        assert (np.diff(y_sweep) >= -1e-9).all()

    def test_transform_returns_valid_probability_simplex(self):
        logits, labels = _make_validation_set()
        cal = fit("isotonic_regression", logits, labels)
        probs = cal.transform(logits)
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-9)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


# ---------------------------------------------------------------------------
# Save + load round trip (R5.6 — also covered by PBT in task 4.7)
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_temperature_scaling_round_trip(self, tmp_path: Path):
        logits, labels = _make_validation_set(seed=11)
        cal = fit("temperature_scaling", logits, labels)
        out = tmp_path / "resnet18_calibrator_test.json"
        save(cal, out)

        # File exists and is valid plain-text JSON.
        text = out.read_text()
        parsed = json.loads(text)
        assert parsed["method"] == "temperature_scaling"
        assert "temperature" in parsed

        loaded = load(out)
        assert isinstance(loaded, TemperatureScalingCalibrator)
        np.testing.assert_allclose(
            loaded.transform(logits),
            cal.transform(logits),
            atol=1e-6,
        )

    def test_isotonic_regression_round_trip(self, tmp_path: Path):
        logits, labels = _make_validation_set(seed=13)
        cal = fit("isotonic_regression", logits, labels)
        out = tmp_path / "resnet18_calibrator_iso.json"
        save(cal, out)

        parsed = json.loads(out.read_text())
        assert parsed["method"] == "isotonic_regression"
        assert "breakpoints" in parsed
        # On-disk breakpoints sorted by x ascending (per R5.3 plain text + sorted).
        xs = [bp["x"] for bp in parsed["breakpoints"]]
        assert xs == sorted(xs)

        loaded = load(out)
        assert isinstance(loaded, IsotonicRegressionCalibrator)
        np.testing.assert_allclose(
            loaded.transform(logits),
            cal.transform(logits),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Missing / corrupt file fallback (R5.5)
# ---------------------------------------------------------------------------


class TestLoadFallback:
    def test_missing_file_returns_none_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        path = tmp_path / "does_not_exist.json"
        with caplog.at_level(logging.WARNING, logger="calibrate"):
            result = load(path)
        assert result is None
        assert any(
            "missing" in record.getMessage() and str(path) in record.getMessage()
            for record in caplog.records
        ), f"expected missing-file WARNING, got: {[r.getMessage() for r in caplog.records]}"

    def test_corrupt_json_returns_none_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        path = tmp_path / "corrupt.json"
        path.write_text("{this is not valid json")
        with caplog.at_level(logging.WARNING, logger="calibrate"):
            result = load(path)
        assert result is None
        assert any(
            "corrupt" in record.getMessage() and str(path) in record.getMessage()
            for record in caplog.records
        ), f"expected corrupt-file WARNING, got: {[r.getMessage() for r in caplog.records]}"

    def test_unknown_method_returns_none_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        path = tmp_path / "unknown.json"
        path.write_text(json.dumps({"method": "not_a_real_method"}))
        with caplog.at_level(logging.WARNING, logger="calibrate"):
            result = load(path)
        assert result is None
        # Falls into the from_dict branch which raises ValueError; caught
        # in load() and reported as corrupt.
        assert any("corrupt" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# from_dict dispatch
# ---------------------------------------------------------------------------


class TestFromDictDispatch:
    def test_dispatches_to_temperature_scaling(self):
        d = {"method": "temperature_scaling", "temperature": 1.5}
        cal = Calibrator.from_dict(d)
        assert isinstance(cal, TemperatureScalingCalibrator)
        assert cal.temperature == 1.5

    def test_dispatches_to_isotonic_regression(self):
        d = {
            "method": "isotonic_regression",
            "n_classes": 2,
            "breakpoints": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.4},
                {"x": 1.0, "y": 1.0},
            ],
        }
        cal = Calibrator.from_dict(d)
        assert isinstance(cal, IsotonicRegressionCalibrator)
        assert cal.n_classes == 2

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown calibration method"):
            Calibrator.from_dict({"method": "garbage"})


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestFitInputValidation:
    def test_non_2d_logits_raises(self):
        with pytest.raises(ValueError, match="2D"):
            fit("temperature_scaling", np.zeros(10), np.zeros(10, dtype=np.int64))

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            fit(
                "temperature_scaling",
                np.zeros((10, 2)),
                np.zeros(9, dtype=np.int64),
            )

    def test_label_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 2\)"):
            fit(
                "temperature_scaling",
                np.zeros((4, 2)),
                np.array([0, 1, 2, 0], dtype=np.int64),
            )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown calibration method"):
            fit("rubbish", np.zeros((4, 2)), np.zeros(4, dtype=np.int64))
