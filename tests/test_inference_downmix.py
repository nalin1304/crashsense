"""Tests for the multi-channel downmix path in ``_coerce_audio`` (R4.4).

Spatial audio features are deferred (``docs/spatial_audio_decision.md``),
so the Audio_Detector's supported channel count is one. Anything outside
that range — including stereo, quad, and 7.1 input — is downmixed to
mono via channel-wise mean and logged at INFO with the correlation id.

These tests do not require the ESC-50 corpus to be present.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from backend.audio_model.inference import _coerce_audio


# --- helpers ---------------------------------------------------------------


def _const_channels(length: int, values: list[float]) -> np.ndarray:
    """Build a (length, n_channels) array where channel i is filled with values[i]."""
    cols = [np.full(length, v, dtype=np.float32) for v in values]
    return np.stack(cols, axis=1)


# --- core downmix correctness ---------------------------------------------


class TestDownmixCorrectness:
    def test_mono_1d_passthrough(self):
        x = np.linspace(-0.5, 0.5, 1024, dtype=np.float32)
        out = _coerce_audio(x)
        assert out.ndim == 1
        assert out.dtype == np.float32
        assert out.shape == x.shape
        np.testing.assert_array_equal(out, x)

    def test_stereo_downmixes_to_channel_mean(self):
        n = 2048
        stereo = _const_channels(n, [0.2, 0.8])
        out = _coerce_audio(stereo)
        assert out.ndim == 1
        assert out.shape == (n,)
        # mean(0.2, 0.8) == 0.5
        np.testing.assert_allclose(out, np.full(n, 0.5, dtype=np.float32))

    def test_four_channel_downmixes_to_channel_mean(self):
        n = 1024
        quad = _const_channels(n, [0.0, 0.25, 0.5, 1.0])
        out = _coerce_audio(quad)
        # mean = (0 + 0.25 + 0.5 + 1.0) / 4 = 0.4375
        np.testing.assert_allclose(out, np.full(n, 0.4375, dtype=np.float32))
        assert out.shape == (n,)

    def test_eight_channel_downmixes_to_channel_mean(self):
        n = 512
        values = [0.1 * i for i in range(8)]  # 0.0, 0.1, ..., 0.7
        eight = _const_channels(n, values)
        out = _coerce_audio(eight)
        expected = float(np.mean(values))
        np.testing.assert_allclose(
            out, np.full(n, expected, dtype=np.float32), rtol=1e-6
        )
        assert out.shape == (n,)

    def test_downmix_preserves_length(self):
        # The number of samples (axis 0) must survive downmix unchanged.
        for n in (128, 1024, 22050):
            for channels in (2, 3, 4, 6, 8):
                arr = _const_channels(n, [0.5] * channels)
                out = _coerce_audio(arr)
                assert out.shape == (n,), (
                    f"downmix changed length for {channels}-channel input "
                    f"of length {n}"
                )

    def test_single_channel_2d_squeezed_without_logging(self, caplog):
        # Shape (N, 1) is effectively mono — should not emit a downmix log.
        n = 1024
        arr = np.linspace(-1.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
        with caplog.at_level(logging.INFO, logger="inference"):
            out = _coerce_audio(arr)
        assert out.shape == (n,)
        assert not any("audio_downmixed_to_mono" in m for m in caplog.messages)


# --- logging contract (R4.4) ----------------------------------------------


class TestDownmixLogging:
    def test_logs_info_with_channel_count(self, caplog):
        stereo = _const_channels(1024, [0.1, 0.2])
        with caplog.at_level(logging.INFO, logger="inference"):
            _coerce_audio(stereo)
        msgs = [m for m in caplog.messages if "audio_downmixed_to_mono" in m]
        assert len(msgs) == 1, f"expected exactly one downmix log, got {msgs}"
        assert "original_channels=2" in msgs[0]
        assert "correlation_id=" in msgs[0]

    def test_log_level_is_info(self, caplog):
        quad = _const_channels(512, [0.0, 0.0, 0.0, 0.0])
        with caplog.at_level(logging.INFO, logger="inference"):
            _coerce_audio(quad)
        records = [r for r in caplog.records if "audio_downmixed_to_mono" in r.message]
        assert records, "expected a downmix log record"
        assert all(r.levelno == logging.INFO for r in records)

    def test_log_uses_correlation_id_when_set(self, caplog):
        # When the request-scoped correlation id is set, the downmix log
        # should carry it instead of the placeholder dash.
        from backend.api.logging_config import request_id_ctx

        cid = "11111111-2222-3333-4444-555555555555"
        token = request_id_ctx.set(cid)
        try:
            with caplog.at_level(logging.INFO, logger="inference"):
                _coerce_audio(_const_channels(256, [0.0, 0.0]))
        finally:
            request_id_ctx.reset(token)

        assert any(
            "audio_downmixed_to_mono" in m and f"correlation_id={cid}" in m
            for m in caplog.messages
        )

    def test_log_uses_dash_when_correlation_id_unset(self, caplog):
        # Outside the FastAPI request scope the context var is the
        # default sentinel ("-"), and that is what the log should carry.
        with caplog.at_level(logging.INFO, logger="inference"):
            _coerce_audio(_const_channels(256, [0.0, 0.0]))
        assert any(
            "audio_downmixed_to_mono" in m and "correlation_id=-" in m
            for m in caplog.messages
        )


# --- input validation -----------------------------------------------------


class TestUnsupportedShapes:
    def test_three_dimensional_array_rejected(self):
        # 3-D input is not a supported audio shape and should still raise.
        arr = np.zeros((10, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D mono PCM"):
            _coerce_audio(arr)

    def test_unsupported_input_type_rejected(self):
        with pytest.raises(TypeError, match="unsupported audio input type"):
            _coerce_audio(123)  # type: ignore[arg-type]
