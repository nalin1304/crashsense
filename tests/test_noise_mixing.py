"""Tests for highway noise synthesis + SNR mixing."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from backend.audio_model.noise_mixing import (
    mix_at_snr,
    mix_with_highway_noise,
    random_snr_db,
    synthesize_highway_noise,
)


class TestSynthesize:
    def test_returns_correct_length(self):
        out = synthesize_highway_noise(22050 * 3, sr=22050, seed=0)
        assert out.shape == (22050 * 3,)
        assert out.dtype == np.float32

    def test_normalized_within_unit_range(self):
        out = synthesize_highway_noise(22050, seed=42)
        assert np.max(np.abs(out)) <= 1.0

    def test_deterministic_with_seed(self):
        a = synthesize_highway_noise(22050, seed=7)
        b = synthesize_highway_noise(22050, seed=7)
        assert np.array_equal(a, b)


class TestMixAtSnr:
    def _signal(self) -> np.ndarray:
        # Pure tone at 1kHz, 22050 Hz sr, 1 second
        t = np.arange(22050) / 22050.0
        return np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5

    def test_higher_snr_means_more_signal(self):
        sig = self._signal()
        noise = synthesize_highway_noise(sig.shape[0], seed=1)
        loud = mix_at_snr(sig, noise, snr_db=30.0)
        quiet = mix_at_snr(sig, noise, snr_db=0.0)
        # At 30 dB SNR the noise is barely audible; at 0 dB they're equal.
        # Compare difference from clean signal
        loud_dist = float(np.mean((loud - sig) ** 2))
        quiet_dist = float(np.mean((quiet - sig) ** 2))
        assert loud_dist < quiet_dist

    def test_no_clipping_after_mix(self):
        sig = self._signal()
        noise = synthesize_highway_noise(sig.shape[0], seed=1)
        for snr in (-10, 0, 10, 30):
            mixed = mix_at_snr(sig, noise, snr_db=snr)
            assert np.max(np.abs(mixed)) <= 1.0

    def test_noise_tiled_or_trimmed_to_signal_length(self):
        sig = self._signal()
        short_noise = synthesize_highway_noise(sig.shape[0] // 3, seed=1)
        out = mix_at_snr(sig, short_noise, snr_db=10.0)
        assert out.shape == sig.shape

    def test_mix_with_highway_noise_changes_signal(self):
        sig = self._signal()
        out = mix_with_highway_noise(sig, snr_db=10.0, seed=42)
        assert out.shape == sig.shape
        assert not np.allclose(sig, out)


class TestRandomSnr:
    @given(st.integers(min_value=0, max_value=1_000_000))
    @settings(max_examples=20, deadline=None)
    def test_within_range(self, seed):
        rng = np.random.default_rng(seed)
        snr = random_snr_db(rng, low=0.0, high=25.0)
        assert 0.0 <= snr <= 25.0


class TestMixClip:
    """Unit tests for the public :func:`mix_clip` surface introduced in R2.1."""

    @staticmethod
    def _clean(seed: int = 0) -> np.ndarray:
        # 1 second of a 1 kHz tone at 22050 Hz, low amplitude so even a 0 dB
        # mix stays well under the 0.99 soft-limit ceiling. Keeping the limiter
        # disengaged lets us recover the scaled noise from ``mixed - clean``.
        t = np.arange(22050) / 22050.0
        return (np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.1)

    @staticmethod
    def _measured_snr_db(clean: np.ndarray, mixed: np.ndarray) -> float:
        # Recover scaled noise from the mix and compute power-based SNR.
        noise = mixed.astype(np.float64) - clean.astype(np.float64)
        p_clean = float(np.mean(clean.astype(np.float64) ** 2))
        p_noise = float(np.mean(noise ** 2))
        return 10.0 * np.log10(p_clean / max(p_noise, 1e-30))

    @pytest.mark.parametrize("target_snr_db", [0.0, 10.0, 20.0])
    def test_measured_snr_within_half_db(self, target_snr_db: float) -> None:
        from backend.audio_model.noise_mixing import mix_clip

        clean = self._clean()
        noise = synthesize_highway_noise(clean.shape[0], seed=1)
        mixed = mix_clip(clean, noise, target_snr_db=target_snr_db)
        measured = self._measured_snr_db(clean, mixed)
        assert abs(measured - target_snr_db) <= 0.5, (
            f"measured SNR {measured:.3f} dB outside ±0.5 dB of "
            f"target {target_snr_db} dB"
        )

    def test_silent_clean_returns_clean_unchanged(self) -> None:
        from backend.audio_model.noise_mixing import mix_clip

        clean = np.zeros(22050, dtype=np.float32)
        noise = synthesize_highway_noise(clean.shape[0], seed=2)
        out = mix_clip(clean, noise, target_snr_db=10.0)
        assert out.dtype == clean.dtype
        assert out.shape == clean.shape
        assert np.array_equal(out, clean)

    def test_silent_noise_returns_clean_unchanged(self) -> None:
        from backend.audio_model.noise_mixing import mix_clip

        clean = self._clean()
        silent_noise = np.zeros_like(clean)
        out = mix_clip(clean, silent_noise, target_snr_db=10.0)
        assert out.dtype == clean.dtype
        assert out.shape == clean.shape
        assert np.array_equal(out, clean)

    def test_preserves_dtype_and_length(self) -> None:
        from backend.audio_model.noise_mixing import mix_clip

        clean = self._clean()
        noise = synthesize_highway_noise(clean.shape[0] // 2, seed=3)
        out = mix_clip(clean, noise, target_snr_db=10.0)
        assert out.dtype == np.float32
        assert out.shape == clean.shape

    def test_no_clipping_after_mix(self) -> None:
        from backend.audio_model.noise_mixing import mix_clip

        clean = self._clean()
        noise = synthesize_highway_noise(clean.shape[0], seed=4)
        for snr in (-10.0, 0.0, 10.0, 30.0):
            mixed = mix_clip(clean, noise, target_snr_db=snr)
            assert np.max(np.abs(mixed)) <= 1.0
