"""
Synthesize highway road noise and mix it into training/eval samples at varied SNR.

Real-world deployment scenario: a sensor mounted on a highway captures
crashes embedded in 75-85 dB road roar (tire/wind/distant traffic). Our
training data is mostly clean recordings of the event of interest, so the
domain gap is large.

Public surface (R2.1):
  * :func:`mix_clip` — mix arbitrary clean+noise signals at a target SNR
    using power-based scaling, returning an array with the same dtype,
    sample rate (implicit), channel count, and length as ``clean_signal``.
  * :func:`synthesize_highway_noise` — procedurally generate a highway-noise
    stem of the requested length.
  * :func:`random_snr_db` — uniformly sample an SNR from a configurable range.

The legacy helpers ``mix_with_highway_noise`` and ``mix_at_snr`` remain
importable so existing callers keep working, but they are thin wrappers
around :func:`mix_clip` and the internal length-matching helper. New code
should depend on :func:`mix_clip` only.

Used by:
  * mix_dataset.py — bake mixed variants into data/raw_audio_mixed/
  * SNRMixupDataset — on-the-fly mixing during training
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("noise_mixing")

REPO_ROOT = Path(__file__).resolve().parents[2]
NOISE_DIR = REPO_ROOT / "data" / "raw_audio" / "noise"

# Power below this threshold (in float64) is treated as silence. Equivalent
# to an RMS of ~3e-5, which is well below 16-bit PCM quantization noise.
_SILENCE_POWER_EPS = 1e-9

# Soft peak limiter ceiling — keeps mixed output strictly below full scale.
_PEAK_CEILING = 0.99

__all__ = ["mix_clip", "synthesize_highway_noise", "random_snr_db"]


# ---------------------------------------------------------------------------
# Synthetic highway noise (public)
# ---------------------------------------------------------------------------


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink noise (1/f spectrum) of length ``n``."""
    # Method: shape white noise in the frequency domain.
    n_fft = int(2 ** np.ceil(np.log2(max(n, 256))))
    white = rng.standard_normal(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    freqs[0] = 1e-6  # avoid div-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_fft).astype(np.float32)[:n]
    return pink / (np.max(np.abs(pink)) + 1e-9)


def _highway_rumble(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency rumble component centered around 80-200 Hz."""
    t = np.arange(n, dtype=np.float32) / sr
    f1, f2, f3 = (
        rng.uniform(60, 120),
        rng.uniform(140, 220),
        rng.uniform(280, 420),
    )
    rumble = (
        np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi)) * 0.5
        + np.sin(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi)) * 0.3
        + np.sin(2 * np.pi * f3 * t + rng.uniform(0, 2 * np.pi)) * 0.2
    )
    # Slow amplitude envelope (vehicles passing)
    env = 0.7 + 0.3 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
    return (rumble * env).astype(np.float32)


def synthesize_highway_noise(
    n_samples: int, sr: int = 22050, seed: int | None = None
) -> np.ndarray:
    """Procedurally generate ``n_samples`` of highway road noise."""
    rng = np.random.default_rng(seed)
    pink = _pink_noise(n_samples, rng) * 0.6
    rumble = _highway_rumble(n_samples, sr, rng) * 0.4
    bg = pink + rumble
    return (bg / (np.max(np.abs(bg)) + 1e-9)).astype(np.float32)


def random_snr_db(
    rng: np.random.Generator | None = None,
    low: float = 0.0,
    high: float = 25.0,
) -> float:
    """Random SNR drawn uniformly within a realistic operating range."""
    rng = rng or np.random.default_rng()
    return float(rng.uniform(low, high))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _match_length(noise: np.ndarray, n: int) -> np.ndarray:
    """Tile (then crop) ``noise`` along axis 0 so its length equals ``n``."""
    if noise.shape[0] == n:
        return noise
    if noise.shape[0] < n:
        reps = n // noise.shape[0] + 1
        if noise.ndim == 1:
            noise = np.tile(noise, reps)
        else:
            tile_shape = (reps,) + (1,) * (noise.ndim - 1)
            noise = np.tile(noise, tile_shape)
    return noise[:n]


def _signal_power(x: np.ndarray) -> float:
    """Mean-square power computed in float64 for numerical stability."""
    return float(np.mean(x.astype(np.float64) ** 2))


# ---------------------------------------------------------------------------
# Public mixer (R2.1, R2.2)
# ---------------------------------------------------------------------------


def mix_clip(
    clean_signal: np.ndarray,
    noise_signal: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """Mix ``noise_signal`` into ``clean_signal`` at the target SNR (dB).

    Power-based scaling: the noise is scaled so that

        10 * log10(power(clean_signal) / power(scaled_noise)) ≈ target_snr_db

    with the mixed output clipped softly to ``±0.99`` full-scale to avoid
    integer-PCM wraparound when the result is later quantized. The peak
    limiter scales clean and noise by the same factor, so the output SNR
    is preserved within ±0.5 dB of ``target_snr_db`` whenever both inputs
    are non-silent (R2.2).

    Args:
        clean_signal: 1-D (or N-D) numerical array. Acts as the reference
            for dtype, length, and channel count.
        noise_signal: noise stem to mix in. Tiled/cropped along axis 0 to
            match ``clean_signal.shape[0]``.
        target_snr_db: desired signal-to-noise ratio in decibels.

    Returns:
        ``np.ndarray`` with the same dtype, shape, and length as
        ``clean_signal``. If either input has effectively zero power
        (treated as silence), a copy of ``clean_signal`` is returned
        unchanged.
    """
    if clean_signal.ndim == 0:
        raise ValueError("clean_signal must be at least 1-D")
    if noise_signal.ndim == 0:
        raise ValueError("noise_signal must be at least 1-D")

    original_dtype = clean_signal.dtype
    clean_f64 = clean_signal.astype(np.float64, copy=False)

    # Silent clean → return clean unchanged (defensive copy).
    clean_power = _signal_power(clean_f64)
    if clean_power <= _SILENCE_POWER_EPS:
        return np.array(clean_signal, dtype=original_dtype, copy=True)

    # Tile / crop noise so its length axis matches the clean signal.
    matched_noise = _match_length(noise_signal, clean_signal.shape[0])
    noise_f64 = matched_noise.astype(np.float64, copy=False)

    noise_power = _signal_power(noise_f64)
    if noise_power <= _SILENCE_POWER_EPS:
        return np.array(clean_signal, dtype=original_dtype, copy=True)

    # Power-based scaling: scale^2 * P_noise == P_clean / 10^(snr/10)
    target_noise_power = clean_power / (10.0 ** (target_snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))

    mixed = clean_f64 + noise_f64 * scale

    # Soft peak limiter — applies the same factor to clean and noise so the
    # resulting SNR is unchanged.
    peak = float(np.max(np.abs(mixed)))
    if peak > _PEAK_CEILING:
        mixed = mixed * (_PEAK_CEILING / peak)

    return mixed.astype(original_dtype, copy=False)


# ---------------------------------------------------------------------------
# Backward-compatible shims (kept importable for legacy callers; not in
# ``__all__``). New code should depend on :func:`mix_clip` only.
# ---------------------------------------------------------------------------


def mix_at_snr(
    signal: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    """Legacy alias for :func:`mix_clip`. Kept for backward compatibility."""
    return mix_clip(signal, noise, snr_db)


def mix_with_highway_noise(
    samples: np.ndarray,
    sr: int = 22050,
    snr_db: float = 15.0,
    seed: int | None = None,
) -> np.ndarray:
    """Mix synthetic highway noise into ``samples`` at target SNR.

    Lower SNR = noisier (harder). Real highways register around 5-15 dB SNR
    for impulsive crash events; sensors close to the road can be at 0 dB.
    """
    bg = synthesize_highway_noise(samples.shape[0], sr=sr, seed=seed)
    return mix_clip(samples, bg, snr_db)
