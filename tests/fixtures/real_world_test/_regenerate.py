"""Regenerate the CI smoke fixture corpus.

Produces a small set of synthetic WAV files used to exercise the
``evaluate_real_world.py`` I/O path in CI.  Output is **deterministic** and
byte-identical across runs on the same numpy/scipy versions: every generator
that draws random samples does so through a freshly-seeded
``np.random.default_rng(42)``, and every deterministic generator uses pure
formulae over ``np.arange``.

Usage::

    python tests/fixtures/real_world_test/_regenerate.py

This script does **not** edit ``labels.csv`` or ``README.md``; those are
hand-maintained alongside the audio so reviewers can see the intentionally
invalid rows in diff form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050  # Hz, mono, 16-bit PCM (matches spectrogram_gen.SAMPLE_RATE)
SEED = 42
ROOT = Path(__file__).resolve().parent


def _to_int16(signal: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and quantize to int16 PCM."""
    clipped = np.clip(signal, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, _to_int16(signal))


def _gen_crash_0001() -> np.ndarray:
    """5 s linear sine sweep 200 Hz -> 4000 Hz with attack-decay envelope."""
    duration_s = 5.0
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    f0, f1 = 200.0, 4000.0
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / duration_s)
    sweep = np.sin(phase)
    env = np.ones(n)
    rise = int(0.1 * SAMPLE_RATE)
    decay = int(0.5 * SAMPLE_RATE)
    env[:rise] = np.linspace(0.0, 1.0, rise)
    env[-decay:] = np.linspace(1.0, 0.0, decay)
    return 0.6 * sweep * env


def _gen_crash_0002() -> np.ndarray:
    """3 s white-noise burst, normalized to 0.5 peak."""
    rng = np.random.default_rng(SEED)
    n = int(SAMPLE_RATE * 3.0)
    noise = rng.standard_normal(n)
    noise = noise / np.max(np.abs(noise))
    return 0.5 * noise


def _gen_noise_0001() -> np.ndarray:
    """4 s pink noise via 1/sqrt(f) spectral shaping of seeded white noise."""
    rng = np.random.default_rng(SEED)
    n = int(SAMPLE_RATE * 4.0)
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    shape = np.zeros_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * shape, n=n)
    pink = pink / np.max(np.abs(pink))
    return 0.4 * pink


def _gen_noise_0002() -> np.ndarray:
    """6 s near-silence with two low-amplitude tone bursts."""
    duration_s = 6.0
    n = int(SAMPLE_RATE * duration_s)
    out = np.zeros(n)
    # Burst 1: 440 Hz tone, 0.3 s starting at 1.5 s
    b1_start = int(1.5 * SAMPLE_RATE)
    b1_len = int(0.3 * SAMPLE_RATE)
    t1 = np.arange(b1_len) / SAMPLE_RATE
    out[b1_start : b1_start + b1_len] = 0.05 * np.sin(2.0 * np.pi * 440.0 * t1)
    # Burst 2: 660 Hz tone, 0.2 s starting at 4.0 s
    b2_start = int(4.0 * SAMPLE_RATE)
    b2_len = int(0.2 * SAMPLE_RATE)
    t2 = np.arange(b2_len) / SAMPLE_RATE
    out[b2_start : b2_start + b2_len] = 0.05 * np.sin(2.0 * np.pi * 660.0 * t2)
    return out


def _gen_crash_short() -> np.ndarray:
    """1.5 s sine tone -- intentionally below the 3.0 s minimum.

    Referenced from ``labels.csv`` to exercise the ``duration_out_of_range``
    error path in ``evaluate_real_world.py``.
    """
    n = int(SAMPLE_RATE * 1.5)
    t = np.arange(n) / SAMPLE_RATE
    return 0.4 * np.sin(2.0 * np.pi * 880.0 * t)


def main() -> None:
    _write(ROOT / "crash" / "rw_crash_0001.wav", _gen_crash_0001())
    _write(ROOT / "crash" / "rw_crash_0002.wav", _gen_crash_0002())
    _write(ROOT / "noise" / "rw_noise_0001.wav", _gen_noise_0001())
    _write(ROOT / "noise" / "rw_noise_0002.wav", _gen_noise_0002())
    _write(ROOT / "crash" / "rw_crash_short.wav", _gen_crash_short())


if __name__ == "__main__":
    main()
