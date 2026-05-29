"""Regenerate the **placeholder** ambient capture corpus.

This script writes three short, synthetic WAV files to
``data/ambient_long/`` so the rest of the toolchain (manifest schema,
``curate_ambient.py``, ``ambient_stats.py``) has something to read in CI.
The committed WAVs are stubs — they are not real ambient highway
recordings and must not be referenced by any preservation-gate metric.

Real recordings (≥24 cumulative hours across ≥3 sessions, R6.1) are
ingested manually via ``scripts/curate_ambient.py``; see
``data/ambient_long/README.md``.

Determinism contract
--------------------

Every stochastic generator draws through a freshly-seeded
``np.random.default_rng(42)`` and every deterministic generator uses pure
formulae over ``np.arange``. Two back-to-back runs on the same
numpy/scipy versions therefore produce **byte-identical** WAVs.

This script does **not** edit ``manifest.json`` or ``README.md``; the
manifest is hand-maintained next to the audio so reviewers can see the
``is_placeholder`` flag in diff form.

Usage::

    python data/ambient_long/_regenerate_placeholders.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Stub clips are 22050 Hz mono 16-bit so they share the canonical project
# sample rate (matches ``backend.audio_model.spectrogram_gen.SAMPLE_RATE``)
# and stay tiny (~440 KB per 10 s clip) so the repo stays light.
SAMPLE_RATE = 22050
DURATION_S = 10.0
SEED = 42
ROOT = Path(__file__).resolve().parent


def _to_int16(signal: np.ndarray) -> np.ndarray:
    """Clip to [-1, 1] and quantize to int16 PCM."""
    clipped = np.clip(signal, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, _to_int16(signal))


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink noise via 1/sqrt(f) shaping of seeded white noise."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    shape = np.zeros_like(freqs)
    shape[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * shape, n=n)
    pink = pink / max(np.max(np.abs(pink)), 1e-9)
    return pink


def _low_rumble(n: int, frequency_hz: float) -> np.ndarray:
    """Pure-tone low-frequency rumble at ``frequency_hz``."""
    t = np.arange(n) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * frequency_hz * t)


def _gen_session(seed_offset: int, rumble_hz: float) -> np.ndarray:
    """Pink noise + low rumble, normalized to peak ≈ 0.4."""
    rng = np.random.default_rng(SEED + seed_offset)
    n = int(SAMPLE_RATE * DURATION_S)
    signal = 0.7 * _pink_noise(n, rng) + 0.3 * _low_rumble(n, rumble_hz)
    signal = signal / max(np.max(np.abs(signal)), 1e-9)
    return 0.4 * signal


def main() -> None:
    # Three sessions, distinct rumble frequencies so the placeholders are
    # visually distinguishable in a spectrogram even though they share the
    # same generator family.
    _write(ROOT / "ambient_001.wav", _gen_session(seed_offset=0, rumble_hz=60.0))
    _write(ROOT / "ambient_002.wav", _gen_session(seed_offset=1, rumble_hz=80.0))
    _write(ROOT / "ambient_003.wav", _gen_session(seed_offset=2, rumble_hz=100.0))


if __name__ == "__main__":
    main()
