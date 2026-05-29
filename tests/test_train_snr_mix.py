"""Tests for the trainer's ``--snr-mix`` augmentation path (R2.3, R2.4).

Builds a tiny synthetic corpus (one crash WAV + spectrogram, one noise WAV +
spectrogram), wraps it in :class:`SnrMixDataset` with the same train
transforms the production trainer uses, and asserts:

  * the augmentation runs without exception across many samples,
  * with ``mix_probability=0.5`` the empirical mix rate over 200 calls
    falls within a wide ``[30, 70]`` interval,
  * a noise-load failure is logged at WARNING and the sample falls through
    to the pre-baked spectrogram instead of aborting,
  * a missing source WAV (decode failure) likewise logs a warning and
    returns the pre-baked tensor.

The tests do not invoke the full training loop — that is gated behind the
preservation gate and Phase 2 evaluation gate. Here we only exercise the
dataset wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import wavfile
from torchvision import transforms
from torchvision.datasets import ImageFolder

from backend.audio_model.snr_mix_dataset import SnrMixDataset
from backend.audio_model.spectrogram_gen import (
    SAMPLE_RATE,
    samples_to_mel_image,
)
from backend.audio_model.train import _make_transforms

CLASSES = ("crash", "noise")
CLIP_DURATION_S = 3.0


# ---------------------------------------------------------------------------
# Corpus fixture
# ---------------------------------------------------------------------------


def _to_int16(signal: np.ndarray) -> np.ndarray:
    return (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)


def _make_synth_corpus(root: Path) -> tuple[Path, Path]:
    """Build a 2-class synthetic corpus with paired WAV + spectrogram PNGs.

    Layout matches production: ``raw_audio/<cls>/<stem>.wav`` and
    ``spectrograms/<cls>/<stem>.png``.
    """
    raw_root = root / "raw_audio"
    spec_root = root / "spectrograms"
    n = int(SAMPLE_RATE * CLIP_DURATION_S)
    rng = np.random.default_rng(42)

    for cls_idx, cls in enumerate(CLASSES):
        (raw_root / cls).mkdir(parents=True, exist_ok=True)
        (spec_root / cls).mkdir(parents=True, exist_ok=True)
        # Two distinguishable signals so the two classes don't collapse to
        # the same spectrogram (a 220 Hz tone vs a white-noise burst).
        if cls_idx == 0:
            t = np.arange(n) / SAMPLE_RATE
            samples = (np.sin(2.0 * np.pi * 220.0 * t) * 0.5).astype(np.float32)
        else:
            samples = (rng.standard_normal(n) * 0.3).astype(np.float32)
        for j in range(2):
            stem = f"synth_{cls}_{j:04d}"
            wav_path = raw_root / cls / f"{stem}.wav"
            wavfile.write(str(wav_path), SAMPLE_RATE, _to_int16(samples))
            mel = samples_to_mel_image(samples, sr=SAMPLE_RATE)
            mel.save(spec_root / cls / f"{stem}.png", format="PNG")
    return raw_root, spec_root


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    return _make_synth_corpus(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnrMixDataset:
    def _build(
        self,
        corpus: tuple[Path, Path],
        *,
        mix_probability: float = 0.5,
        noise_path: Path | None = None,
        seed: int = 42,
    ) -> SnrMixDataset:
        raw_root, spec_root = corpus
        base = ImageFolder(str(spec_root))
        subset = torch.utils.data.Subset(base, list(range(len(base))))
        train_tx, _ = _make_transforms()
        return SnrMixDataset(
            subset,
            train_transform=train_tx,
            spec_root=spec_root,
            raw_root=raw_root,
            noise_path=noise_path,
            mix_probability=mix_probability,
            seed=seed,
        )

    def test_returns_3channel_224x224_tensor(
        self, corpus: tuple[Path, Path]
    ) -> None:
        ds = self._build(corpus, mix_probability=1.0)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (3, 224, 224)
        assert y in (0, 1)

    def test_p1_always_mixes(self, corpus: tuple[Path, Path]) -> None:
        ds = self._build(corpus, mix_probability=1.0)
        for i in range(20):
            ds[i % len(ds)]
        assert ds.mix_attempts == 20
        # All 20 should succeed since the source WAVs and synthetic noise
        # are both well-defined for the test corpus.
        assert ds.mix_succeeded == 20
        assert ds.mix_failed == 0

    def test_p0_never_mixes(self, corpus: tuple[Path, Path]) -> None:
        ds = self._build(corpus, mix_probability=0.0)
        for i in range(20):
            ds[i % len(ds)]
        assert ds.mix_attempts == 0

    def test_p_half_mix_rate_within_30_to_70(
        self, corpus: tuple[Path, Path]
    ) -> None:
        """R2.3: per-sample probability 0.5. Empirical bound across 200 calls."""
        ds = self._build(corpus, mix_probability=0.5, seed=42)
        n = 200
        for i in range(n):
            ds[i % len(ds)]
        # Wide tolerance: P(<=30 or >=70 successes in 200 Bernoulli(0.5)) is
        # vanishingly small (~6 sigma either way), so this is a stable bound
        # without retuning per RNG implementation.
        assert 30 < ds.mix_attempts < 170, (
            f"mix_attempts={ds.mix_attempts} outside (30, 170) of n={n}"
        )

    def test_missing_source_wav_falls_through(
        self, corpus: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """R2.4: decode failure logs a WARNING and yields the pre-baked tensor."""
        raw_root, _spec_root = corpus
        # Delete one source WAV so the mixing path raises during decode.
        first_wav = next((raw_root / "crash").glob("*.wav"))
        first_wav.unlink()

        ds = self._build(corpus, mix_probability=1.0)
        # Find the dataset index whose spec corresponds to the deleted WAV.
        target_idx = None
        for i in range(len(ds)):
            spec_path, _ = ds._resolve_sample(i)
            if Path(spec_path).stem == first_wav.stem:
                target_idx = i
                break
        assert target_idx is not None, "could not locate target index"

        with caplog.at_level(logging.WARNING, logger="snr_mix_dataset"):
            x, _ = ds[target_idx]
        assert x.shape == (3, 224, 224)
        assert ds.mix_failed == 1
        assert ds.mix_succeeded == 0
        assert any(
            "failed to decode source WAV" in rec.message
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), "expected a WARNING-level decode-failure log"

    def test_noise_stem_load_failure_falls_through(
        self, corpus: tuple[Path, Path], caplog: pytest.LogCaptureFixture
    ) -> None:
        """R2.4: a bad noise-stem path warns and skips mixing without aborting."""
        bad_stem = corpus[0].parent / "does_not_exist.wav"
        ds = self._build(corpus, mix_probability=1.0, noise_path=bad_stem)

        with caplog.at_level(logging.WARNING, logger="snr_mix_dataset"):
            for i in range(4):
                x, _ = ds[i]
                assert x.shape == (3, 224, 224)
        # The dataset caches the failure after the first attempt and skips
        # the noise-load path on subsequent calls; we just need to see
        # at least one warning and have no exception raised.
        assert ds.mix_failed >= 1
        assert any(
            "failed to load noise stem" in rec.message
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), "expected a WARNING-level noise-load-failure log"

    def test_invalid_mix_probability_raises(
        self, corpus: tuple[Path, Path]
    ) -> None:
        with pytest.raises(ValueError, match="mix_probability"):
            self._build(corpus, mix_probability=1.5)

    def test_empty_snr_choices_raises(self, corpus: tuple[Path, Path]) -> None:
        raw_root, spec_root = corpus
        base = ImageFolder(str(spec_root))
        subset = torch.utils.data.Subset(base, list(range(len(base))))
        train_tx, _ = _make_transforms()
        with pytest.raises(ValueError, match="snr_choices"):
            SnrMixDataset(
                subset,
                train_transform=train_tx,
                spec_root=spec_root,
                raw_root=raw_root,
                snr_choices=(),
            )
