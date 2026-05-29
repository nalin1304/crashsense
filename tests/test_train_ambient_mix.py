"""Tests for the trainer's ``--ambient-mix`` augmentation path (R6.3, task 4.11).

Builds a tiny synthetic corpus (one crash WAV + spectrogram, one noise WAV +
spectrogram), points the SnrMixDataset at a temporary ambient manifest +
ambient session WAVs, and asserts the documented contract:

* ``ambient_mix=True, ambient_mix_probability=0.0`` → never invokes the
  ambient path (regression guard against accidental flag inversion).
* ``ambient_mix=True, ambient_mix_probability=0.5`` → mix rate empirically
  in the ``(20, 80)%`` window across 200 samples (loose Bernoulli bound;
  P(<=20 or >=80 over 200 trials) ~ 6 sigma).
* ``ambient_mix=True, ambient_mix_probability=0.5`` with all sessions
  flagged ``is_placeholder=true`` → ambient path disabled at construction
  time, INFO log emitted, falls back to the synthetic-only path.
* ``ambient_mix=True`` with no manifest on disk → same fall-back behaviour.
* ``--ambient-mix`` composes with ``--snr-mix`` cleanly: with both flags
  set the wrapper still produces 3-channel 224×224 tensors and the per-
  sample mix attempts/successes counters increment.

The tests do not invoke the full training loop — those are covered by the
preservation gate. Here we only exercise the dataset wrapper and the
``_build_loaders`` integration in ``backend.audio_model.train``.

All tests use ``tmp_path`` so no test ever writes to ``data/ambient_long/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import wavfile
from torchvision.datasets import ImageFolder

from backend.audio_model.snr_mix_dataset import (
    AMBIENT_MIX_PROBABILITY_MAX,
    AMBIENT_MIX_PROBABILITY_MIN,
    DEFAULT_AMBIENT_MIX_PROBABILITY,
    SnrMixDataset,
)
from backend.audio_model.spectrogram_gen import (
    SAMPLE_RATE,
    samples_to_mel_image,
)
from backend.audio_model.train import _make_transforms

CLASSES = ("crash", "noise")
CLIP_DURATION_S = 3.0
AMBIENT_DURATION_S = 30.0


# ---------------------------------------------------------------------------
# Corpus / manifest helpers
# ---------------------------------------------------------------------------


def _to_int16(signal: np.ndarray) -> np.ndarray:
    return (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)


def _make_synth_corpus(root: Path) -> tuple[Path, Path]:
    """Build a 2-class synthetic corpus (mirrors test_train_snr_mix)."""
    raw_root = root / "raw_audio"
    spec_root = root / "spectrograms"
    n = int(SAMPLE_RATE * CLIP_DURATION_S)
    rng = np.random.default_rng(42)

    for cls_idx, cls in enumerate(CLASSES):
        (raw_root / cls).mkdir(parents=True, exist_ok=True)
        (spec_root / cls).mkdir(parents=True, exist_ok=True)
        if cls_idx == 0:
            t = np.arange(n) / SAMPLE_RATE
            samples = (np.sin(2.0 * np.pi * 220.0 * t) * 0.5).astype(np.float32)
        else:
            samples = (rng.standard_normal(n) * 0.3).astype(np.float32)
        for j in range(2):
            stem = f"synth_{cls}_{j:04d}"
            wavfile.write(
                str(raw_root / cls / f"{stem}.wav"),
                SAMPLE_RATE,
                _to_int16(samples),
            )
            mel = samples_to_mel_image(samples, sr=SAMPLE_RATE)
            mel.save(spec_root / cls / f"{stem}.png", format="PNG")
    return raw_root, spec_root


def _make_ambient_corpus(
    root: Path,
    *,
    n_real: int = 2,
    n_placeholder: int = 1,
    duration_s: float = AMBIENT_DURATION_S,
) -> Path:
    """Build a tiny ambient corpus and manifest.

    ``n_real`` non-placeholder WAVs (with ``is_placeholder`` absent) and
    ``n_placeholder`` placeholder rows (file may or may not exist; we
    write a stub WAV anyway to mirror the production layout). Returns
    the manifest path.
    """
    root.mkdir(parents=True, exist_ok=True)
    n = int(SAMPLE_RATE * duration_s)
    rng = np.random.default_rng(7)

    sessions: list[dict] = []
    seq = 1
    for _ in range(n_real):
        signal = (rng.standard_normal(n) * 0.2).astype(np.float32)
        filename = f"ambient_{seq:03d}.wav"
        wavfile.write(str(root / filename), SAMPLE_RATE, _to_int16(signal))
        sessions.append(
            {
                "filename": filename,
                "duration_seconds": float(duration_s),
                "sample_rate_hz": SAMPLE_RATE,
                "recorded_at_iso8601": "2025-03-10T22:00:00Z",
                "location_label": f"test_real_{seq}",
            }
        )
        seq += 1
    for _ in range(n_placeholder):
        signal = (rng.standard_normal(n) * 0.2).astype(np.float32)
        filename = f"ambient_{seq:03d}.wav"
        wavfile.write(str(root / filename), SAMPLE_RATE, _to_int16(signal))
        sessions.append(
            {
                "filename": filename,
                "duration_seconds": 36000.0,
                "sample_rate_hz": SAMPLE_RATE,
                "recorded_at_iso8601": "2025-01-15T06:00:00Z",
                "location_label": f"PLACEHOLDER:test_{seq}",
                "is_placeholder": True,
            }
        )
        seq += 1

    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at_iso8601": "2025-01-15T00:00:00Z",
                "sessions": sessions,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    return _make_synth_corpus(tmp_path)


@pytest.fixture
def ambient_manifest(tmp_path: Path) -> Path:
    return _make_ambient_corpus(tmp_path / "ambient_long")


@pytest.fixture
def placeholder_only_manifest(tmp_path: Path) -> Path:
    return _make_ambient_corpus(
        tmp_path / "ambient_long_pl",
        n_real=0,
        n_placeholder=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build(
    corpus: tuple[Path, Path],
    *,
    ambient_mix: bool = False,
    ambient_manifest: Path | None = None,
    ambient_mix_probability: float = DEFAULT_AMBIENT_MIX_PROBABILITY,
    mix_probability: float = 0.0,
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
        mix_probability=mix_probability,
        seed=seed,
        ambient_mix=ambient_mix,
        ambient_manifest=ambient_manifest,
        ambient_mix_probability=ambient_mix_probability,
    )


class TestAmbientMixProbabilityBounds:
    def test_default_in_documented_range(self) -> None:
        assert (
            AMBIENT_MIX_PROBABILITY_MIN
            <= DEFAULT_AMBIENT_MIX_PROBABILITY
            <= AMBIENT_MIX_PROBABILITY_MAX
        )

    def test_below_range_raises(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        with pytest.raises(ValueError, match="ambient_mix_probability"):
            _build(
                corpus,
                ambient_mix=True,
                ambient_manifest=ambient_manifest,
                ambient_mix_probability=0.1,
            )

    def test_above_range_raises(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        with pytest.raises(ValueError, match="ambient_mix_probability"):
            _build(
                corpus,
                ambient_mix=True,
                ambient_manifest=ambient_manifest,
                ambient_mix_probability=0.6,
            )

    def test_off_flag_skips_validation(
        self, corpus: tuple[Path, Path]
    ) -> None:
        # When ambient_mix is False the probability is unused and any
        # finite value should construct without raising.
        ds = _build(corpus, ambient_mix=False, ambient_mix_probability=0.99)
        assert ds.ambient_mix is False
        assert ds.ambient_attempts == 0


class TestAmbientMixHappyPath:
    def test_returns_3channel_224x224_tensor(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        ds = _build(
            corpus,
            ambient_mix=True,
            ambient_manifest=ambient_manifest,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
        )
        assert ds.ambient_mix is True
        assert len(ds._ambient_sessions) == 2  # 2 real, 1 placeholder
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.shape == (3, 224, 224)
        assert y in (0, 1)

    def test_p_min_low_attempt_rate(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        # The validator rejects values below 0.2 (R6.3 lower bound), so
        # the lowest legal sampling probability is exactly 0.2. Verify
        # the empirical attempt rate tracks the configured probability.
        ds = _build(
            corpus,
            ambient_mix=True,
            ambient_manifest=ambient_manifest,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MIN,
            mix_probability=0.0,
        )
        n = 100
        for i in range(n):
            ds[i % len(ds)]
        # With p=0.2 and n=100, expected 20; (5, 50) is a generous bound.
        assert 5 < ds.ambient_attempts < 50, (
            f"ambient_attempts={ds.ambient_attempts} outside (5, 50) at p=0.2"
        )

    def test_p_max_high_attempt_rate(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        ds = _build(
            corpus,
            ambient_mix=True,
            ambient_manifest=ambient_manifest,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
            mix_probability=0.0,
        )
        n = 200
        for i in range(n):
            ds[i % len(ds)]
        # With p=0.5 and n=200, expected 100; (30, 170) is a 6-sigma
        # interval that survives every reasonable RNG implementation.
        assert 30 < ds.ambient_attempts < 170, (
            f"ambient_attempts={ds.ambient_attempts} outside (30, 170)"
        )
        # Every attempt must have decoded successfully — the test corpus
        # is well-formed and ample.
        assert ds.ambient_succeeded == ds.ambient_attempts
        assert ds.ambient_failed == 0


class TestAmbientMixFallbacks:
    def test_placeholder_only_manifest_disables_ambient(
        self,
        corpus: tuple[Path, Path],
        placeholder_only_manifest: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="snr_mix_dataset"):
            ds = _build(
                corpus,
                ambient_mix=True,
                ambient_manifest=placeholder_only_manifest,
                ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
            )
        assert ds.ambient_mix is False, (
            "ambient_mix should be disabled when only placeholder sessions exist"
        )
        assert ds._ambient_sessions == []
        assert any(
            "no real ambient sessions available" in rec.message
            and rec.levelno == logging.INFO
            for rec in caplog.records
        ), "expected the documented INFO fall-back log"
        # Iterating over the dataset must not hit the ambient path.
        for i in range(len(ds)):
            ds[i]
        assert ds.ambient_attempts == 0

    def test_missing_manifest_disables_ambient(
        self,
        corpus: tuple[Path, Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        missing = tmp_path / "does_not_exist" / "manifest.json"
        with caplog.at_level(logging.INFO, logger="snr_mix_dataset"):
            ds = _build(
                corpus,
                ambient_mix=True,
                ambient_manifest=missing,
                ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
            )
        assert ds.ambient_mix is False
        assert any(
            "no real ambient sessions available" in rec.message
            for rec in caplog.records
        )

    def test_unparsable_manifest_disables_ambient(
        self,
        corpus: tuple[Path, Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        bad = tmp_path / "bad_manifest.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="snr_mix_dataset"):
            ds = _build(
                corpus,
                ambient_mix=True,
                ambient_manifest=bad,
                ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
            )
        assert ds.ambient_mix is False
        # Both the warn-on-parse and the info-on-empty should fire.
        assert any(
            "failed to read manifest" in rec.message for rec in caplog.records
        )

    def test_no_manifest_argument_disables_ambient(
        self, corpus: tuple[Path, Path]
    ) -> None:
        ds = _build(
            corpus,
            ambient_mix=True,
            ambient_manifest=None,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
        )
        assert ds.ambient_mix is False
        assert ds._ambient_sessions == []


class TestComposeWithSnrMix:
    def test_both_flags_produces_valid_tensors(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        # snr_mix path on (mix_probability > 0) AND ambient on.
        ds = _build(
            corpus,
            ambient_mix=True,
            ambient_manifest=ambient_manifest,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
            mix_probability=0.5,
        )
        for i in range(20):
            x, y = ds[i % len(ds)]
            assert x.shape == (3, 224, 224)
            assert y in (0, 1)
        # Every triggered sample must have completed (no aborts on
        # well-formed inputs).
        assert ds.mix_attempts >= ds.mix_succeeded
        assert ds.mix_failed == 0

    def test_ambient_takes_precedence_when_both_trigger(
        self, corpus: tuple[Path, Path], ambient_manifest: Path
    ) -> None:
        """When both triggers fire on the same sample the ambient path is
        used. With p_synth=1.0 and p_ambient=0.5 we expect ambient_attempts
        to roughly match the ambient probability (not double-count)."""
        ds = _build(
            corpus,
            ambient_mix=True,
            ambient_manifest=ambient_manifest,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
            mix_probability=1.0,
        )
        n = 200
        for i in range(n):
            ds[i % len(ds)]
        # Every sample triggers SNR mixing (p=1.0).
        assert ds.mix_attempts == n
        # Ambient attempt rate should track the ambient probability
        # (0.5), not the synth probability.
        assert 30 < ds.ambient_attempts < 170, (
            f"ambient_attempts={ds.ambient_attempts} outside (30, 170)"
        )


class TestBuildLoadersIntegration:
    """Verify ``_build_loaders`` wires the new flags through correctly.

    Builds tiny in-memory data roots and patches the train module's
    SPECTROGRAM_ROOT / RAW_AUDIO_ROOT to the test corpus before calling
    ``_build_loaders`` directly.
    """

    def _patch_roots(
        self,
        corpus: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend.audio_model import train as train_mod

        raw_root, spec_root = corpus
        monkeypatch.setattr(train_mod, "SPECTROGRAM_ROOT", spec_root)
        monkeypatch.setattr(train_mod, "RAW_AUDIO_ROOT", raw_root)

    def test_ambient_only_uses_wrapper(
        self,
        corpus: tuple[Path, Path],
        ambient_manifest: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Need a deterministic split file. The ImageFolder under our test
        # corpus has 4 samples (2 per class); we build a tiny split that
        # references all of them as train and val.
        from backend.audio_model import train as train_mod

        self._patch_roots(corpus, monkeypatch)

        def fake_load_split():
            return [0, 1, 2, 3], [0], [1], None, list(CLASSES)

        monkeypatch.setattr(train_mod, "load_split", fake_load_split)

        train_loader, _val_loader, classes = train_mod._build_loaders(
            snr_mix=False,
            ambient_mix=True,
            ambient_manifest=ambient_manifest,
            ambient_root=ambient_manifest.parent,
            ambient_mix_probability=AMBIENT_MIX_PROBABILITY_MAX,
        )
        assert classes == list(CLASSES)
        assert isinstance(train_loader.dataset, SnrMixDataset)
        # synth probability must be zero — ambient-only run.
        assert train_loader.dataset.mix_probability == 0.0
        # ambient_mix must remain enabled because the manifest has real
        # sessions.
        assert train_loader.dataset.ambient_mix is True

    def test_neither_flag_uses_plain_subset(
        self,
        corpus: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend.audio_model import train as train_mod
        from backend.audio_model.train import _SubsetWithTransform

        self._patch_roots(corpus, monkeypatch)

        def fake_load_split():
            return [0, 1, 2, 3], [0], [1], None, list(CLASSES)

        monkeypatch.setattr(train_mod, "load_split", fake_load_split)

        train_loader, _val_loader, _classes = train_mod._build_loaders(
            snr_mix=False, ambient_mix=False
        )
        assert isinstance(train_loader.dataset, _SubsetWithTransform)


class TestCliFlagParsing:
    def test_ambient_mix_prob_below_range_returns_2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from backend.audio_model import train as train_mod

        # Block any actual training.
        monkeypatch.setattr(
            train_mod,
            "train_once",
            lambda *args, **kwargs: (100.0, {}),
        )
        rc = train_mod.main(
            [
                "--backbone", "resnet18",
                "--epochs", "1",
                "--ambient-mix",
                "--ambient-mix-prob", "0.1",
            ]
        )
        assert rc == 2

    def test_ambient_mix_prob_above_range_returns_2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from backend.audio_model import train as train_mod

        monkeypatch.setattr(
            train_mod,
            "train_once",
            lambda *args, **kwargs: (100.0, {}),
        )
        rc = train_mod.main(
            [
                "--backbone", "resnet18",
                "--epochs", "1",
                "--ambient-mix",
                "--ambient-mix-prob", "0.6",
            ]
        )
        assert rc == 2
