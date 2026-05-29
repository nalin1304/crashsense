"""
On-the-fly SNR-mixed dataset wrapper for the trainer (R2.3, R2.4, R6.3).

Wraps the training subset built from the pre-baked spectrograms in
``data/spectrograms/`` so each sample can optionally be re-rendered from a
noise-mixed copy of its source WAV in ``data/raw_audio/`` before being run
through the existing train transforms.

Per ``__getitem__`` call:
  1. Draw two independent Bernoulli decisions:
       * ``do_synth ~ Bernoulli(mix_probability)`` — SNR-mix path (R2.3,
         default ``p=0.5``).
       * ``do_ambient ~ Bernoulli(ambient_mix_probability)`` — ambient-mix
         path (R6.3, ``p ∈ [0.2, 0.5]``), only consulted when
         ``ambient_mix=True`` and at least one non-placeholder session is
         loaded from the ambient manifest.
  2. If either trigger fires: locate the source WAV at
     ``raw_root / <class>/<stem>.wav`` (mirroring the pre-baked spectrogram
     layout), decode it via ``librosa.load`` at the spectrogram-pipeline
     sample rate, draw ``target_snr_db`` uniformly from ``snr_choices``
     (default ``{0, 10, 20}`` dB), mix in either a synthetic highway-noise
     stem or a randomly-selected ambient-session segment via
     :func:`mix_clip`, and re-render the mel image via
     :func:`samples_to_mel_image`. When both triggers fire, ambient takes
     precedence — its richer real-world spectrum is preferred over the
     synthetic stem.
  3. Otherwise (or on failure — see below) fall through to the pre-baked
     PNG spectrogram, exactly as the unmixed trainer would have used it.

Failure handling (R2.4, R6.3): if the source WAV cannot be decoded, the
selected ambient session cannot be loaded, or the noise stem cannot be
loaded, the sample falls through to the pre-baked path and a WARNING is
logged with the offending filename. Training continues without raising.
Noise length is matched to the clean clip length by :func:`mix_clip`
itself (tile + crop), so a noise stem shorter than the clean clip is not
an error.

The noise source is one of:
  - **ambient** (R6.3): when ``ambient_mix=True`` and the ambient trigger
    fires, a random non-placeholder session is selected from the manifest
    at ``ambient_manifest`` (default ``data/ambient_long/manifest.json``)
    and a random window of the requested length is loaded via
    ``librosa.load(..., offset=..., duration=...)``. Placeholder rows
    (``is_placeholder=true``) are skipped per the contract documented in
    ``data/ambient_long/README.md``.
  - **synthetic highway noise** (default fall-back): synthesised fresh
    per call via :func:`synthesize_highway_noise` when ``noise_path`` is
    ``None``, drawing a per-call seed from the dataset's RNG so the
    synthetic stem varies across samples.
  - **fixed noise WAV**: loaded once from the WAV file at ``noise_path``
    when provided through the trainer's ``--snr-noise-stem`` flag.

The class exposes ``mix_attempts`` / ``mix_succeeded`` / ``mix_failed``
counters for SNR-mix bookkeeping and ``ambient_attempts`` /
``ambient_succeeded`` / ``ambient_failed`` for ambient-mix bookkeeping.
These are best-effort and not thread-safe across DataLoader workers (the
existing trainer uses ``num_workers=0`` so this is fine in practice).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .noise_mixing import mix_clip, synthesize_highway_noise
from .spectrogram_gen import SAMPLE_RATE, samples_to_mel_image

LOG = logging.getLogger("snr_mix_dataset")

DEFAULT_SNR_CHOICES: tuple[float, ...] = (0.0, 10.0, 20.0)
DEFAULT_MIX_PROBABILITY = 0.5

#: Mid-point of the R6.3 [0.2, 0.5] ambient sampling-probability range.
DEFAULT_AMBIENT_MIX_PROBABILITY = 0.3

#: Inclusive R6.3 bounds on the ambient sampling probability. The CLI flag
#: parser in ``train.py`` enforces these as well; we re-validate here so
#: the dataset is safe to construct from arbitrary call sites.
AMBIENT_MIX_PROBABILITY_MIN = 0.2
AMBIENT_MIX_PROBABILITY_MAX = 0.5


class SnrMixDataset(torch.utils.data.Dataset):
    """Training-side dataset that randomly SNR-mixes audio before spectrogramming."""

    def __init__(
        self,
        subset: torch.utils.data.Subset,
        train_transform: transforms.Compose,
        spec_root: str | Path,
        raw_root: str | Path,
        noise_path: str | Path | None = None,
        snr_choices: tuple[float, ...] = DEFAULT_SNR_CHOICES,
        mix_probability: float = DEFAULT_MIX_PROBABILITY,
        seed: int = 42,
        ambient_mix: bool = False,
        ambient_manifest: str | Path | None = None,
        ambient_root: str | Path | None = None,
        ambient_mix_probability: float = DEFAULT_AMBIENT_MIX_PROBABILITY,
    ) -> None:
        if not (0.0 <= mix_probability <= 1.0):
            raise ValueError(
                f"mix_probability must be in [0, 1], got {mix_probability}"
            )
        if not snr_choices:
            raise ValueError("snr_choices must contain at least one SNR value")
        if ambient_mix and not (
            AMBIENT_MIX_PROBABILITY_MIN
            <= ambient_mix_probability
            <= AMBIENT_MIX_PROBABILITY_MAX
        ):
            raise ValueError(
                f"ambient_mix_probability must be in "
                f"[{AMBIENT_MIX_PROBABILITY_MIN}, "
                f"{AMBIENT_MIX_PROBABILITY_MAX}] (R6.3), got "
                f"{ambient_mix_probability}"
            )

        self.subset = subset
        self.transform = train_transform
        self.spec_root = Path(spec_root)
        self.raw_root = Path(raw_root)
        self.noise_path = Path(noise_path) if noise_path is not None else None
        self.snr_choices = tuple(float(s) for s in snr_choices)
        self.mix_probability = float(mix_probability)
        self._rng = np.random.default_rng(seed)
        self._cached_noise: np.ndarray | None = None
        self._cached_noise_failed = False

        # Test instrumentation — counters track per-call mix outcomes.
        self.mix_attempts = 0
        self.mix_succeeded = 0
        self.mix_failed = 0

        # ----- Ambient-mix configuration (R6.3) ----------------------------
        # ``_ambient_sessions`` is the list of non-placeholder sessions
        # parsed from the ambient manifest. When the list is empty (no
        # manifest, manifest unparsable, or only placeholder rows), we log
        # an INFO line once and disable the ambient path so subsequent
        # ``__getitem__`` calls fall back to the synthetic-only flow per
        # the contract in the docstring.
        self.ambient_mix = bool(ambient_mix)
        self.ambient_mix_probability = float(ambient_mix_probability)
        self._ambient_manifest_path = (
            Path(ambient_manifest) if ambient_manifest is not None else None
        )
        self._ambient_root = (
            Path(ambient_root)
            if ambient_root is not None
            else (self._ambient_manifest_path.parent
                  if self._ambient_manifest_path is not None
                  else None)
        )
        self._ambient_sessions: list[dict] = []
        if self.ambient_mix:
            self._ambient_sessions = self._load_ambient_sessions()
            if not self._ambient_sessions:
                LOG.info(
                    "no real ambient sessions available, skipping ambient mix"
                )
                self.ambient_mix = False
        self.ambient_attempts = 0
        self.ambient_succeeded = 0
        self.ambient_failed = 0

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        spec_path, label = self._resolve_sample(idx)
        do_synth = bool(self._rng.random() < self.mix_probability)
        do_ambient = (
            self.ambient_mix
            and bool(self._rng.random() < self.ambient_mix_probability)
        )
        if do_synth or do_ambient:
            self.mix_attempts += 1
            mixed_image: Image.Image | None = None
            if do_ambient:
                self.ambient_attempts += 1
                mixed_image = self._render_mixed_image(
                    spec_path, ambient=True
                )
                if mixed_image is not None:
                    self.ambient_succeeded += 1
                else:
                    self.ambient_failed += 1
            if mixed_image is None and do_synth:
                # Either ambient was not requested for this sample, or it
                # was requested and failed — fall through to the synthetic
                # path so we still get an SNR-mixed sample whenever the
                # synth trigger fired.
                mixed_image = self._render_mixed_image(spec_path, ambient=False)
            if mixed_image is not None:
                self.mix_succeeded += 1
                return self.transform(mixed_image), label
            self.mix_failed += 1
            # Fall through to the pre-baked spectrogram on any mixing failure.
        return self._load_prebaked(spec_path), label

    # ------------------------------------------------------------------
    # Path / sample helpers
    # ------------------------------------------------------------------

    def _resolve_sample(self, idx: int) -> tuple[str, int]:
        base = self.subset.dataset  # ImageFolder
        return base.samples[self.subset.indices[idx]]

    def _spec_to_wav(self, spec_path: str) -> Path:
        """Map ``spec_root/<cls>/<stem>.png`` -> ``raw_root/<cls>/<stem>.wav``."""
        spec = Path(spec_path)
        try:
            rel = spec.relative_to(self.spec_root)
        except ValueError:
            # Fallback: assume the immediate parent is the class directory.
            rel = Path(spec.parent.name) / spec.name
        return self.raw_root / rel.with_suffix(".wav")

    # ------------------------------------------------------------------
    # Image generation paths
    # ------------------------------------------------------------------

    def _load_prebaked(self, spec_path: str) -> torch.Tensor:
        with Image.open(spec_path) as pil:
            rgb = pil.convert("RGB")
            return self.transform(rgb)

    def _render_mixed_image(
        self, spec_path: str, *, ambient: bool = False
    ) -> Image.Image | None:
        wav_path = self._spec_to_wav(spec_path)
        clean = self._load_clean_wav(wav_path)
        if clean is None:
            return None
        if ambient:
            noise = self._load_ambient_segment(clean.shape[0])
        else:
            noise = self._load_noise(clean.shape[0])
        if noise is None:
            return None
        target_snr_db = float(self._rng.choice(self.snr_choices))
        mixed = mix_clip(clean, noise, target_snr_db=target_snr_db)
        return samples_to_mel_image(mixed, sr=SAMPLE_RATE)

    def _load_clean_wav(self, wav_path: Path) -> np.ndarray | None:
        try:
            samples, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            LOG.warning(
                "snr_mix: failed to decode source WAV %s (%s: %s); "
                "falling back to pre-baked spectrogram",
                wav_path.name,
                exc.__class__.__name__,
                exc,
            )
            return None
        if samples.size == 0:
            LOG.warning(
                "snr_mix: source WAV %s decoded to zero samples; "
                "falling back to pre-baked spectrogram",
                wav_path.name,
            )
            return None
        return samples.astype(np.float32, copy=False)

    def _load_noise(self, n_samples: int) -> np.ndarray | None:
        if self.noise_path is None:
            # Synthetic stem — vary the seed per call so each mix gets its
            # own realization of pink + rumble.
            seed = int(self._rng.integers(0, 2**31 - 1))
            return synthesize_highway_noise(n_samples, sr=SAMPLE_RATE, seed=seed)

        if self._cached_noise_failed:
            return None
        if self._cached_noise is None:
            try:
                samples, _ = librosa.load(
                    str(self.noise_path), sr=SAMPLE_RATE, mono=True
                )
            except Exception as exc:
                LOG.warning(
                    "snr_mix: failed to load noise stem %s (%s: %s); "
                    "skipping SNR mixing for this run",
                    self.noise_path.name,
                    exc.__class__.__name__,
                    exc,
                )
                self._cached_noise_failed = True
                return None
            if samples.size == 0:
                LOG.warning(
                    "snr_mix: noise stem %s decoded to zero samples; "
                    "skipping SNR mixing for this run",
                    self.noise_path.name,
                )
                self._cached_noise_failed = True
                return None
            self._cached_noise = samples.astype(np.float32, copy=False)
        # mix_clip handles tile + crop to ``n_samples``.
        return self._cached_noise

    # ------------------------------------------------------------------
    # Ambient-mix helpers (R6.3, task 4.11)
    # ------------------------------------------------------------------

    def _load_ambient_sessions(self) -> list[dict]:
        """Read the ambient manifest and return the non-placeholder sessions.

        Returns an empty list (which disables the ambient path in
        ``__init__``) on any of:

        * manifest path is ``None`` or does not exist on disk,
        * manifest is unparsable JSON or has the wrong top-level shape,
        * every session is flagged ``is_placeholder=true`` (matches the
          contract documented in ``data/ambient_long/README.md``),
        * the resolved ``ambient_root`` does not exist.

        Each returned dict carries the manifest fields plus a precomputed
        absolute ``_path`` and ``_duration_seconds`` (defaults to 0.0 when
        the manifest does not provide one — used as a soft hint when
        choosing a random window offset, not as a hard length check).
        """
        path = self._ambient_manifest_path
        if path is None or not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning(
                "ambient_mix: failed to read manifest %s (%s: %s)",
                path,
                exc.__class__.__name__,
                exc,
            )
            return []
        if not isinstance(manifest, dict):
            LOG.warning(
                "ambient_mix: manifest %s is not a JSON object", path
            )
            return []
        sessions = manifest.get("sessions")
        if not isinstance(sessions, list):
            LOG.warning(
                "ambient_mix: manifest %s missing 'sessions' list", path
            )
            return []

        root = self._ambient_root or path.parent
        loaded: list[dict] = []
        for entry in sessions:
            if not isinstance(entry, dict):
                continue
            # R6.3: skip any placeholder row — these are synthetic and the
            # README forbids them feeding into a model.
            if entry.get("is_placeholder") is True:
                continue
            filename = entry.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            wav_path = (root / filename).resolve()
            duration = entry.get("duration_seconds")
            try:
                duration_f = float(duration) if duration is not None else 0.0
            except (TypeError, ValueError):
                duration_f = 0.0
            loaded.append(
                {
                    "filename": filename,
                    "_path": wav_path,
                    "_duration_seconds": duration_f,
                }
            )
        return loaded

    def _load_ambient_segment(self, n_samples: int) -> np.ndarray | None:
        """Pick a random non-placeholder ambient session and return a window.

        Tries every loaded session (in random order, without replacement)
        until one decodes successfully — this keeps a single corrupt file
        from disabling the entire ambient path. On total failure, returns
        ``None`` and emits a single WARNING per call so the caller falls
        through to the synthetic stem.

        The window offset is sampled uniformly within the available file
        length (``duration - target_seconds`` clamped to ``[0, ∞)``); the
        resulting segment is right-padded by ``mix_clip``'s tile-and-crop
        helper if it ends up shorter than requested.
        """
        if not self._ambient_sessions:
            return None
        target_seconds = float(n_samples) / float(SAMPLE_RATE)
        order = list(self._rng.permutation(len(self._ambient_sessions)))
        last_exc: Exception | None = None
        last_path: Path | None = None
        for ix in order:
            session = self._ambient_sessions[ix]
            wav_path: Path = session["_path"]
            duration: float = session["_duration_seconds"]
            # Pick a random start offset; if the manifest reported no
            # duration, start at 0 and let librosa.load consume what is
            # available.
            if duration > target_seconds + 0.01:
                offset = float(
                    self._rng.uniform(0.0, duration - target_seconds)
                )
            else:
                offset = 0.0
            try:
                samples, _ = librosa.load(
                    str(wav_path),
                    sr=SAMPLE_RATE,
                    mono=True,
                    offset=offset,
                    duration=target_seconds,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per R6.3
                last_exc = exc
                last_path = wav_path
                continue
            if samples.size == 0:
                last_exc = ValueError("zero-length decode")
                last_path = wav_path
                continue
            return samples.astype(np.float32, copy=False)

        # Every session failed to decode — log once and let the caller
        # fall back to the synthetic path.
        LOG.warning(
            "ambient_mix: all %d ambient sessions failed to decode "
            "(last attempt %s: %s)",
            len(self._ambient_sessions),
            last_path.name if last_path is not None else "<unknown>",
            last_exc,
        )
        return None
