"""
Mel spectrogram generation (Requirement 2).

Loads WAVs from data/raw_audio/<class>/<name>.wav and writes 224x224 PNGs to
data/spectrograms/<class>/<name>.png. Resamples to 22050 Hz, downmixes to
mono, pads/trims to exactly 3.0 seconds, computes a 128-band mel spectrogram
with an 8000 Hz upper bound.

Implementation notes
--------------------
We avoid matplotlib for batch generation (it is ~50x slower than PIL for
this case). Instead we render the mel power spectrogram to an array, apply
a magma-like colormap, and resize to 224x224 with PIL.

Public API:
  - audio_to_mel_spectrogram(wav_path, output_path) — single-file write
  - wav_to_mel_image(wav_path) -> PIL.Image — in-memory for inference
  - samples_to_mel_image(samples, sr) -> PIL.Image — in-memory for inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
from PIL import Image

LOG = logging.getLogger("spectrogram_gen")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_AUDIO_ROOT = REPO_ROOT / "data" / "raw_audio"
SPECTROGRAM_ROOT = REPO_ROOT / "data" / "spectrograms"

SAMPLE_RATE = 22050
N_MELS = 128
F_MAX = 8000
CLIP_DURATION_S = 3.0
TARGET_LEN = int(SAMPLE_RATE * CLIP_DURATION_S)
IMG_SIZE = 224


# Magma colormap (roughly), 8 control points -> linear interpolation.
# Saves us from importing matplotlib in the hot path.
_MAGMA_KEYPOINTS = np.array([
    [0,   0,   3],
    [27,  12,  65],
    [74,  20,  101],
    [127, 30,  120],
    [174, 50,  113],
    [218, 78,  91],
    [248, 130, 75],
    [253, 191, 110],
    [251, 252, 191],
], dtype=np.float32)


def _build_magma_lut() -> np.ndarray:
    n = _MAGMA_KEYPOINTS.shape[0]
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        idx = np.searchsorted(xs, t, side="right") - 1
        idx = max(0, min(n - 2, int(idx)))
        a = _MAGMA_KEYPOINTS[idx]
        b = _MAGMA_KEYPOINTS[idx + 1]
        frac = (t - xs[idx]) / max(1e-9, xs[idx + 1] - xs[idx])
        out[i] = np.clip(a + (b - a) * frac, 0, 255).astype(np.uint8)
    return out


_MAGMA_LUT = _build_magma_lut()


def fix_length(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] < TARGET_LEN:
        return np.pad(samples, (0, TARGET_LEN - samples.shape[0]), mode="constant")
    return samples[:TARGET_LEN]


def samples_to_mel_db(samples: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a log-power mel spectrogram as a float32 array."""
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=SAMPLE_RATE)
    samples = fix_length(samples.astype(np.float32))
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        fmax=F_MAX,
    )
    return librosa.power_to_db(mel, ref=np.max)


def _mel_to_pil(mel_db: np.ndarray, size: int = IMG_SIZE) -> Image.Image:
    # Normalize to 0..1 then to 0..255 indices into the magma LUT.
    lo = float(np.min(mel_db))
    hi = float(np.max(mel_db))
    span = hi - lo if hi > lo else 1.0
    norm = (mel_db - lo) / span
    norm = np.clip(norm, 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
    rgb = _MAGMA_LUT[idx]
    # Flip vertically so low frequencies are at the bottom (matplotlib convention).
    rgb = np.flipud(rgb)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((size, size), Image.BILINEAR)


def wav_to_mel_image(wav_path: str | Path) -> Image.Image:
    samples, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    return _mel_to_pil(samples_to_mel_db(samples, sr))


def samples_to_mel_image(samples: np.ndarray, sr: int = SAMPLE_RATE) -> Image.Image:
    return _mel_to_pil(samples_to_mel_db(samples, sr))


def audio_to_mel_spectrogram(wav_path: str | Path, output_path: str | Path) -> Path:
    """Write a 224x224 mel spectrogram PNG for the given WAV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = wav_to_mel_image(wav_path)
    img.save(output_path, format="PNG")
    return output_path


def write_spectrogram_for_wav(wav_path: Path, out_dir: Path) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (wav_path.stem + ".png")
    if out_path.exists():
        return out_path
    try:
        audio_to_mel_spectrogram(wav_path, out_path)
    except Exception as exc:
        LOG.warning("skip %s: %s", wav_path.name, exc)
        return None
    return out_path


def process_classes(classes: Iterable[str] = ("crash", "noise")) -> int:
    total = 0
    for cls in classes:
        in_dir = RAW_AUDIO_ROOT / cls
        out_dir = SPECTROGRAM_ROOT / cls
        if not in_dir.is_dir():
            LOG.warning("missing input dir %s", in_dir)
            continue
        wavs = sorted(in_dir.glob("*.wav"))
        LOG.info("class %s: %d files", cls, len(wavs))
        n_done = 0
        for wav in wavs:
            if write_spectrogram_for_wav(wav, out_dir) is not None:
                total += 1
                n_done += 1
                if n_done % 50 == 0:
                    LOG.info("  %s: %d/%d done", cls, n_done, len(wavs))
    return total


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Generate 224x224 mel spectrogram PNGs")
    parser.add_argument("--class", dest="classes", nargs="*", default=["crash", "noise"])
    args = parser.parse_args(argv)
    n = process_classes(args.classes)
    LOG.info("wrote %d spectrograms to %s", n, SPECTROGRAM_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
