"""
PyTorch dataset wrapping raw WAVs for AST training.

Loads on-the-fly with librosa, resamples to 16 kHz mono, pads/trims to a
fixed window, and runs the AST feature extractor.

Reuses backend/audio_model/data_split.py for source-aware partitioning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .ast_model import AST_SAMPLE_RATE
from .data_split import load_split, source_key

LOG = logging.getLogger("ast_dataset")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_AUDIO_ROOT = REPO_ROOT / "data" / "raw_audio"

CLIP_DURATION_S = 5.0  # AST handles up to 10 s; 5 s = enough for crash + context
TARGET_SAMPLES = int(AST_SAMPLE_RATE * CLIP_DURATION_S)


def _wav_path_for_spec(spec_path: str) -> Path:
    """Recover the WAV path from the spectrogram path."""
    p = Path(spec_path)
    cls = p.parent.name           # crash | noise
    stem = p.stem
    return RAW_AUDIO_ROOT / cls / f"{stem}.wav"


def _load_audio_mono_16k(wav_path: Path) -> np.ndarray:
    samples, _ = librosa.load(str(wav_path), sr=AST_SAMPLE_RATE, mono=True)
    samples = samples.astype(np.float32)
    if samples.shape[0] < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - samples.shape[0]
        samples = np.pad(samples, (0, pad), mode="constant")
    elif samples.shape[0] > TARGET_SAMPLES:
        samples = samples[:TARGET_SAMPLES]
    return samples


class _ASTSplit(Dataset):
    """Yields (input_values_tensor, label, path) tuples ready for AST."""

    def __init__(self, indices: list[int], samples: list[tuple[str, int]],
                 feature_extractor, augment: bool = False):
        self.indices = indices
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        spec_path, label = self.samples[self.indices[idx]]
        wav_path = _wav_path_for_spec(spec_path)
        audio = _load_audio_mono_16k(wav_path)

        if self.augment:
            # Light SpecAugment-style noise on the raw waveform
            if np.random.rand() < 0.5:
                audio = audio + np.random.normal(0, 0.005, audio.shape).astype(np.float32)
            if np.random.rand() < 0.5 and audio.shape[0] > 1000:
                # Random ~50 ms gain dropout
                start = np.random.randint(0, audio.shape[0] - 800)
                audio[start:start + 800] *= np.random.uniform(0.0, 0.5)

        feats = self.feature_extractor(
            audio, sampling_rate=AST_SAMPLE_RATE, return_tensors="pt"
        )
        # input_values shape: (1, n_frames, 128)  -> squeeze the batch dim
        return feats["input_values"].squeeze(0), label, str(wav_path)


def build_loaders(feature_extractor, batch_size: int = 8, augment_train: bool = True):
    train_idx, val_idx, test_idx, samples, classes = load_split()
    train_ds = _ASTSplit(train_idx, samples, feature_extractor, augment=augment_train)
    val_ds = _ASTSplit(val_idx, samples, feature_extractor, augment=False)
    test_ds = _ASTSplit(test_idx, samples, feature_extractor, augment=False)

    from torch.utils.data import DataLoader
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        classes,
    )
