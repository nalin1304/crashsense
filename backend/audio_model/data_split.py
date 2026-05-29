"""
Source-aware train/test split for CrashSense.

The naive 80/20 random split over spectrograms leaks information: a clip
"foo.wav" can be in train and its pitch-shifted variant "foo_pitch+2.wav"
in val, so the model just needs to recognize the source. To get an honest
generalization measurement we split by SOURCE — every variant of a given
recording lives in the same partition.

Public API:
    load_split() -> (train_indices, val_indices, test_indices, samples, classes)

The split is deterministic via SEED. The held-out TEST set contains entire
source recordings never seen during training in any form.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from torchvision.datasets import ImageFolder

LOG = logging.getLogger("data_split")

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECTROGRAM_ROOT = REPO_ROOT / "data" / "spectrograms"

SEED = 42
TEST_SOURCES_PER_CLASS = 100
VAL_FRACTION_OF_REMAINDER = 0.15  # of the non-test sources

_AUG_SUFFIXES = ("_pitch+2", "_pitch-2", "_stretch9", "_stretch11",
                 "_noise12", "_noise18")


def source_key(filename: str) -> str:
    """Strip augmentation suffix from a stem to recover the original source name."""
    stem = Path(filename).stem
    for s in _AUG_SUFFIXES:
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem


def load_split() -> tuple[list[int], list[int], list[int], list[tuple[str, int]], list[str]]:
    """Build a deterministic source-aware (train, val, test) index split."""
    base = ImageFolder(str(SPECTROGRAM_ROOT))
    classes: list[str] = list(base.classes)
    samples: list[tuple[str, int]] = list(base.samples)

    # Group sample indices by (class_label, source_key)
    by_source: dict[tuple[int, str], list[int]] = defaultdict(list)
    for idx, (path, label) in enumerate(samples):
        key = source_key(path)
        by_source[(label, key)].append(idx)

    rng = random.Random(SEED)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for label, _cls_name in enumerate(classes):
        sources = sorted(k for (l, k) in by_source.keys() if l == label)
        rng.shuffle(sources)
        n_test = min(TEST_SOURCES_PER_CLASS, len(sources) // 5)
        test_sources = set(sources[:n_test])
        remaining = sources[n_test:]
        n_val = max(1, int(len(remaining) * VAL_FRACTION_OF_REMAINDER))
        val_sources = set(remaining[:n_val])
        train_sources = set(remaining[n_val:])

        for src in test_sources:
            test_idx.extend(by_source[(label, src)])
        for src in val_sources:
            val_idx.extend(by_source[(label, src)])
        for src in train_sources:
            train_idx.extend(by_source[(label, src)])

    LOG.info(
        "split: train=%d val=%d test=%d (sources held out per class: %d)",
        len(train_idx), len(val_idx), len(test_idx), TEST_SOURCES_PER_CLASS,
    )
    return train_idx, val_idx, test_idx, samples, classes
