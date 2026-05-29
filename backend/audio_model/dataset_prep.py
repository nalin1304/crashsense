"""
Dataset acquisition for the CrashSense audio detector (Requirement 1).

Strategy
--------
We compose the training set from three CC-licensed sources:

1. **ESC-50** (Karol J. Piczak, CC-BY-NC) — 2,000 clips across 50 categories,
   downloaded from GitHub.
2. **UrbanSound8K** (Justin Salamon et al., CC-BY-NC) — 8,732 clips of urban
   sounds across 10 categories, mirrored on the Hugging Face datasets hub.
3. **Freesound** crash recordings — a small set of CC-BY collision audio
   pulled from the archive.org Freesound mirror.

Category mapping is chosen so that (a) the crash class contains transient
collision-like sounds, and (b) the noise class contains the kinds of
sustained mechanical / ambient sounds that would otherwise fool the model
("hard negatives"). Notably, *engine* sounds go into the NOISE class so
the classifier cannot take the lazy "vehicle = crash" shortcut.

  CRASH ← ESC-50 glass_breaking, fireworks, gun-shot from US8K,
          car_horn (US8K + ESC-50), Freesound collision recordings
  NOISE ← ESC-50 wind, rain, crackling_fire, traffic-y stuff,
          UrbanSound8K air_conditioner, engine_idling, drilling,
          jackhammer, siren, street_music (hard negatives)

Output layout:
  data/raw_audio/crash/*.wav
  data/raw_audio/noise/*.wav
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

LOG = logging.getLogger("dataset_prep")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw_audio"
CRASH_DIR = DATA_DIR / "crash"
NOISE_DIR = DATA_DIR / "noise"
CACHE_DIR = REPO_ROOT / ".cache"
MANIFEST_PATH = DATA_DIR / "_manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"

# Seed for any deterministic split / shuffling done by this module. Documented
# in design.md as the seed-42 reproducibility contract (R31.4, P16).
SPLIT_SEED = 42

# Default split fractions for the deterministic dry-run preview. These match
# the train / val / test ratios used elsewhere in the audio_model package.
SPLIT_TRAIN_FRACTION = 0.70
SPLIT_VAL_FRACTION = 0.15
# Test fraction is the remainder (1 - train - val) so the three partitions
# always cover the full file list.

MIN_CLIPS_PER_CLASS = 500
MAX_CLIPS_PER_CLASS = 4000

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip"

# ESC-50 category mapping (correct names this time)
ESC50_CRASH_CATEGORIES = ["glass_breaking", "car_horn", "fireworks"]
ESC50_NOISE_CATEGORIES = [
    "wind", "rain", "crackling_fire", "engine",
    "chainsaw", "vacuum_cleaner", "washing_machine",
]

# UrbanSound8K category mapping (10 classes, IDs 0-9)
US8K_CRASH_CLASSES = {1: "car_horn", 6: "gun_shot"}
US8K_NOISE_CLASSES = {
    0: "air_conditioner",
    4: "drilling",
    5: "engine_idling",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

# Freesound (via archive.org Freesound mirror) — curated CC-BY crash recordings.
FREESOUND_CRASH_IDS = ["332056", "332057", "332058", "332059", "332060", "332061"]

UA = {"User-Agent": "Mozilla/5.0 (CrashSense data fetcher)"}


# ------------------------------------------------------------------ helpers
def ensure_dirs() -> None:
    CRASH_DIR.mkdir(parents=True, exist_ok=True)
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def class_count(class_dir: Path) -> int:
    return len(list(class_dir.glob("*.wav")))


def normalize_wav(samples: np.ndarray, sr: int, dst: Path,
                  min_dur: float = 3.0, max_dur: float = 10.0) -> bool:
    """Write `samples` to `dst` as mono WAV with duration clamped to [3, 10] s."""
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / float(sr)
    if duration < min_dur:
        pad = int(min_dur * sr) - samples.shape[0]
        samples = np.pad(samples, (0, pad), mode="constant")
    elif duration > max_dur:
        samples = samples[: int(max_dur * sr)]
    peak = float(np.max(np.abs(samples)))
    if peak > 1e-6:
        samples = samples * (0.95 / peak)
    sf.write(str(dst), samples, sr, subtype="PCM_16")
    return True


# ------------------------------------------------------------------ ESC-50
def download_esc50() -> Path:
    target = CACHE_DIR / "esc50"
    target.mkdir(parents=True, exist_ok=True)
    extracted = list(target.glob("ESC-50-*"))
    if extracted:
        LOG.info("ESC-50 cached at %s", extracted[0])
        return extracted[0]

    LOG.info("Downloading ESC-50 (~600 MB) ...")
    resp = requests.get(ESC50_URL, timeout=300, stream=True)
    resp.raise_for_status()
    blob = io.BytesIO()
    total = 0
    last_log = time.monotonic()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        blob.write(chunk)
        total += len(chunk)
        if time.monotonic() - last_log > 5.0:
            LOG.info("downloaded %.1f MB", total / (1024 * 1024))
            last_log = time.monotonic()
    LOG.info("downloaded %.1f MB total; extracting", total / (1024 * 1024))
    with zipfile.ZipFile(blob) as zf:
        zf.extractall(target)
    extracted = list(target.glob("ESC-50-*"))
    if not extracted:
        raise RuntimeError("ESC-50 archive did not extract as expected")
    return extracted[0]


def copy_esc50_classes(esc_root: Path) -> tuple[int, int]:
    meta = esc_root / "meta" / "esc50.csv"
    audio = esc_root / "audio"
    if not meta.exists() or not audio.exists():
        raise RuntimeError(f"ESC-50 layout unexpected at {esc_root}")
    crash_set = set(ESC50_CRASH_CATEGORIES)
    noise_set = set(ESC50_NOISE_CATEGORIES)
    crash, noise = 0, 0
    with meta.open() as fh:
        for row in csv.DictReader(fh):
            wav = audio / row["filename"]
            if not wav.exists():
                continue
            try:
                data, sr = sf.read(str(wav), always_2d=False)
            except Exception as exc:
                LOG.warning("skip %s: %s", wav.name, exc)
                continue
            cat = row["category"]
            if cat in crash_set and class_count(CRASH_DIR) < MAX_CLIPS_PER_CLASS:
                normalize_wav(data, sr, CRASH_DIR / f"esc_{wav.stem}.wav")
                crash += 1
            elif cat in noise_set and class_count(NOISE_DIR) < MAX_CLIPS_PER_CLASS:
                normalize_wav(data, sr, NOISE_DIR / f"esc_{wav.stem}.wav")
                noise += 1
    return crash, noise


# ------------------------------------------------------------------ UrbanSound8K
def _hf_rows_page(offset: int, length: int = 100) -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=danavery%2Furbansound8K"
        "&config=default&split=train"
        f"&offset={offset}&length={length}"
    )
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("rows", [])


def fetch_urbansound8k(target_per_class: int = 1500, max_pages: int = 90) -> tuple[int, int]:
    """Stream UrbanSound8K via Hugging Face's datasets-server rows API.

    Each page is 100 rows; we keep going until both class targets are met
    or we hit `max_pages` (~9000 rows = full dataset). Audio URLs are
    short-lived signed URLs so we fetch them immediately.
    """
    LOG.info("Fetching UrbanSound8K via Hugging Face datasets-server ...")
    crash_added, noise_added = 0, 0
    seen_signatures: set[tuple[int, str]] = set()

    def _consume(row_dict: dict) -> None:
        nonlocal crash_added, noise_added
        cls_id = int(row_dict.get("classID", -1))
        slice_name = row_dict.get("slice_file_name", "")
        sig = (cls_id, slice_name)
        if sig in seen_signatures:
            return
        seen_signatures.add(sig)

        target_dir = None
        if cls_id in US8K_CRASH_CLASSES and class_count(CRASH_DIR) < target_per_class:
            target_dir = CRASH_DIR
            label = "crash"
        elif cls_id in US8K_NOISE_CLASSES and class_count(NOISE_DIR) < target_per_class:
            target_dir = NOISE_DIR
            label = "noise"
        else:
            return

        # The audio src is provided in the row payload
        audio_list = row_dict.get("audio") or []
        if not audio_list:
            return
        url = audio_list[0].get("src")
        if not url:
            return

        # Filename may collide across folds; include classID + slice for safety.
        dst_name = f"us8k_{cls_id}_{slice_name.replace(' ', '_')}"
        if not dst_name.endswith(".wav"):
            dst_name += ".wav"
        dst = target_dir / dst_name
        if dst.exists():
            return
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = r.read()
            data, sr = sf.read(io.BytesIO(payload), always_2d=False)
        except Exception as exc:
            LOG.debug("skip %s: %s", slice_name, exc)
            return
        normalize_wav(data, sr, dst)
        if label == "crash":
            crash_added += 1
        else:
            noise_added += 1

    for page in range(max_pages):
        offset = page * 100
        try:
            rows = _hf_rows_page(offset, 100)
        except Exception as exc:
            LOG.warning("US8K page %d failed: %s", page, exc)
            time.sleep(1.5)
            continue
        if not rows:
            break

        # Fetch the per-row audio assets in parallel
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_consume, r["row"]) for r in rows]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception:
                    pass

        if (page + 1) % 5 == 0:
            LOG.info("US8K page %d: crash=%d noise=%d",
                     page + 1, class_count(CRASH_DIR), class_count(NOISE_DIR))
        if class_count(CRASH_DIR) >= target_per_class and class_count(NOISE_DIR) >= target_per_class:
            break
    return crash_added, noise_added


# ------------------------------------------------------------------ Freesound
def fetch_freesound_collisions() -> int:
    """Pull CC-BY collision recordings from archive.org's Freesound mirror."""
    import re
    LOG.info("Fetching Freesound collision recordings via archive.org ...")
    saved = 0
    for sid in FREESOUND_CRASH_IDS:
        try:
            html = urllib.request.urlopen(
                urllib.request.Request(
                    f"https://archive.org/details/Freesound-{sid}", headers=UA
                ), timeout=20).read().decode("utf-8", errors="ignore")
        except Exception as exc:
            LOG.warning("Freesound-%s metadata: %s", sid, exc)
            continue
        pat = re.compile(rf"/download/Freesound-{sid}/[A-Za-z0-9_\-\.%]+\.(?:mp3|ogg|wav|flac)")
        files = sorted(set(pat.findall(html)))
        if not files:
            continue
        # Prefer MP3 for compactness
        choice = next((f for f in files if f.lower().endswith(".mp3")), files[0])
        url = "https://archive.org" + choice
        try:
            payload = urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=30).read()
            data, sr = sf.read(io.BytesIO(payload), always_2d=False)
        except Exception as exc:
            LOG.warning("Freesound-%s download: %s", sid, exc)
            continue
        dst = CRASH_DIR / f"freesound_{sid}.wav"
        normalize_wav(data, sr, dst)
        saved += 1
        LOG.info("  saved %s", dst.name)
    return saved


# ------------------------------------------------------------------ Augmentation
def augment_pitch_shift(samples: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import librosa
    return librosa.effects.pitch_shift(samples, sr=sr, n_steps=semitones)


def augment_time_stretch(samples: np.ndarray, rate: float) -> np.ndarray:
    import librosa
    return librosa.effects.time_stretch(samples, rate=rate)


def augment_noise(samples: np.ndarray, snr_db: float = 18.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-6:
        return samples
    noise_rms = rms / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0, noise_rms, samples.shape).astype(np.float32)
    return samples + noise


def augment_class(class_dir: Path, target: int, max_per_source: int = 4) -> int:
    """Generate augmented variants until the directory has `target` files."""
    sources = sorted(p for p in class_dir.glob("*.wav")
                     if not any(t in p.name for t in
                                ("_pitch", "_stretch", "_noise12", "_noise18")))
    if not sources:
        return 0
    LOG.info("augmenting %s -> target %d (have %d sources)",
             class_dir.name, target, len(sources))
    augmentations = [
        ("pitch+2", lambda s, sr: augment_pitch_shift(s, sr, 2)),
        ("pitch-2", lambda s, sr: augment_pitch_shift(s, sr, -2)),
        ("stretch9", lambda s, sr: augment_time_stretch(s, 0.9)),
        ("stretch11", lambda s, sr: augment_time_stretch(s, 1.1)),
        ("noise18", lambda s, sr: augment_noise(s, 18.0)),
        ("noise12", lambda s, sr: augment_noise(s, 12.0)),
    ]
    added = 0
    # Round-robin through augmentations so coverage is balanced.
    for aug_idx in range(min(max_per_source, len(augmentations))):
        if class_count(class_dir) >= target:
            break
        tag, fn = augmentations[aug_idx]
        for src in sources:
            if class_count(class_dir) >= target:
                break
            try:
                data, sr = sf.read(str(src), always_2d=False)
                if data.ndim == 2:
                    data = data.mean(axis=1)
                data = data.astype(np.float32, copy=False)
            except Exception:
                continue
            dst = class_dir / f"{src.stem}_{tag}.wav"
            if dst.exists():
                continue
            try:
                aug = fn(data, sr)
                normalize_wav(aug, sr, dst)
                added += 1
            except Exception as exc:
                LOG.debug("aug %s->%s failed: %s", src.name, tag, exc)
    return added


def balance_classes(target: int) -> None:
    crash_files = sorted(CRASH_DIR.glob("*.wav"))
    noise_files = sorted(NOISE_DIR.glob("*.wav"))
    final = min(len(crash_files), len(noise_files), target)
    if final == 0:
        return
    # Shuffle deterministically before truncating so we don't always drop the
    # same kind of clip (e.g. only augmentations).
    rng = random.Random(42)
    for files in (crash_files, noise_files):
        rng.shuffle(files)
        for extra in files[final:]:
            extra.unlink(missing_ok=True)


# ------------------------------------------------------------------ source manifest + dry run
_AUG_SUFFIXES = (
    "_pitch+2", "_pitch-2",
    "_stretch9", "_stretch11",
    "_noise12", "_noise18",
)


def _source_key_from_stem(stem: str) -> str:
    """Strip augmentation suffix from a filename stem.

    Mirrors `data_split.source_key` so dry-run splits and the trainer
    agree on which files share a source recording.
    """
    for suffix in _AUG_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _infer_source_uri(filename: str) -> str:
    """Infer a stable, unique source URI from a filename.

    Filenames in `data/raw_audio/` follow conventions established by the
    download path:
      - `esc_<original>.wav`         (ESC-50 GitHub mirror)
      - `us8k_<classID>_<slice>.wav` (UrbanSound8K via Hugging Face)
      - `freesound_<id>.wav`         (Freesound via archive.org)
      - `audioset_<videoID>.wav`     (AudioSet)

    The goal here is determinism + uniqueness, not URI fidelity. When no
    real upstream URI can be derived, we fall back to `local:<filename>`
    which still uniquely identifies the row.
    """
    stem = Path(filename).stem
    base = _source_key_from_stem(stem)

    if base.startswith("esc_"):
        rest = base[len("esc_") :]
        return f"esc50:{rest}"
    if base.startswith("us8k_"):
        rest = base[len("us8k_") :]
        return f"urbansound8k:{rest}"
    if base.startswith("freesound_"):
        rest = base[len("freesound_") :]
        return f"freesound:{rest}"
    if base.startswith("audioset_"):
        rest = base[len("audioset_") :]
        return f"audioset:{rest}"
    return f"local:{base}"


def _scan_class_dir(class_dir: Path) -> list[str]:
    """Return sorted WAV filenames in `class_dir` (basename only)."""
    if not class_dir.exists():
        return []
    return sorted(p.name for p in class_dir.glob("*.wav"))


def build_split_preview(
    crash_files: list[str],
    noise_files: list[str],
    seed: int = SPLIT_SEED,
    train_fraction: float = SPLIT_TRAIN_FRACTION,
    val_fraction: float = SPLIT_VAL_FRACTION,
) -> dict[str, dict[str, list[str]]]:
    """Compute a deterministic train/val/test split grouped by source.

    Variants of the same source recording (e.g. `clip_pitch+2`) are kept
    in the same partition to avoid leakage, matching the source-aware
    split policy in `data_split.py`.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for label, files in (("crash", crash_files), ("noise", noise_files)):
        # Group filenames by source recording.
        by_source: dict[str, list[str]] = defaultdict(list)
        for fn in files:
            by_source[_source_key_from_stem(Path(fn).stem)].append(fn)

        # Deterministic source ordering, then deterministic shuffle.
        sources = sorted(by_source.keys())
        rng = random.Random(seed)
        rng.shuffle(sources)

        n = len(sources)
        if n == 0:
            continue
        n_train = int(n * train_fraction)
        n_val = int(n * val_fraction)
        # Anything left after train + val is test.
        train_sources = sources[:n_train]
        val_sources = sources[n_train : n_train + n_val]
        test_sources = sources[n_train + n_val :]

        for src in train_sources:
            splits["train"].extend(by_source[src])
        for src in val_sources:
            splits["val"].extend(by_source[src])
        for src in test_sources:
            splits["test"].extend(by_source[src])

    # Sort within each partition so output is byte-identical across runs.
    for partition in splits:
        splits[partition] = sorted(splits[partition])

    return {label: splits[label] for label in sorted(splits.keys())}


def build_dry_run_payload(
    crash_files: list[str] | None = None,
    noise_files: list[str] | None = None,
    seed: int = SPLIT_SEED,
) -> dict:
    """Build the deterministic dry-run JSON payload (sorted at every level).

    When `crash_files` / `noise_files` are not provided, scans the on-disk
    `data/raw_audio/{crash,noise}/` directories.
    """
    if crash_files is None:
        crash_files = _scan_class_dir(CRASH_DIR)
    if noise_files is None:
        noise_files = _scan_class_dir(NOISE_DIR)

    splits = build_split_preview(crash_files, noise_files, seed=seed)

    all_files = sorted(set(crash_files) | set(noise_files))
    source_uris = sorted({_infer_source_uri(fn) for fn in all_files})

    return {
        "seed": seed,
        "source_uris": source_uris,
        "splits": splits,
    }


def emit_dry_run(stream=sys.stdout, seed: int = SPLIT_SEED) -> dict:
    """Write the deterministic dry-run JSON to `stream` and return the payload."""
    payload = build_dry_run_payload(seed=seed)
    text = json.dumps(payload, sort_keys=True, indent=2)
    stream.write(text)
    stream.write("\n")
    return payload


def write_source_manifest(
    crash_files: list[str] | None = None,
    noise_files: list[str] | None = None,
    manifest_path: Path = MANIFEST_PATH,
    generated_at: datetime | None = None,
) -> dict:
    """Emit `_manifest.json` listing every training/validation source URI.

    The output is deterministic given the same input file list: sources are
    sorted by filename and the JSON is dumped with `sort_keys=True`.
    """
    if crash_files is None:
        crash_files = _scan_class_dir(CRASH_DIR)
    if noise_files is None:
        noise_files = _scan_class_dir(NOISE_DIR)

    all_files = sorted(set(crash_files) | set(noise_files))
    sources = [
        {"filename": fn, "source_uri": _infer_source_uri(fn)}
        for fn in all_files
    ]

    if generated_at is None:
        generated_at = datetime.now(timezone.utc)
    # Render as RFC 3339 / ISO 8601 with explicit "Z" suffix.
    generated_at_iso = generated_at.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_iso8601": generated_at_iso,
        "sources": sources,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.write("\n")
    return manifest


# ------------------------------------------------------------------ entry point
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Build CrashSense crash/noise dataset")
    parser.add_argument("--target", type=int, default=2000,
                        help="Target clips per class after augmentation")
    parser.add_argument("--us8k-target", type=int, default=1500,
                        help="UrbanSound8K target per class before augmentation")
    parser.add_argument("--no-us8k", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Emit a deterministic split JSON (sorted keys) for the current "
            "on-disk crash/noise files and exit. Does not download or copy "
            "files. Output is byte-identical across consecutive invocations "
            "with the same seed (R31.4, P16)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help=f"Seed used by --dry-run split shuffling (default {SPLIT_SEED})",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        emit_dry_run(stream=sys.stdout, seed=args.seed)
        return 0

    ensure_dirs()
    started = time.monotonic()

    # 1. ESC-50 — small but dependable
    esc_root = download_esc50()
    crash, noise = copy_esc50_classes(esc_root)
    LOG.info("seeded from ESC-50: crash=%d noise=%d", crash, noise)

    # 2. Freesound real collision recordings
    fs_added = fetch_freesound_collisions()
    LOG.info("seeded from Freesound: crash=%d", fs_added)

    # 3. UrbanSound8K — the large urban-sound corpus
    if not args.no_us8k:
        u_crash, u_noise = fetch_urbansound8k(target_per_class=args.us8k_target)
        LOG.info("seeded from UrbanSound8K: crash=%d noise=%d", u_crash, u_noise)

    LOG.info("after real-data seeding: crash=%d noise=%d",
             class_count(CRASH_DIR), class_count(NOISE_DIR))

    # 4. Augmentation if we still need more
    if not args.no_augment:
        target = max(args.target, MIN_CLIPS_PER_CLASS)
        augment_class(CRASH_DIR, target)
        augment_class(NOISE_DIR, target)

    # 5. Balance to the smaller class
    balance_classes(args.target)
    final_crash = class_count(CRASH_DIR)
    final_noise = class_count(NOISE_DIR)
    LOG.info("FINAL: crash=%d noise=%d (took %.1fs)",
             final_crash, final_noise, time.monotonic() - started)

    # 6. Emit the source manifest so downstream tooling (e.g.
    #    check_real_world_disjoint.py) can verify training corpora vs the
    #    held-out real-world set without re-scanning every file (R31.5).
    manifest = write_source_manifest()
    LOG.info("wrote source manifest %s (%d sources)",
             MANIFEST_PATH, len(manifest["sources"]))

    if final_crash < MIN_CLIPS_PER_CLASS or final_noise < MIN_CLIPS_PER_CLASS:
        LOG.warning("below 500-per-class minimum (Requirement 1.4)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
