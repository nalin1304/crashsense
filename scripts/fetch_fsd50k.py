#!/usr/bin/env python3
"""
Fetch crash-relevant audio from FSD50K (Fhrozen/FSD50k mirror on HF).

FSD50K is the gold-standard audio dataset for sound event recognition:
50,000 Freesound clips (CC-licensed) with strong human annotations using
the AudioSet ontology vocabulary.

We fetch from the dev (training) and eval (test) splits, keeping only clips
whose labels intersect our crash/noise vocabularies and don't overlap both.

Usage:
    python scripts/fetch_fsd50k.py --target-crash 1500 --target-noise 1500
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

LOG = logging.getLogger("fetch_fsd50k")

REPO_ROOT = Path(__file__).resolve().parents[1]
CRASH_DIR = REPO_ROOT / "data" / "raw_audio" / "crash"
NOISE_DIR = REPO_ROOT / "data" / "raw_audio" / "noise"

UA = {"User-Agent": "Mozilla/5.0 (CrashSense FSD50K fetcher)"}
HF_BASE = "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main"

CRASH_LABELS = {
    "Glass", "Crash_cymbal",  # NB: Crash_cymbal is musical, exclude in EXCLUDE
    "Crashing", "Smash", "Crunch", "Slam", "Bang", "Boom",
    "Burst_or_pop", "Explosion", "Gunshot_and_gunfire", "Fireworks",
    "Vehicle_horn_and_car_horn_and_honking", "Skidding", "Shatter",
    "Glass_breaking",
}

NOISE_LABELS = {
    "Wind", "Rain", "Engine", "Vehicle", "Car_passing_by", "Truck",
    "Motorcycle", "Traffic_noise_and_roadway_noise", "Police_car_(siren)",
    "Ambulance_(siren)", "Fire_engine_and_fire_truck_(siren)",
    "Air_conditioning", "Hammer", "Power_tool", "Drill",
    "Sawing", "Chainsaw", "Vacuum_cleaner", "Helicopter", "Aircraft",
    "Bus", "Train",
}

EXCLUDE_LABELS = {
    "Music", "Speech", "Singing", "Drum", "Drum_kit", "Cymbal",
    "Crash_cymbal", "Snare_drum", "Bass_drum", "Tabla",
    "Bass_guitar", "Electric_guitar", "Acoustic_guitar", "Synthesizer",
    "Piano", "Vocal_music", "Trumpet", "Saxophone", "Tambourine",
    "Tuning_fork", "Marimba_and_xylophone", "Cowbell",
    "Children_playing", "Children_shouting", "Crying_and_sobbing",
    "Laughter", "Sigh", "Whoop",
}


def _classify(labels_str: str) -> str | None:
    labels = set(labels_str.strip('"').split(','))
    if labels & EXCLUDE_LABELS:
        return None
    in_crash = bool(labels & CRASH_LABELS)
    in_noise = bool(labels & NOISE_LABELS)
    if in_crash and not in_noise:
        return "crash"
    if in_noise and not in_crash:
        return "noise"
    return None


def _load_label_csv(split: str) -> list[tuple[str, str]]:
    """Return [(filename, labels_str), ...] from FSD50K dev or eval CSV."""
    url = f"{HF_BASE}/labels/{split}.csv"
    LOG.info("downloading labels: %s", url)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        csv_text = r.read().decode("utf-8")
    rows = []
    for line in csv_text.splitlines()[1:]:
        parts = line.split(',', 2)
        if len(parts) < 3:
            continue
        rows.append((parts[0], parts[1]))
    return rows


def _save(url: str, dst: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = r.read()
        data, sr = sf.read(io.BytesIO(payload), always_2d=False)
    except Exception as exc:
        LOG.debug("download fail %s: %s", dst.name, exc)
        return False
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    if len(data) > sr * 10:
        data = data[: sr * 10]
    if len(data) < sr * 3:
        data = np.pad(data, (0, sr * 3 - len(data)), mode="constant")
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data * (0.95 / peak)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), data, sr, subtype="PCM_16")
    return True


def fetch(target_crash: int, target_noise: int) -> tuple[int, int]:
    crash_added, noise_added = 0, 0
    started = time.monotonic()

    for split, dirname in [("dev", "FSD50K.dev_audio"),
                           ("eval", "FSD50K.eval_audio")]:
        if crash_added >= target_crash and noise_added >= target_noise:
            break
        try:
            label_rows = _load_label_csv(split)
        except Exception as exc:
            LOG.warning("%s labels: %s", split, exc)
            continue

        # Keep only clips whose label classification is clean.
        candidates: list[tuple[str, str]] = []  # (cls, fname)
        for fname, labels_str in label_rows:
            cls = _classify(labels_str)
            if cls is None:
                continue
            candidates.append((cls, fname))
        LOG.info("[%s] %d eligible candidates", split, len(candidates))

        # Hugging Face hosts FSD50K clips on the LFS-backed resolve URL.
        # Path pattern: clips/<split>/<fname>.wav
        def _process_one(item):
            nonlocal crash_added, noise_added
            cls, fname = item
            if cls == "crash" and crash_added >= target_crash:
                return
            if cls == "noise" and noise_added >= target_noise:
                return
            url = f"{HF_BASE}/clips/{split}/{fname}.wav"
            dst_dir = CRASH_DIR if cls == "crash" else NOISE_DIR
            dst = dst_dir / f"fsd50k_{fname}.wav"
            if dst.exists():
                if cls == "crash":
                    crash_added += 1
                else:
                    noise_added += 1
                return
            ok = _save(url, dst)
            if ok:
                if cls == "crash":
                    crash_added += 1
                else:
                    noise_added += 1

        # Cap how many candidates we even look at to avoid scanning all 40K.
        max_to_scan = (target_crash + target_noise) * 4
        candidates = candidates[:max_to_scan]

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(_process_one, c) for c in candidates]
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                except Exception:
                    pass
                if i % 100 == 0:
                    LOG.info("[%s] processed %d/%d crash=%d noise=%d (%.0fs)",
                             split, i, len(candidates),
                             crash_added, noise_added,
                             time.monotonic() - started)
                if crash_added >= target_crash and noise_added >= target_noise:
                    break

    return crash_added, noise_added


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Pull crash-relevant clips from FSD50K")
    parser.add_argument("--target-crash", type=int, default=1500)
    parser.add_argument("--target-noise", type=int, default=1500)
    args = parser.parse_args(argv)

    crash, noise = fetch(args.target_crash, args.target_noise)
    LOG.info("FETCHED crash=%d noise=%d", crash, noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
