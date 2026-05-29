#!/usr/bin/env python3
"""
Fetch crash-relevant audio from AudioSet (via Hugging Face mirror).

AudioSet is the standard for environmental audio recognition, and unlike
ESC-50 / UrbanSound8K its clips are real YouTube recordings — including
actual highway crashes, glass shattering in real environments, sirens, etc.
This is the closest we can get to in-domain training data without scraping
YouTube ourselves.

Filters by human-readable label. Clips whose label set intersects the
configured CRASH_LABELS go into data/raw_audio/crash/, NOISE_LABELS into
noise/. Mixed-label clips (e.g. "Glass + Music") are skipped to keep the
labels clean.

Usage:
    python scripts/fetch_audioset_crash.py --target-crash 200 --target-noise 200
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

LOG = logging.getLogger("fetch_audioset")

REPO_ROOT = Path(__file__).resolve().parents[1]
CRASH_DIR = REPO_ROOT / "data" / "raw_audio" / "crash"
NOISE_DIR = REPO_ROOT / "data" / "raw_audio" / "noise"

UA = {"User-Agent": "Mozilla/5.0 (CrashSense AudioSet fetcher)"}

# AudioSet ontology labels (the strings AudioSet itself uses)
CRASH_LABELS = {
    "Crash cymbal",  # NOPE - musical, exclude
}
# Use sets explicitly. CRASH_INCLUDE => any label in this set marks clip as crash
CRASH_INCLUDE = {
    "Vehicle horn, car horn, honking",
    "Skidding",
    "Glass",
    "Shatter",
    "Crash",                    # the real "crash" label in the ontology
    "Crunch",
    "Gunshot, gunfire",
    "Explosion",
    "Boom",
    "Smash, crash",
    "Slam",
    "Thump, thud",
    "Bang",
    "Burst, pop",
    "Fireworks",
    "Eruption",
    "Bicycle bell",
}

NOISE_INCLUDE = {
    "Wind",
    "Rain",
    "Traffic noise, roadway noise",
    "Engine",
    "Motor vehicle (road)",
    "Car",
    "Vehicle",
    "Race car, auto racing",
    "Truck",
    "Motorcycle",
    "Bus",
    "Police car (siren)",
    "Ambulance (siren)",
    "Fire engine, fire truck (siren)",
    "Civil defense siren",
    "Emergency vehicle",
    "Air conditioning",
    "Power tool",
    "Drill",
    "Hammer",
    "Sawing",
    "Chainsaw",
    "Vacuum cleaner",
    "Helicopter",
    "Aircraft",
}

# Labels that make a clip ineligible regardless of which side it'd otherwise fall on.
EXCLUDE_LABELS = {
    "Music",
    "Speech",
    "Singing",
    "Drum",
    "Drum kit",
    "Cymbal",
    "Crash cymbal",
    "Snare drum",
    "Tabla",
    "Bass guitar",
    "Electric guitar",
    "Acoustic guitar",
    "Synthesizer",
    "Piano",
    "Vocal music",
    "Singing bowl",
    "Trumpet",
    "Saxophone",
    "Beatboxing",
    "Children playing",
    "Children shouting",
    "Crying, sobbing",
    "Laughter",
    "Whoop",
}


def _hf_rows_page(config: str, split: str, offset: int, length: int = 100) -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=agkphysics%2FAudioSet"
        f"&config={config}&split={split}"
        f"&offset={offset}&length={length}"
    )
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return __import__("json").loads(r.read()).get("rows", [])


def _classify(human_labels: list[str]) -> str | None:
    if not human_labels:
        return None
    label_set = set(human_labels)
    if label_set & EXCLUDE_LABELS:
        return None
    in_crash = bool(label_set & CRASH_INCLUDE)
    in_noise = bool(label_set & NOISE_INCLUDE)
    if in_crash and not in_noise:
        return "crash"
    if in_noise and not in_crash:
        return "noise"
    return None


def _save(src: str, dst: Path) -> bool:
    try:
        req = urllib.request.Request(src, headers=UA)
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


def fetch(target_crash: int, target_noise: int,
          configs: list[tuple[str, str]] = (
              ("balanced", "train"), ("balanced", "test"), ("full", "unbal_train"),
          )) -> tuple[int, int]:
    crash_added, noise_added = 0, 0
    seen: set[str] = set()
    started = time.monotonic()

    for config, split in configs:
        if crash_added >= target_crash and noise_added >= target_noise:
            break
        for page in range(0, 200):
            if crash_added >= target_crash and noise_added >= target_noise:
                break
            offset = page * 100
            try:
                rows = _hf_rows_page(config, split, offset, 100)
            except Exception as exc:
                LOG.warning("page %d failed: %s", page, exc)
                time.sleep(2)
                continue
            if not rows:
                break

            jobs = []
            with ThreadPoolExecutor(max_workers=6) as pool:
                for row in rows:
                    r = row["row"]
                    vid = r.get("video_id", "")
                    labels = r.get("human_labels") or []
                    if vid in seen:
                        continue
                    seen.add(vid)
                    cls = _classify(labels)
                    if cls is None:
                        continue
                    if cls == "crash" and crash_added >= target_crash:
                        continue
                    if cls == "noise" and noise_added >= target_noise:
                        continue
                    audio_list = r.get("audio") or []
                    if not audio_list:
                        continue
                    src = audio_list[0].get("src")
                    if not src:
                        continue
                    dst_dir = CRASH_DIR if cls == "crash" else NOISE_DIR
                    dst = dst_dir / f"audioset_{vid}.wav"
                    if dst.exists():
                        continue
                    jobs.append((cls, dst, src))

                futures = {pool.submit(_save, src, dst): (cls, dst) for cls, dst, src in jobs}
                for f in as_completed(futures):
                    cls, dst = futures[f]
                    try:
                        ok = f.result()
                    except Exception:
                        ok = False
                    if not ok:
                        continue
                    if cls == "crash":
                        crash_added += 1
                    else:
                        noise_added += 1

            if (page + 1) % 5 == 0:
                LOG.info("[%s/%s] page %d crash=%d noise=%d (%.0fs)",
                         config, split, page + 1, crash_added, noise_added,
                         time.monotonic() - started)

    return crash_added, noise_added


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Pull crash-relevant audio from AudioSet")
    parser.add_argument("--target-crash", type=int, default=300)
    parser.add_argument("--target-noise", type=int, default=300)
    args = parser.parse_args(argv)
    crash, noise = fetch(args.target_crash, args.target_noise)
    LOG.info("FETCHED crash=%d noise=%d", crash, noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
