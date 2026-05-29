#!/usr/bin/env python3
"""
Pull real crash audio samples from archive.org's Freesound mirror.

These supplement the ESC-50-derived training set with actual collision/crash
recordings, and ship a single representative clip into the frontend so the
dashboard's AudioMonitor can play a real sample instead of a synthetic one.
"""

from __future__ import annotations

import io
import logging
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

LOG = logging.getLogger("fetch_real_audio")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CRASH = REPO_ROOT / "data" / "raw_audio" / "crash"
FRONTEND_PUBLIC = REPO_ROOT / "frontend" / "public"

# Curated archive.org mirrors of CC-BY Freesound clips relevant to highway crashes.
SOURCES = [
    ("332059", "Collision_Reverb-332059", "qubodup, CC-BY"),
    ("332056", "Fast_Collision-332056", "qubodup, CC-BY"),
]

UA = {"User-Agent": "Mozilla/5.0 (CrashSense data fetcher)"}


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _list_assets(item_id: str) -> list[str]:
    html = _fetch(f"https://archive.org/details/Freesound-{item_id}").decode(
        "utf-8", errors="ignore"
    )
    pat = re.compile(rf"/download/Freesound-{item_id}/[A-Za-z0-9_\-\.%]+\.(?:mp3|ogg|wav|flac)")
    return sorted(set(pat.findall(html)))


def _pick_audio(paths: list[str], stem: str) -> str | None:
    # Prefer mp3, then ogg, that match our stem.
    pref = [p for p in paths if stem in p and p.lower().endswith(".mp3")]
    pref += [p for p in paths if stem in p and p.lower().endswith(".ogg")]
    pref += [p for p in paths if p.lower().endswith(".mp3")]
    return pref[0] if pref else None


def _save_as_wav(content: bytes, dst_path: Path) -> bool:
    try:
        data, sr = sf.read(io.BytesIO(content), always_2d=False)
    except Exception as exc:
        LOG.warning("decode failed for %s: %s", dst_path.name, exc)
        return False
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    if len(data) > sr * 10:
        data = data[: sr * 10]
    if len(data) < sr * 3:
        pad = sr * 3 - len(data)
        data = np.pad(data, (0, pad), mode="constant")
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data * (0.95 / peak)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), data, sr, subtype="PCM_16")
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    RAW_CRASH.mkdir(parents=True, exist_ok=True)
    FRONTEND_PUBLIC.mkdir(parents=True, exist_ok=True)

    saved = []
    for item_id, stem, license_note in SOURCES:
        try:
            assets = _list_assets(item_id)
        except Exception as exc:
            LOG.warning("could not list %s: %s", item_id, exc)
            continue
        choice = _pick_audio(assets, stem)
        if not choice:
            LOG.warning("no audio asset for %s", item_id)
            continue
        url = "https://archive.org" + choice
        LOG.info("fetching %s ... (%s)", url, license_note)
        try:
            payload = _fetch(url)
        except Exception as exc:
            LOG.warning("download failed: %s", exc)
            continue
        dst = RAW_CRASH / f"freesound_{item_id}.wav"
        if _save_as_wav(payload, dst):
            saved.append(dst)
            LOG.info("  saved %s (%d bytes)", dst.name, dst.stat().st_size)

    if not saved:
        LOG.error("no real audio samples were fetched")
        return 1

    # Ship one representative sample to the frontend public folder for the
    # AudioMonitor (Requirement 18).
    demo_dst = FRONTEND_PUBLIC / "dashcam-crash.wav"
    src = saved[0]
    demo_dst.write_bytes(src.read_bytes())
    LOG.info("dashboard demo clip -> %s", demo_dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
