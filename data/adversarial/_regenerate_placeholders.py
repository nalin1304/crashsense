"""Regenerate the synthetic placeholder Adversarial_Suite corpus.

Implements the *stub* portion of task 4.12 in
``.kiro/specs/crashsense-hardening/tasks.md``. The production target is
≥50 real WAV clips per category (R7.1); for now this script writes 5
short, category-distinct synthetic WAVs per category so the rest of the
hardening epic — `evaluate_adversarial.py`, `curate_adversarial.py`,
`check_real_world_disjoint.py` extension — has something concrete to
exercise. Real clips are added later via
``scripts/curate_adversarial.py add``.

Output format
-------------
* Mono PCM, 22050 Hz, 16-bit (matches ``spectrogram_gen.SAMPLE_RATE``).
* 4.0 s per clip, well inside the R7.1 [3.0, 30.0] s window.
* Filenames ``<category>_placeholder_<NN>.wav`` so they sort lexically
  and never collide with future real captures named ``<category>_<id>``
  per the curation tool.

Determinism
-----------
Every clip is generated from a category-specific
``np.random.default_rng(SEED + offset)`` plus pure formulae over
``np.arange``. Two consecutive runs on the same numpy/scipy versions
produce byte-identical WAVs. If a regeneration changes any committed
byte, the diff is the signal.

Usage::

    python data/adversarial/_regenerate_placeholders.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# ------------------------------------------------------------------ constants
SAMPLE_RATE = 22050  # Hz, mono, 16-bit PCM
DURATION_S = 4.0  # well inside [3.0, 30.0]
CLIPS_PER_CATEGORY = 5  # placeholder count; production target is ≥50
SEED = 42

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("horns", "fireworks", "tire_blowouts", "airbrakes")


# ------------------------------------------------------------------ helpers
def _to_int16(signal: np.ndarray) -> np.ndarray:
    clipped = np.clip(signal, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, _to_int16(signal))


def _t(n: int) -> np.ndarray:
    return np.arange(n) / SAMPLE_RATE


# ------------------------------------------------------------------ generators
def _gen_horn(idx: int) -> np.ndarray:
    """Vehicle horn — repeating square-wave bursts at 380–520 Hz with harmonics."""
    n = int(SAMPLE_RATE * DURATION_S)
    out = np.zeros(n, dtype=np.float64)
    # Distinct fundamental per clip so the 5 placeholders are not identical
    # frequency content.
    fundamental = 380.0 + 30.0 * idx  # 380, 410, 440, 470, 500 Hz
    burst_s = 0.35
    gap_s = 0.15
    burst_n = int(burst_s * SAMPLE_RATE)
    gap_n = int(gap_s * SAMPLE_RATE)
    pos = 0
    while pos + burst_n <= n:
        t = _t(burst_n)
        # Square wave with first three harmonics weighted to avoid aliasing.
        wave = (
            np.sign(np.sin(2.0 * np.pi * fundamental * t))
            + 0.5 * np.sign(np.sin(2.0 * np.pi * 2.0 * fundamental * t))
            + 0.25 * np.sign(np.sin(2.0 * np.pi * 3.0 * fundamental * t))
        )
        wave /= np.max(np.abs(wave))
        # Smooth attack/decay so the WAV does not click.
        env = np.ones(burst_n)
        ramp = int(0.02 * SAMPLE_RATE)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        out[pos : pos + burst_n] = 0.55 * wave * env
        pos += burst_n + gap_n
    return out


def _gen_firework(idx: int) -> np.ndarray:
    """Firework — two white-noise bursts with exponential decay envelopes."""
    rng = np.random.default_rng(SEED + 100 + idx)
    n = int(SAMPLE_RATE * DURATION_S)
    out = np.zeros(n, dtype=np.float64)

    # Two bursts at slightly different positions per clip so the placeholders
    # are not identical.
    burst_starts_s = (0.4 + 0.05 * idx, 2.1 + 0.07 * idx)
    burst_durations_s = (1.3, 1.1)
    for start_s, dur_s in zip(burst_starts_s, burst_durations_s):
        start = int(start_s * SAMPLE_RATE)
        length = int(dur_s * SAMPLE_RATE)
        if start + length > n:
            length = n - start
        if length <= 0:
            continue
        noise = rng.standard_normal(length)
        # Exponential decay (fast attack, ~0.4 s e-fold).
        env = np.exp(-np.arange(length) / (0.4 * SAMPLE_RATE))
        # Short attack ramp so we do not click.
        ramp = int(0.005 * SAMPLE_RATE)
        env[:ramp] = np.linspace(0.0, env[ramp], ramp) if ramp > 0 else env[:ramp]
        out[start : start + length] += 0.7 * noise * env
    # Normalize to avoid int16 clipping while keeping the bursts loud.
    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out *= 0.85 / peak
    return out


def _gen_tire_blowout(idx: int) -> np.ndarray:
    """Tire blowout — short downward chirp 2 kHz → 200 Hz with broadband noise."""
    rng = np.random.default_rng(SEED + 200 + idx)
    n = int(SAMPLE_RATE * DURATION_S)
    out = np.zeros(n, dtype=np.float64)

    # Place the blowout impulse at 0.5 s + idx*0.1 s for variety.
    impact_start_s = 0.5 + 0.1 * idx
    impact_dur_s = 0.6
    start = int(impact_start_s * SAMPLE_RATE)
    length = int(impact_dur_s * SAMPLE_RATE)
    if start + length > n:
        length = n - start
    t = _t(length) if length > 0 else np.zeros(0)

    # Linear downward chirp 2000 → 200 Hz over the impact window.
    if length > 0:
        f0, f1 = 2000.0, 200.0
        phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / impact_dur_s)
        chirp = np.sin(phase)
        noise = rng.standard_normal(length)
        # Quick attack, slow exponential decay.
        env = np.exp(-np.arange(length) / (0.25 * SAMPLE_RATE))
        ramp = int(0.003 * SAMPLE_RATE)
        if ramp > 0:
            env[:ramp] = np.linspace(0.0, env[ramp], ramp)
        out[start : start + length] = (0.55 * chirp + 0.35 * noise) * env

    # Add a low-level rolling-tire whoosh through the rest of the clip.
    rolling_noise = rng.standard_normal(n) * 0.05
    out += rolling_noise

    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out *= 0.8 / peak
    return out


def _gen_airbrake(idx: int) -> np.ndarray:
    """Air brake — band-limited white noise 'whoosh' with slow amplitude envelope."""
    rng = np.random.default_rng(SEED + 300 + idx)
    n = int(SAMPLE_RATE * DURATION_S)
    out = np.zeros(n, dtype=np.float64)

    # One or two whoosh segments, varied by index.
    # idx 0,2,4 -> single long whoosh; idx 1,3 -> two short whooshes.
    if idx % 2 == 0:
        segments = ((0.7, 2.6),)
    else:
        segments = ((0.4, 1.0), (2.0, 1.0))

    for start_s, dur_s in segments:
        start = int(start_s * SAMPLE_RATE)
        length = int(dur_s * SAMPLE_RATE)
        if start + length > n:
            length = n - start
        if length <= 0:
            continue
        white = rng.standard_normal(length)

        # Apply a 1-pole low-pass at ~3 kHz via simple IIR to give it the
        # band-limited "whoosh" character without scipy.signal.
        rc = 1.0 / (2.0 * np.pi * 3000.0)
        dt = 1.0 / SAMPLE_RATE
        alpha = dt / (rc + dt)
        filtered = np.empty_like(white)
        filtered[0] = white[0] * alpha
        for i in range(1, length):
            filtered[i] = filtered[i - 1] + alpha * (white[i] - filtered[i - 1])

        # Bell-shaped amplitude envelope (sine half-wave) — slow rise, slow fall.
        env = np.sin(np.linspace(0.0, np.pi, length))
        out[start : start + length] += 0.7 * filtered * env

    peak = np.max(np.abs(out))
    if peak > 1e-9:
        out *= 0.75 / peak
    return out


_GENERATORS = {
    "horns": _gen_horn,
    "fireworks": _gen_firework,
    "tire_blowouts": _gen_tire_blowout,
    "airbrakes": _gen_airbrake,
}


# ------------------------------------------------------------------ manifest
def _manifest_entry(category: str, idx: int) -> dict:
    """Schema-stable manifest entry for one placeholder clip.

    The fields here mirror what ``scripts/curate_adversarial.py add`` writes
    for real clips, with ``is_placeholder=true`` and a synthetic ``source_uri``.
    Keeping the same schema means the manifest can be append-only as real
    clips replace placeholders one by one.
    """
    return {
        "filename": f"{category}_placeholder_{idx:02d}.wav",
        "source_uri": f"synthetic:{category}_placeholder_{idx:02d}",
        "license": "CC0-1.0 (synthetic)",
        "duration_s": DURATION_S,
        "recorded_at_iso8601": "2025-01-15T00:00:00Z",
        "is_placeholder": True,
    }


def _write_manifest(manifest_path: Path) -> None:
    """Write or refresh ``data/adversarial/manifest.json`` for the placeholders.

    If the manifest already exists with non-placeholder entries we *preserve*
    them so this script is idempotent and safe to re-run after real clips
    have been ingested. Only the placeholder rows for each category are
    rewritten.
    """
    existing = {category: [] for category in CATEGORIES}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                prior = json.load(fh)
            for category, rows in (prior.get("categories") or {}).items():
                if category not in existing:
                    continue
                for row in rows:
                    if isinstance(row, dict) and not row.get("is_placeholder"):
                        existing[category].append(row)
        except (OSError, json.JSONDecodeError):
            existing = {category: [] for category in CATEGORIES}

    categories = {}
    for category in CATEGORIES:
        rows = list(existing[category])  # preserve real clips
        for idx in range(1, CLIPS_PER_CATEGORY + 1):
            rows.append(_manifest_entry(category, idx))
        # Sort by filename for stable diffs.
        rows.sort(key=lambda r: r.get("filename", ""))
        categories[category] = rows

    manifest = {
        "schema_version": "1.0.0",
        # The generated_at value is the WAV provenance timestamp baked into
        # ``_manifest_entry``; keep it deterministic so the placeholders +
        # the manifest both diff cleanly.
        "generated_at_iso8601": "2025-01-15T00:00:00Z",
        "categories": categories,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.write("\n")


# ------------------------------------------------------------------ entry point
def main() -> None:
    for category in CATEGORIES:
        gen = _GENERATORS[category]
        for idx in range(1, CLIPS_PER_CATEGORY + 1):
            signal = gen(idx - 1)
            assert signal.shape == (int(SAMPLE_RATE * DURATION_S),), (
                f"{category} clip {idx} has unexpected length {signal.shape}"
            )
            path = ROOT / category / f"{category}_placeholder_{idx:02d}.wav"
            _write(path, signal)
    _write_manifest(ROOT / "manifest.json")
    # Nudge the operator about the placeholder vs. real distinction.
    print(
        f"Wrote {len(CATEGORIES) * CLIPS_PER_CATEGORY} placeholder clips and "
        f"refreshed {ROOT / 'manifest.json'}.\n"
        f"Production target is ≥50 real clips per category (R7.1). Use "
        f"`python scripts/curate_adversarial.py add ...` to ingest real audio."
    )


if __name__ == "__main__":
    main()
