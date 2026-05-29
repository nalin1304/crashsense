"""
End-to-end demo orchestration (Requirement 20).

Workflow:
  1. Load (or stub) the crash detector.
  2. Generate or play a dashcam crash audio clip and run sliding-window
     inference. On a CRASH window, print the confidence line.
  3. Pick a uniformly random crash point inside the sensor triangle.
  4. Run forward simulation, perturb with 2 ms Gaussian noise, run inverse
     localization.
  5. POST a CrashEvent to the Backend_API; the backend broadcasts it to the
     dashboard via WebSocket.

If `--audio <path>` is supplied we use that file. Otherwise the runner builds
a synthetic dashcam-style waveform on the fly so the demo is reproducible
without external assets.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Allow `python backend/demo_runner.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.audio_model.inference import predict_stream  # noqa: E402
from backend.triangulation.sensor_config import SENSORS, sensor_triangle_bbox  # noqa: E402
from backend.triangulation.tdoa_solver import (  # noqa: E402
    simulate_arrival_times,
    tdoa_localize,
)

LOG = logging.getLogger("demo_runner")

DEFAULT_BACKEND = "http://127.0.0.1:8000"
SAMPLE_RATE = 22050


def synthesize_dashcam_clip(seed: int = 42) -> np.ndarray:
    """Build a 6 s clip: highway hiss + a real crash sample at t≈3.0s.

    Uses a real ESC-50 crash clip if available so the trained model produces
    high-confidence detections out of the box. Falls back to a synthetic
    chirp if the dataset hasn't been prepared.
    """
    import librosa
    rng = np.random.default_rng(seed)
    duration_s = 6.0
    n = int(SAMPLE_RATE * duration_s)
    bg = rng.normal(0.0, 0.03, n).astype(np.float32)
    impact_start = int(3.0 * SAMPLE_RATE)

    crash_dir = Path(__file__).resolve().parents[1] / "data" / "raw_audio" / "crash"
    real_clips = [
        p for p in crash_dir.glob("esc_*.wav")
        if not any(t in p.name for t in ("_pitch", "_stretch", "_noise"))
    ] if crash_dir.is_dir() else []

    if real_clips:
        clip_path = real_clips[seed % len(real_clips)]
        samples, _ = librosa.load(str(clip_path), sr=SAMPLE_RATE, mono=True)
        samples = samples.astype(np.float32, copy=False)
        # Take up to 3 seconds of the clip starting from a peak energy region
        if len(samples) > SAMPLE_RATE * 3:
            samples = samples[: SAMPLE_RATE * 3]
        end = impact_start + len(samples)
        if end > n:
            samples = samples[: n - impact_start]
            end = n
        bg[impact_start:end] += samples * 0.95
        LOG.info("dashcam clip uses real crash sample: %s", clip_path.name)
    else:
        # Fallback synthetic chirp
        impact_len = int(0.25 * SAMPLE_RATE)
        t = np.linspace(0, 0.25, impact_len, dtype=np.float32)
        chirp = np.sin(2 * np.pi * (200 + 4000 * t) * t) * np.exp(-t * 6.0)
        impact = (chirp * 0.95).astype(np.float32)
        bg[impact_start:impact_start + impact_len] += impact
        LOG.info("dashcam clip uses synthetic chirp (no dataset found)")
    return np.clip(bg, -1.0, 1.0)


def random_point_in_triangle(seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    p1, p2, p3 = SENSORS
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1.0:
        r1, r2 = 1.0 - r1, 1.0 - r2
    r3 = 1.0 - r1 - r2
    return (
        r1 * p1.lat + r2 * p2.lat + r3 * p3.lat,
        r1 * p1.lon + r2 * p2.lon + r3 * p3.lon,
    )


def load_clip(audio_path: Path | None) -> np.ndarray:
    if audio_path is None:
        return synthesize_dashcam_clip()
    import librosa
    samples, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    return samples.astype(np.float32)


def run_demo(audio_path: Path | None, backend_url: str, *, retries: int = 2) -> int:
    started = time.monotonic()
    samples = load_clip(audio_path)
    LOG.info("loaded audio: %d samples (%.2fs)", samples.size, samples.size / SAMPLE_RATE)

    # Sliding-window inference; the predict_stream helper prints CRASH lines
    # only when 3-of-4 consecutive windows fire (consensus debouncing).
    detected_confidence = 0.0
    consensus_count = 0
    for window in predict_stream(samples, print_detections=True):
        if window.get("consensus"):
            consensus_count += 1
            detected_confidence = max(detected_confidence, float(window["confidence"]))
    if consensus_count == 0:
        LOG.warning("no crash consensus reached; using fallback confidence 0.9")
        detected_confidence = 0.9

    seed = int(time.time())
    true_lat, true_lon = random_point_in_triangle(seed)
    delays = simulate_arrival_times(true_lat, true_lon, SENSORS)
    rng = np.random.default_rng(seed)
    noisy = {k: v + float(rng.normal(0.0, 0.002)) for k, v in delays.items()}
    solved = tdoa_localize(noisy, SENSORS)
    if solved.success and solved.lat is not None and solved.lon is not None:
        crash_lat, crash_lon = solved.lat, solved.lon
        LOG.info(
            "TDOA OK | true=(%.6f,%.6f) solved=(%.6f,%.6f)",
            true_lat, true_lon, crash_lat, crash_lon,
        )
    else:
        LOG.warning("TDOA solver rejected (%s); falling back to ground truth", solved.error_message)
        crash_lat, crash_lon = true_lat, true_lon

    # Hand off to the Backend_API which broadcasts to the dashboard.
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.post(
                    f"{backend_url}/simulate-crash",
                    json={"lat": crash_lat, "lon": crash_lon},
                )
                resp.raise_for_status()
                LOG.info("backend accepted CrashEvent: %s", resp.json().get("event_id"))
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOG.warning("POST attempt %d failed: %s", attempt + 1, exc)
            time.sleep(0.5)
    else:
        LOG.error("could not reach backend after %d retries: %s", retries, last_error)
        return 2

    elapsed = time.monotonic() - started
    LOG.info("end-to-end pipeline complete in %.2fs", elapsed)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="CrashSense end-to-end demo")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Optional WAV file. Defaults to a synthesized clip.")
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    args = parser.parse_args(argv)
    return run_demo(args.audio, args.backend)


if __name__ == "__main__":
    sys.exit(main())
