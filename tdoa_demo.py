#!/usr/bin/env python3
"""
TDOA proof-of-concept demo (Requirement 7 criterion 7).

Picks a uniformly-random crash point inside the sensor triangle, simulates
true arrival times, perturbs each per-sensor time by independent Gaussian
noise (sigma = 2 ms), runs the inverse solver, and prints the true coordinate,
the solved coordinate, and the localization error in meters.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running from the repo root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from geopy.distance import geodesic

from backend.triangulation.sensor_config import (
    SENSORS,
    sensor_triangle_bbox,
)
from backend.triangulation.tdoa_solver import (
    LocalizationResult,
    simulate_arrival_times,
    tdoa_localize,
)

NOISE_STD_S = 0.002  # 2 milliseconds


def random_point_in_triangle(seed: int | None = None) -> tuple[float, float]:
    """Uniformly sample a point inside the triangle formed by the first three sensors."""
    rng = random.Random(seed)
    p1, p2, p3 = SENSORS[:3]
    r1 = rng.random()
    r2 = rng.random()
    if r1 + r2 > 1.0:
        r1, r2 = 1.0 - r1, 1.0 - r2
    r3 = 1.0 - r1 - r2
    lat = r1 * p1.lat + r2 * p2.lat + r3 * p3.lat
    lon = r1 * p1.lon + r2 * p2.lon + r3 * p3.lon
    return lat, lon


def run_once(seed: int | None = None, *, noise_std_s: float = NOISE_STD_S) -> LocalizationResult:
    rng = np.random.default_rng(seed)
    true_lat, true_lon = random_point_in_triangle(seed)
    delays = simulate_arrival_times(true_lat, true_lon, SENSORS)
    noisy = {k: v + float(rng.normal(0.0, noise_std_s)) for k, v in delays.items()}
    result = tdoa_localize(noisy, SENSORS)

    print(f"True coordinate:   ({true_lat:.6f}, {true_lon:.6f})")
    if not result.success:
        print(f"Solver failed: {result.error_message}")
        return result
    err_m = geodesic((true_lat, true_lon), (result.lat, result.lon)).meters
    print(f"Solved coordinate: ({result.lat:.6f}, {result.lon:.6f})")
    print(f"Localization error: {err_m:.1f} m")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CrashSense TDOA demo")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args(argv)
    rc = 0
    for i in range(args.trials):
        seed = None if args.seed is None else args.seed + i
        result = run_once(seed)
        if not result.success:
            rc = 1
        if args.trials > 1 and i + 1 < args.trials:
            print("---")
    return rc


if __name__ == "__main__":
    sys.exit(main())
