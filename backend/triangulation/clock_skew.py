"""
Per-sensor clock-skew estimation for the TDOA solver.

Implements R13.1 (Clock_Skew_Estimator) from the CrashSense Hardening
spec — task 5.6.

Public API
----------
estimate_offset(sensor_id, ntp_or_ptp_samples) -> dict
    Compute and cache the latest skew estimate for ``sensor_id`` from a
    list of microsecond offsets observed from NTP/PTP queries. Returns a
    dictionary with keys:

    - ``offset_us`` (float): median of the samples — robust against
      outliers that creep into NTP/PTP measurements over a noisy WAN.
    - ``offset_us_bound`` (float, non-negative): the 1-σ-times-two
      uncertainty floor:

          ``offset_us_bound = max(2 * stdev(samples), measurement_floor_us)``

      The floor (default 10 µs) keeps the bound non-zero even when the
      sample stream is artificially clean (e.g. a synthetic test or a
      lab PTP grandmaster). This is what lets R13.4 / P15 hold:
      ``offset_us_bound ≥ 2·σ`` for every σ_us < 100.
    - ``last_measured_at_iso8601`` (str): UTC timestamp of this call,
      written as ``YYYY-MM-DDTHH:MM:SSZ`` to match the convention used by
      other reporting paths in the codebase.

get_cached_offset(sensor_id) -> dict | None
    Return the most recent estimate for ``sensor_id`` or ``None`` if no
    estimate has been recorded yet. The returned dict is a fresh copy;
    callers can mutate it freely.

clear_cache() -> None
    Wipe the in-memory cache. Intended for test isolation.

set_measurement_floor_us(floor_us) -> None / get_measurement_floor_us()
    Optional knobs for tests that want to vary the floor without
    monkey-patching the constant. The default floor (10 µs) is what
    production should use.

Design notes
------------
- The cache is a process-local dict keyed by ``sensor_id``. The Phase 3
  design (§3.3) does not require cross-process sharing; the TDOA solver
  reads the cache from the same process that runs the periodic NTP/PTP
  poller.
- We use ``statistics.median`` and ``statistics.stdev`` from the stdlib
  rather than NumPy. The samples are typically dozens to hundreds of
  floats, so this is well below any NumPy speedup threshold and avoids
  pulling NumPy into a module that triangulation otherwise does not
  need.
- ``stdev`` requires at least two samples; we surface that as a
  ``ValueError`` so callers do not silently get a zero-bound estimate
  from a single sample.
"""

from __future__ import annotations

import logging
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger("clock_skew")

# Default measurement floor: 10 µs. Picked so the bound dominates
# typical NTP-over-LAN noise floors (~1 µs) and PTP boundary-clock
# residuals (~100 ns). Production deployments should leave this alone.
_DEFAULT_MEASUREMENT_FLOOR_US: float = 10.0

# Module-level state. Both are guarded by GIL access patterns — single
# producer (the NTP/PTP poller) and many readers (the TDOA solver) are
# safe with plain dict get/set on CPython. If Phase 3 introduces an
# async poller we revisit.
_cache: Dict[str, Dict[str, Any]] = {}
_measurement_floor_us: float = _DEFAULT_MEASUREMENT_FLOOR_US


# --- Validation helpers --------------------------------------------------


def _validate_sensor_id(sensor_id: Any) -> str:
    if not isinstance(sensor_id, str):
        raise ValueError(
            f"sensor_id: expected str, got {type(sensor_id).__name__}"
        )
    if not sensor_id:
        raise ValueError("sensor_id: must be non-empty")
    return sensor_id


def _validate_samples(samples: Any) -> List[float]:
    """Coerce ``samples`` into a list of finite floats with len >= 2."""
    if isinstance(samples, (str, bytes)):
        # str/bytes are iterable but never the right type here.
        raise ValueError(
            f"ntp_or_ptp_samples: expected sequence of floats, "
            f"got {type(samples).__name__}"
        )
    if not isinstance(samples, Iterable):
        raise ValueError(
            f"ntp_or_ptp_samples: expected iterable, got {type(samples).__name__}"
        )

    out: List[float] = []
    for i, s in enumerate(samples):
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            raise ValueError(
                f"ntp_or_ptp_samples[{i}]: expected real number, "
                f"got {type(s).__name__}"
            )
        sf = float(s)
        if not math.isfinite(sf):
            raise ValueError(
                f"ntp_or_ptp_samples[{i}]: must be finite, got {s!r}"
            )
        out.append(sf)

    if len(out) < 2:
        raise ValueError(
            f"ntp_or_ptp_samples: need at least 2 samples to estimate "
            f"a skew bound, got {len(out)}"
        )
    return out


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC with explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- Public API ----------------------------------------------------------


def estimate_offset(
    sensor_id: str,
    ntp_or_ptp_samples: Iterable[float],
) -> Dict[str, Any]:
    """
    Estimate the clock offset for ``sensor_id`` from NTP/PTP samples.

    Parameters
    ----------
    sensor_id : str
        Sensor identifier. Must be a non-empty string. Used as the cache
        key — different sensors do not share state.
    ntp_or_ptp_samples : Iterable[float]
        Microsecond offsets measured by the NTP/PTP client. At least two
        samples are required so the standard deviation is well-defined.

    Returns
    -------
    dict
        ``{"offset_us": float, "offset_us_bound": float,
           "last_measured_at_iso8601": str}``. The dict is a fresh copy
        of the cached value — callers can mutate it without disturbing
        future ``get_cached_offset`` reads.

    Raises
    ------
    ValueError
        If ``sensor_id`` is not a non-empty string, or if any sample is
        non-finite, the wrong type, or there are fewer than two samples.
    """
    sid = _validate_sensor_id(sensor_id)
    sample_list = _validate_samples(ntp_or_ptp_samples)

    offset_us = float(statistics.median(sample_list))
    sigma_us = float(statistics.stdev(sample_list))
    offset_us_bound = max(2.0 * sigma_us, _measurement_floor_us)

    estimate = {
        "offset_us": offset_us,
        "offset_us_bound": offset_us_bound,
        "last_measured_at_iso8601": _utc_now_iso(),
    }

    # Store an independent copy so the cache is decoupled from whatever
    # the caller does with the returned dict.
    _cache[sid] = dict(estimate)

    LOG.debug(
        "clock_skew_estimated",
        extra={
            "sensor_id": sid,
            "n_samples": len(sample_list),
            "offset_us": offset_us,
            "offset_us_bound": offset_us_bound,
        },
    )
    return estimate


def get_cached_offset(sensor_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent estimate for ``sensor_id``, or ``None``.

    The returned dict is a copy — mutating it does not affect the cache.
    """
    sid = _validate_sensor_id(sensor_id)
    cached = _cache.get(sid)
    if cached is None:
        return None
    return dict(cached)


def clear_cache() -> None:
    """Wipe the in-memory estimate cache. Intended for test isolation."""
    _cache.clear()


def set_measurement_floor_us(floor_us: float) -> None:
    """
    Override the measurement floor (µs). Intended for tests.

    The floor must be a finite non-negative real number. Resetting back
    to the default is a call to :func:`reset_measurement_floor_us`.
    """
    if isinstance(floor_us, bool) or not isinstance(floor_us, (int, float)):
        raise ValueError(
            f"floor_us: expected real number, got {type(floor_us).__name__}"
        )
    f = float(floor_us)
    if not math.isfinite(f):
        raise ValueError(f"floor_us: must be finite, got {floor_us!r}")
    if f < 0.0:
        raise ValueError(f"floor_us: must be non-negative, got {f}")
    global _measurement_floor_us
    _measurement_floor_us = f


def reset_measurement_floor_us() -> None:
    """Restore the measurement floor to the default (10 µs)."""
    global _measurement_floor_us
    _measurement_floor_us = _DEFAULT_MEASUREMENT_FLOOR_US


def get_measurement_floor_us() -> float:
    """Return the current measurement floor (µs)."""
    return _measurement_floor_us
