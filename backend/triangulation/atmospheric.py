"""
Atmospheric correction for the TDOA solver.

Implements R11 (Atmospheric_Corrector) from the CrashSense Hardening spec.

Public API
----------
effective_speed_of_sound(temperature_c, humidity_pct, wind_vector_mps,
                         propagation_unit_vector) -> float
    Effective speed of sound (m/s) with a wind-projection term:

        c_eff = c_still(T, RH) + dot(wind_vec, prop_vec)

    where c_still is the Simon (1965) approximation already exposed by
    ``backend.triangulation.sensor_config.speed_of_sound_at``.

default_atmospheric_inputs() -> dict
    The R11.6 substitution defaults used when measured atmospherics are
    unavailable: T = 20 °C, RH = 50 %, wind = (0, 0, 0). Pure data, no
    logging — safe to call from anywhere.

apply_default_atmospheric_inputs(correlation_id) -> dict
    Same defaults as above, but emits a single INFO log line tagged with
    the caller-supplied ``correlation_id``. Callers should invoke this
    at most once per CrashEvent so the log volume tracks crash count
    rather than atmospheric-lookup count (R11.6).

Properties (validated by ``tests/pbt/test_atmospheric_properties.py`` in
task 5.3 of the hardening plan):

- P6 (R11.3) — temperature monotonicity: with humidity and wind held
  fixed, ``effective_speed_of_sound`` is strictly increasing in
  ``temperature_c``. Direct consequence of ``c_still`` being linear in
  T with a positive slope (0.606 m/s/°C) and the wind term being
  independent of T.

- P7 (R11.4) — wind sign-correctness: when ``dot(wind, prop) > 0`` the
  returned speed exceeds ``c_still``; when the dot product is negative,
  the returned speed is below ``c_still``. Direct consequence of the
  additive wind-projection term.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

from .sensor_config import speed_of_sound_at

LOG = logging.getLogger("atmospheric")

Vector3 = Tuple[float, float, float]

# --- Validation bounds (R11.2) -------------------------------------------
_TEMP_MIN_C = -40.0
_TEMP_MAX_C = 60.0
_RH_MIN_PCT = 0.0
_RH_MAX_PCT = 100.0
_WIND_COMPONENT_MIN_MPS = -50.0
_WIND_COMPONENT_MAX_MPS = 50.0
_PROP_UNIT_NORM_MIN = 0.999
_PROP_UNIT_NORM_MAX = 1.001

# --- Substitution defaults (R11.6) ---------------------------------------
_DEFAULT_TEMPERATURE_C: float = 20.0
_DEFAULT_HUMIDITY_PCT: float = 50.0
_DEFAULT_WIND_VECTOR_MPS: Vector3 = (0.0, 0.0, 0.0)


def _validate_scalar(value: float, lo: float, hi: float, field: str) -> float:
    """Range-check a scalar input. Booleans and non-finite values rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field}: expected real number, got {type(value).__name__}"
        )
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{field}: must be finite, got {value!r}")
    if not (lo <= v <= hi):
        raise ValueError(f"{field}: must be in [{lo}, {hi}], got {v}")
    return v


def _validate_vector3(vec, field: str) -> Vector3:
    """Coerce a 3-element sequence of finite floats."""
    if not isinstance(vec, (tuple, list)):
        raise ValueError(
            f"{field}: expected 3-tuple of floats, got {type(vec).__name__}"
        )
    if len(vec) != 3:
        raise ValueError(
            f"{field}: expected 3 components, got {len(vec)}"
        )
    out: list[float] = []
    for i, c in enumerate(vec):
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            raise ValueError(
                f"{field}[{i}]: expected real number, got {type(c).__name__}"
            )
        cf = float(c)
        if not math.isfinite(cf):
            raise ValueError(f"{field}[{i}]: must be finite, got {c!r}")
        out.append(cf)
    return (out[0], out[1], out[2])


def effective_speed_of_sound(
    temperature_c: float,
    humidity_pct: float,
    wind_vector_mps: Vector3,
    propagation_unit_vector: Vector3,
) -> float:
    """
    Effective speed of sound including a wind-projection term.

    Returns ``c_still(T, RH) + dot(wind, prop)`` where ``c_still`` is the
    temperature/humidity-corrected still-air speed from
    ``sensor_config.speed_of_sound_at``.

    All inputs are validated per R11.2; out-of-range or non-finite values
    raise ``ValueError`` naming the offending field. Booleans are
    rejected so a stray ``True``/``False`` cannot silently be coerced
    to 1.0/0.0.

    Parameters
    ----------
    temperature_c : float
        Air temperature in degrees Celsius. Range [-40.0, 60.0].
    humidity_pct : float
        Relative humidity in percent. Range [0.0, 100.0].
    wind_vector_mps : (float, float, float)
        Wind vector in metres per second. Each component in [-50.0, 50.0].
    propagation_unit_vector : (float, float, float)
        Direction of acoustic propagation as a unit vector. L2 norm in
        [0.999, 1.001] to allow for floating-point round-off.

    Returns
    -------
    float
        Effective speed of sound in metres per second. Always positive
        for inputs inside the validated ranges (worst case
        ≈ 307 - 87 = 220 m/s).
    """
    t_c = _validate_scalar(temperature_c, _TEMP_MIN_C, _TEMP_MAX_C, "temperature_c")
    rh = _validate_scalar(humidity_pct, _RH_MIN_PCT, _RH_MAX_PCT, "humidity_pct")

    wind = _validate_vector3(wind_vector_mps, "wind_vector_mps")
    for i, comp in enumerate(wind):
        if not (_WIND_COMPONENT_MIN_MPS <= comp <= _WIND_COMPONENT_MAX_MPS):
            raise ValueError(
                f"wind_vector_mps[{i}]: must be in "
                f"[{_WIND_COMPONENT_MIN_MPS}, {_WIND_COMPONENT_MAX_MPS}], got {comp}"
            )

    prop = _validate_vector3(propagation_unit_vector, "propagation_unit_vector")
    norm = math.sqrt(prop[0] * prop[0] + prop[1] * prop[1] + prop[2] * prop[2])
    if not (_PROP_UNIT_NORM_MIN <= norm <= _PROP_UNIT_NORM_MAX):
        raise ValueError(
            f"propagation_unit_vector: L2 norm must be in "
            f"[{_PROP_UNIT_NORM_MIN}, {_PROP_UNIT_NORM_MAX}], got {norm:.6f}"
        )

    c_still = speed_of_sound_at(t_c, rh)
    wind_proj = wind[0] * prop[0] + wind[1] * prop[1] + wind[2] * prop[2]
    return float(c_still + wind_proj)


def default_atmospheric_inputs() -> dict:
    """
    Return the R11.6 substitution defaults without logging.

    Use case: callers that already have measured atmospherics and only
    need a starter dict to merge into. For the logged variant that
    should fire once per CrashEvent, use
    :func:`apply_default_atmospheric_inputs`.
    """
    return {
        "temperature_c": _DEFAULT_TEMPERATURE_C,
        "humidity_pct": _DEFAULT_HUMIDITY_PCT,
        "wind_vector_mps": _DEFAULT_WIND_VECTOR_MPS,
    }


def apply_default_atmospheric_inputs(correlation_id: str | None = None) -> dict:
    """
    Return R11.6 defaults and emit a single INFO log line.

    Intended to be called at most once per CrashEvent (R11.6) so the log
    volume tracks crash count rather than atmospheric-lookup count.

    Parameters
    ----------
    correlation_id : str | None
        The CrashEvent correlation id to attach to the log record. ``None``
        is allowed but discouraged — it makes incident triage harder.
    """
    LOG.info(
        "atmospheric_defaults_substituted",
        extra={
            "correlation_id": correlation_id,
            "temperature_c": _DEFAULT_TEMPERATURE_C,
            "humidity_pct": _DEFAULT_HUMIDITY_PCT,
            "wind_vector_mps": _DEFAULT_WIND_VECTOR_MPS,
        },
    )
    return default_atmospheric_inputs()
