"""
TDOA forward simulation and overdetermined inverse localization.

Improvements over the previous version:
  * Supports an arbitrary sensor count (≥ 3). With 4+ sensors the system
    is overdetermined; we solve the least-squares problem directly via
    `scipy.optimize.least_squares` instead of `fsolve`, which is more
    robust to noise and can reject single-sensor outliers.
  * Temperature- and humidity-corrected speed of sound (Simon 1965).
  * Optional ``atmospheric_inputs`` argument (R11.5) — when supplied,
    the solver swaps the constant ``SPEED_OF_SOUND`` for the effective
    speed derived from measured temperature, humidity, and wind. The
    API layer is responsible for sourcing the inputs (and for gating
    on ``deployment.mode == "production"``); the solver itself is
    mode-agnostic.
  * Clock-skew aware: each sensor record carries a microsecond-scale
    offset that the forward simulator applies and the solver can be
    configured to estimate jointly.
  * Multipath/reflection resilience via Huber loss instead of plain L2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from geopy.distance import geodesic
from scipy.optimize import least_squares

from .sensor_config import (
    SPEED_OF_SOUND,
    Sensor,
    speed_of_sound_at,
    sensor_triangle_bbox,
)

LOG = logging.getLogger("tdoa")


def _sensor_view(sensor) -> tuple[str, float, float, float]:
    """Return (sensor_id, lat, lon, clock_offset_us) for any sensor record shape."""
    if isinstance(sensor, Sensor):
        return sensor.sensor_id, sensor.lat, sensor.lon, sensor.clock_offset_us
    if isinstance(sensor, dict):
        return (
            sensor.get("id") or sensor.get("name") or "",
            float(sensor["lat"]),
            float(sensor["lon"]),
            float(sensor.get("clock_offset_us", 0.0)),
        )
    return (
        getattr(sensor, "sensor_id", None) or getattr(sensor, "name", ""),
        float(sensor.lat),
        float(sensor.lon),
        float(getattr(sensor, "clock_offset_us", 0.0)),
    )


def _speed_of_sound(temperature_c: float | None, humidity_pct: float | None) -> float:
    if temperature_c is None and humidity_pct is None:
        return SPEED_OF_SOUND
    return speed_of_sound_at(
        temperature_c if temperature_c is not None else 20.0,
        humidity_pct if humidity_pct is not None else 50.0,
    )


def _effective_c_from_atmospheric_inputs(atmospheric_inputs: Mapping[str, Any]) -> float:
    """Derive a single scalar speed of sound from a Phase-3 atmospheric record.

    Accepts the dict shape returned by
    :func:`backend.triangulation.atmospheric.default_atmospheric_inputs`
    (and the same shape produced by the production atmospheric pipeline):

        {
            "temperature_c": float,
            "humidity_pct":  float,
            "wind_vector_mps": (vx, vy, vz),  # ignored — see note below
        }

    The wind term is intentionally **not** applied here. Applying it
    correctly requires a per-sensor-pair propagation direction (the
    ``c_eff = c_still + dot(wind, prop)`` form from R11.4), which in
    turn depends on the source location the solver is trying to
    estimate. Resolving that chicken-and-egg in the solver requires a
    fixed-point iteration that we defer to a follow-up task; for now
    we apply only the temperature- and humidity-correction (which is
    the dominant contributor on highway-scale baselines and does not
    depend on geometry). The wind component is preserved on the input
    dict so callers and tests can still inspect it.

    Validates the T/RH bounds via the same range-checks as
    ``effective_speed_of_sound``: if the inputs are out of range the
    underlying call raises ``ValueError`` with a field-named message.
    """
    # Local import keeps the module dependency graph acyclic — the
    # atmospheric module already depends on sensor_config, and this
    # solver also depends on sensor_config but not on atmospheric.
    from .atmospheric import effective_speed_of_sound

    t_c = atmospheric_inputs.get("temperature_c", 20.0)
    rh = atmospheric_inputs.get("humidity_pct", 50.0)
    # Wind term suppressed by passing the zero vector; the propagation
    # vector below is therefore irrelevant but still has to satisfy the
    # unit-norm validator.
    return float(effective_speed_of_sound(t_c, rh, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))


def simulate_arrival_times(
    crash_lat: float, crash_lon: float, sensors: Iterable[Sensor],
    *,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    apply_clock_offsets: bool = True,
) -> dict[str, float]:
    """Per-sensor arrival time (seconds) for a known crash coordinate.

    With apply_clock_offsets=True the returned times include each sensor's
    simulated clock skew — that's the realistic case the inverse solver
    has to deal with.
    """
    c = _speed_of_sound(temperature_c, humidity_pct)
    out: dict[str, float] = {}
    for sensor in sensors:
        name, s_lat, s_lon, clock_offset_us = _sensor_view(sensor)
        distance_m = geodesic((crash_lat, crash_lon), (s_lat, s_lon)).meters
        t = distance_m / c
        if apply_clock_offsets:
            t += clock_offset_us * 1e-6
        out[name] = t
    return out


@dataclass
class LocalizationResult:
    success: bool
    lat: float | None = None
    lon: float | None = None
    error_message: str | None = None
    rms_residual_seconds: float | None = None  # quality of fit
    inliers: int | None = None                  # sensors used after outlier rejection
    # R13.3: when the failure is "clock_skew_exceeded" the offending sensor
    # id is surfaced as a structured field so callers do not have to parse
    # ``error_message``. None on success and on every other failure mode.
    offending_sensor_id: str | None = None

    def to_dict(self) -> dict:
        if self.success:
            return {
                "success": True,
                "lat": self.lat,
                "lon": self.lon,
                "rms_residual_seconds": self.rms_residual_seconds,
                "inliers": self.inliers,
            }
        payload: dict = {
            "success": False,
            "error": self.error_message or "localization failed",
        }
        if self.offending_sensor_id is not None:
            payload["offending_sensor_id"] = self.offending_sensor_id
        return payload


def tdoa_localize(
    time_delays: Mapping[str, float], sensors: Iterable[Sensor],
    *,
    temperature_c: float | None = None,
    humidity_pct: float | None = None,
    atmospheric_inputs: Mapping[str, Any] | None = None,
    use_huber: bool = True,
    reject_outliers: bool = True,
    bbox_padding_deg: float = 0.001,
    multipath_residual_threshold_s: float | None = 0.005,
    clock_skew_estimates: Mapping[str, Mapping[str, float]] | None = None,
    sensor_offset_bounds: Mapping[str, float] | None = None,
    sensor_timeout_ms: int = 1000,
) -> LocalizationResult:
    """
    Overdetermined inverse TDOA via nonlinear least squares.

    With N sensors and 2 unknowns (lat, lon) we have N-1 TDOA residuals
    relative to the reference (smallest-arrival) sensor. For N ≥ 4 the
    problem is overdetermined and we use `scipy.optimize.least_squares`
    with optional Huber loss for robustness to multipath outliers.

    Sensor count contract (R14.1): the solver accepts ``N ∈ [3, 16]``
    and minimises a least-squares residual over **every** sensor whose
    arrival time has been reported. Configurations outside that range
    are rejected at entry — ``N < 3`` returns
    ``error_message="need at least 3 sensors"`` (legacy contract); ``N
    > 16`` returns ``error_message="too many sensors (N > 16)"``.

    Single-sensor-failure tolerance (R14.2 / R14.3 / R14.4). The solver
    itself does not observe wall-clock arrival times; it operates on
    the ``time_delays`` dict the caller hands in. A *missing* sensor
    is one whose ``sensor_id`` appears in ``sensors`` but **not** in
    ``time_delays`` — by convention the upstream pipeline drops sensors
    that have not reported within ``sensor_timeout_ms`` (integer in
    [100, 5000] ms, default 1000) before it calls us. We then enforce
    the failure-tolerance contract:

    * ``N == 3`` and one or more sensors missing → fail fast with
      ``error_message="insufficient_sensors"`` (R14.3).
    * ``N >= 4`` and exactly one sensor missing → solve with the
      remaining ``N - 1`` sensors (R14.2, drop-one tolerant).
    * ``N >= 4`` and two or more sensors missing →
      ``error_message="insufficient_sensors"`` (R14.4).

    The ``sensor_timeout_ms`` knob is validated and surfaced here for
    callers that want to plumb a single configuration value through;
    its semantic enforcement (which sensors have or have not reported)
    is the upstream pipeline's responsibility because only that layer
    has the wall-clock context. The solver therefore treats the
    ``time_delays`` dict it receives as authoritative.

    Permutation symmetry (R14.5): the residual sum is a sum of squares
    over participating sensors, so it is invariant under any
    permutation π applied jointly to ``sensors`` and ``time_delays``.
    The reference-sensor pick is keyed on arrival time, not input
    order, and Python's sort is stable so ties resolve
    deterministically. The multi-start seed list is derived from the
    participating-sensor bbox / centroid (both permutation-invariant).
    Net effect: ``tdoa_localize(π(td), π(s))`` matches
    ``tdoa_localize(td, s)`` to numerical precision for any π.

    If reject_outliers=True, after the first solve we re-fit excluding any
    sensor with residual > 3·median absolute deviation. This drops the
    pathological reflection cases.

    Speed of sound resolution (R11.5):

    * If ``atmospheric_inputs`` is supplied (a dict in the shape returned
      by ``backend.triangulation.atmospheric.default_atmospheric_inputs``),
      its ``temperature_c`` / ``humidity_pct`` drive the effective speed
      of sound via ``effective_speed_of_sound``. The wind component is
      currently not applied at the solver level — see
      :func:`_effective_c_from_atmospheric_inputs` for the rationale.
      ``atmospheric_inputs`` takes precedence over the legacy
      ``temperature_c`` / ``humidity_pct`` keyword arguments when both
      are supplied (this matches the request-context plumbing pattern
      used by the API layer).
    * Otherwise the legacy ``temperature_c`` / ``humidity_pct`` keywords
      are honoured if provided.
    * Otherwise the constant ``SPEED_OF_SOUND`` (≈ 343 m/s @ 20 °C) is
      used. This preserves the pre-R11 behaviour for every existing
      caller — the new keyword is opt-in.

    The solver itself is mode-agnostic: gating on
    ``deployment.mode == "production"`` is the caller's responsibility
    (the API route reads the active deployment and decides whether to
    look up measured atmospherics or substitute the R11.6 defaults).

    Multipath rejection (R12.1–R12.4):

    The optimizer is launched from several seed locations (the cluster
    centroid plus the four bbox corners). Each successful run produces a
    candidate solution; the **per-residual RMS** ``sqrt(mean(r²))`` is
    computed on the candidate's inlier residuals (after the existing
    3·MAD outlier pass, if enabled). Candidates whose RMS exceeds
    ``multipath_residual_threshold_s`` are rejected as inconsistent with
    a single-source TDOA model — the most common cause is a strong
    reflected (multipath) path that no direct-source point can explain.

    Note on terminology: R12.2 names this quantity the "residual L2
    norm". The design explicitly fixes the default at 5 ms — described
    as "the order of magnitude of one sample at 22050 Hz over typical
    baselines", i.e. a per-residual scale — and a vector L2 sum would
    grow as σ·√N and clip 6-sensor deployments at the 5 ms boundary
    (regressing the R14.6 / R31.1 noise-budget contracts). We therefore
    interpret R12.2 as a per-residual norm: ``sqrt(mean(r²))``. The
    legacy field ``rms_residual_seconds`` already exposes this same
    quantity to downstream consumers.

    From the surviving candidates we pick the one with the **smallest**
    residual RMS; ties are broken by the candidate's geodesic distance
    to the centroid of the participating sensors (real crashes happen
    inside the sensor cluster). When **no** candidate qualifies, the
    solver returns ``LocalizationResult(success=False,
    error_message="multipath_rejected")`` and does **not** populate
    ``lat`` / ``lon``.

    Setting ``multipath_residual_threshold_s=None`` disables the filter
    entirely (legacy behaviour); this is intended for diagnostics only
    and is not the path taken by production callers.

    Clock skew (R13.2, R13.3):

    * ``clock_skew_estimates`` is an optional mapping from ``sensor_id``
      to the dict shape returned by
      :func:`backend.triangulation.clock_skew.estimate_offset` —
      ``{"offset_us": float, "offset_us_bound": float, ...}``. When
      supplied, ``offset_us * 1e-6`` (seconds) is **subtracted** from
      that sensor's arrival time before residual computation, so the
      solver works with skew-corrected delays.
    * ``sensor_offset_bounds`` is an optional mapping from ``sensor_id``
      to its declared ``SensorRecord.clock_offset_us_bound`` (also in
      microseconds). Callers source this from the active deployment
      (``get_active_deployment().sensors``); the solver itself stays
      deployment-agnostic.
    * For every sensor present in **both** mappings, if
      ``clock_skew_estimates[sid]["offset_us_bound"]`` exceeds
      ``sensor_offset_bounds[sid]`` the solver fails fast with
      ``LocalizationResult(success=False,
      error_message="clock_skew_exceeded", offending_sensor_id=sid)``
      and does **not** populate ``lat`` / ``lon`` (R13.3). A sensor
      missing from either mapping is left alone — the enforcement is
      strictly opt-in. Sensors that are present in
      ``clock_skew_estimates`` but **not** in ``time_delays`` (i.e. not
      participating in this fix) are also skipped: R13.3 only requires
      enforcement on participating sensors.
    * Both arguments default to ``None``, preserving the pre-R13
      behaviour for every existing caller.
    """
    if multipath_residual_threshold_s is not None:
        if not (0.0001 <= multipath_residual_threshold_s <= 0.1):
            raise ValueError(
                "multipath_residual_threshold_s must be in [0.0001, 0.1] s "
                "(R12.2); got %r" % (multipath_residual_threshold_s,)
            )

    # ---- sensor_timeout_ms validation (R14.2) --------------------------
    # The knob is metadata for the solver — wall-clock missing-sensor
    # detection happens in the upstream pipeline that builds
    # ``time_delays``. We still validate the range here so a single
    # config value can be plumbed end-to-end without callers having to
    # repeat the bounds check.
    if not (100 <= int(sensor_timeout_ms) <= 5000):
        raise ValueError(
            "sensor_timeout_ms must be an integer in [100, 5000] ms "
            "(R14.2); got %r" % (sensor_timeout_ms,)
        )

    if atmospheric_inputs is not None:
        c = _effective_c_from_atmospheric_inputs(atmospheric_inputs)
    else:
        c = _speed_of_sound(temperature_c, humidity_pct)
    sensor_list = list(sensors)
    if len(sensor_list) < 3:
        return LocalizationResult(False, error_message="need at least 3 sensors")
    # R14.1: cap the deployment at 16 sensors. Anything larger is
    # rejected at entry — the optimizer is permutation-invariant but
    # we still want to keep the runtime bounded and the contract
    # matched to the SensorDeployment Pydantic schema (which caps at
    # 16 via ``Field(min_length=3, max_length=16)``).
    if len(sensor_list) > 16:
        return LocalizationResult(
            False, error_message="too many sensors (N > 16)",
        )

    items: list[tuple[str, float, float, float]] = []  # (name, lat, lon, t)
    for sensor in sensor_list:
        name, s_lat, s_lon, _offset_us = _sensor_view(sensor)
        if name not in time_delays:
            continue
        items.append((name, s_lat, s_lon, float(time_delays[name])))

    # ---- single-sensor-failure tolerance (R14.2 / R14.3 / R14.4) -------
    # ``N`` here is the number of sensors the deployment expects, not
    # the number that actually reported. ``missing`` is the count of
    # expected sensors whose arrival time was not handed to us — the
    # caller's missing-within-timeout enforcement is what populates
    # this. We then apply the rule table from the docstring.
    n_expected = len(sensor_list)
    n_reporting = len(items)
    n_missing = n_expected - n_reporting
    if n_expected == 3 and n_missing >= 1:
        # R14.3: 3-sensor deployment, any missing → fail fast.
        return LocalizationResult(
            False, error_message="insufficient_sensors",
        )
    if n_expected >= 4 and n_missing >= 2:
        # R14.4: 4+-sensor deployment, drop-one tolerant only.
        return LocalizationResult(
            False, error_message="insufficient_sensors",
        )

    if len(items) < 3:
        return LocalizationResult(False, error_message="need at least 3 matching delays")

    # ---- clock-skew enforcement (R13.3) --------------------------------
    # Run *before* the skew correction so a deployment that knows a
    # sensor has degraded fails fast rather than silently using a
    # partially-corrected delay. Only sensors that are both
    # ``participating`` (have an entry in ``time_delays``) and present
    # in both mappings are checked. We walk ``items`` in its existing
    # order so the offending-sensor choice is deterministic when more
    # than one sensor exceeds its bound.
    if clock_skew_estimates is not None and sensor_offset_bounds is not None:
        for name, _s_lat, _s_lon, _s_t in items:
            estimate = clock_skew_estimates.get(name)
            if estimate is None:
                continue
            declared_bound = sensor_offset_bounds.get(name)
            if declared_bound is None:
                continue
            measured_bound = float(estimate.get("offset_us_bound", 0.0))
            if measured_bound > float(declared_bound):
                return LocalizationResult(
                    False,
                    error_message="clock_skew_exceeded",
                    offending_sensor_id=name,
                )

    # ---- clock-skew correction (R13.2) ---------------------------------
    # Subtract the estimated offset (µs → s) from each participating
    # sensor's arrival time. ``items`` is rebuilt with corrected times
    # so every downstream computation (reference selection, residuals,
    # tie-break centroid) sees the same numbers.
    if clock_skew_estimates is not None:
        corrected: list[tuple[str, float, float, float]] = []
        for name, s_lat, s_lon, s_t in items:
            estimate = clock_skew_estimates.get(name)
            if estimate is not None:
                offset_s = float(estimate.get("offset_us", 0.0)) * 1e-6
                s_t = s_t - offset_s
            corrected.append((name, s_lat, s_lon, s_t))
        items = corrected

    # Reference sensor = smallest arrival time
    items.sort(key=lambda r: r[3])
    ref_name, ref_lat, ref_lon, ref_t = items[0]

    # ---- bbox / centroid derived from the *participating* sensors -------
    # The legacy ``sensor_triangle_bbox`` / ``sensor_triangle_centroid``
    # helpers read the module-level ``SENSORS`` global, which won't agree
    # with the caller's deployment when the API hands us a custom list.
    # For multipath tie-breaking we want the centroid of the sensors that
    # actually contributed delays (R12.3 — "active sensor centroid").
    item_lats = [it[1] for it in items]
    item_lons = [it[2] for it in items]
    active_centroid_lat = sum(item_lats) / len(items)
    active_centroid_lon = sum(item_lons) / len(items)
    min_lat_active, max_lat_active = min(item_lats), max(item_lats)
    min_lon_active, max_lon_active = min(item_lons), max(item_lons)

    # The bounding-box used for the optimizer is still derived from the
    # module-level helper — that preserves the pre-R12 behaviour for the
    # 3-sensor demo and the highway-corridor layout. We fall back to the
    # active-only bbox if the helpers raise (e.g. when the SENSORS global
    # has been swapped out under us).
    try:
        min_lat_g, max_lat_g, min_lon_g, max_lon_g = sensor_triangle_bbox()
        # If the active set is *outside* the global bbox (custom deployment
        # passed in directly) widen the bbox to cover both — otherwise the
        # optimizer's bounds clip the seed corners and we lose the multi-
        # start coverage.
        min_lat = min(min_lat_g, min_lat_active)
        max_lat = max(max_lat_g, max_lat_active)
        min_lon = min(min_lon_g, min_lon_active)
        max_lon = max(max_lon_g, max_lon_active)
    except Exception:  # pragma: no cover — defensive
        min_lat, max_lat = min_lat_active, max_lat_active
        min_lon, max_lon = min_lon_active, max_lon_active

    bounds = (
        [min_lat - bbox_padding_deg, min_lon - bbox_padding_deg],
        [max_lat + bbox_padding_deg, max_lon + bbox_padding_deg],
    )

    others = items[1:]

    def residuals_at(lat: float, lon: float, items_subset) -> np.ndarray:
        d_ref = geodesic((lat, lon), (ref_lat, ref_lon)).meters
        out = np.zeros(len(items_subset), dtype=np.float64)
        for i, (_, s_lat, s_lon, s_t) in enumerate(items_subset):
            d_i = geodesic((lat, lon), (s_lat, s_lon)).meters
            tdoa_observed = s_t - ref_t
            tdoa_modeled = (d_i - d_ref) / c
            out[i] = tdoa_modeled - tdoa_observed
        return out

    # ---- multi-start optimisation (R12.1, R12.3) ------------------------
    # Seed the optimizer from the cluster centroid plus the four bbox
    # corners. With Huber loss + a 2-unknown problem each start is cheap
    # (sub-millisecond on a typical CPU) so 5 starts is fine. Multiple
    # starts give us coverage when the residual surface has more than one
    # local minimum (the classic phantom-vs-direct-path scenario).
    seed_points: list[tuple[float, float]] = [
        (active_centroid_lat, active_centroid_lon),
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, min_lon),
        (max_lat, max_lon),
    ]

    candidates: list[tuple[float, float, float, int]] = []
    # Each candidate tuple = (lat, lon, residual_rms_seconds, inliers).

    for seed_lat, seed_lon in seed_points:
        # Clip seeds inside the optimizer bounds (the corner seeds sit on
        # the bbox boundary, so we just nudge them inward).
        guess = np.array(
            [
                min(max(seed_lat, bounds[0][0]), bounds[1][0]),
                min(max(seed_lon, bounds[0][1]), bounds[1][1]),
            ],
            dtype=np.float64,
        )

        local_others = others

        def residual_fn(pos, _items=local_others):
            return residuals_at(float(pos[0]), float(pos[1]), _items)

        try:
            sol = least_squares(
                residual_fn,
                guess,
                bounds=bounds,
                loss="huber" if use_huber else "linear",
                f_scale=0.005,  # 5 ms — typical clock-jitter scale
            )
        except Exception:
            continue
        if not sol.success:
            continue

        lat_solved, lon_solved = float(sol.x[0]), float(sol.x[1])

        # Outlier rejection pass — drop sensors with residual > 3·MAD
        # and re-fit. This is the same logic that lived in the original
        # solver; it now runs per multi-start candidate.
        if reject_outliers and len(local_others) >= 4:
            first_residuals = residuals_at(lat_solved, lon_solved, local_others)
            mad = float(
                np.median(np.abs(first_residuals - np.median(first_residuals))) + 1e-9
            )
            kept = [
                it for it, r in zip(local_others, first_residuals)
                if abs(r) <= max(3 * mad, 0.005)
            ]
            if len(kept) >= 3 and len(kept) < len(local_others):
                def residual_fn_kept(pos, _items=kept):
                    return residuals_at(float(pos[0]), float(pos[1]), _items)
                try:
                    sol2 = least_squares(
                        residual_fn_kept, sol.x, bounds=bounds,
                        loss="huber" if use_huber else "linear",
                        f_scale=0.005,
                    )
                    if sol2.success:
                        lat_solved = float(sol2.x[0])
                        lon_solved = float(sol2.x[1])
                        local_others = kept
                except Exception:
                    pass

        final_residuals = residuals_at(lat_solved, lon_solved, local_others)
        if len(final_residuals) == 0:
            continue
        # ``residual_metric`` is the per-residual scale of the candidate
        # fit. R12.2 names it the "L2 norm" but the design explicitly
        # describes the 5 ms default as "the order of magnitude of one
        # sample at 22050 Hz over typical baselines" — a *per-residual*
        # quantity, not a vector-sum quantity. We therefore use RMS
        # (= ||r||_2 / sqrt(N)) so the threshold's interpretation does
        # not depend on the number of sensors. This keeps the legacy
        # σ = 2 ms noise-budget gate (R31.1, R14.6) green: a correct fit
        # to N noisy sensors yields RMS ≈ σ regardless of N, whereas the
        # raw vector L2 grows as σ·√N and would clip 6-sensor deployments
        # at the 5 ms boundary.
        residual_metric = float(np.sqrt(np.mean(final_residuals ** 2)))

        # Reject solutions that drift outside the cluster bbox — these are
        # almost always optimizer artifacts from the corner seeds.
        if not (
            min_lat - bbox_padding_deg <= lat_solved <= max_lat + bbox_padding_deg
            and min_lon - bbox_padding_deg <= lon_solved <= max_lon + bbox_padding_deg
        ):
            continue

        candidates.append((lat_solved, lon_solved, residual_metric, len(local_others) + 1))

    if not candidates:
        # Either every start failed to converge or every result fell
        # outside the bbox — the latter is the dominant case for highly
        # corrupted inputs. Surface a non-multipath error so callers can
        # tell the two failure modes apart.
        return LocalizationResult(
            False, error_message="no candidate solution converged inside bbox",
        )

    # ---- multipath filter (R12.2, R12.4) --------------------------------
    if multipath_residual_threshold_s is not None:
        qualifying = [
            cand for cand in candidates
            if cand[2] <= multipath_residual_threshold_s
        ]
    else:
        qualifying = list(candidates)

    if not qualifying:
        # R12.4: signal multipath rejection without lat/lon. ``inliers``
        # and ``rms_residual_seconds`` are intentionally left None so the
        # caller can't accidentally treat the rejected fit as a partial
        # success.
        return LocalizationResult(
            False, error_message="multipath_rejected",
        )

    # ---- pick best surviving candidate (R12.3) --------------------------
    # Smallest residual RMS wins; tie-break by geodesic distance to the
    # active sensor centroid. We bucket the RMS to 1 ns so numerically-
    # equal multi-start hits aren't ordered by floating-point noise.
    def _tie_break_key(cand: tuple[float, float, float, int]) -> tuple[float, float]:
        lat_c, lon_c, rms, _inliers = cand
        # Round RMS to 1 ns to collapse multi-start duplicates onto the
        # same primary key — this makes the centroid-distance secondary
        # key actually fire when two starts converge to numerically
        # equivalent residuals.
        rms_bucket = round(rms, 9)  # 1 ns
        dist_to_centroid_m = geodesic(
            (lat_c, lon_c), (active_centroid_lat, active_centroid_lon),
        ).meters
        return (rms_bucket, dist_to_centroid_m)

    qualifying.sort(key=_tie_break_key)
    best_lat, best_lon, best_rms, best_inliers = qualifying[0]
    # Re-derive RMS for the chosen candidate so the legacy
    # ``rms_residual_seconds`` field remains consistent with what the
    # filter actually compared against. We need the inlier subset that
    # produced ``best_rms``; recompute it by re-running the residual
    # evaluation against ``others`` and re-applying the same MAD filter
    # (cheap — a few microseconds).
    final_residuals_best = residuals_at(best_lat, best_lon, others)
    if reject_outliers and len(others) >= 4:
        mad_best = float(
            np.median(np.abs(final_residuals_best - np.median(final_residuals_best)))
            + 1e-9
        )
        kept_best = [
            r for r in final_residuals_best if abs(r) <= max(3 * mad_best, 0.005)
        ]
        if len(kept_best) >= 3:
            final_residuals_best = np.asarray(kept_best, dtype=np.float64)
    rms = (
        float(np.sqrt(np.mean(final_residuals_best ** 2)))
        if len(final_residuals_best) else 0.0
    )

    return LocalizationResult(
        True,
        lat=best_lat,
        lon=best_lon,
        rms_residual_seconds=rms,
        inliers=best_inliers,
    )
