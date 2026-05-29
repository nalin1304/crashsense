"""Per-session and aggregate statistics for the long-stretch ambient corpus
(R6.2 / R6.4 — ``backend/audio_model/ambient_stats.py``, task 4.9 in
``.kiro/specs/crashsense-hardening/tasks.md``).

The script reads ``data/ambient_long/manifest.json`` (populated by task 4.8 /
``scripts/curate_ambient.py``) and, for every **non-placeholder** session,
loads the WAV via :mod:`soundfile` and computes:

* ``mean_rms``                     — root-mean-square of the waveform
* ``a_weighted_spl_dbfs``          — mean RMS in dBFS *after* the standard
                                     IEC 61672 A-weighting IIR filter
* ``spectral_centroid_hz``         — distribution {mean, p50, p95, max}
                                     (``librosa.feature.spectral_centroid``)
* ``zero_crossing_rate``           — distribution {mean, p50, p95, max}
                                     (``librosa.feature.zero_crossing_rate``)

Per-session entries are written to ``reports/ambient_stats.json``; aggregate
metrics are duration-weighted across processed sessions. Sessions with
``is_placeholder == true`` are **counted** in ``n_sessions_skipped_placeholder``
but never feed into a metric (the placeholder bytes are synthetic — see
``data/ambient_long/README.md``).

R6.4 fail-open contract: a missing-on-disk, non-WAV, decode-error, or
shape-error session is appended to ``reports/ambient_stats.errors.csv`` and
the remaining sessions continue processing. The script never raises on a
single bad row — only fatal pre-flight problems (missing manifest,
unparsable manifest) hit a non-zero exit code.

Determinism contract:

* Sessions sorted by ``filename`` before iteration.
* Per-session feature reductions go through stable percentile / mean
  reductions on float32 arrays.
* Output JSON serialised with ``json.dumps(..., sort_keys=True, indent=2)``
  so byte-identical inputs produce byte-identical bytes.

CLI::

    python -m backend.audio_model.ambient_stats \\
        [--manifest PATH] \\
        [--out reports/ambient_stats.json] \\
        [--errors reports/ambient_stats.errors.csv]

Exit codes:

    0 — report written successfully (zero or more per-session errors
        appended to the errors CSV)
    1 — fatal pre-flight error (manifest missing, manifest unparsable,
        manifest schema malformed)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Direct execution (``python backend/audio_model/ambient_stats.py``) needs
# the repo root on sys.path before any first-party imports.
if __name__ == "__main__" and __package__ in (None, ""):
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

LOG = logging.getLogger("ambient_stats")

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "ambient_long" / "manifest.json"
DEFAULT_AMBIENT_ROOT = REPO_ROOT / "data" / "ambient_long"
DEFAULT_OUT = REPO_ROOT / "reports" / "ambient_stats.json"
DEFAULT_ERRORS = REPO_ROOT / "reports" / "ambient_stats.errors.csv"

#: Output schema version; bumped when fields change shape.
SCHEMA_VERSION = "1.0.0"

#: Closed enum of error reasons recorded in the errors CSV (R6.4).
ERROR_REASONS = (
    "missing_on_disk",
    "non_wav_extension",
    "decode_error",
    "manifest_entry_invalid",
    "empty_signal",
)

#: Order of columns in the errors CSV. Stable for downstream tooling.
ERROR_CSV_COLUMNS: tuple[str, ...] = (
    "filename",
    "reason",
    "detail",
    "excluded_at_iso8601",
)

#: Floor for log-of-zero protection when computing dBFS. -120 dBFS is well
#: below the noise floor of any real recording so it cannot be confused
#: with a measured value.
DBFS_FLOOR = -120.0


# ---------------------------------------------------------------------------
# A-weighting filter (IEC 61672 / Wikipedia analogue prototype)
# ---------------------------------------------------------------------------


def _a_weighting_iir(sample_rate_hz: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(b, a)`` IIR coefficients implementing IEC 61672 A-weighting.

    Built from the standard analogue prototype (poles at 20.598997 Hz,
    107.65265 Hz, 737.86223 Hz, 12194.217 Hz; double pole at the lowest and
    highest frequencies) digitised via the bilinear transform. The filter
    is normalised to 0 dB at 1 kHz, matching the IEC 61672 reference.

    Implemented locally (rather than via a third-party package) so the
    dependency surface stays at scipy + numpy, both already pinned.
    """
    # scipy is imported lazily so ``--help`` doesn't require it.
    from scipy.signal import bilinear

    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    a1000_db = 1.9997  # nominal A-weight at 1 kHz, used to normalise

    # Continuous-time numerator (s^4) and denominator polynomials.
    nums = [(2 * np.pi * f4) ** 2 * (10 ** (a1000_db / 20)), 0, 0, 0, 0]
    dens = np.polymul(
        [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
    )
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])

    b, a = bilinear(nums, dens, sample_rate_hz)
    return b.astype(np.float64), a.astype(np.float64)


def a_weighted_signal(signal: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Apply IEC 61672 A-weighting to a mono float32 signal.

    Returns a float64 array of the same length as ``signal``.
    """
    from scipy.signal import lfilter

    b, a = _a_weighting_iir(sample_rate_hz)
    return lfilter(b, a, np.asarray(signal, dtype=np.float64))


# ---------------------------------------------------------------------------
# Feature reductions
# ---------------------------------------------------------------------------


def _reduce_distribution(values: np.ndarray) -> dict:
    """Reduce a 1-D feature stream to ``{mean, p50, p95, max}``.

    Uses ``np.percentile`` with the default linear interpolation so
    distribution numbers are reproducible across runs.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(arr.max()),
    }


def _rms(signal: np.ndarray) -> float:
    """Plain-old root-mean-square of a 1-D float signal."""
    arr = np.asarray(signal, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(arr * arr))))


def _rms_to_dbfs(rms: float) -> float:
    """Convert a linear RMS value (0..1 full-scale) to dBFS.

    A floor of ``DBFS_FLOOR`` is returned for non-positive RMS so the
    serialised JSON never carries ``-inf``.
    """
    if rms <= 0.0:
        return DBFS_FLOOR
    db = 20.0 * math.log10(rms)
    return max(db, DBFS_FLOOR)


def compute_session_stats(signal: np.ndarray, sample_rate_hz: int) -> dict:
    """Compute the four R6.2 metric blocks for a single session.

    ``signal`` is expected as a mono float array in ``[-1, 1]`` (the
    convention used by the WAV loader below). The function does not mutate
    ``signal``; a copy is taken if A-weighting requires a different dtype.

    Returns a dict with keys ``mean_rms``, ``a_weighted_spl_dbfs``,
    ``spectral_centroid_hz``, ``zero_crossing_rate``.
    """
    # Lazy import — keeps argparse error paths import-free.
    import librosa

    if signal.size == 0:
        raise ValueError("signal is empty")

    mean_rms = _rms(signal)

    weighted = a_weighted_signal(signal, sample_rate_hz)
    a_weighted_spl_dbfs = _rms_to_dbfs(_rms(weighted))

    # ``librosa.feature.spectral_centroid`` returns shape (1, n_frames). We
    # flatten before reducing to a scalar distribution.
    centroid = librosa.feature.spectral_centroid(
        y=signal.astype(np.float32, copy=False),
        sr=sample_rate_hz,
    )[0]
    spectral_centroid_hz = _reduce_distribution(centroid)

    zcr = librosa.feature.zero_crossing_rate(
        y=signal.astype(np.float32, copy=False),
    )[0]
    zero_crossing_rate = _reduce_distribution(zcr)

    return {
        "mean_rms": float(mean_rms),
        "a_weighted_spl_dbfs": float(a_weighted_spl_dbfs),
        "spectral_centroid_hz": spectral_centroid_hz,
        "zero_crossing_rate": zero_crossing_rate,
    }


# ---------------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------------


def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load ``path`` as mono float32 in ``[-1, 1]``.

    Multi-channel input is downmixed to mono via the per-frame mean.
    Returns ``(signal, sample_rate_hz)``. Lazy soundfile import.
    """
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    return data, int(sample_rate)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Duration-weighted mean. Falls back to plain mean if all weights == 0."""
    if not values:
        return 0.0
    if sum(weights) <= 0.0:
        return float(np.mean(values))
    arr_v = np.asarray(values, dtype=np.float64)
    arr_w = np.asarray(weights, dtype=np.float64)
    return float(np.sum(arr_v * arr_w) / np.sum(arr_w))


def _aggregate(per_session: list[dict]) -> dict:
    """Combine per-session metrics into a duration-weighted aggregate.

    Distribution fields (``spectral_centroid_hz``, ``zero_crossing_rate``)
    are aggregated as a duration-weighted mean of the per-session
    distribution scalars (mean / p50 / p95 / max). This is an approximation
    — pooling raw frames across multi-hour sessions is RAM-prohibitive — but
    it is monotone in the underlying values and reproducible.
    """
    if not per_session:
        return {
            "mean_rms": 0.0,
            "a_weighted_spl_dbfs": 0.0,
            "spectral_centroid_hz": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "zero_crossing_rate": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        }

    weights = [float(s.get("_duration_seconds", 0.0)) for s in per_session]

    rms_values = [float(s["mean_rms"]) for s in per_session]
    spl_values = [float(s["a_weighted_spl_dbfs"]) for s in per_session]

    def _agg_dist(key: str) -> dict:
        return {
            stat: _weighted_mean(
                [float(s[key][stat]) for s in per_session],
                weights,
            )
            for stat in ("mean", "p50", "p95", "max")
        }

    return {
        "mean_rms": _weighted_mean(rms_values, weights),
        "a_weighted_spl_dbfs": _weighted_mean(spl_values, weights),
        "spectral_centroid_hz": _agg_dist("spectral_centroid_hz"),
        "zero_crossing_rate": _agg_dist("zero_crossing_rate"),
    }


# ---------------------------------------------------------------------------
# Errors CSV writer
# ---------------------------------------------------------------------------


class _ErrorsCsv:
    """Append-only writer for ``reports/ambient_stats.errors.csv`` (R6.4).

    Header is written when the file is empty so re-runs extend the existing
    diagnostic trail rather than overwriting it.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(ERROR_CSV_COLUMNS))
        if self._path.stat().st_size == 0:
            self._writer.writeheader()
            self._fh.flush()

    def append(self, *, filename: str, reason: str, detail: str = "") -> None:
        if reason not in ERROR_REASONS:
            raise ValueError(f"unknown reason {reason!r}; expected one of {ERROR_REASONS}")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._writer.writerow(
            {
                "filename": filename,
                "reason": reason,
                "detail": detail,
                "excluded_at_iso8601": ts,
            }
        )
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "_ErrorsCsv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict:
    """Load and structurally validate the ambient manifest.

    Raises ``FileNotFoundError`` if the manifest is missing, ``ValueError``
    if it is unparsable or has the wrong shape. Both are pre-flight fatal
    (R6.4 only mandates fail-open at the *session* level).
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            manifest = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"could not parse manifest at {path}: {exc}"
            ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest at {path} is not a JSON object")
    if "sessions" not in manifest or not isinstance(manifest["sessions"], list):
        raise ValueError(f"manifest at {path} is missing a 'sessions' list")
    return manifest


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------


def analyse(
    *,
    manifest_path: Path,
    ambient_root: Path,
    out_path: Path,
    errors_path: Path,
) -> int:
    """Run the analysis end-to-end. Returns the process exit code.

    Pure I/O orchestration; per-session feature extraction lives in
    :func:`compute_session_stats`. R6.4 fail-open semantics are enforced
    here: every per-session error is recorded to the errors CSV and the
    loop continues.
    """
    try:
        manifest = _load_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sessions: list[dict] = list(manifest.get("sessions") or [])
    # R6.4 — sort by filename for deterministic iteration. Stable on equal
    # keys (Python's sort is stable) so manifest order is preserved within
    # ties, but the test corpus uses unique filenames so this only matters
    # as a determinism guarantee.
    sessions.sort(key=lambda entry: str(entry.get("filename") or ""))

    per_session_records: list[dict] = []
    n_total = len(sessions)
    n_skipped_placeholder = 0
    n_processed = 0

    with _ErrorsCsv(errors_path) as errors:
        for entry in sessions:
            filename_obj = entry.get("filename")
            filename = str(filename_obj) if filename_obj else ""

            # ---- skip placeholders without touching disk -------------------
            if entry.get("is_placeholder") is True:
                n_skipped_placeholder += 1
                continue

            # ---- structural sanity on the manifest entry ------------------
            if not filename:
                errors.append(
                    filename="<unknown>",
                    reason="manifest_entry_invalid",
                    detail="missing 'filename' field",
                )
                continue

            wav_path = ambient_root / filename
            if not wav_path.exists():
                errors.append(
                    filename=filename,
                    reason="missing_on_disk",
                    detail=str(wav_path),
                )
                continue
            if wav_path.suffix.lower() != ".wav":
                errors.append(
                    filename=filename,
                    reason="non_wav_extension",
                    detail=wav_path.suffix or "<no extension>",
                )
                continue

            # ---- decode + feature extraction (R6.2) ----------------------
            try:
                signal, sample_rate_hz = _load_mono_wav(wav_path)
            except Exception as exc:  # noqa: BLE001 — fail-open per R6.4
                errors.append(
                    filename=filename,
                    reason="decode_error",
                    detail=f"{type(exc).__name__}: {exc}",
                )
                continue

            if signal.size == 0:
                errors.append(
                    filename=filename,
                    reason="empty_signal",
                    detail="soundfile returned a zero-length array",
                )
                continue

            try:
                stats = compute_session_stats(signal, sample_rate_hz)
            except Exception as exc:  # noqa: BLE001 — fail-open per R6.4
                errors.append(
                    filename=filename,
                    reason="decode_error",
                    detail=f"feature extraction failed: "
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            duration_seconds = float(signal.size) / float(sample_rate_hz)
            record = {
                "filename": filename,
                "duration_seconds": duration_seconds,
                "sample_rate_hz": int(sample_rate_hz),
                **stats,
                # ``_duration_seconds`` is private — used by the aggregator
                # and stripped from the JSON output below.
                "_duration_seconds": duration_seconds,
            }
            per_session_records.append(record)
            n_processed += 1

    aggregate = _aggregate(per_session_records)

    # Strip the private duration field before serialising.
    public_per_session = []
    for rec in per_session_records:
        public_per_session.append(
            {k: v for k, v in rec.items() if not k.startswith("_")}
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_iso8601": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "n_sessions_total": int(n_total),
        "n_sessions_skipped_placeholder": int(n_skipped_placeholder),
        "n_sessions_processed": int(n_processed),
        "per_session": public_per_session,
        "aggregate": aggregate,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ambient_stats",
        description=(
            "Compute per-session and aggregate statistics for the long-stretch "
            "ambient capture corpus (R6.2)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to manifest.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--ambient-root",
        type=Path,
        default=DEFAULT_AMBIENT_ROOT,
        help=f"WAV root directory (default: {DEFAULT_AMBIENT_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"report output path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS,
        help=f"errors CSV output path (default: {DEFAULT_ERRORS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    return analyse(
        manifest_path=Path(args.manifest),
        ambient_root=Path(args.ambient_root),
        out_path=Path(args.out),
        errors_path=Path(args.errors),
    )


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "SCHEMA_VERSION",
    "ERROR_REASONS",
    "ERROR_CSV_COLUMNS",
    "DBFS_FLOOR",
    "a_weighted_signal",
    "compute_session_stats",
    "analyse",
    "main",
]
