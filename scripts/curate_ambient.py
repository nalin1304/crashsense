"""Ingestion helper for the long-stretch ambient capture corpus.

This script is *tooling*, not corpus population. It records provenance
for ambient highway WAVs and appends session entries to
``data/ambient_long/manifest.json`` per R6.1 (task 4.8 in
``.kiro/specs/crashsense-hardening/tasks.md``):

> ≥3 distinct recording sessions, ≥24 cumulative hours of ambient
> highway audio, manifest with ``filename``, ``duration_seconds``,
> ``sample_rate_hz``, ``recorded_at_iso8601``, ``location_label``.

The helper does **not** record audio itself — the operator captures the
audio with their own equipment, then passes the resulting WAV via
``--audio-file``.

Subcommands
-----------

``add``       Probe a WAV via ``soundfile.info()`` for its duration and
              sample rate, validate them (duration > 60 s, sample rate
              in ``{16000, 22050, 44100, 48000}``), and append a
              session entry to ``manifest.json``. ``--copy-as-managed``
              copies the file to ``data/ambient_long/ambient_NNN.wav``
              using the next available 3-digit sequence; without it the
              helper records the session under the file's existing
              basename.

``validate``  Walk every entry in ``manifest.json``, confirm the file
              exists on disk, and assert the manifest's
              ``duration_seconds`` matches the probed duration within
              1 second. Placeholder sessions (``is_placeholder == true``)
              are skipped because their durations are intentionally
              aspirational stub values.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable so this script can be run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------ constants
DEFAULT_MANIFEST = REPO_ROOT / "data" / "ambient_long" / "manifest.json"
DEFAULT_AMBIENT_ROOT = REPO_ROOT / "data" / "ambient_long"

# R6.1: minimum sanity for a real ambient session. Ten seconds of stub
# audio is much shorter than this, but stubs are not added by this tool —
# they are written by ``_regenerate_placeholders.py`` and carry the
# ``is_placeholder`` flag, so the duration floor for real ingests stays
# strict.
MIN_DURATION_S = 60.0

# Allowed sample rates. 22050 Hz matches
# ``backend.audio_model.spectrogram_gen.SAMPLE_RATE``; the others are
# common capture rates we accept and downsample later.
ALLOWED_SAMPLE_RATES = (16000, 22050, 44100, 48000)

# Filename naming convention for managed copies.
AMBIENT_FILENAME_RE = re.compile(r"^ambient_(?P<seq>\d{3})\.wav$")

# Match of the ``validate`` duration tolerance — soundfile reports frames
# at sample-rate granularity, so 1 s is comfortably above any rounding
# noise while still catching truncated or extended recordings.
DURATION_MATCH_TOLERANCE_S = 1.0


# ------------------------------------------------------------------ helpers
def _utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_iso8601(value: str) -> str:
    """Round-trip ``value`` through ``datetime.fromisoformat`` and re-render.

    Accepts both ``"2025-01-15T00:00:00Z"`` and the explicit-offset form
    ``"2025-01-15T00:00:00+00:00"``. Always re-emits with the trailing
    ``Z`` for consistency with the rest of the tooling.
    """
    raw = value.strip()
    if raw.endswith("Z"):
        # Python <3.11 cannot parse the "Z" suffix directly.
        parseable = raw[:-1] + "+00:00"
    else:
        parseable = raw
    try:
        parsed = datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise ValueError(f"invalid ISO 8601 timestamp {raw!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise ValueError(
            f"timestamp {raw!r} has no timezone — must be UTC (Z or +00:00)"
        )
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_manifest(manifest_path: Path) -> dict:
    """Load ``manifest.json`` (return a fresh skeleton if missing)."""
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return {
            "schema_version": "1.0.0",
            "generated_at_iso8601": _utc_now_iso(),
            "sessions": [],
        }
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("schema_version", "1.0.0")
    manifest.setdefault("generated_at_iso8601", _utc_now_iso())
    manifest.setdefault("sessions", [])
    if not isinstance(manifest["sessions"], list):
        raise ValueError(
            f"manifest at {manifest_path} has non-list 'sessions' field"
        )
    return manifest


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    """Write ``manifest`` with sorted keys + trailing newline (deterministic)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _next_managed_filename(sessions: list[dict]) -> str:
    """Compute the next ``ambient_NNN.wav`` filename across all sessions."""
    highest = 0
    for entry in sessions:
        match = AMBIENT_FILENAME_RE.match(entry.get("filename", "") or "")
        if match is None:
            continue
        seq = int(match.group("seq"))
        if seq > highest:
            highest = seq
    return f"ambient_{highest + 1:03d}.wav"


def _probe_wav(path: Path) -> tuple[float, int]:
    """Return ``(duration_seconds, sample_rate_hz)`` via ``soundfile.info``.

    Imported lazily so the script can still expose ``--help`` and run
    its argparse error paths without ``soundfile`` installed.
    """
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate), int(info.samplerate)


# ------------------------------------------------------------------ subcommands
def cmd_add(args: argparse.Namespace) -> int:
    """``add`` subcommand. Returns process exit code."""
    manifest_path = Path(args.manifest).resolve()
    ambient_root = Path(args.ambient_root).resolve()

    # ---- 1. Argument validation ---------------------------------------------
    audio_file = Path(args.audio_file).resolve()
    if not audio_file.exists():
        print(
            f"error: --audio-file does not exist: {audio_file}",
            file=sys.stderr,
        )
        return 4
    if audio_file.suffix.lower() != ".wav":
        print(
            f"error: --audio-file must be a .wav file (got {audio_file.suffix})",
            file=sys.stderr,
        )
        return 4

    location_label = args.location_label.strip()
    if not location_label:
        print("error: --location-label must not be empty", file=sys.stderr)
        return 2

    # Recorded-at: argparse default of `None` -> use current UTC.
    if args.recorded_at:
        try:
            recorded_at_iso = _validate_iso8601(args.recorded_at)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        recorded_at_iso = _utc_now_iso()

    # ---- 2. Probe the WAV ---------------------------------------------------
    try:
        duration_seconds, sample_rate_hz = _probe_wav(audio_file)
    except Exception as exc:  # noqa: BLE001
        print(
            f"error: could not probe {audio_file} via soundfile: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 4

    if duration_seconds <= MIN_DURATION_S:
        print(
            f"error: duration {duration_seconds:.2f}s is below the "
            f"{MIN_DURATION_S}s sanity floor for an ambient session "
            "(R6.1 expects long-stretch recordings)",
            file=sys.stderr,
        )
        return 2
    if sample_rate_hz not in ALLOWED_SAMPLE_RATES:
        print(
            f"error: sample rate {sample_rate_hz} Hz not in allowed set "
            f"{ALLOWED_SAMPLE_RATES}",
            file=sys.stderr,
        )
        return 2

    # ---- 3. Resolve filename + optional managed copy ------------------------
    manifest = _read_manifest(manifest_path)
    sessions: list[dict] = manifest["sessions"]

    if args.copy_as_managed:
        filename = _next_managed_filename(sessions)
        dest = ambient_root / filename
        if dest.exists():
            # Defensive — _next_managed_filename should never collide, but
            # if the manifest is out of sync with the directory, refuse.
            print(
                f"error: managed destination {dest} already exists "
                "(manifest may be out of sync with the directory)",
                file=sys.stderr,
            )
            return 4
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_file, dest)
        print(f"copied {audio_file} -> {dest}")
    else:
        filename = audio_file.name

    # Reject duplicates (same filename already in manifest).
    if any(entry.get("filename") == filename for entry in sessions):
        print(
            f"error: filename {filename!r} already in manifest at "
            f"{manifest_path}",
            file=sys.stderr,
        )
        return 3

    # ---- 4. Append the session entry ---------------------------------------
    new_entry = {
        "filename": filename,
        "duration_seconds": float(duration_seconds),
        "sample_rate_hz": int(sample_rate_hz),
        "recorded_at_iso8601": recorded_at_iso,
        "location_label": location_label,
    }
    sessions.append(new_entry)
    manifest["generated_at_iso8601"] = _utc_now_iso()
    _write_manifest(manifest_path, manifest)

    print(
        f"appended session to {manifest_path}: "
        f"filename={filename} duration_seconds={duration_seconds:.2f} "
        f"sample_rate_hz={sample_rate_hz}"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """``validate`` subcommand. Returns process exit code."""
    manifest_path = Path(args.manifest).resolve()
    ambient_root = Path(args.ambient_root).resolve()

    if not manifest_path.exists():
        print(f"error: {manifest_path} does not exist", file=sys.stderr)
        return 1

    try:
        manifest = _read_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"error: could not parse {manifest_path}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    sessions: list[dict] = manifest.get("sessions") or []
    n_real = 0
    n_placeholder = 0
    errors: list[str] = []

    for idx, entry in enumerate(sessions):
        filename = entry.get("filename")
        if not filename:
            errors.append(f"sessions[{idx}]: missing 'filename'")
            continue

        if entry.get("is_placeholder"):
            n_placeholder += 1
            # Placeholder rows still need to point at a file on disk so
            # ambient_stats.py can decide to skip them rather than choke.
            path = ambient_root / filename
            if not path.exists():
                errors.append(
                    f"sessions[{idx}] ({filename}): placeholder file "
                    f"missing on disk at {path}"
                )
            continue

        n_real += 1
        path = ambient_root / filename
        if not path.exists():
            errors.append(
                f"sessions[{idx}] ({filename}): file missing on disk at {path}"
            )
            continue

        try:
            probed_duration, probed_rate = _probe_wav(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"sessions[{idx}] ({filename}): could not probe "
                f"({type(exc).__name__}: {exc})"
            )
            continue

        claimed = float(entry.get("duration_seconds", 0.0))
        if abs(probed_duration - claimed) > DURATION_MATCH_TOLERANCE_S:
            errors.append(
                f"sessions[{idx}] ({filename}): manifest claims "
                f"duration_seconds={claimed:.3f} but soundfile probed "
                f"{probed_duration:.3f} (Δ > {DURATION_MATCH_TOLERANCE_S}s)"
            )

        claimed_rate = int(entry.get("sample_rate_hz", 0))
        if claimed_rate != probed_rate:
            errors.append(
                f"sessions[{idx}] ({filename}): manifest claims "
                f"sample_rate_hz={claimed_rate} but soundfile probed "
                f"{probed_rate}"
            )

    print(
        f"{manifest_path}: {len(sessions)} sessions "
        f"({n_real} real, {n_placeholder} placeholder, {len(errors)} errors)"
    )
    for err in errors:
        print(f"  ERROR  {err}", file=sys.stderr)
    return 0 if not errors else 1


# ------------------------------------------------------------------ argparse
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curate_ambient",
        description=(
            "Provenance-recording ingestion helper for the long-stretch "
            "ambient capture corpus (R6.1)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # add ----------------------------------------------------------------
    p_add = sub.add_parser("add", help="record one new session in manifest.json")
    p_add.add_argument(
        "--audio-file",
        required=True,
        help="path to the WAV recording to register",
    )
    p_add.add_argument(
        "--location-label",
        required=True,
        help="short human-readable tag, e.g. I-5_north_milepost_142",
    )
    p_add.add_argument(
        "--recorded-at",
        default=None,
        help="ISO 8601 UTC timestamp when the recording was captured "
        "(default: current UTC time)",
    )
    p_add.add_argument(
        "--copy-as-managed",
        action="store_true",
        help="copy the source WAV to data/ambient_long/ambient_NNN.wav "
        "with the next available 3-digit sequence; without this flag "
        "the helper records the session under the file's existing basename",
    )
    p_add.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"default: {DEFAULT_MANIFEST}",
    )
    p_add.add_argument(
        "--ambient-root",
        default=str(DEFAULT_AMBIENT_ROOT),
        help=f"default: {DEFAULT_AMBIENT_ROOT}",
    )
    p_add.set_defaults(func=cmd_add)

    # validate -----------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help=(
            "walk manifest.json, confirm each file exists, and assert "
            "duration_seconds matches the probed value within 1 second"
        ),
    )
    p_validate.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )
    p_validate.add_argument(
        "--ambient-root",
        default=str(DEFAULT_AMBIENT_ROOT),
    )
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
