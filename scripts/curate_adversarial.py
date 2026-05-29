"""Provenance-recording tool for the Adversarial_Suite (R7.1).

Implements the curation surface called out in task 4.12 of
``.kiro/specs/crashsense-hardening/tasks.md``. Like
``scripts/curate_youtube_clips.py`` this script is *tooling*, not corpus
generation: an annotator obtains a real WAV clip from somewhere
(field recording, FreeSound, archive.org, ...) and feeds it to
``add`` along with the licence string and an optional source URI.

Each clip is validated for:

* ``.wav`` extension (R7.1 — adversarial corpus is WAV-only).
* Duration in the inclusive range 3.0 to 30.0 seconds (R7.1).
* Mono (1 channel) at 22050 Hz, 16-bit PCM (matches
  ``backend/audio_model/spectrogram_gen.SAMPLE_RATE`` so the existing
  inference code can ingest it without resampling).

On success the clip is copied into
``data/adversarial/<category>/<next_filename>.wav`` and a row is
appended to ``data/adversarial/manifest.json``.

Subcommands
-----------
``add``      Ingest one new clip with licence + provenance.
``list``     Print per-category counts and the rows in the manifest.
``validate`` Walk the manifest, confirm every file is present on disk
             and the recorded duration matches the actual WAV.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = REPO_ROOT / "data" / "adversarial"
DEFAULT_MANIFEST = DEFAULT_CORPUS_ROOT / "manifest.json"

# R7.1: the four adversarial categories. Adding a fifth means changing the
# spec, not just this list.
CATEGORIES = ("horns", "fireworks", "tire_blowouts", "airbrakes")

# R7.1 duration window.
MIN_DURATION_S = 3.0
MAX_DURATION_S = 30.0

# Required audio format: mono PCM, 22050 Hz, 16-bit (matches
# spectrogram_gen.SAMPLE_RATE). Real recordings rarely arrive in this
# format directly; the annotator is expected to resample with sox / ffmpeg
# / librosa before invoking ``add``.
REQUIRED_SAMPLE_RATE_HZ = 22050
REQUIRED_CHANNELS = 1
REQUIRED_SUBTYPE = "PCM_16"

# Filename convention: ``<category>_<NNNN>.wav`` with a 4-digit zero-padded
# sequence number per category. Distinct from the placeholder pattern
# ``<category>_placeholder_<NN>.wav`` so the two coexist without collisions.
_REAL_FILENAME_RE = re.compile(
    r"^(?P<category>horns|fireworks|tire_blowouts|airbrakes)"
    r"_(?P<seq>\d{4})\.wav$"
)


# ------------------------------------------------------------------ helpers
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_manifest() -> dict:
    """Return a manifest skeleton that ``--validate`` will accept."""
    return {
        "schema_version": "1.0.0",
        "generated_at_iso8601": _utc_now_iso(),
        "categories": {category: [] for category in CATEGORIES},
    }


def _load_manifest(path: Path) -> dict:
    """Load the manifest, returning a fresh skeleton if the file is missing."""
    if not path.exists():
        return _empty_manifest()
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    # Defensive: legacy / hand-edited manifests may omit a category.
    categories = manifest.setdefault("categories", {})
    for category in CATEGORIES:
        categories.setdefault(category, [])
    return manifest


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort each category by filename for stable diffs across runs.
    categories = manifest.get("categories") or {}
    for category in CATEGORIES:
        rows = categories.get(category) or []
        rows.sort(key=lambda r: r.get("filename", ""))
        categories[category] = rows
    manifest["categories"] = categories
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.write("\n")


def _next_real_filename(category: str, manifest: dict) -> str:
    """Compute the next ``<category>_<NNNN>.wav`` filename for ``category``.

    Walks the manifest entries already present for ``category`` and finds
    the highest 4-digit sequence number among real (non-placeholder) rows.
    Placeholders use a different name pattern and are skipped.
    """
    highest = 0
    for row in manifest.get("categories", {}).get(category, []):
        match = _REAL_FILENAME_RE.match(row.get("filename", "") or "")
        if match is None:
            continue
        if match.group("category") != category:
            continue
        seq = int(match.group("seq"))
        if seq > highest:
            highest = seq
    return f"{category}_{highest + 1:04d}.wav"


def _probe_wav(path: Path) -> tuple[float, int, int, str]:
    """Return ``(duration_s, sample_rate, channels, subtype)`` for a WAV file.

    Imports ``soundfile`` lazily so the ``list`` subcommand and the
    happy-path tests for argument validation can run without it.
    Raises any ``soundfile`` exception unchanged so the caller can decide
    how to surface it.
    """
    import soundfile as sf  # noqa: PLC0415  (lazy by design)

    info = sf.info(str(path))
    duration_s = info.frames / info.samplerate if info.samplerate else 0.0
    return duration_s, info.samplerate, info.channels, info.subtype


# ------------------------------------------------------------------ validation
class _ValidationFailure(Exception):
    """Raised by ``_validate_audio_file`` with an exit-code-friendly message."""

    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


def _validate_audio_file(path: Path) -> float:
    """Check ``path`` matches the adversarial-corpus audio invariants.

    Returns the measured duration in seconds on success, or raises
    ``_ValidationFailure`` with a stable exit code:

    * 4 — extension or existence problem
    * 5 — duration / format / sample-rate / channel / subtype problem
    """
    if not path.exists():
        raise _ValidationFailure(f"audio file does not exist: {path}", 4)
    if path.suffix.lower() != ".wav":
        raise _ValidationFailure(
            f"audio file must be a .wav (got {path.suffix!r})", 4
        )

    try:
        duration_s, sample_rate, channels, subtype = _probe_wav(path)
    except Exception as exc:  # soundfile.LibsndfileError, OSError, ...
        raise _ValidationFailure(
            f"cannot read WAV {path}: {exc.__class__.__name__}: {exc}", 5
        ) from exc

    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        raise _ValidationFailure(
            f"duration {duration_s:.3f}s out of range "
            f"[{MIN_DURATION_S}, {MAX_DURATION_S}] (R7.1)",
            5,
        )
    if sample_rate != REQUIRED_SAMPLE_RATE_HZ:
        raise _ValidationFailure(
            f"sample rate {sample_rate} Hz != required "
            f"{REQUIRED_SAMPLE_RATE_HZ} Hz (resample with sox/ffmpeg/librosa)",
            5,
        )
    if channels != REQUIRED_CHANNELS:
        raise _ValidationFailure(
            f"channel count {channels} != required {REQUIRED_CHANNELS} (mono)",
            5,
        )
    if subtype != REQUIRED_SUBTYPE:
        raise _ValidationFailure(
            f"subtype {subtype!r} != required {REQUIRED_SUBTYPE!r} (16-bit PCM)",
            5,
        )

    return duration_s


# ------------------------------------------------------------------ subcommands
def cmd_add(args: argparse.Namespace) -> int:
    audio_file = Path(args.audio_file).resolve()
    manifest_path = Path(args.manifest).resolve()
    corpus_root = Path(args.corpus_root).resolve()
    category: str = args.category

    if category not in CATEGORIES:
        # argparse `choices=` already enforces this, but defensive in case
        # this function is called programmatically.
        print(
            f"error: category {category!r} not in {CATEGORIES}",
            file=sys.stderr,
        )
        return 2

    if not args.license or not args.license.strip():
        print("error: --license must be a non-empty string", file=sys.stderr)
        return 2

    try:
        duration_s = _validate_audio_file(audio_file)
    except _ValidationFailure as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return exc.exit_code

    manifest = _load_manifest(manifest_path)
    filename = _next_real_filename(category, manifest)
    source_uri = (
        args.source_uri.strip()
        if args.source_uri
        else f"local:{audio_file.name}"
    )

    # Reject duplicate source_uri within the manifest — same provenance
    # uniqueness invariant as the real-world manifest.
    for cat_rows in manifest.get("categories", {}).values():
        for row in cat_rows:
            if row.get("source_uri") == source_uri:
                print(
                    f"error: source_uri {source_uri!r} already present in "
                    f"manifest (filename={row.get('filename')!r})",
                    file=sys.stderr,
                )
                return 3

    dest_dir = corpus_root / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        # Should never happen with the auto-incrementing sequence number,
        # but better to fail loudly than silently overwrite real audio.
        print(
            f"error: destination {dest} already exists (manifest is out of "
            f"sync with the corpus directory)",
            file=sys.stderr,
        )
        return 4
    shutil.copy2(audio_file, dest)

    new_row = {
        "filename": filename,
        "source_uri": source_uri,
        "license": args.license.strip(),
        "duration_s": round(float(duration_s), 3),
        "recorded_at_iso8601": _utc_now_iso(),
        "is_placeholder": False,
    }
    manifest.setdefault("categories", {}).setdefault(category, []).append(new_row)
    manifest["last_updated_iso8601"] = _utc_now_iso()
    _save_manifest(manifest_path, manifest)

    print(
        f"added {category}/{filename} (duration={duration_s:.3f}s, "
        f"license={new_row['license']!r}, source_uri={source_uri!r})"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    categories = manifest.get("categories", {})

    print(f"{manifest_path}")
    print(f"  schema_version: {manifest.get('schema_version', '?')}")
    print(f"  generated_at_iso8601: {manifest.get('generated_at_iso8601', '?')}")
    print()
    total = 0
    placeholder_total = 0
    for category in CATEGORIES:
        rows = categories.get(category, [])
        n = len(rows)
        placeholders = sum(1 for r in rows if r.get("is_placeholder"))
        real = n - placeholders
        total += n
        placeholder_total += placeholders
        print(
            f"  {category:<14} {n:>4} clips "
            f"({real} real, {placeholders} placeholder)"
        )
    print()
    print(
        f"  TOTAL: {total} ({total - placeholder_total} real, "
        f"{placeholder_total} placeholder)"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Walk the manifest and confirm every entry resolves to a valid WAV."""
    manifest_path = Path(args.manifest).resolve()
    corpus_root = Path(args.corpus_root).resolve()

    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)
    categories = manifest.get("categories", {})

    failures: list[str] = []
    for category in CATEGORIES:
        for row in categories.get(category, []):
            filename = row.get("filename", "")
            if not filename:
                failures.append(f"{category}: row missing filename: {row!r}")
                continue
            path = corpus_root / category / filename
            if not path.exists():
                failures.append(f"{category}/{filename}: file missing on disk")
                continue
            try:
                duration_s, sr, channels, subtype = _probe_wav(path)
            except Exception as exc:  # soundfile errors
                failures.append(
                    f"{category}/{filename}: cannot probe WAV: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                continue

            recorded_duration = row.get("duration_s")
            if recorded_duration is not None and abs(
                recorded_duration - duration_s
            ) > 0.05:
                failures.append(
                    f"{category}/{filename}: manifest duration "
                    f"{recorded_duration:.3f}s != actual {duration_s:.3f}s"
                )
            # Check that the file still meets the corpus invariants.
            if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
                failures.append(
                    f"{category}/{filename}: duration {duration_s:.3f}s out "
                    f"of range [{MIN_DURATION_S}, {MAX_DURATION_S}]"
                )
            if sr != REQUIRED_SAMPLE_RATE_HZ:
                failures.append(
                    f"{category}/{filename}: sample rate {sr} Hz != required "
                    f"{REQUIRED_SAMPLE_RATE_HZ} Hz"
                )
            if channels != REQUIRED_CHANNELS:
                failures.append(
                    f"{category}/{filename}: channel count {channels} != "
                    f"required {REQUIRED_CHANNELS}"
                )
            if subtype != REQUIRED_SUBTYPE:
                failures.append(
                    f"{category}/{filename}: subtype {subtype!r} != required "
                    f"{REQUIRED_SUBTYPE!r}"
                )

    if failures:
        print(
            f"FAIL: {len(failures)} validation problem(s) in {manifest_path}",
            file=sys.stderr,
        )
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    total = sum(len(categories.get(cat, [])) for cat in CATEGORIES)
    print(f"OK: validated {total} entries across {len(CATEGORIES)} categories")
    return 0


# ------------------------------------------------------------------ argparse
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curate_adversarial",
        description=(
            "Provenance-recording tool for the Adversarial_Suite (R7.1). "
            "Validates duration, format, and licence; copies the WAV into "
            "data/adversarial/<category>/ and appends a row to manifest.json."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # add ----------------------------------------------------------------
    p_add = sub.add_parser("add", help="ingest one new clip with provenance")
    p_add.add_argument("--category", choices=CATEGORIES, required=True)
    p_add.add_argument(
        "--audio-file", required=True, help="path to the source .wav clip"
    )
    p_add.add_argument(
        "--license",
        required=True,
        help='e.g. "CC-BY-4.0", "CC0-1.0", "Standard FreeSound License"',
    )
    p_add.add_argument(
        "--source-uri",
        default=None,
        help='optional provenance URI; defaults to "local:<filename>"',
    )
    p_add.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)}",
    )
    p_add.add_argument(
        "--corpus-root",
        default=str(DEFAULT_CORPUS_ROOT),
        help=f"default: {DEFAULT_CORPUS_ROOT.relative_to(REPO_ROOT)}",
    )
    p_add.set_defaults(func=cmd_add)

    # list ---------------------------------------------------------------
    p_list = sub.add_parser("list", help="print per-category counts")
    p_list.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_list.set_defaults(func=cmd_list)

    # validate -----------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help="walk manifest, confirm all files present and durations match",
    )
    p_validate.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p_validate.add_argument(
        "--corpus-root", default=str(DEFAULT_CORPUS_ROOT)
    )
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
