"""YouTube clip-curation tool for the Real_World_Test_Set.

This script is *tooling*, not corpus population. It helps a human annotator
record provenance for hand-cut YouTube clips and append rows to
``data/real_world_test/labels.csv``. It does **not** download YouTube
videos — the annotator runs ``yt-dlp`` themselves and feeds the resulting
clip into ``--audio-file`` (or appends the row first and adds the WAV
later).

Implements: R1.1, R1.2, R1.3 (task 1.2 in
``.kiro/specs/crashsense-hardening/tasks.md``).

Subcommands
-----------

``add``        Record one new clip in ``labels.csv``. Validates the
               YouTube ``video_id`` format, the clip duration window
               (3.0–30.0 s), and rejects any clip whose ``video_id``
               already appears in the AudioSet manifest at
               ``data/raw_audio/_manifest.json`` (R1.3 — disjoint with
               training corpora) or whose ``source_uri`` is already in
               ``labels.csv`` (R1.2 — uniqueness invariant).

``list``       Pretty-print ``labels.csv`` with a per-label count.

``validate``   Parse ``labels.csv`` through
               :func:`backend.audio_model.real_world_schema.parse_labels_csv`
               and exit non-zero if any row is rejected.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable so this script can be run from anywhere
# without PYTHONPATH gymnastics.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.audio_model.real_world_schema import (  # noqa: E402  (path setup above)
    RealWorldLabelRow,
    parse_labels_csv,
)

# ------------------------------------------------------------------ constants
DEFAULT_LABELS_CSV = REPO_ROOT / "data" / "real_world_test" / "labels.csv"
DEFAULT_AUDIO_ROOT = REPO_ROOT / "data" / "real_world_test"
DEFAULT_RAW_MANIFEST = REPO_ROOT / "data" / "raw_audio" / "_manifest.json"

CSV_FIELDNAMES = [
    "filename",
    "label",
    "annotator_id",
    "annotated_at_iso8601",
    "source_uri",
]

# YouTube video IDs are exactly 11 characters in [A-Za-z0-9_-].
# https://developers.google.com/youtube/v3/docs/videos
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# R1.1 / R1.2: clip duration must lie in [3.0, 30.0] seconds inclusive.
MIN_DURATION_S = 3.0
MAX_DURATION_S = 30.0

# Filename naming convention from design §3.1.1 — `rw_<label>_<NNNN>.wav`.
RW_FILENAME_RE = re.compile(r"^rw_(?P<label>crash|noise)_(?P<seq>\d{4})\.wav$")


# ------------------------------------------------------------------ helpers
def _utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_existing_rows(labels_csv: Path) -> list[dict[str, str]]:
    """Read raw rows from ``labels_csv`` (returns ``[]`` if file is missing/empty)."""
    if not labels_csv.exists() or labels_csv.stat().st_size == 0:
        return []
    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _next_filename(rows: list[dict[str, str]], label: str) -> str:
    """Compute the next ``rw_<label>_<NNNN>.wav`` filename.

    Walks every row in ``rows`` (including rows for the *other* label) and
    finds the highest 4-digit sequence number for the requested ``label``.
    Returns ``rw_<label>_0001.wav`` when no prior row exists for the label.

    Counter is per-label (a separate sequence is maintained for ``crash``
    and ``noise``), matching the layout in design §3.1.1.
    """
    highest = 0
    for row in rows:
        match = RW_FILENAME_RE.match(row.get("filename", "") or "")
        if match is None:
            continue
        if match.group("label") != label:
            continue
        seq = int(match.group("seq"))
        if seq > highest:
            highest = seq
    return f"rw_{label}_{highest + 1:04d}.wav"


def _load_audioset_video_ids(manifest_path: Path) -> set[str]:
    """Extract YouTube video IDs from ``data/raw_audio/_manifest.json``.

    AudioSet entries are written by ``dataset_prep.py`` with the
    ``audioset:<videoID>`` URI prefix (see ``_infer_source_uri``).
    Other prefixes (``esc50:``, ``urbansound8k:``, ``freesound:``,
    ``local:``) do not carry YouTube IDs and are ignored.

    Also tolerates a future ``youtube:`` prefix in case AudioSet
    re-encoding ever emits raw YouTube URIs.

    Returns:
        Set of 11-character YouTube IDs found in the manifest. Returns an
        empty set when the manifest does not exist (an annotator may run
        this tool before any training data has been pulled).
    """
    if not manifest_path.exists():
        return set()
    try:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # Be permissive on read-only "is this id taken" lookups: a corrupt
        # manifest should not prevent the annotator from working — but it
        # also should not silently allow overlap, so re-raise.
        raise

    ids: set[str] = set()
    sources = manifest.get("sources") or []
    for entry in sources:
        uri = (entry.get("source_uri") or "").strip()
        if uri.startswith("audioset:"):
            ids.add(uri[len("audioset:") :])
        elif uri.startswith("youtube:"):
            # Strip optional "@start-end" suffix.
            rest = uri[len("youtube:") :]
            ids.add(rest.split("@", 1)[0])
    return ids


def _format_window(start_s: float, end_s: float) -> str:
    """Render the ``@start-end`` window for ``source_uri``.

    Uses ``repr``-equivalent formatting via ``"{:g}"`` so an integer-valued
    float still renders compactly (``12`` instead of ``12.0``). The
    annotator copy/pastes timestamps from a video player, so a ``120-150``
    URI is more legible than ``120.0-150.0``.
    """
    return f"{start_s:g}-{end_s:g}"


# ------------------------------------------------------------------ subcommands
def cmd_add(args: argparse.Namespace) -> int:
    """``add`` subcommand. Returns process exit code."""
    labels_csv = Path(args.labels_csv).resolve()
    audio_root = Path(args.audio_root).resolve()
    raw_manifest = Path(args.raw_manifest).resolve()

    # ---- 1. Argument validation ---------------------------------------------
    video_id = args.video_id.strip()
    if not VIDEO_ID_RE.match(video_id):
        print(
            f"error: invalid video_id {video_id!r}: must be exactly 11 "
            "characters in [A-Za-z0-9_-]",
            file=sys.stderr,
        )
        return 2

    if args.start_s < 0:
        print(
            f"error: --start-s must be >= 0 (got {args.start_s})",
            file=sys.stderr,
        )
        return 2
    if args.end_s <= args.start_s:
        print(
            f"error: --end-s must be > --start-s "
            f"(got start={args.start_s}, end={args.end_s})",
            file=sys.stderr,
        )
        return 2
    duration = args.end_s - args.start_s
    if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
        print(
            f"error: clip duration {duration:.3f}s out of range "
            f"[{MIN_DURATION_S}, {MAX_DURATION_S}] (R1.1)",
            file=sys.stderr,
        )
        return 2

    # ---- 2. R1.3 disjointness — reject AudioSet-overlapping IDs --------------
    audioset_ids = _load_audioset_video_ids(raw_manifest)
    if video_id in audioset_ids:
        print(
            f"error: video_id {video_id!r} is already in the AudioSet "
            f"manifest at {raw_manifest} — Real_World_Test_Set must be "
            "disjoint with training corpora (R1.3)",
            file=sys.stderr,
        )
        return 3

    # ---- 3. Build provenance fields -----------------------------------------
    source_uri = f"youtube:{video_id}@{_format_window(args.start_s, args.end_s)}"

    existing_rows = _read_existing_rows(labels_csv)
    existing_uris = {row.get("source_uri", "").strip() for row in existing_rows}
    if source_uri in existing_uris:
        print(
            f"error: source_uri {source_uri!r} is already in {labels_csv} "
            "(R1.2 uniqueness)",
            file=sys.stderr,
        )
        return 3

    filename = _next_filename(existing_rows, args.label)

    new_row = {
        "filename": filename,
        "label": args.label,
        "annotator_id": args.annotator_id,
        "annotated_at_iso8601": _utc_now_iso(),
        "source_uri": source_uri,
    }

    # Validate against the canonical schema before touching the disk so we
    # never write a row that ``parse_labels_csv`` would reject. This catches
    # bad annotator IDs early.
    try:
        RealWorldLabelRow.model_validate(new_row)
    except Exception as exc:  # pydantic.ValidationError, but stay generic
        print(f"error: row failed schema validation: {exc}", file=sys.stderr)
        return 2

    # ---- 4. Optional audio file copy ----------------------------------------
    if args.audio_file:
        src = Path(args.audio_file).resolve()
        if not src.exists():
            print(f"error: --audio-file does not exist: {src}", file=sys.stderr)
            return 4
        if src.suffix.lower() != ".wav":
            print(
                f"error: --audio-file must be a .wav file (got {src.suffix})",
                file=sys.stderr,
            )
            return 4
        dest_dir = audio_root / args.label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        shutil.copy2(src, dest)
        print(f"copied {src} -> {dest}")

    # ---- 5. Append the row --------------------------------------------------
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not labels_csv.exists() or labels_csv.stat().st_size == 0
    with labels_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)

    # ---- 6. Sidecar provenance log ------------------------------------------
    # The label row is the canonical provenance, but R1.2 only asks for
    # filename/label/annotator/timestamp/source_uri. License + raw URL are
    # useful but go beyond the schema, so they live in a sidecar JSON-Lines
    # file next to labels.csv. Keeps the CSV portable.
    sidecar = labels_csv.with_suffix(".provenance.jsonl")
    sidecar_entry = {
        "filename": filename,
        "video_id": video_id,
        "start_s": args.start_s,
        "end_s": args.end_s,
        "license": args.license,
        "annotator_id": args.annotator_id,
        "label": args.label,
        "source_uri": source_uri,
        "source_url": args.source_url
        or f"https://www.youtube.com/watch?v={video_id}&t={int(args.start_s)}s",
        "recorded_at_iso8601": new_row["annotated_at_iso8601"],
    }
    with sidecar.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sidecar_entry, sort_keys=True) + "\n")

    print(f"appended row to {labels_csv}: {filename}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """``list`` subcommand."""
    labels_csv = Path(args.labels_csv).resolve()
    rows = _read_existing_rows(labels_csv)
    if not rows:
        print(f"{labels_csv}: empty or missing")
        return 0

    counts: Counter = Counter(row.get("label", "?") for row in rows)
    print(f"{labels_csv} ({len(rows)} rows)")
    for label in sorted(counts):
        print(f"  {label}: {counts[label]}")
    print()
    print(f"{'filename':<24}  {'label':<6}  {'annotator':<16}  source_uri")
    print(f"{'-' * 24:<24}  {'-' * 6:<6}  {'-' * 16:<16}  {'-' * 40}")
    for row in rows:
        print(
            f"{row.get('filename', ''):<24}  "
            f"{row.get('label', ''):<6}  "
            f"{row.get('annotator_id', ''):<16}  "
            f"{row.get('source_uri', '')}"
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """``validate`` subcommand. Exits non-zero if any row is rejected."""
    labels_csv = Path(args.labels_csv).resolve()
    if not labels_csv.exists():
        print(f"error: {labels_csv} does not exist", file=sys.stderr)
        return 1

    rows, errors = parse_labels_csv(labels_csv)
    print(f"{labels_csv}: {len(rows)} valid, {len(errors)} rejected")
    for err in errors:
        print(
            f"  REJECT  filename={err.filename!r:<28}  "
            f"reason={err.reason}  detail={err.detail}",
            file=sys.stderr,
        )
    return 0 if not errors else 1


# ------------------------------------------------------------------ argparse
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="curate_youtube_clips",
        description=(
            "Provenance-recording tool for the Real_World_Test_Set "
            "(R1.1, R1.2, R1.3). Does not download from YouTube."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # add ----------------------------------------------------------------
    p_add = sub.add_parser("add", help="record one new clip in labels.csv")
    p_add.add_argument("--video-id", required=True, help="11-char YouTube ID")
    p_add.add_argument("--start-s", type=float, required=True)
    p_add.add_argument("--end-s", type=float, required=True)
    p_add.add_argument("--label", choices=("crash", "noise"), required=True)
    p_add.add_argument(
        "--license",
        required=True,
        help='e.g. "CC-BY-4.0" or "Standard YouTube License"',
    )
    p_add.add_argument("--annotator-id", required=True)
    p_add.add_argument(
        "--source-url",
        default=None,
        help="optional full YouTube URL (defaults to "
        "https://www.youtube.com/watch?v=<video_id>&t=<start>s)",
    )
    p_add.add_argument(
        "--audio-file",
        default=None,
        help="optional path to the .wav clip; copied to "
        "data/real_world_test/<label>/<filename> if supplied",
    )
    p_add.add_argument(
        "--labels-csv",
        default=str(DEFAULT_LABELS_CSV),
        help=f"default: {DEFAULT_LABELS_CSV}",
    )
    p_add.add_argument(
        "--audio-root",
        default=str(DEFAULT_AUDIO_ROOT),
        help=f"default: {DEFAULT_AUDIO_ROOT}",
    )
    p_add.add_argument(
        "--raw-manifest",
        default=str(DEFAULT_RAW_MANIFEST),
        help=f"default: {DEFAULT_RAW_MANIFEST}",
    )
    p_add.set_defaults(func=cmd_add)

    # list ---------------------------------------------------------------
    p_list = sub.add_parser("list", help="print labels.csv with per-label counts")
    p_list.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    p_list.set_defaults(func=cmd_list)

    # validate -----------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help="parse labels.csv via real_world_schema and exit non-zero on errors",
    )
    p_validate.add_argument("--labels-csv", default=str(DEFAULT_LABELS_CSV))
    p_validate.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
