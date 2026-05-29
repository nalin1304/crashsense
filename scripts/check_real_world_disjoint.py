#!/usr/bin/env python3
"""
Verify the Real_World_Test_Set is disjoint with the training corpora.

Implements R1.3 (zero filename / source_uri / YouTube video_id overlap with
the training and AudioSet-derived corpora) and is invoked by
``make preservation-gate`` (R31.5, design §3.1.4).

CLI::

    python scripts/check_real_world_disjoint.py \\
        [--labels-csv data/real_world_test/labels.csv] \\
        [--training-manifest data/raw_audio/_manifest.json] \\
        [--adversarial-manifest data/adversarial/manifest.json]

Algorithm:
    1. Parse the real-world label manifest with
       ``backend.audio_model.real_world_schema.parse_labels_csv``. Per-row
       parse errors (returned as the second element of the tuple) are
       *already excluded* from downstream evaluation, so they are ignored
       here too — disjointness is only meaningful for rows that would be
       evaluated.
    2. Load the training source manifest emitted by ``dataset_prep.py``
       (task 1.4). Build two sets:
         - ``training_filenames``  = {sources[i].filename}
         - ``training_source_uris`` = {sources[i].source_uri}
    3. Compute three set intersections:
         a. real_world_filenames    ∩ training_filenames
         b. real_world_source_uris  ∩ training_source_uris
         c. real_world_video_ids    ∩ training_audioset_video_ids
            (extracted from any real-world ``source_uri`` matching
            ``^youtube:([A-Za-z0-9_-]{11})@`` and any training source URI
            matching ``^audioset:([A-Za-z0-9_-]{11})$`` — AudioSet uses
            YouTube-format 11-character video IDs.)
    4. *Adversarial extension* (task 4.12): if
       ``data/adversarial/manifest.json`` exists, also check that no
       ``source_uri`` in any of the adversarial categories is present in
       the training manifest. The check is **additive** — its absence does
       not break the original real-world disjointness behaviour, so legacy
       checkouts without an adversarial corpus still pass.
    5. On any non-empty intersection: print the offending entries to stderr
       and exit 1.
    6. On any missing input file: print a clear error to stderr and exit 1.
    7. On success: print
       ``OK: real-world corpus disjoint with training corpora (X labels
       checked)`` to stdout and exit 0.

Pure standard library plus ``backend.audio_model.real_world_schema`` (which
itself depends only on pydantic).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make the repo root importable so this script runs from any working
# directory (CI, local checkout, etc.) without PYTHONPATH gymnastics.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.audio_model.real_world_schema import (  # noqa: E402
    RealWorldLabelRow,
    parse_labels_csv,
)

DEFAULT_LABELS_CSV = REPO_ROOT / "data" / "real_world_test" / "labels.csv"
DEFAULT_TRAINING_MANIFEST = REPO_ROOT / "data" / "raw_audio" / "_manifest.json"
DEFAULT_ADVERSARIAL_MANIFEST = (
    REPO_ROOT / "data" / "adversarial" / "manifest.json"
)

# YouTube / AudioSet IDs are exactly 11 characters from [A-Za-z0-9_-].
# - Real-world rows produced by ``scripts/curate_youtube_clips.py`` (task 1.2)
#   use ``youtube:<video_id>@<start_s>-<end_s>`` so the same video sliced into
#   different windows still has a unique ``source_uri``.
# - AudioSet entries written by ``dataset_prep.py::_infer_source_uri`` use
#   ``audioset:<video_id>``.
_YOUTUBE_VIDEO_ID_RE = re.compile(r"^youtube:([A-Za-z0-9_-]{11})@")
_AUDIOSET_VIDEO_ID_RE = re.compile(r"^audioset:([A-Za-z0-9_-]{11})$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the real-world test corpus shares zero filenames, "
            "source URIs, or YouTube video IDs with the training corpora. "
            "Exits 1 on any overlap or missing input."
        )
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help=(
            "Real_World_Test_Set label manifest "
            f"(default: {DEFAULT_LABELS_CSV.relative_to(REPO_ROOT)})"
        ),
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=DEFAULT_TRAINING_MANIFEST,
        help=(
            "Training source manifest emitted by dataset_prep.py "
            f"(default: {DEFAULT_TRAINING_MANIFEST.relative_to(REPO_ROOT)})"
        ),
    )
    parser.add_argument(
        "--adversarial-manifest",
        type=Path,
        default=DEFAULT_ADVERSARIAL_MANIFEST,
        help=(
            "Adversarial_Suite manifest written by curate_adversarial.py. "
            "If absent, the adversarial cross-check is skipped (the check "
            "is additive and does not break legacy checkouts). "
            f"Default: {DEFAULT_ADVERSARIAL_MANIFEST.relative_to(REPO_ROOT)}"
        ),
    )
    return parser


def _extract_youtube_video_id(source_uri: str) -> str | None:
    """Return the 11-character YouTube video id from a real-world source URI.

    Returns ``None`` when the URI does not follow the
    ``youtube:<video_id>@<start>-<end>`` convention (e.g. archive.org URLs,
    local recordings).
    """
    match = _YOUTUBE_VIDEO_ID_RE.match(source_uri)
    return match.group(1) if match else None


def _extract_audioset_video_id(source_uri: str) -> str | None:
    """Return the 11-character video id from an ``audioset:<video_id>`` URI."""
    match = _AUDIOSET_VIDEO_ID_RE.match(source_uri)
    return match.group(1) if match else None


def _load_training_manifest(path: Path) -> tuple[set[str], set[str]]:
    """Load ``_manifest.json`` and return ``(filenames, source_uris)``.

    Raises ``FileNotFoundError`` if the manifest is missing,
    ``json.JSONDecodeError`` if it is unparseable, and ``ValueError`` if it
    is missing the expected ``sources`` array.
    """
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError(
            f"training manifest {path} is missing a 'sources' array"
        )

    filenames: set[str] = set()
    source_uris: set[str] = set()
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("filename")
        uri = entry.get("source_uri")
        if isinstance(fn, str) and fn:
            filenames.add(fn)
        if isinstance(uri, str) and uri:
            source_uris.add(uri)
    return filenames, source_uris


def _real_world_video_ids(rows: list[RealWorldLabelRow]) -> dict[str, str]:
    """Return ``{video_id: source_uri}`` for real-world rows that name a YouTube clip."""
    out: dict[str, str] = {}
    for row in rows:
        vid = _extract_youtube_video_id(row.source_uri)
        if vid is not None and vid not in out:
            # Keep the first occurrence so the diagnostic identifies a
            # concrete row even if the same video is referenced twice
            # (parse_labels_csv already rejects duplicate source_uris,
            # but two different windows of the same video are allowed).
            out[vid] = row.source_uri
    return out


def _training_audioset_video_ids(source_uris: set[str]) -> set[str]:
    """Return the set of 11-character AudioSet video IDs in the training manifest."""
    ids: set[str] = set()
    for uri in source_uris:
        vid = _extract_audioset_video_id(uri)
        if vid is not None:
            ids.add(vid)
    return ids


def _load_adversarial_source_uris(
    manifest_path: Path,
) -> dict[str, list[tuple[str, str]]]:
    """Return ``{source_uri: [(category, filename), ...]}`` from the adversarial manifest.

    Returns an empty dict when the manifest does not exist (the adversarial
    cross-check is additive — see module docstring). Synthetic placeholder
    rows are excluded because their ``source_uri`` (``synthetic:...``)
    cannot collide with anything in the training set; including them would
    only produce noise in the cross-check.

    Raises ``json.JSONDecodeError`` if the manifest exists but is unreadable
    so the operator notices a corrupt manifest rather than silently passing.
    """
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    categories = manifest.get("categories") or {}
    by_uri: dict[str, list[tuple[str, str]]] = {}
    for category, rows in categories.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("is_placeholder"):
                continue
            uri = row.get("source_uri")
            filename = row.get("filename")
            if not isinstance(uri, str) or not uri:
                continue
            if not isinstance(filename, str):
                filename = "<unknown>"
            by_uri.setdefault(uri, []).append((category, filename))
    return by_uri


def _print_overlap(label: str, items: list[str]) -> None:
    """Render an overlapping-set diagnostic to stderr in a stable order."""
    print(f"FAIL: {label} ({len(items)}):", file=sys.stderr)
    for item in sorted(items):
        print(f"  - {item}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    labels_csv: Path = args.labels_csv
    training_manifest: Path = args.training_manifest

    if not labels_csv.exists():
        print(
            f"FAIL: real-world labels CSV not found: {labels_csv}",
            file=sys.stderr,
        )
        return 1
    if not training_manifest.exists():
        print(
            f"FAIL: training manifest not found: {training_manifest} "
            f"(run dataset_prep.py to generate it)",
            file=sys.stderr,
        )
        return 1

    # 1. Parse real-world labels. Per-row errors are already excluded from
    #    evaluation, so they are not relevant to disjointness either.
    try:
        rows, _errors = parse_labels_csv(labels_csv)
    except OSError as exc:
        print(f"FAIL: cannot read {labels_csv}: {exc}", file=sys.stderr)
        return 1

    # 2. Load training manifest.
    try:
        training_filenames, training_source_uris = _load_training_manifest(
            training_manifest
        )
    except json.JSONDecodeError as exc:
        print(
            f"FAIL: invalid JSON in {training_manifest}: {exc}",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"FAIL: cannot read {training_manifest}: {exc}", file=sys.stderr)
        return 1

    real_world_filenames = {row.filename for row in rows}
    real_world_source_uris = {row.source_uri for row in rows}
    real_world_video_ids = _real_world_video_ids(rows)
    training_audioset_video_ids = _training_audioset_video_ids(
        training_source_uris
    )

    # 3. Compute the three intersections.
    filename_overlap = sorted(real_world_filenames & training_filenames)
    source_uri_overlap = sorted(real_world_source_uris & training_source_uris)
    video_id_overlap_ids = sorted(
        set(real_world_video_ids.keys()) & training_audioset_video_ids
    )

    # 4. Adversarial extension (task 4.12). The check is additive: when the
    #    manifest does not exist (e.g. a checkout that has not run task 4.12
    #    yet) we silently skip rather than fail. Real placeholder rows have
    #    ``synthetic:`` source URIs which cannot collide with anything in
    #    the training set, so they are excluded inside the loader.
    adversarial_manifest: Path = args.adversarial_manifest
    adversarial_overlap: list[tuple[str, str, str]] = []  # (uri, category, filename)
    if adversarial_manifest.exists():
        try:
            adversarial_by_uri = _load_adversarial_source_uris(
                adversarial_manifest
            )
        except json.JSONDecodeError as exc:
            print(
                f"FAIL: invalid JSON in {adversarial_manifest}: {exc}",
                file=sys.stderr,
            )
            return 1
        except OSError as exc:
            print(
                f"FAIL: cannot read {adversarial_manifest}: {exc}",
                file=sys.stderr,
            )
            return 1
        for uri, locations in adversarial_by_uri.items():
            if uri in training_source_uris:
                for category, filename in locations:
                    adversarial_overlap.append((uri, category, filename))

    # 5. Report any overlaps and exit non-zero.
    if (
        filename_overlap
        or source_uri_overlap
        or video_id_overlap_ids
        or adversarial_overlap
    ):
        if filename_overlap:
            _print_overlap("filename overlap with training set", filename_overlap)
        if source_uri_overlap:
            _print_overlap(
                "source_uri overlap with training set", source_uri_overlap
            )
        if video_id_overlap_ids:
            # Pair each colliding YouTube video id with the real-world URI
            # that referenced it so the operator can locate the offending
            # row directly.
            paired = [
                f"{vid} (real-world source: {real_world_video_ids[vid]})"
                for vid in video_id_overlap_ids
            ]
            _print_overlap(
                "AudioSet/YouTube video_id overlap with training set", paired
            )
        if adversarial_overlap:
            paired = [
                f"{uri} (adversarial: {category}/{filename})"
                for (uri, category, filename) in sorted(adversarial_overlap)
            ]
            _print_overlap(
                "adversarial source_uri overlap with training set", paired
            )
        return 1

    # 6. Success.
    print(
        f"OK: real-world corpus disjoint with training corpora "
        f"({len(rows)} labels checked)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
