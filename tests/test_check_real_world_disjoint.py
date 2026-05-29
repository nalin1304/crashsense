"""Tests for ``scripts/check_real_world_disjoint.py`` (R1.3, R31.5).

Covers the cases called out by the task spec:
  - success (zero overlap)
  - filename collision with the training set
  - source_uri collision with the training set
  - YouTube video_id collision with the AudioSet portion of the training set
  - missing labels CSV
  - missing training manifest

The script is invoked as a subprocess to exercise the real CLI surface and
exit codes, exactly as ``make preservation-gate`` will call it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_real_world_disjoint.py"

_LABELS_HEADER = (
    "filename,label,annotator_id,annotated_at_iso8601,source_uri\n"
)


def _run(labels_csv: Path, training_manifest: Path) -> subprocess.CompletedProcess[str]:
    # Force the adversarial manifest argument to a path that does not exist
    # so existing test cases never accidentally cross-check against the
    # repo's real ``data/adversarial/manifest.json``. The cross-check is
    # additive and skipped when the manifest is absent.
    nonexistent = labels_csv.parent / "_unused_adversarial_manifest.json"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--labels-csv",
            str(labels_csv),
            "--training-manifest",
            str(training_manifest),
            "--adversarial-manifest",
            str(nonexistent),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_with_adversarial(
    labels_csv: Path,
    training_manifest: Path,
    adversarial_manifest: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--labels-csv",
            str(labels_csv),
            "--training-manifest",
            str(training_manifest),
            "--adversarial-manifest",
            str(adversarial_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_labels(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    """Write a labels.csv with the given (filename, label, annotator_id, ts, source_uri) rows."""
    body = _LABELS_HEADER + "".join(
        f"{fn},{label},{annot},{ts},{uri}\n"
        for (fn, label, annot, ts, uri) in rows
    )
    path.write_text(body, encoding="utf-8")


def _write_training_manifest(
    path: Path, sources: list[tuple[str, str]]
) -> None:
    """Write a `_manifest.json` with the given (filename, source_uri) pairs."""
    manifest = {
        "schema_version": "1.0.0",
        "generated_at_iso8601": "2025-01-15T14:30:00Z",
        "sources": [
            {"filename": fn, "source_uri": uri} for (fn, uri) in sources
        ],
    }
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def _write_adversarial_manifest(
    path: Path,
    *,
    horns: list[dict] | None = None,
    fireworks: list[dict] | None = None,
    tire_blowouts: list[dict] | None = None,
    airbrakes: list[dict] | None = None,
) -> None:
    """Write a `data/adversarial/manifest.json` for the cross-check tests.

    Categories default to empty lists so each test only specifies the
    rows it cares about.
    """
    manifest = {
        "schema_version": "1.0.0",
        "generated_at_iso8601": "2025-01-15T00:00:00Z",
        "categories": {
            "horns": horns or [],
            "fireworks": fireworks or [],
            "tire_blowouts": tire_blowouts or [],
            "airbrakes": airbrakes or [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )


# ----------------------------------------------------------------------------
# Success case
# ----------------------------------------------------------------------------


def test_success_no_overlap(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    manifest = tmp_path / "_manifest.json"

    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "youtube:abcDEFghi01@5.0-15.0",
            ),
            (
                "rw_noise_0001.wav",
                "noise",
                "ann1",
                "2025-01-15T14:31:00Z",
                "youtube:zzzzzzzzzz1@0.0-10.0",
            ),
            (
                "rw_crash_0002.wav",
                "crash",
                "ann2",
                "2025-01-15T15:00:00Z",
                "archive:org/some-clip-42",
            ),
        ],
    )
    _write_training_manifest(
        manifest,
        [
            ("audioset_videoXXXXXX.wav", "audioset:videoXXXXXX"),
            ("us8k_3_4.wav", "urbansound8k:3_4"),
            ("esc_clip.wav", "esc50:clip"),
        ],
    )

    res = _run(labels, manifest)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
    assert "3 labels checked" in res.stdout


# ----------------------------------------------------------------------------
# Filename collision
# ----------------------------------------------------------------------------


def test_filename_collision(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    manifest = tmp_path / "_manifest.json"

    # The real-world filename "audioset_-1PZQg5Gi8A.wav" intentionally
    # matches a filename present in the training manifest.
    _write_labels(
        labels,
        [
            (
                "audioset_-1PZQg5Gi8A.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/independent-source-uri",
            ),
        ],
    )
    _write_training_manifest(
        manifest,
        [
            ("audioset_-1PZQg5Gi8A.wav", "audioset:differentVidI"),
            ("audioset_otherVideo01.wav", "audioset:otherVideo01"),
        ],
    )

    res = _run(labels, manifest)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "filename overlap" in res.stderr
    assert "audioset_-1PZQg5Gi8A.wav" in res.stderr


# ----------------------------------------------------------------------------
# Source URI collision
# ----------------------------------------------------------------------------


def test_source_uri_collision(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    manifest = tmp_path / "_manifest.json"

    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "freesound:12345",
            ),
        ],
    )
    _write_training_manifest(
        manifest,
        [
            ("freesound_12345.wav", "freesound:12345"),
        ],
    )

    res = _run(labels, manifest)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "source_uri overlap" in res.stderr
    assert "freesound:12345" in res.stderr


# ----------------------------------------------------------------------------
# AudioSet ↔ YouTube video_id cross-collision
# ----------------------------------------------------------------------------


def test_audioset_youtube_video_id_cross_collision(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    manifest = tmp_path / "_manifest.json"

    # Real-world clip is a YouTube window of video "abcDEFghi01"; the
    # training set contains the same 11-char ID under the audioset: scheme.
    # The two source_uris are different strings, so only the video_id
    # cross-check catches the leak.
    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "youtube:abcDEFghi01@5.0-15.0",
            ),
        ],
    )
    _write_training_manifest(
        manifest,
        [
            ("audioset_abcDEFghi01.wav", "audioset:abcDEFghi01"),
            ("audioset_otherVideo01.wav", "audioset:otherVideo01"),
        ],
    )

    res = _run(labels, manifest)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "video_id overlap" in res.stderr
    assert "abcDEFghi01" in res.stderr
    # The diagnostic should also surface the offending real-world source_uri.
    assert "youtube:abcDEFghi01@5.0-15.0" in res.stderr


def test_unrelated_youtube_video_id_does_not_collide(tmp_path: Path) -> None:
    """Real-world YouTube clip whose video_id is not in AudioSet is fine."""
    labels = tmp_path / "labels.csv"
    manifest = tmp_path / "_manifest.json"

    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "youtube:fresh1videoX@5.0-15.0",
            ),
        ],
    )
    _write_training_manifest(
        manifest,
        [
            ("audioset_otherVideo01.wav", "audioset:otherVideo01"),
        ],
    )

    res = _run(labels, manifest)
    assert res.returncode == 0, res.stderr


# ----------------------------------------------------------------------------
# Missing inputs
# ----------------------------------------------------------------------------


def test_missing_labels_csv(tmp_path: Path) -> None:
    manifest = tmp_path / "_manifest.json"
    _write_training_manifest(manifest, [("audioset_a.wav", "audioset:a")])

    res = _run(tmp_path / "does_not_exist.csv", manifest)
    assert res.returncode == 1
    assert "labels CSV not found" in res.stderr


def test_missing_training_manifest(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/some-clip",
            ),
        ],
    )

    res = _run(labels, tmp_path / "missing_manifest.json")
    assert res.returncode == 1
    assert "training manifest not found" in res.stderr


# ----------------------------------------------------------------------------
# Combined overlap reports every category
# ----------------------------------------------------------------------------


def test_multiple_overlap_categories_reported(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    manifest = tmp_path / "_manifest.json"

    _write_labels(
        labels,
        [
            # filename collision
            (
                "audioset_collide00001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/independent-1",
            ),
            # source_uri collision
            (
                "rw_crash_0002.wav",
                "crash",
                "ann1",
                "2025-01-15T14:31:00Z",
                "freesound:99",
            ),
            # video_id cross-collision (exactly 11-char YouTube IDs)
            (
                "rw_crash_0003.wav",
                "crash",
                "ann1",
                "2025-01-15T14:32:00Z",
                "youtube:videoIDABC0@0.0-10.0",
            ),
        ],
    )
    _write_training_manifest(
        manifest,
        [
            ("audioset_collide00001.wav", "audioset:differentVid"),
            ("freesound_99.wav", "freesound:99"),
            ("audioset_videoIDABC0.wav", "audioset:videoIDABC0"),
        ],
    )

    res = _run(labels, manifest)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "filename overlap" in res.stderr
    assert "source_uri overlap" in res.stderr
    assert "video_id overlap" in res.stderr


# ----------------------------------------------------------------------------
# Adversarial cross-check (task 4.12 extension)
# ----------------------------------------------------------------------------


def test_adversarial_source_uri_overlap_with_training_is_rejected(
    tmp_path: Path,
) -> None:
    """An adversarial clip whose source_uri lives in the training manifest fails."""
    labels = tmp_path / "labels.csv"
    training_manifest = tmp_path / "_manifest.json"
    adversarial_manifest = tmp_path / "adversarial_manifest.json"

    # Real-world labels are clean — no overlap there.
    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/clean-1",
            ),
        ],
    )

    # Training manifest contains "freesound:55555".
    _write_training_manifest(
        training_manifest,
        [
            ("freesound_55555.wav", "freesound:55555"),
            ("audioset_other.wav", "audioset:otherVideo01"),
        ],
    )

    # Adversarial manifest re-uses the same source_uri in `horns/`.
    _write_adversarial_manifest(
        adversarial_manifest,
        horns=[
            {
                "filename": "horns_0001.wav",
                "source_uri": "freesound:55555",
                "license": "CC-BY-4.0",
                "duration_s": 5.0,
                "recorded_at_iso8601": "2025-01-15T14:30:00Z",
                "is_placeholder": False,
            }
        ],
    )

    res = _run_with_adversarial(labels, training_manifest, adversarial_manifest)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "adversarial source_uri overlap" in res.stderr
    assert "freesound:55555" in res.stderr
    assert "horns/horns_0001.wav" in res.stderr


def test_adversarial_placeholders_do_not_trigger_overlap(tmp_path: Path) -> None:
    """Synthetic placeholder rows are excluded from the cross-check."""
    labels = tmp_path / "labels.csv"
    training_manifest = tmp_path / "_manifest.json"
    adversarial_manifest = tmp_path / "adversarial_manifest.json"

    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/clean-1",
            ),
        ],
    )

    # Training entry whose URI happens to match a `synthetic:...` value —
    # this would only ever happen via deliberate sabotage, but it is the
    # cleanest way to prove placeholders are filtered out.
    _write_training_manifest(
        training_manifest,
        [
            (
                "synthetic_collide.wav",
                "synthetic:horns_placeholder_01",
            ),
        ],
    )
    _write_adversarial_manifest(
        adversarial_manifest,
        horns=[
            {
                "filename": "horns_placeholder_01.wav",
                "source_uri": "synthetic:horns_placeholder_01",
                "license": "CC0-1.0 (synthetic)",
                "duration_s": 4.0,
                "recorded_at_iso8601": "2025-01-15T00:00:00Z",
                "is_placeholder": True,
            }
        ],
    )

    res = _run_with_adversarial(labels, training_manifest, adversarial_manifest)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "adversarial" not in res.stderr.lower()


def test_adversarial_manifest_absent_is_skipped(tmp_path: Path) -> None:
    """Without an adversarial manifest, the original behaviour is preserved."""
    labels = tmp_path / "labels.csv"
    training_manifest = tmp_path / "_manifest.json"
    missing_adversarial = tmp_path / "no_such_adversarial_manifest.json"

    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/clean-1",
            ),
        ],
    )
    _write_training_manifest(
        training_manifest,
        [("audioset_other.wav", "audioset:otherVideo01")],
    )
    assert not missing_adversarial.exists()

    res = _run_with_adversarial(labels, training_manifest, missing_adversarial)
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert "OK" in res.stdout


def test_adversarial_manifest_corrupt_json_fails_clearly(tmp_path: Path) -> None:
    """A corrupt adversarial manifest should fail with a clear message."""
    labels = tmp_path / "labels.csv"
    training_manifest = tmp_path / "_manifest.json"
    bad_adversarial = tmp_path / "adversarial_manifest.json"

    _write_labels(
        labels,
        [
            (
                "rw_crash_0001.wav",
                "crash",
                "ann1",
                "2025-01-15T14:30:00Z",
                "archive:org/clean-1",
            ),
        ],
    )
    _write_training_manifest(
        training_manifest,
        [("audioset_other.wav", "audioset:otherVideo01")],
    )
    bad_adversarial.write_text("{ this is not json", encoding="utf-8")

    res = _run_with_adversarial(labels, training_manifest, bad_adversarial)
    assert res.returncode == 1, (res.stdout, res.stderr)
    assert "invalid JSON" in res.stderr
