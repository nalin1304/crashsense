"""Tests for ``scripts/curate_youtube_clips.py``.

Covers happy-path provenance recording and every documented rejection path
(R1.1 duration window, R1.2 source_uri uniqueness, R1.3 disjointness with
the AudioSet manifest, plus the YouTube ``video_id`` format check).

The tests use a ``tmp_path`` labels.csv and a ``tmp_path`` raw manifest so
no test ever writes to the real ``data/`` tree.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import curate_youtube_clips as tool


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def labels_csv(tmp_path: Path) -> Path:
    return tmp_path / "labels.csv"


@pytest.fixture
def raw_manifest_empty(tmp_path: Path) -> Path:
    """Manifest with no AudioSet entries — overlap check is a no-op."""
    path = tmp_path / "raw_audio" / "_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at_iso8601": "2025-01-15T00:00:00Z",
                "sources": [
                    {"filename": "esc_x.wav", "source_uri": "esc50:x"},
                    {"filename": "us8k_1.wav", "source_uri": "urbansound8k:1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def raw_manifest_with_audioset(tmp_path: Path) -> Path:
    """Manifest containing one AudioSet entry for video_id ``aaaaaaaaaaa``."""
    path = tmp_path / "raw_audio" / "_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_at_iso8601": "2025-01-15T00:00:00Z",
                "sources": [
                    {
                        "filename": "audioset_aaaaaaaaaaa.wav",
                        "source_uri": "audioset:aaaaaaaaaaa",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_args(
    labels_csv: Path,
    raw_manifest: Path,
    *,
    video_id: str = "dQw4w9WgXcQ",
    start_s: float = 10.0,
    end_s: float = 15.0,
    label: str = "crash",
) -> list[str]:
    return [
        "add",
        "--video-id", video_id,
        "--start-s", str(start_s),
        "--end-s", str(end_s),
        "--label", label,
        "--license", "CC-BY-4.0",
        "--annotator-id", "alice",
        "--labels-csv", str(labels_csv),
        "--raw-manifest", str(raw_manifest),
    ]


# ------------------------------------------------------------------ happy path
class TestAddHappyPath:
    def test_first_row_writes_header_and_row(self, labels_csv, raw_manifest_empty):
        rc = tool.main(_base_args(labels_csv, raw_manifest_empty))
        assert rc == 0
        assert labels_csv.exists()

        with labels_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["filename"] == "rw_crash_0001.wav"
        assert rows[0]["label"] == "crash"
        assert rows[0]["annotator_id"] == "alice"
        assert rows[0]["source_uri"] == "youtube:dQw4w9WgXcQ@10-15"
        # Timestamp must be RFC 3339 / ISO 8601 with explicit Z suffix.
        assert rows[0]["annotated_at_iso8601"].endswith("Z")

    def test_filename_counter_increments_per_label(
        self, labels_csv, raw_manifest_empty
    ):
        # Two crashes and a noise — the counter must be per-label.
        tool.main(_base_args(labels_csv, raw_manifest_empty))
        tool.main(
            _base_args(
                labels_csv,
                raw_manifest_empty,
                video_id="abcdefghijk",
                start_s=0.0,
                end_s=4.0,
                label="crash",
            )
        )
        tool.main(
            _base_args(
                labels_csv,
                raw_manifest_empty,
                video_id="lmnopqrstuv",
                start_s=0.0,
                end_s=4.0,
                label="noise",
            )
        )
        with labels_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        names = [r["filename"] for r in rows]
        assert names == [
            "rw_crash_0001.wav",
            "rw_crash_0002.wav",
            "rw_noise_0001.wav",
        ]

    def test_audio_file_copied_when_supplied(
        self, labels_csv, raw_manifest_empty, tmp_path
    ):
        wav = tmp_path / "src.wav"
        wav.write_bytes(b"RIFFxxxxWAVE")  # bytes are not parsed; copy is opaque
        audio_root = tmp_path / "real_world_test"

        rc = tool.main(
            _base_args(labels_csv, raw_manifest_empty)
            + ["--audio-file", str(wav), "--audio-root", str(audio_root)]
        )
        assert rc == 0

        copied = audio_root / "crash" / "rw_crash_0001.wav"
        assert copied.exists()
        assert copied.read_bytes() == b"RIFFxxxxWAVE"

    def test_sidecar_provenance_jsonl_written(self, labels_csv, raw_manifest_empty):
        tool.main(
            _base_args(labels_csv, raw_manifest_empty)
            + ["--source-url", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        )
        sidecar = labels_csv.with_suffix(".provenance.jsonl")
        assert sidecar.exists()

        lines = sidecar.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["video_id"] == "dQw4w9WgXcQ"
        assert entry["start_s"] == 10.0
        assert entry["end_s"] == 15.0
        assert entry["license"] == "CC-BY-4.0"
        assert entry["filename"] == "rw_crash_0001.wav"
        assert (
            entry["source_url"]
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )

    def test_missing_raw_manifest_is_treated_as_empty(
        self, labels_csv, tmp_path
    ):
        # An annotator may run this tool before any training data has been
        # pulled — `_load_audioset_video_ids` should treat the absent
        # manifest as "no overlap possible".
        absent = tmp_path / "does_not_exist.json"
        rc = tool.main(_base_args(labels_csv, absent))
        assert rc == 0
        assert labels_csv.exists()


# ------------------------------------------------------------------ rejections
class TestAddRejections:
    @pytest.mark.parametrize(
        "video_id",
        [
            "tooShort",       # <11 chars
            "dQw4w9WgXcQQ",   # >11 chars
            "dQw4w9!gXcQ",    # disallowed character
        ],
    )
    def test_invalid_video_id_format(
        self, labels_csv, raw_manifest_empty, video_id, capsys
    ):
        rc = tool.main(
            _base_args(labels_csv, raw_manifest_empty, video_id=video_id)
        )
        assert rc == 2
        assert "invalid video_id" in capsys.readouterr().err
        assert not labels_csv.exists()

    @pytest.mark.parametrize(
        "start_s, end_s",
        [
            (10.0, 12.5),   # 2.5s - too short (R1.1)
            (10.0, 41.0),   # 31s  - too long (R1.1)
            (10.0, 10.0),   # zero-length - end must be > start
            (10.0, 5.0),    # negative - end must be > start
            (-1.0, 5.0),    # negative start
        ],
    )
    def test_invalid_window(
        self, labels_csv, raw_manifest_empty, start_s, end_s, capsys
    ):
        rc = tool.main(
            _base_args(
                labels_csv,
                raw_manifest_empty,
                start_s=start_s,
                end_s=end_s,
            )
        )
        assert rc == 2
        assert capsys.readouterr().err  # some message printed
        assert not labels_csv.exists()

    def test_rejects_audioset_overlap(
        self, labels_csv, raw_manifest_with_audioset, capsys
    ):
        rc = tool.main(
            _base_args(
                labels_csv,
                raw_manifest_with_audioset,
                video_id="aaaaaaaaaaa",  # collides with manifest entry
            )
        )
        assert rc == 3
        assert "AudioSet manifest" in capsys.readouterr().err
        assert not labels_csv.exists()

    def test_rejects_duplicate_source_uri(
        self, labels_csv, raw_manifest_empty
    ):
        # First add succeeds.
        rc1 = tool.main(_base_args(labels_csv, raw_manifest_empty))
        assert rc1 == 0
        # Same video_id + same window -> identical source_uri -> reject.
        rc2 = tool.main(_base_args(labels_csv, raw_manifest_empty))
        assert rc2 == 3
        # Only one row should have been written.
        with labels_csv.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1

    def test_audio_file_not_wav(
        self, labels_csv, raw_manifest_empty, tmp_path, capsys
    ):
        bogus = tmp_path / "src.mp3"
        bogus.write_bytes(b"not-a-wav")
        rc = tool.main(
            _base_args(labels_csv, raw_manifest_empty)
            + ["--audio-file", str(bogus)]
        )
        assert rc == 4
        assert ".wav" in capsys.readouterr().err
        assert not labels_csv.exists()

    def test_audio_file_missing(
        self, labels_csv, raw_manifest_empty, tmp_path, capsys
    ):
        rc = tool.main(
            _base_args(labels_csv, raw_manifest_empty)
            + ["--audio-file", str(tmp_path / "nope.wav")]
        )
        assert rc == 4
        assert "does not exist" in capsys.readouterr().err
        assert not labels_csv.exists()


# ------------------------------------------------------------------ list / validate
class TestList:
    def test_list_empty_file(self, labels_csv, capsys):
        rc = tool.main(["list", "--labels-csv", str(labels_csv)])
        assert rc == 0
        assert "empty or missing" in capsys.readouterr().out

    def test_list_after_adds(self, labels_csv, raw_manifest_empty, capsys):
        tool.main(_base_args(labels_csv, raw_manifest_empty))
        tool.main(
            _base_args(
                labels_csv,
                raw_manifest_empty,
                video_id="lmnopqrstuv",
                start_s=0.0,
                end_s=4.0,
                label="noise",
            )
        )
        capsys.readouterr()  # drain "appended row" prints

        rc = tool.main(["list", "--labels-csv", str(labels_csv)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 rows" in out
        assert "crash: 1" in out
        assert "noise: 1" in out
        assert "rw_crash_0001.wav" in out
        assert "rw_noise_0001.wav" in out


class TestValidate:
    def test_validate_clean_csv(self, labels_csv, raw_manifest_empty):
        tool.main(_base_args(labels_csv, raw_manifest_empty))
        rc = tool.main(["validate", "--labels-csv", str(labels_csv)])
        assert rc == 0

    def test_validate_missing_csv(self, tmp_path, capsys):
        rc = tool.main(
            ["validate", "--labels-csv", str(tmp_path / "missing.csv")]
        )
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err

    def test_validate_flags_bad_row(self, labels_csv, raw_manifest_empty, capsys):
        # Seed a valid row so the file + header exist.
        tool.main(_base_args(labels_csv, raw_manifest_empty))
        capsys.readouterr()

        # Append a malformed row directly (bypassing argparse validation).
        with labels_csv.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=tool.CSV_FIELDNAMES)
            writer.writerow(
                {
                    "filename": "rw_crash_0002.wav",
                    "label": "crash",
                    "annotator_id": "bad!@#",  # disallowed chars
                    "annotated_at_iso8601": "2025-01-15T00:00:00Z",
                    "source_uri": "youtube:zzzzzzzzzzz@0-5",
                }
            )

        rc = tool.main(["validate", "--labels-csv", str(labels_csv)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "1 valid, 1 rejected" in captured.out
        assert "annotator_id" in captured.err
