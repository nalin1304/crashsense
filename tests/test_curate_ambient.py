"""Tests for ``scripts/curate_ambient.py``.

Covers the documented happy paths and rejection paths for both the
``add`` and ``validate`` subcommands. Every test uses ``tmp_path`` so
no test ever writes to the real ``data/ambient_long/`` tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts import curate_ambient as tool


# ------------------------------------------------------------------ helpers
def _make_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate: int = 22050,
) -> Path:
    """Write a deterministic mono int16 WAV at ``path``.

    The contents are a pure-tone 440 Hz sine — pre-existing project tests
    use the same generator family. The test only cares that
    ``soundfile.info()`` reports the expected duration/rate, so the
    waveform shape is irrelevant.
    """
    n = int(round(sample_rate * duration_s))
    t = np.arange(n) / sample_rate
    signal = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    pcm = (signal * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, pcm)
    return path


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "ambient_long" / "manifest.json"


@pytest.fixture
def ambient_root(tmp_path: Path) -> Path:
    root = tmp_path / "ambient_long"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _add_args(
    audio_file: Path,
    manifest_path: Path,
    ambient_root: Path,
    *,
    location_label: str = "I-5_north_milepost_142",
    recorded_at: str | None = "2025-03-10T22:00:00Z",
    copy_as_managed: bool = False,
) -> list[str]:
    args = [
        "add",
        "--audio-file", str(audio_file),
        "--location-label", location_label,
        "--manifest", str(manifest_path),
        "--ambient-root", str(ambient_root),
    ]
    if recorded_at is not None:
        args += ["--recorded-at", recorded_at]
    if copy_as_managed:
        args += ["--copy-as-managed"]
    return args


# ------------------------------------------------------------------ add: success
class TestAddHappyPath:
    def test_first_session_writes_manifest_with_probed_fields(
        self, tmp_path, manifest_path, ambient_root
    ):
        wav = _make_wav(
            tmp_path / "raw" / "session_a.wav",
            duration_s=120.0,
            sample_rate=22050,
        )

        rc = tool.main(_add_args(wav, manifest_path, ambient_root))
        assert rc == 0
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "1.0.0"
        assert isinstance(manifest["sessions"], list)
        assert len(manifest["sessions"]) == 1
        entry = manifest["sessions"][0]
        # Filename defaults to source basename when --copy-as-managed is off.
        assert entry["filename"] == "session_a.wav"
        # Duration probed from the file (not from a CLI arg).
        assert abs(entry["duration_seconds"] - 120.0) < 0.05
        assert entry["sample_rate_hz"] == 22050
        assert entry["recorded_at_iso8601"] == "2025-03-10T22:00:00Z"
        assert entry["location_label"] == "I-5_north_milepost_142"
        # Real ingests must not carry the placeholder flag.
        assert "is_placeholder" not in entry

    def test_copy_as_managed_renames_to_sequence_filename(
        self, tmp_path, manifest_path, ambient_root
    ):
        wav = _make_wav(
            tmp_path / "raw" / "weird basename.wav",
            duration_s=90.0,
        )

        rc = tool.main(
            _add_args(
                wav, manifest_path, ambient_root, copy_as_managed=True
            )
        )
        assert rc == 0
        # File copied to managed name, original untouched.
        assert (ambient_root / "ambient_001.wav").exists()
        assert wav.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["sessions"][0]["filename"] == "ambient_001.wav"

    def test_managed_sequence_increments_across_calls(
        self, tmp_path, manifest_path, ambient_root
    ):
        wav_a = _make_wav(tmp_path / "raw" / "a.wav", duration_s=70.0)
        wav_b = _make_wav(tmp_path / "raw" / "b.wav", duration_s=80.0)

        tool.main(
            _add_args(wav_a, manifest_path, ambient_root, copy_as_managed=True)
        )
        tool.main(
            _add_args(wav_b, manifest_path, ambient_root, copy_as_managed=True)
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = [s["filename"] for s in manifest["sessions"]]
        assert names == ["ambient_001.wav", "ambient_002.wav"]

    def test_recorded_at_defaults_to_utc_now(
        self, tmp_path, manifest_path, ambient_root
    ):
        wav = _make_wav(tmp_path / "raw" / "a.wav", duration_s=70.0)

        rc = tool.main(
            _add_args(
                wav,
                manifest_path,
                ambient_root,
                recorded_at=None,  # exercise the default
            )
        )
        assert rc == 0

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ts = manifest["sessions"][0]["recorded_at_iso8601"]
        # Must be RFC 3339 with explicit Z suffix; full parsing is
        # exercised in the rejection tests below.
        assert ts.endswith("Z")
        assert len(ts) == len("2025-01-15T00:00:00Z")

    def test_appends_to_existing_manifest(
        self, tmp_path, manifest_path, ambient_root
    ):
        # Seed a manifest that already has a placeholder entry; the
        # ingest helper must preserve it untouched.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        seeded = {
            "schema_version": "1.0.0",
            "generated_at_iso8601": "2025-01-15T00:00:00Z",
            "sessions": [
                {
                    "filename": "ambient_999.wav",
                    "duration_seconds": 36000.0,
                    "sample_rate_hz": 22050,
                    "recorded_at_iso8601": "2025-01-15T06:00:00Z",
                    "location_label": "PLACEHOLDER:dont_touch",
                    "is_placeholder": True,
                }
            ],
        }
        manifest_path.write_text(json.dumps(seeded), encoding="utf-8")

        wav = _make_wav(tmp_path / "raw" / "real.wav", duration_s=70.0)
        rc = tool.main(_add_args(wav, manifest_path, ambient_root))
        assert rc == 0

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Placeholder still present, real session appended.
        names = [s["filename"] for s in manifest["sessions"]]
        assert names == ["ambient_999.wav", "real.wav"]
        # Placeholder flag preserved.
        assert manifest["sessions"][0]["is_placeholder"] is True


# ------------------------------------------------------------------ add: rejections
class TestAddRejections:
    def test_rejects_too_short_file(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        # 30 s is below the 60 s sanity floor for a real ambient session.
        wav = _make_wav(tmp_path / "raw" / "short.wav", duration_s=30.0)

        rc = tool.main(_add_args(wav, manifest_path, ambient_root))
        assert rc == 2
        err = capsys.readouterr().err
        assert "below the" in err
        assert "60" in err
        assert not manifest_path.exists()

    def test_rejects_bad_sample_rate(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        # 11025 Hz is not in the allowed set.
        wav = _make_wav(
            tmp_path / "raw" / "weird_rate.wav",
            duration_s=80.0,
            sample_rate=11025,
        )

        rc = tool.main(_add_args(wav, manifest_path, ambient_root))
        assert rc == 2
        err = capsys.readouterr().err
        assert "sample rate 11025" in err
        assert not manifest_path.exists()

    def test_rejects_missing_audio_file(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        rc = tool.main(
            _add_args(
                tmp_path / "raw" / "nope.wav",
                manifest_path,
                ambient_root,
            )
        )
        assert rc == 4
        assert "does not exist" in capsys.readouterr().err
        assert not manifest_path.exists()

    def test_rejects_non_wav_extension(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        bogus = tmp_path / "raw" / "bogus.mp3"
        bogus.parent.mkdir(parents=True, exist_ok=True)
        bogus.write_bytes(b"not-a-wav")

        rc = tool.main(_add_args(bogus, manifest_path, ambient_root))
        assert rc == 4
        assert ".wav" in capsys.readouterr().err
        assert not manifest_path.exists()

    def test_rejects_invalid_recorded_at(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        wav = _make_wav(tmp_path / "raw" / "a.wav", duration_s=70.0)
        rc = tool.main(
            _add_args(
                wav,
                manifest_path,
                ambient_root,
                recorded_at="not-a-timestamp",
            )
        )
        assert rc == 2
        assert "ISO 8601" in capsys.readouterr().err
        assert not manifest_path.exists()

    def test_rejects_recorded_at_without_timezone(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        wav = _make_wav(tmp_path / "raw" / "a.wav", duration_s=70.0)
        rc = tool.main(
            _add_args(
                wav,
                manifest_path,
                ambient_root,
                recorded_at="2025-03-10T22:00:00",  # missing Z / offset
            )
        )
        assert rc == 2
        assert "no timezone" in capsys.readouterr().err

    def test_rejects_duplicate_filename(
        self, tmp_path, manifest_path, ambient_root
    ):
        wav = _make_wav(tmp_path / "raw" / "dup.wav", duration_s=70.0)
        rc1 = tool.main(_add_args(wav, manifest_path, ambient_root))
        assert rc1 == 0
        # Same basename -> would collide in manifest.
        rc2 = tool.main(_add_args(wav, manifest_path, ambient_root))
        assert rc2 == 3
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["sessions"]) == 1


# ------------------------------------------------------------------ validate
class TestValidate:
    def test_validate_happy_path(
        self, tmp_path, manifest_path, ambient_root
    ):
        # Seed a real session via `add` so the manifest exactly matches
        # the file on disk.
        wav = _make_wav(tmp_path / "raw" / "a.wav", duration_s=120.0)
        tool.main(
            _add_args(
                wav, manifest_path, ambient_root, copy_as_managed=True
            )
        )

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
            ]
        )
        assert rc == 0

    def test_validate_flags_missing_file(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        # Manifest references a file that doesn't exist on disk.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "generated_at_iso8601": "2025-01-15T00:00:00Z",
                    "sessions": [
                        {
                            "filename": "ghost.wav",
                            "duration_seconds": 120.0,
                            "sample_rate_hz": 22050,
                            "recorded_at_iso8601": "2025-01-15T06:00:00Z",
                            "location_label": "no_such_file",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "ghost.wav" in err
        assert "missing on disk" in err

    def test_validate_flags_duration_mismatch(
        self, tmp_path, manifest_path, ambient_root, capsys
    ):
        # File exists but the manifest claims a wildly wrong duration.
        wav_path = ambient_root / "real.wav"
        _make_wav(wav_path, duration_s=80.0)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "generated_at_iso8601": "2025-01-15T00:00:00Z",
                    "sessions": [
                        {
                            "filename": "real.wav",
                            # Claim 200s while the file is only 80s.
                            "duration_seconds": 200.0,
                            "sample_rate_hz": 22050,
                            "recorded_at_iso8601": "2025-01-15T06:00:00Z",
                            "location_label": "lying_manifest",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "duration_seconds=200.000" in err
        assert "probed 80." in err

    def test_validate_skips_placeholder_durations(
        self, tmp_path, manifest_path, ambient_root
    ):
        # Placeholder rows have aspirational durations (10 hours) but
        # the on-disk WAVs are tiny stubs. validate must skip the
        # duration check for placeholders.
        _make_wav(ambient_root / "stub.wav", duration_s=10.0)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "generated_at_iso8601": "2025-01-15T00:00:00Z",
                    "sessions": [
                        {
                            "filename": "stub.wav",
                            "duration_seconds": 36000.0,  # 10 hours, fake
                            "sample_rate_hz": 22050,
                            "recorded_at_iso8601": "2025-01-15T06:00:00Z",
                            "location_label": "PLACEHOLDER:stub",
                            "is_placeholder": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
            ]
        )
        assert rc == 0

    def test_validate_missing_manifest(self, tmp_path, capsys):
        rc = tool.main(
            ["validate", "--manifest", str(tmp_path / "missing.json")]
        )
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err
