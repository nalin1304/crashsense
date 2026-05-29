"""Tests for ``scripts/curate_adversarial.py``.

Covers happy-path ingest plus every documented rejection path (R7.1
audio-format and duration window, manifest source_uri uniqueness,
filename allocation, validate / list subcommands).

Each test uses a ``tmp_path`` corpus root + manifest path so no test
ever writes to the real ``data/adversarial/`` tree. WAV inputs are
synthesised on the fly with ``scipy.io.wavfile`` to avoid committing
new fixture files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from scripts import curate_adversarial as tool

SAMPLE_RATE_HZ = 22050  # matches tool.REQUIRED_SAMPLE_RATE_HZ


# ------------------------------------------------------------------ helpers
def _write_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    channels: int = 1,
    dtype: str = "int16",
    seed: int = 0,
) -> Path:
    """Write a synthetic WAV with the requested properties."""
    rng = np.random.default_rng(seed)
    n = int(round(sample_rate_hz * duration_s))
    if channels == 1:
        samples = (rng.standard_normal(n) * 0.2).astype(np.float64)
    else:
        samples = (rng.standard_normal((n, channels)) * 0.2).astype(np.float64)

    samples = np.clip(samples, -1.0, 1.0)

    if dtype == "int16":
        samples = (samples * 32767.0).astype(np.int16)
    elif dtype == "int32":
        samples = (samples * 2147483647.0).astype(np.int32)
    elif dtype == "float32":
        samples = samples.astype(np.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate_hz, samples)
    return path


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_add_args(
    audio_file: Path,
    *,
    category: str = "horns",
    license_str: str = "CC-BY-4.0",
    source_uri: str | None = None,
    manifest: Path,
    corpus_root: Path,
) -> list[str]:
    args = [
        "add",
        "--category", category,
        "--audio-file", str(audio_file),
        "--license", license_str,
        "--manifest", str(manifest),
        "--corpus-root", str(corpus_root),
    ]
    if source_uri is not None:
        args += ["--source-uri", source_uri]
    return args


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(corpus_root, manifest_path)`` rooted in ``tmp_path``."""
    corpus_root = tmp_path / "adversarial"
    corpus_root.mkdir()
    manifest = corpus_root / "manifest.json"
    return corpus_root, manifest


# ------------------------------------------------------------------ happy path
class TestAddHappyPath:
    def test_first_real_clip_creates_manifest_and_copies_file(
        self, tmp_path: Path, corpus: tuple[Path, Path]
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "src.wav", duration_s=5.0)

        rc = tool.main(
            _base_add_args(
                wav,
                category="horns",
                source_uri="freesound:1",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 0

        # File was copied with the 4-digit naming convention.
        copied = corpus_root / "horns" / "horns_0001.wav"
        assert copied.exists()
        assert copied.stat().st_size == wav.stat().st_size

        # Manifest entry has every required field.
        m = _read_manifest(manifest)
        assert m["schema_version"] == "1.0.0"
        assert "horns" in m["categories"]
        rows = m["categories"]["horns"]
        assert len(rows) == 1
        row = rows[0]
        assert row["filename"] == "horns_0001.wav"
        assert row["source_uri"] == "freesound:1"
        assert row["license"] == "CC-BY-4.0"
        assert row["is_placeholder"] is False
        assert abs(row["duration_s"] - 5.0) < 0.05
        assert row["recorded_at_iso8601"].endswith("Z")

    def test_filename_counter_increments_per_category(
        self, tmp_path: Path, corpus: tuple[Path, Path]
    ) -> None:
        corpus_root, manifest = corpus
        for idx in range(1, 4):
            wav = _write_wav(tmp_path / f"src_{idx}.wav", duration_s=4.0, seed=idx)
            assert (
                tool.main(
                    _base_add_args(
                        wav,
                        category="fireworks",
                        source_uri=f"freesound:fw_{idx}",
                        manifest=manifest,
                        corpus_root=corpus_root,
                    )
                )
                == 0
            )
        # And one in a different category to confirm sequences are independent.
        wav = _write_wav(tmp_path / "blowout.wav", duration_s=4.0, seed=99)
        assert (
            tool.main(
                _base_add_args(
                    wav,
                    category="tire_blowouts",
                    source_uri="freesound:tb_1",
                    manifest=manifest,
                    corpus_root=corpus_root,
                )
            )
            == 0
        )
        m = _read_manifest(manifest)
        fireworks_names = [r["filename"] for r in m["categories"]["fireworks"]]
        blowout_names = [r["filename"] for r in m["categories"]["tire_blowouts"]]
        assert fireworks_names == [
            "fireworks_0001.wav",
            "fireworks_0002.wav",
            "fireworks_0003.wav",
        ]
        assert blowout_names == ["tire_blowouts_0001.wav"]

    def test_default_source_uri_is_local_filename(
        self, tmp_path: Path, corpus: tuple[Path, Path]
    ) -> None:
        """When --source-uri is omitted, default to ``local:<filename>``."""
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "horn_clip.wav", duration_s=4.0)
        rc = tool.main(
            _base_add_args(
                wav,
                category="horns",
                source_uri=None,
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 0
        m = _read_manifest(manifest)
        assert m["categories"]["horns"][0]["source_uri"] == "local:horn_clip.wav"

    def test_minimum_duration_3s_is_accepted(
        self, tmp_path: Path, corpus: tuple[Path, Path]
    ) -> None:
        """Boundary check: exactly 3.0s is valid (R7.1 inclusive lower bound)."""
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "min.wav", duration_s=3.0)
        rc = tool.main(
            _base_add_args(
                wav,
                category="airbrakes",
                source_uri="freesound:min",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 0


# ------------------------------------------------------------------ rejections
class TestAddRejections:
    @pytest.mark.parametrize("duration_s", [0.5, 2.99, 30.01, 60.0])
    def test_rejects_duration_out_of_range(
        self, tmp_path: Path, corpus: tuple[Path, Path], duration_s: float, capsys
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "bad.wav", duration_s=duration_s)
        rc = tool.main(
            _base_add_args(
                wav,
                source_uri="freesound:bad",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 5
        assert "duration" in capsys.readouterr().err
        # No row written, no file copied.
        assert not manifest.exists()
        assert not (corpus_root / "horns" / "horns_0001.wav").exists()

    def test_rejects_non_wav_extension(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        bogus = tmp_path / "src.mp3"
        bogus.write_bytes(b"not-a-wav")
        rc = tool.main(
            _base_add_args(
                bogus,
                source_uri="freesound:mp3",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 4
        assert ".wav" in capsys.readouterr().err

    def test_rejects_missing_audio_file(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        rc = tool.main(
            _base_add_args(
                tmp_path / "does_not_exist.wav",
                source_uri="freesound:missing",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 4
        assert "does not exist" in capsys.readouterr().err

    def test_rejects_wrong_sample_rate(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "wrong_sr.wav", duration_s=4.0, sample_rate_hz=44100)
        rc = tool.main(
            _base_add_args(
                wav,
                source_uri="freesound:sr",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 5
        assert "sample rate" in capsys.readouterr().err

    def test_rejects_stereo(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "stereo.wav", duration_s=4.0, channels=2)
        rc = tool.main(
            _base_add_args(
                wav,
                source_uri="freesound:stereo",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 5
        assert "channel count" in capsys.readouterr().err

    def test_rejects_wrong_subtype(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        """A 32-bit PCM WAV is rejected because subtype != PCM_16."""
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "wrong_sub.wav", duration_s=4.0, dtype="int32")
        rc = tool.main(
            _base_add_args(
                wav,
                source_uri="freesound:subtype",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 5
        # Either subtype or PCM_16 should appear in the error.
        err = capsys.readouterr().err
        assert "subtype" in err.lower() or "pcm_16" in err.lower()

    def test_rejects_duplicate_source_uri(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        wav1 = _write_wav(tmp_path / "src_a.wav", duration_s=4.0, seed=1)
        wav2 = _write_wav(tmp_path / "src_b.wav", duration_s=4.0, seed=2)
        rc1 = tool.main(
            _base_add_args(
                wav1,
                source_uri="freesound:dup",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc1 == 0
        capsys.readouterr()
        rc2 = tool.main(
            _base_add_args(
                wav2,
                source_uri="freesound:dup",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc2 == 3
        assert "already present" in capsys.readouterr().err

        m = _read_manifest(manifest)
        assert len(m["categories"]["horns"]) == 1
        assert not (corpus_root / "horns" / "horns_0002.wav").exists()

    def test_rejects_duplicate_source_uri_across_categories(
        self, tmp_path: Path, corpus: tuple[Path, Path]
    ) -> None:
        """Same provenance URI is unique across the entire manifest, not per-category."""
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "src.wav", duration_s=4.0)
        assert (
            tool.main(
                _base_add_args(
                    wav,
                    category="horns",
                    source_uri="freesound:cross",
                    manifest=manifest,
                    corpus_root=corpus_root,
                )
            )
            == 0
        )
        rc = tool.main(
            _base_add_args(
                wav,
                category="airbrakes",
                source_uri="freesound:cross",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 3

    def test_rejects_empty_license(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "src.wav", duration_s=4.0)
        rc = tool.main(
            _base_add_args(
                wav,
                license_str="   ",
                source_uri="freesound:lic",
                manifest=manifest,
                corpus_root=corpus_root,
            )
        )
        assert rc == 2
        assert "license" in capsys.readouterr().err.lower()


# ------------------------------------------------------------------ list / validate
class TestList:
    def test_list_empty_manifest(
        self, corpus: tuple[Path, Path], capsys
    ) -> None:
        _, manifest = corpus
        rc = tool.main(["list", "--manifest", str(manifest)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "horns" in out
        assert "TOTAL: 0" in out

    def test_list_distinguishes_real_and_placeholder(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus

        # Seed manifest with a placeholder row by hand; mirrors what
        # `_regenerate_placeholders.py` writes.
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "generated_at_iso8601": "2025-01-15T00:00:00Z",
                    "categories": {
                        "horns": [
                            {
                                "filename": "horns_placeholder_01.wav",
                                "source_uri": "synthetic:horns_placeholder_01",
                                "license": "CC0-1.0 (synthetic)",
                                "duration_s": 4.0,
                                "recorded_at_iso8601": "2025-01-15T00:00:00Z",
                                "is_placeholder": True,
                            }
                        ],
                        "fireworks": [],
                        "tire_blowouts": [],
                        "airbrakes": [],
                    },
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Add one real clip.
        wav = _write_wav(tmp_path / "real.wav", duration_s=4.0)
        assert (
            tool.main(
                _base_add_args(
                    wav,
                    category="horns",
                    source_uri="freesound:real",
                    manifest=manifest,
                    corpus_root=corpus_root,
                )
            )
            == 0
        )
        capsys.readouterr()

        rc = tool.main(["list", "--manifest", str(manifest)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 real, 1 placeholder" in out
        assert "TOTAL: 2 (1 real, 1 placeholder)" in out


class TestValidate:
    def test_validate_passes_on_clean_corpus(
        self, tmp_path: Path, corpus: tuple[Path, Path]
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "src.wav", duration_s=4.0)
        assert (
            tool.main(
                _base_add_args(
                    wav,
                    source_uri="freesound:1",
                    manifest=manifest,
                    corpus_root=corpus_root,
                )
            )
            == 0
        )

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest),
                "--corpus-root", str(corpus_root),
            ]
        )
        assert rc == 0

    def test_validate_fails_on_missing_manifest(
        self, tmp_path: Path, capsys
    ) -> None:
        rc = tool.main(
            [
                "validate",
                "--manifest", str(tmp_path / "missing.json"),
                "--corpus-root", str(tmp_path),
            ]
        )
        assert rc == 1
        assert "manifest not found" in capsys.readouterr().err

    def test_validate_flags_missing_file(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "src.wav", duration_s=4.0)
        assert (
            tool.main(
                _base_add_args(
                    wav,
                    source_uri="freesound:1",
                    manifest=manifest,
                    corpus_root=corpus_root,
                )
            )
            == 0
        )
        # Delete the file to simulate corruption.
        (corpus_root / "horns" / "horns_0001.wav").unlink()

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest),
                "--corpus-root", str(corpus_root),
            ]
        )
        assert rc == 1
        assert "missing on disk" in capsys.readouterr().err

    def test_validate_flags_duration_mismatch(
        self, tmp_path: Path, corpus: tuple[Path, Path], capsys
    ) -> None:
        """Tampering with the manifest duration is caught by validate."""
        corpus_root, manifest = corpus
        wav = _write_wav(tmp_path / "src.wav", duration_s=4.0)
        assert (
            tool.main(
                _base_add_args(
                    wav,
                    source_uri="freesound:1",
                    manifest=manifest,
                    corpus_root=corpus_root,
                )
            )
            == 0
        )
        # Hand-edit the manifest duration to something obviously wrong.
        m = _read_manifest(manifest)
        m["categories"]["horns"][0]["duration_s"] = 99.0
        manifest.write_text(json.dumps(m, sort_keys=True, indent=2), encoding="utf-8")

        rc = tool.main(
            [
                "validate",
                "--manifest", str(manifest),
                "--corpus-root", str(corpus_root),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        # The mismatch surfaces because the recorded duration is far outside
        # both the 50 ms tolerance and the [3, 30] s window.
        assert (
            "manifest duration" in err
            or "duration" in err
        )
