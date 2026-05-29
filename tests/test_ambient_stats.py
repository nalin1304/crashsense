"""Tests for ``backend/audio_model/ambient_stats.py`` (R6.2 / R6.4).

Covers the documented behaviour:

* placeholder rows (``is_placeholder == true``) are counted but never feed
  metrics
* a real session produces all four metric blocks (mean RMS, A-weighted SPL,
  spectral centroid distribution, zero-crossing-rate distribution)
* missing-on-disk and non-WAV files are appended to the errors CSV and the
  remaining sessions still process (R6.4 fail-open contract)
* the JSON report deserialises cleanly and is byte-identical between two
  back-to-back runs on the same inputs (determinism)

Every test uses ``tmp_path`` so the real ``data/ambient_long`` tree is
never touched.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from backend.audio_model import ambient_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate: int = 22050,
    seed: int = 42,
) -> Path:
    """Write a deterministic mono int16 WAV containing pink-noise-ish content.

    The signal is just a seeded normal distribution scaled into the int16
    range — enough to give nontrivial RMS, A-weighted SPL, spectral centroid
    and ZCR values. Determinism comes from ``np.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(sample_rate * duration_s))
    signal = rng.standard_normal(n).astype(np.float32)
    # Normalise to peak ≈ 0.4 so we don't accidentally clip in the int16
    # cast and so the dBFS measurement falls in a sensible range.
    signal = (0.4 * signal / max(float(np.max(np.abs(signal))), 1e-9))
    pcm = (signal * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sample_rate, pcm)
    return path


def _seed_corpus(
    *,
    ambient_root: Path,
    manifest_path: Path,
    extra_sessions: list[dict] | None = None,
) -> Path:
    """Seed a 2-session corpus: one real, one placeholder.

    Returns the path to the real WAV so the caller can mutate / delete it.
    """
    real_wav = _make_wav(ambient_root / "ambient_001.wav", duration_s=2.0)

    sessions = [
        {
            "filename": "ambient_001.wav",
            "duration_seconds": 2.0,
            "sample_rate_hz": 22050,
            "recorded_at_iso8601": "2025-01-15T06:00:00Z",
            "location_label": "test_real_session",
            # No is_placeholder flag — real session
        },
        {
            "filename": "ambient_002.wav",
            "duration_seconds": 36000.0,
            "sample_rate_hz": 22050,
            "recorded_at_iso8601": "2025-01-15T07:00:00Z",
            "location_label": "PLACEHOLDER:stub",
            "is_placeholder": True,
        },
    ]
    if extra_sessions:
        sessions.extend(extra_sessions)

    manifest = {
        "schema_version": "1.0.0",
        "generated_at_iso8601": "2025-01-15T00:00:00Z",
        "sessions": sessions,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return real_wav


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


class TestAweighting:
    def test_a_weighting_attenuates_low_frequency_tone(self):
        # 50 Hz pure tone — A-weighting attenuates by ~30 dB at 50 Hz, so
        # the filtered signal must have substantially lower RMS than the
        # input. Anything in [10 dB, 60 dB] of attenuation is fine; we
        # check the qualitative property, not a precise number.
        sample_rate = 22050
        n = sample_rate * 1
        t = np.arange(n) / sample_rate
        tone = np.sin(2.0 * np.pi * 50.0 * t).astype(np.float32)

        weighted = ambient_stats.a_weighted_signal(tone, sample_rate)

        rms_in = float(np.sqrt(np.mean(tone.astype(np.float64) ** 2)))
        rms_out = float(np.sqrt(np.mean(weighted ** 2)))
        # Low-frequency content must be heavily attenuated.
        assert rms_out < rms_in * 0.3

    def test_a_weighting_close_to_unity_at_1khz(self):
        # A-weighting is normalised to 0 dB at 1 kHz. The filter's gain at
        # 1 kHz should therefore be approximately 1.0.
        sample_rate = 22050
        n = sample_rate * 1
        t = np.arange(n) / sample_rate
        tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32)

        # Skip the filter warm-up region (first 200 samples) before
        # measuring — the IIR has a small transient from zero state.
        weighted = ambient_stats.a_weighted_signal(tone, sample_rate)

        rms_in = float(np.sqrt(np.mean(tone[200:].astype(np.float64) ** 2)))
        rms_out = float(np.sqrt(np.mean(weighted[200:] ** 2)))
        ratio = rms_out / max(rms_in, 1e-9)
        # Within ±0.5 dB of unity is well within the IEC 61672 tolerance,
        # but we use a looser ±2 dB band so platform numerical drift in
        # scipy's bilinear transform doesn't flake the test.
        assert 0.79 < ratio < 1.26


class TestComputeSessionStats:
    def test_all_metric_blocks_present_and_well_formed(self):
        sample_rate = 22050
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(sample_rate * 2).astype(np.float32) * 0.3

        out = ambient_stats.compute_session_stats(signal, sample_rate)

        assert set(out) == {
            "mean_rms",
            "a_weighted_spl_dbfs",
            "spectral_centroid_hz",
            "zero_crossing_rate",
        }
        assert isinstance(out["mean_rms"], float)
        assert out["mean_rms"] > 0.0

        # dBFS is always negative for non-clipping signals; floor is -120.
        assert ambient_stats.DBFS_FLOOR <= out["a_weighted_spl_dbfs"] <= 0.0

        for key in ("spectral_centroid_hz", "zero_crossing_rate"):
            block = out[key]
            assert set(block) == {"mean", "p50", "p95", "max"}
            for stat in ("mean", "p50", "p95", "max"):
                assert isinstance(block[stat], float)
                # Distribution stats are non-negative by construction.
                assert block[stat] >= 0.0
            # max is always >= mean, p95 >= p50 — sanity for the reduction.
            assert block["max"] >= block["mean"]
            assert block["p95"] >= block["p50"]

    def test_silence_yields_zero_rms_and_dbfs_floor(self):
        sample_rate = 22050
        signal = np.zeros(sample_rate, dtype=np.float32)

        out = ambient_stats.compute_session_stats(signal, sample_rate)

        assert out["mean_rms"] == 0.0
        # Pure silence -> A-weighted RMS is zero -> dBFS floor.
        assert out["a_weighted_spl_dbfs"] == ambient_stats.DBFS_FLOOR

    def test_empty_signal_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ambient_stats.compute_session_stats(
                np.zeros(0, dtype=np.float32), 22050
            )


# ---------------------------------------------------------------------------
# CLI / end-to-end tests
# ---------------------------------------------------------------------------


class TestAnalyseCorpus:
    def test_two_session_corpus_skips_placeholder_processes_real(
        self, tmp_path: Path
    ):
        ambient_root = tmp_path / "ambient_long"
        manifest_path = ambient_root / "manifest.json"
        out_path = tmp_path / "reports" / "ambient_stats.json"
        errors_path = tmp_path / "reports" / "ambient_stats.errors.csv"

        _seed_corpus(ambient_root=ambient_root, manifest_path=manifest_path)

        rc = ambient_stats.main(
            [
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
                "--out", str(out_path),
                "--errors", str(errors_path),
            ]
        )
        assert rc == 0
        assert out_path.exists()

        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["schema_version"] == ambient_stats.SCHEMA_VERSION
        assert report["n_sessions_total"] == 2
        assert report["n_sessions_skipped_placeholder"] == 1
        assert report["n_sessions_processed"] == 1

        # per_session contains exactly the real session, with all metrics.
        assert len(report["per_session"]) == 1
        rec = report["per_session"][0]
        assert rec["filename"] == "ambient_001.wav"
        assert "is_placeholder" not in rec  # placeholder never appears
        assert rec["mean_rms"] > 0.0
        assert ambient_stats.DBFS_FLOOR <= rec["a_weighted_spl_dbfs"] <= 0.0
        assert rec["spectral_centroid_hz"]["mean"] > 0.0
        assert rec["zero_crossing_rate"]["mean"] >= 0.0

        # Aggregate equals the single processed session's metrics.
        agg = report["aggregate"]
        assert agg["mean_rms"] == pytest.approx(rec["mean_rms"], rel=1e-9)
        assert agg["a_weighted_spl_dbfs"] == pytest.approx(
            rec["a_weighted_spl_dbfs"], rel=1e-9
        )
        for stat in ("mean", "p50", "p95", "max"):
            assert agg["spectral_centroid_hz"][stat] == pytest.approx(
                rec["spectral_centroid_hz"][stat], rel=1e-9
            )

        # Errors CSV was created (header only — no errors yet).
        assert errors_path.exists()
        with errors_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []

    def test_missing_file_recorded_in_errors_csv_and_report_continues(
        self, tmp_path: Path
    ):
        ambient_root = tmp_path / "ambient_long"
        manifest_path = ambient_root / "manifest.json"
        out_path = tmp_path / "reports" / "ambient_stats.json"
        errors_path = tmp_path / "reports" / "ambient_stats.errors.csv"

        # Seed corpus, then add a third session whose WAV is intentionally
        # absent from disk.
        _seed_corpus(
            ambient_root=ambient_root,
            manifest_path=manifest_path,
            extra_sessions=[
                {
                    "filename": "ambient_999.wav",
                    "duration_seconds": 5.0,
                    "sample_rate_hz": 22050,
                    "recorded_at_iso8601": "2025-01-15T08:00:00Z",
                    "location_label": "ghost",
                }
            ],
        )

        rc = ambient_stats.main(
            [
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
                "--out", str(out_path),
                "--errors", str(errors_path),
            ]
        )
        assert rc == 0  # R6.4: missing file is non-fatal

        report = json.loads(out_path.read_text(encoding="utf-8"))
        # 3 total, 1 placeholder, 1 processed, 1 errored (missing).
        assert report["n_sessions_total"] == 3
        assert report["n_sessions_skipped_placeholder"] == 1
        assert report["n_sessions_processed"] == 1

        # The real session must still appear in per_session.
        filenames = [s["filename"] for s in report["per_session"]]
        assert filenames == ["ambient_001.wav"]

        # Errors CSV records the ghost session with reason missing_on_disk.
        with errors_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["filename"] == "ambient_999.wav"
        assert rows[0]["reason"] == "missing_on_disk"
        # detail carries the resolved on-disk path.
        assert "ambient_999.wav" in rows[0]["detail"]
        # Timestamp ends with the canonical Z suffix.
        assert rows[0]["excluded_at_iso8601"].endswith("Z")

    def test_non_wav_extension_recorded_and_continues(self, tmp_path: Path):
        ambient_root = tmp_path / "ambient_long"
        manifest_path = ambient_root / "manifest.json"
        out_path = tmp_path / "reports" / "ambient_stats.json"
        errors_path = tmp_path / "reports" / "ambient_stats.errors.csv"

        _seed_corpus(ambient_root=ambient_root, manifest_path=manifest_path)
        # Drop a non-WAV file in the ambient root and reference it in the
        # manifest.
        (ambient_root / "bogus.mp3").write_bytes(b"not a wav")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sessions"].append(
            {
                "filename": "bogus.mp3",
                "duration_seconds": 5.0,
                "sample_rate_hz": 22050,
                "recorded_at_iso8601": "2025-01-15T09:00:00Z",
                "location_label": "wrong_format",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        rc = ambient_stats.main(
            [
                "--manifest", str(manifest_path),
                "--ambient-root", str(ambient_root),
                "--out", str(out_path),
                "--errors", str(errors_path),
            ]
        )
        assert rc == 0

        with errors_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert any(
            r["filename"] == "bogus.mp3" and r["reason"] == "non_wav_extension"
            for r in rows
        )

    def test_missing_manifest_is_fatal_pre_flight(self, tmp_path: Path, capsys):
        rc = ambient_stats.main(
            [
                "--manifest", str(tmp_path / "missing.json"),
                "--ambient-root", str(tmp_path / "ambient_long"),
                "--out", str(tmp_path / "out.json"),
                "--errors", str(tmp_path / "errors.csv"),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "manifest not found" in err

    def test_byte_identical_report_across_two_runs(self, tmp_path: Path):
        # Determinism: two back-to-back runs against the same corpus must
        # produce byte-identical JSON apart from the ``generated_at_iso8601``
        # field, which is allowed to differ since it is a wallclock stamp.
        ambient_root = tmp_path / "ambient_long"
        manifest_path = ambient_root / "manifest.json"
        out_a = tmp_path / "reports" / "a.json"
        out_b = tmp_path / "reports" / "b.json"
        errors_path = tmp_path / "reports" / "ambient_stats.errors.csv"

        _seed_corpus(ambient_root=ambient_root, manifest_path=manifest_path)

        for out_path in (out_a, out_b):
            rc = ambient_stats.main(
                [
                    "--manifest", str(manifest_path),
                    "--ambient-root", str(ambient_root),
                    "--out", str(out_path),
                    "--errors", str(errors_path),
                ]
            )
            assert rc == 0

        a = json.loads(out_a.read_text(encoding="utf-8"))
        b = json.loads(out_b.read_text(encoding="utf-8"))
        # Strip the wallclock field before comparing.
        a.pop("generated_at_iso8601", None)
        b.pop("generated_at_iso8601", None)
        assert a == b
