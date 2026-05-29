"""Tests for `dataset_prep.py` --dry-run + source manifest emission.

Validates Requirements 31.4 (seed-42 byte-identical output) and the
manifest schema described in design §3.1.4.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.audio_model import dataset_prep as dp


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def fake_corpus() -> tuple[list[str], list[str]]:
    """A small, deterministic crash/noise file list with augmentation siblings."""
    crash = [
        "esc_glass_breaking_001.wav",
        "esc_glass_breaking_001_pitch+2.wav",
        "esc_glass_breaking_001_stretch9.wav",
        "us8k_6_100648_A.wav",
        "us8k_6_100648_A_noise12.wav",
        "freesound_332056.wav",
        "audioset_abc123.wav",
        "audioset_def456.wav",
    ]
    noise = [
        "esc_wind_010.wav",
        "esc_wind_010_pitch-2.wav",
        "us8k_0_99812_C.wav",
        "us8k_5_55_X.wav",
        "us8k_5_55_X_noise18.wav",
        "us8k_8_77_S.wav",
    ]
    return crash, noise


# ------------------------------------------------------------------ helpers
class TestSourceUriInference:
    def test_esc50_source(self):
        assert dp._infer_source_uri("esc_glass_breaking_001.wav") == "esc50:glass_breaking_001"

    def test_esc50_strips_augmentation(self):
        assert dp._infer_source_uri("esc_glass_breaking_001_pitch+2.wav") == "esc50:glass_breaking_001"

    def test_urbansound8k_source(self):
        assert dp._infer_source_uri("us8k_6_100648_A.wav") == "urbansound8k:6_100648_A"

    def test_freesound_source(self):
        assert dp._infer_source_uri("freesound_332056.wav") == "freesound:332056"

    def test_audioset_source(self):
        assert dp._infer_source_uri("audioset_-aaILOrkII.wav") == "audioset:-aaILOrkII"

    def test_unknown_falls_back_to_local(self):
        assert dp._infer_source_uri("mystery_clip_42.wav") == "local:mystery_clip_42"


class TestSourceKey:
    def test_strips_pitch_suffix(self):
        assert dp._source_key_from_stem("clip_pitch+2") == "clip"

    def test_strips_stretch_suffix(self):
        assert dp._source_key_from_stem("clip_stretch11") == "clip"

    def test_strips_noise_suffix(self):
        assert dp._source_key_from_stem("clip_noise18") == "clip"

    def test_no_suffix_unchanged(self):
        assert dp._source_key_from_stem("clip") == "clip"


# ------------------------------------------------------------------ split builder
class TestBuildSplitPreview:
    def test_keeps_augmentation_siblings_together(self, fake_corpus):
        crash, noise = fake_corpus
        splits = dp.build_split_preview(crash, noise, seed=42)

        # Every base + augmentation pair must end up in the same partition.
        partition_for: dict[str, str] = {}
        for partition, files in splits.items():
            for fn in files:
                partition_for[fn] = partition

        # esc_glass_breaking_001 has three siblings (base + pitch+2 + stretch9)
        siblings = [
            "esc_glass_breaking_001.wav",
            "esc_glass_breaking_001_pitch+2.wav",
            "esc_glass_breaking_001_stretch9.wav",
        ]
        assigned = {partition_for[fn] for fn in siblings}
        assert len(assigned) == 1, f"Augmentation siblings split across partitions: {assigned}"

    def test_partitions_are_disjoint(self, fake_corpus):
        crash, noise = fake_corpus
        splits = dp.build_split_preview(crash, noise, seed=42)
        train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()

    def test_partitions_cover_all_files(self, fake_corpus):
        crash, noise = fake_corpus
        splits = dp.build_split_preview(crash, noise, seed=42)
        all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert all_assigned == set(crash) | set(noise)

    def test_each_partition_is_sorted(self, fake_corpus):
        crash, noise = fake_corpus
        splits = dp.build_split_preview(crash, noise, seed=42)
        for partition, files in splits.items():
            assert files == sorted(files), f"{partition!r} partition is not sorted"


# ------------------------------------------------------------------ dry-run determinism (R31.4, P16)
class TestDryRunDeterminism:
    def test_two_consecutive_runs_byte_identical(self, fake_corpus, monkeypatch):
        crash, noise = fake_corpus
        # Force the dry-run path to use our deterministic corpus instead of
        # whatever is on disk.
        monkeypatch.setattr(dp, "_scan_class_dir", lambda d: crash if "crash" in d.name else noise)

        buf1 = io.StringIO()
        buf2 = io.StringIO()
        dp.emit_dry_run(stream=buf1, seed=42)
        dp.emit_dry_run(stream=buf2, seed=42)

        assert buf1.getvalue() == buf2.getvalue()
        # Sanity: it actually produced something.
        assert len(buf1.getvalue()) > 0

    def test_payload_has_required_top_level_keys(self, fake_corpus, monkeypatch):
        crash, noise = fake_corpus
        monkeypatch.setattr(dp, "_scan_class_dir", lambda d: crash if "crash" in d.name else noise)
        payload = dp.build_dry_run_payload(seed=42)
        assert set(payload.keys()) == {"seed", "splits", "source_uris"}
        assert payload["seed"] == 42
        assert set(payload["splits"].keys()) == {"train", "val", "test"}

    def test_source_uris_are_sorted_and_deduplicated(self, fake_corpus, monkeypatch):
        crash, noise = fake_corpus
        monkeypatch.setattr(dp, "_scan_class_dir", lambda d: crash if "crash" in d.name else noise)
        payload = dp.build_dry_run_payload(seed=42)
        uris = payload["source_uris"]
        assert uris == sorted(uris)
        assert len(uris) == len(set(uris))

    def test_different_seeds_give_different_output(self, fake_corpus, monkeypatch):
        crash, noise = fake_corpus
        monkeypatch.setattr(dp, "_scan_class_dir", lambda d: crash if "crash" in d.name else noise)
        a = dp.build_dry_run_payload(seed=42)
        b = dp.build_dry_run_payload(seed=99)
        # Splits should differ given different seeds (with this corpus size).
        assert a["splits"] != b["splits"]


# ------------------------------------------------------------------ source manifest
class TestSourceManifest:
    def test_writes_manifest_with_required_fields(self, fake_corpus, tmp_path):
        crash, noise = fake_corpus
        manifest_path = tmp_path / "_manifest.json"
        fixed_time = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

        manifest = dp.write_source_manifest(
            crash_files=crash,
            noise_files=noise,
            manifest_path=manifest_path,
            generated_at=fixed_time,
        )

        assert manifest_path.exists()
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert on_disk == manifest
        assert on_disk["schema_version"] == "1.0.0"
        assert on_disk["generated_at_iso8601"] == "2025-01-15T14:30:00Z"
        assert isinstance(on_disk["sources"], list)
        for entry in on_disk["sources"]:
            assert set(entry.keys()) == {"filename", "source_uri"}

    def test_sources_sorted_by_filename(self, fake_corpus, tmp_path):
        crash, noise = fake_corpus
        manifest_path = tmp_path / "_manifest.json"
        manifest = dp.write_source_manifest(
            crash_files=crash,
            noise_files=noise,
            manifest_path=manifest_path,
            generated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        filenames = [e["filename"] for e in manifest["sources"]]
        assert filenames == sorted(filenames)

    def test_manifest_byte_identical_with_fixed_timestamp(self, fake_corpus, tmp_path):
        crash, noise = fake_corpus
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

        dp.write_source_manifest(crash, noise, manifest_path=path_a, generated_at=ts)
        dp.write_source_manifest(crash, noise, manifest_path=path_b, generated_at=ts)

        assert path_a.read_bytes() == path_b.read_bytes()

    def test_manifest_covers_every_corpus_file(self, fake_corpus, tmp_path):
        crash, noise = fake_corpus
        manifest = dp.write_source_manifest(
            crash_files=crash,
            noise_files=noise,
            manifest_path=tmp_path / "_manifest.json",
            generated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        manifest_filenames = {e["filename"] for e in manifest["sources"]}
        assert manifest_filenames == set(crash) | set(noise)


# ------------------------------------------------------------------ CLI integration
class TestCliDryRun:
    def test_main_dry_run_writes_valid_json(self, fake_corpus, monkeypatch, capsys):
        crash, noise = fake_corpus
        monkeypatch.setattr(dp, "_scan_class_dir", lambda d: crash if "crash" in d.name else noise)

        rc = dp.main(["--dry-run"])
        assert rc == 0

        out = capsys.readouterr().out
        # Must be parseable JSON.
        payload = json.loads(out)
        assert payload["seed"] == dp.SPLIT_SEED
        assert "splits" in payload
        assert "source_uris" in payload

    def test_main_dry_run_does_not_call_downloaders(self, fake_corpus, monkeypatch, capsys):
        """`--dry-run` must not perform any network or filesystem mutation."""
        crash, noise = fake_corpus
        monkeypatch.setattr(dp, "_scan_class_dir", lambda d: crash if "crash" in d.name else noise)

        def fail(*_a, **_kw):
            raise AssertionError("dry-run path triggered a downloader / writer")

        monkeypatch.setattr(dp, "download_esc50", fail)
        monkeypatch.setattr(dp, "fetch_freesound_collisions", fail)
        monkeypatch.setattr(dp, "fetch_urbansound8k", fail)
        monkeypatch.setattr(dp, "ensure_dirs", fail)
        monkeypatch.setattr(dp, "write_source_manifest", fail)

        rc = dp.main(["--dry-run"])
        assert rc == 0
