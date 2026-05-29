"""Tests for ``backend/audio_model/evaluate_adversarial.py`` (task 4.13).

Covers the three scenarios called out in the task brief:

* All-placeholder corpus → exit 0 with the "no real adversarial clips
  available" stderr note (Adversarial_Suite is currently fully stubbed
  by task 4.12; we don't want CI to fail because of that).
* Mock model + real-looking corpus that flags every horn clip as a
  crash → exit 1 with ``horns`` listed on stderr (R7.4).
* Clean corpus (model says ``NORMAL`` for every clip) → exit 0 with low
  per-category FPR (R7.3).

The Audio_Detector itself is replaced with a mock for every test; this
keeps the test suite deterministic and independent of whether the
``crash_detector.pth`` checkpoint is present in the sandbox. Audio
files referenced by the manifest only need to exist on disk if the mock
calls back into librosa/soundfile, which it doesn't, so empty
placeholders are sufficient.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from backend.audio_model import evaluate_adversarial as tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clip(corpus_root: Path, category: str, filename: str) -> None:
    """Create an empty placeholder file for a manifest row.

    The script never opens the file — it only passes the path to
    ``inference.predict`` which we mock — so an empty file is enough to
    satisfy any future on-disk existence check we might add.
    """
    target = corpus_root / category / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")


def _write_manifest(
    manifest_path: Path,
    *,
    rows_by_category: dict[str, list[dict]],
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "generated_at_iso8601": "2025-01-15T00:00:00Z",
        "categories": {
            cat: list(rows_by_category.get(cat, []))
            for cat in tool.CATEGORIES
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _placeholder_row(category: str, idx: int) -> dict:
    return {
        "filename": f"{category}_placeholder_{idx:02d}.wav",
        "source_uri": f"synthetic:{category}_placeholder_{idx:02d}",
        "license": "CC0-1.0 (synthetic)",
        "duration_s": 4.0,
        "recorded_at_iso8601": "2025-01-15T00:00:00Z",
        "is_placeholder": True,
    }


def _real_row(category: str, idx: int) -> dict:
    return {
        "filename": f"{category}_{idx:04d}.wav",
        "source_uri": f"freesound:{category}_{idx}",
        "license": "CC-BY-4.0",
        "duration_s": 4.0,
        "recorded_at_iso8601": "2025-01-15T14:30:00Z",
        "is_placeholder": False,
    }


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(corpus_root, manifest_path)`` rooted in ``tmp_path``."""
    corpus_root = tmp_path / "adversarial"
    corpus_root.mkdir()
    return corpus_root, corpus_root / "manifest.json"


def _patch_predict(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decide: Callable[[str], str],
) -> None:
    """Mock ``inference.predict`` to return ``decide(path)`` per call.

    ``decide`` returns either ``"CRASH"`` or ``"NORMAL"``.
    """

    def _fake_predict(path):
        event = decide(str(path))
        return {"event": event, "confidence": 0.9}

    monkeypatch.setattr(tool.inference, "predict", _fake_predict)


def _build_argv(*, manifest: Path, out: Path) -> list[str]:
    return [
        "--manifest", str(manifest),
        "--out", str(out),
        "--git-sha", "deadbee",
    ]


# ---------------------------------------------------------------------------
# Scenario 1: all-placeholder corpus → exit 0
# ---------------------------------------------------------------------------


def test_all_placeholder_corpus_exits_zero_with_skip_message(
    corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    corpus_root, manifest = corpus
    rows = {
        cat: [_placeholder_row(cat, i) for i in range(1, 6)]
        for cat in tool.CATEGORIES
    }
    _write_manifest(manifest, rows_by_category=rows)
    for cat, cat_rows in rows.items():
        for r in cat_rows:
            _make_clip(corpus_root, cat, r["filename"])

    # Even when wired up, predict must never be called for placeholder rows.
    calls: list[str] = []

    def _predict_should_not_be_called(path):
        calls.append(str(path))
        return {"event": "CRASH", "confidence": 0.99}

    monkeypatch.setattr(tool.inference, "predict", _predict_should_not_be_called)

    out_path = tmp_path / "out.json"
    rc = tool.main(_build_argv(manifest=manifest, out=out_path))

    assert rc == 0, "stubbed corpus must not fail CI (task brief)"
    assert calls == [], "placeholder rows must not be scored"

    err = capsys.readouterr().err
    assert "no real adversarial clips available" in err

    # Report still emitted with zero per-category counts.
    report = json.loads(out_path.read_text(encoding="utf-8"))
    for cat in tool.CATEGORIES:
        assert report["per_category"][cat]["n_clips"] == 0
        assert report["per_category"][cat]["n_false_positives"] == 0
        assert report["per_category"][cat]["fpr"] == 0.0
        assert report["per_category"][cat]["n_placeholder_skipped"] == 5
    assert report["overall_fpr"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 2: model flags every horn → exit 1 with horns on stderr
# ---------------------------------------------------------------------------


def test_horns_flagged_on_every_clip_exits_one_with_horns_in_stderr(
    corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    corpus_root, manifest = corpus
    # Realistic shape: each category has a mix of placeholders and real
    # rows. The mock model flags every horns clip as CRASH and is
    # honest on the other categories — only horns should land on stderr.
    rows: dict[str, list[dict]] = {}
    for cat in tool.CATEGORIES:
        cat_rows = [_placeholder_row(cat, 1), _placeholder_row(cat, 2)]
        for i in range(1, 11):
            cat_rows.append(_real_row(cat, i))
        rows[cat] = cat_rows
        for r in cat_rows:
            _make_clip(corpus_root, cat, r["filename"])
    _write_manifest(manifest, rows_by_category=rows)

    def _decide(path: str) -> str:
        # Path includes the category folder; flag every horn clip.
        return "CRASH" if "/horns/" in path else "NORMAL"

    _patch_predict(monkeypatch, decide=_decide)

    out_path = tmp_path / "out.json"
    rc = tool.main(_build_argv(manifest=manifest, out=out_path))

    assert rc == 1, "FPR > 0.10 on horns must fail the gate (R7.4)"

    err = capsys.readouterr().err
    assert "horns" in err
    # The offending FPR (1.0000) should appear so an operator can read
    # the magnitude off the failure message.
    assert "fpr=" in err
    # Other categories must NOT appear in the offender list.
    for cat in ("fireworks", "tire_blowouts", "airbrakes"):
        assert cat not in err

    report = json.loads(out_path.read_text(encoding="utf-8"))
    horns = report["per_category"]["horns"]
    assert horns["n_clips"] == 10
    assert horns["n_false_positives"] == 10
    assert horns["fpr"] == pytest.approx(1.0)
    assert horns["n_placeholder_skipped"] == 2
    for cat in ("fireworks", "tire_blowouts", "airbrakes"):
        assert report["per_category"][cat]["n_false_positives"] == 0
        assert report["per_category"][cat]["fpr"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 3: clean corpus → exit 0 with low FPR
# ---------------------------------------------------------------------------


def test_clean_corpus_exits_zero_with_low_fpr(
    corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    corpus_root, manifest = corpus
    rows: dict[str, list[dict]] = {}
    for cat in tool.CATEGORIES:
        cat_rows = [_real_row(cat, i) for i in range(1, 21)]
        rows[cat] = cat_rows
        for r in cat_rows:
            _make_clip(corpus_root, cat, r["filename"])
    _write_manifest(manifest, rows_by_category=rows)

    # Honest model: never falsely flags. Overall FPR is 0%.
    _patch_predict(monkeypatch, decide=lambda _path: "NORMAL")

    out_path = tmp_path / "out.json"
    rc = tool.main(_build_argv(manifest=manifest, out=out_path))

    assert rc == 0
    err = capsys.readouterr().err
    assert "FAIL" not in err
    assert "no real adversarial clips" not in err

    report = json.loads(out_path.read_text(encoding="utf-8"))
    for cat in tool.CATEGORIES:
        block = report["per_category"][cat]
        assert block["n_clips"] == 20
        assert block["n_false_positives"] == 0
        assert block["fpr"] == 0.0
    assert report["overall_fpr"] == 0.0


# ---------------------------------------------------------------------------
# Boundary: per-category FPR exactly at threshold passes; just over fails
# ---------------------------------------------------------------------------


def test_fpr_at_threshold_passes_just_over_fails(
    corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_root, manifest = corpus

    # 10 real clips per category. 1 false positive on horns → fpr=0.10
    # which equals the threshold and must NOT fail (R7.3 says "less
    # than 0.10" but R7.4 says "exceeds 0.10"; we use strict >).
    rows: dict[str, list[dict]] = {
        cat: [_real_row(cat, i) for i in range(1, 11)]
        for cat in tool.CATEGORIES
    }
    _write_manifest(manifest, rows_by_category=rows)
    for cat, cat_rows in rows.items():
        for r in cat_rows:
            _make_clip(corpus_root, cat, r["filename"])

    flagged_paths = {
        str(corpus_root / "horns" / rows["horns"][0]["filename"]),
    }
    _patch_predict(
        monkeypatch,
        decide=lambda path: "CRASH" if path in flagged_paths else "NORMAL",
    )

    out_path = tmp_path / "out.json"
    rc = tool.main(_build_argv(manifest=manifest, out=out_path))
    assert rc == 0

    # Now flip one more horn clip to push fpr to 0.20 > 0.10 → fail.
    flagged_paths.add(
        str(corpus_root / "horns" / rows["horns"][1]["filename"])
    )
    out_path2 = tmp_path / "out2.json"
    rc2 = tool.main(_build_argv(manifest=manifest, out=out_path2))
    assert rc2 == 1


# ---------------------------------------------------------------------------
# Pre-flight: missing manifest is a fatal error
# ---------------------------------------------------------------------------


def test_missing_manifest_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "out.json"
    rc = tool.main(_build_argv(manifest=tmp_path / "nope.json", out=out_path))
    assert rc == 1
    assert "manifest not found" in capsys.readouterr().err
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# Per-clip resilience: a single bad clip doesn't abort the run
# ---------------------------------------------------------------------------


def test_single_failing_predict_does_not_abort(
    corpus: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    corpus_root, manifest = corpus
    rows = {
        cat: [_real_row(cat, i) for i in range(1, 6)]
        for cat in tool.CATEGORIES
    }
    _write_manifest(manifest, rows_by_category=rows)
    for cat, cat_rows in rows.items():
        for r in cat_rows:
            _make_clip(corpus_root, cat, r["filename"])

    bad = str(corpus_root / "horns" / rows["horns"][0]["filename"])

    def _decide(path):
        if path == bad:
            raise RuntimeError("decode failed")
        return {"event": "NORMAL", "confidence": 0.9}

    def _fake_predict(path):
        # _decide may raise; otherwise it returns a dict.
        result = _decide(str(path))
        return result

    monkeypatch.setattr(tool.inference, "predict", _fake_predict)

    out_path = tmp_path / "out.json"
    rc = tool.main(_build_argv(manifest=manifest, out=out_path))
    assert rc == 0

    report = json.loads(out_path.read_text(encoding="utf-8"))
    # The failing clip is excluded from the count; horns sees 4 instead
    # of 5 with zero false positives.
    assert report["per_category"]["horns"]["n_clips"] == 4
    assert report["per_category"]["horns"]["n_false_positives"] == 0
