"""Tests for the trainer's SNR-mixed checkpoint persistence (R2.6, task 2.4).

Covers the small surface added by ``backend.audio_model.train``:

* :func:`resolve_git_sha` returns the short SHA on success and ``"unknown"``
  on every documented failure mode (subprocess error, non-zero exit, empty
  stdout). We monkeypatch ``subprocess.run`` rather than depend on the
  ambient git checkout.
* :func:`resolve_snr_checkpoint_path` builds
  ``<arch>_snr_<git_sha>.pth`` under
  ``backend/audio_model/checkpoints/`` and rejects empty inputs.
* :func:`assert_distinct_from_baseline` is a no-op for any SNR target by
  construction and would raise if a future refactor collapsed the paths.
* :func:`save_snr_checkpoint` writes to the SNR target, refuses to
  overwrite an existing file (printing an actionable error and raising
  ``FileExistsError``), and emits an INFO log line containing the leading
  16 hex chars of the file's sha256.

These exercises do not invoke any training; they only exercise the
path-resolution + refuse-to-overwrite glue. The torch ``state_dict`` is a
small synthetic tensor map so writes are cheap.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

import pytest
import torch

from backend.audio_model import train as train_mod
from backend.audio_model.train import (
    CHECKPOINT_PATH,
    CHECKPOINTS_DIR,
    assert_distinct_from_baseline,
    resolve_git_sha,
    resolve_snr_checkpoint_path,
    save_snr_checkpoint,
)


# ---------------------------------------------------------------------------
# Fake state_dict (no real model needed for path-resolution tests)
# ---------------------------------------------------------------------------


def _tiny_state() -> dict:
    """A 1-tensor state dict; cheap to torch.save."""
    return {"layer.weight": torch.zeros(2, 2)}


# ---------------------------------------------------------------------------
# resolve_git_sha
# ---------------------------------------------------------------------------


class TestResolveGitSha:
    def test_returns_short_sha_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args[0], returncode=0, stdout="abc1234\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert resolve_git_sha() == "abc1234"

    def test_returns_unknown_on_nonzero_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args[0], returncode=128, stdout="", stderr="not a git repo"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert resolve_git_sha() == "unknown"

    def test_returns_unknown_on_empty_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args[0], returncode=0, stdout="\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert resolve_git_sha() == "unknown"

    def test_returns_unknown_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("git not on PATH")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert resolve_git_sha() == "unknown"

    def test_returns_unknown_on_subprocess_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert resolve_git_sha() == "unknown"


# ---------------------------------------------------------------------------
# resolve_snr_checkpoint_path
# ---------------------------------------------------------------------------


class TestResolveSnrCheckpointPath:
    def test_canonical_naming(self) -> None:
        p = resolve_snr_checkpoint_path("resnet18", "abc1234")
        assert p == CHECKPOINTS_DIR / "resnet18_snr_abc1234.pth"

    def test_efficientnet_naming(self) -> None:
        p = resolve_snr_checkpoint_path("efficientnet_b0", "deadbee")
        assert p.name == "efficientnet_b0_snr_deadbee.pth"
        assert p.parent == CHECKPOINTS_DIR

    def test_unknown_sha_fallback(self) -> None:
        p = resolve_snr_checkpoint_path("resnet18", "unknown")
        assert p.name == "resnet18_snr_unknown.pth"

    def test_rejects_empty_arch(self) -> None:
        with pytest.raises(ValueError, match="arch"):
            resolve_snr_checkpoint_path("", "abc1234")

    def test_rejects_empty_sha(self) -> None:
        with pytest.raises(ValueError, match="git_sha"):
            resolve_snr_checkpoint_path("resnet18", "")


# ---------------------------------------------------------------------------
# assert_distinct_from_baseline
# ---------------------------------------------------------------------------


class TestAssertDistinctFromBaseline:
    def test_distinct_path_passes(self) -> None:
        snr_path = resolve_snr_checkpoint_path("resnet18", "abc1234")
        # Path is in checkpoints/ subdir, baseline is at module root, so
        # they're always distinct by construction.
        assert_distinct_from_baseline(snr_path)

    def test_baseline_path_raises(self) -> None:
        with pytest.raises(AssertionError, match="refusing to overwrite"):
            assert_distinct_from_baseline(CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# save_snr_checkpoint
# ---------------------------------------------------------------------------


class TestSaveSnrCheckpoint:
    def test_writes_to_resolved_path_and_logs_sha256(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Redirect the checkpoints dir so we don't pollute the real
        # backend/audio_model/checkpoints/ folder.
        fake_ckpts = tmp_path / "checkpoints"
        monkeypatch.setattr(train_mod, "CHECKPOINTS_DIR", fake_ckpts)

        with caplog.at_level(logging.INFO, logger="train"):
            target = save_snr_checkpoint(
                _tiny_state(),
                "resnet18",
                95.5,
                git_sha="abc1234",
            )

        assert target == fake_ckpts / "resnet18_snr_abc1234.pth"
        assert target.exists()

        # The log line must include the first 16 hex chars of the sha256.
        expected_prefix = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        assert any(
            f"sha256={expected_prefix}" in rec.getMessage()
            for rec in caplog.records
        ), f"expected sha256={expected_prefix} in logs, got {[r.getMessage() for r in caplog.records]}"

    def test_uses_resolve_git_sha_when_arg_omitted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake_ckpts = tmp_path / "checkpoints"
        monkeypatch.setattr(train_mod, "CHECKPOINTS_DIR", fake_ckpts)
        monkeypatch.setattr(train_mod, "resolve_git_sha", lambda: "feedfac")

        target = save_snr_checkpoint(_tiny_state(), "resnet18", 90.0)
        assert target.name == "resnet18_snr_feedfac.pth"

    def test_refuses_to_overwrite_existing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        fake_ckpts = tmp_path / "checkpoints"
        monkeypatch.setattr(train_mod, "CHECKPOINTS_DIR", fake_ckpts)

        # Pre-create the target so the second save attempt collides.
        save_snr_checkpoint(_tiny_state(), "resnet18", 90.0, git_sha="abc1234")

        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            save_snr_checkpoint(
                _tiny_state(), "resnet18", 95.0, git_sha="abc1234"
            )

        captured = capsys.readouterr()
        assert "refusing to overwrite" in captured.err
        assert "abc1234" in captured.err

    def test_overwrite_true_allows_rewrite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Per-epoch best saves within a single run pass overwrite=True."""
        fake_ckpts = tmp_path / "checkpoints"
        monkeypatch.setattr(train_mod, "CHECKPOINTS_DIR", fake_ckpts)

        save_snr_checkpoint(_tiny_state(), "resnet18", 90.0, git_sha="abc1234")
        # Should not raise.
        target = save_snr_checkpoint(
            _tiny_state(),
            "resnet18",
            92.5,
            git_sha="abc1234",
            overwrite=True,
        )
        # And the val_acc should reflect the second write.
        payload = torch.load(target, map_location="cpu", weights_only=False)
        assert payload["val_acc"] == 92.5

    def test_distinct_from_baseline_assertion_holds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity check: the SNR namespace is structurally separate from
        the legacy ``crash_detector.pth`` baseline (R2.6)."""
        fake_ckpts = tmp_path / "checkpoints"
        monkeypatch.setattr(train_mod, "CHECKPOINTS_DIR", fake_ckpts)

        target = save_snr_checkpoint(
            _tiny_state(), "resnet18", 90.0, git_sha="abc1234"
        )
        assert target.resolve() != CHECKPOINT_PATH.resolve()
        assert "checkpoints" in target.parts
