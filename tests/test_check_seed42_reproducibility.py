"""Tests for ``scripts/check_seed42_reproducibility.py`` (R31.4, P16).

Covers the three behaviours called out in the task:
  - matching outputs (exit 0)
  - differing outputs (exit 1, unified diff in stderr)
  - subprocess failure (exit 1, child stderr surfaced)

`subprocess.run` is monkeypatched so the tests do not actually shell out; we
only verify that the script's two-shot wrapper, diffing, and exit-code logic
behave correctly.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_seed42_reproducibility.py"


# ------------------------------------------------------------------ module loader
def _load_module():
    """Import the script as a module so we can monkeypatch its `subprocess`."""
    spec = importlib.util.spec_from_file_location(
        "check_seed42_reproducibility", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module():
    return _load_module()


# ------------------------------------------------------------------ helpers
def _result(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["python", "-m", "backend.audio_model.dataset_prep"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _make_run_sequence(*results: subprocess.CompletedProcess):
    """Return a fake ``subprocess.run`` that returns ``results`` in order."""
    iterator: Iterator[subprocess.CompletedProcess] = iter(results)
    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):  # noqa: ANN001 - matches subprocess.run signature
        calls.append(list(cmd))
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError("subprocess.run called more times than expected") from exc

    fake_run.calls = calls  # type: ignore[attr-defined]
    return fake_run


# ------------------------------------------------------------------ matching outputs
class TestMatchingOutputs:
    def test_returns_zero_on_byte_identical_stdout(self, script_module, monkeypatch, capsys):
        payload = b'{"seed": 42, "splits": {}}\n'
        fake_run = _make_run_sequence(_result(stdout=payload), _result(stdout=payload))
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main([])
        assert rc == 0
        out = capsys.readouterr()
        assert "OK" in out.out
        assert f"{len(payload)} bytes" in out.out
        assert out.err == ""

    def test_invokes_dataset_prep_module_twice(self, script_module, monkeypatch):
        payload = b'{"seed": 42}\n'
        fake_run = _make_run_sequence(_result(stdout=payload), _result(stdout=payload))
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main([])
        assert rc == 0
        assert len(fake_run.calls) == 2  # type: ignore[attr-defined]
        for cmd in fake_run.calls:  # type: ignore[attr-defined]
            assert cmd[0] == sys.executable
            assert cmd[1:3] == ["-m", "backend.audio_model.dataset_prep"]
            assert "--dry-run" in cmd
            # Default seed is 42.
            assert "--seed" in cmd
            assert cmd[cmd.index("--seed") + 1] == "42"

    def test_custom_seed_is_forwarded(self, script_module, monkeypatch):
        payload = b'{"seed": 7}\n'
        fake_run = _make_run_sequence(_result(stdout=payload), _result(stdout=payload))
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main(["--seed", "7"])
        assert rc == 0
        for cmd in fake_run.calls:  # type: ignore[attr-defined]
            assert cmd[cmd.index("--seed") + 1] == "7"


# ------------------------------------------------------------------ differing outputs
class TestDifferingOutputs:
    def test_returns_one_and_prints_unified_diff_on_mismatch(
        self, script_module, monkeypatch, capsys
    ):
        first = b'{"seed": 42, "splits": {"train": ["a.wav"]}}\n'
        second = b'{"seed": 42, "splits": {"train": ["b.wav"]}}\n'
        fake_run = _make_run_sequence(_result(stdout=first), _result(stdout=second))
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main([])
        assert rc == 1
        out = capsys.readouterr()
        # The script should announce the failure plus emit a unified diff to stderr.
        assert "FAIL" in out.err
        assert "NOT byte-identical" in out.err
        # Unified diff markers.
        assert "--- run1" in out.err
        assert "+++ run2" in out.err
        # Both differing tokens should appear in the diff body.
        assert "a.wav" in out.err
        assert "b.wav" in out.err
        # No success message on stdout.
        assert "OK" not in out.out

    def test_diff_includes_seed_in_filenames(self, script_module, monkeypatch, capsys):
        fake_run = _make_run_sequence(
            _result(stdout=b"hello\n"),
            _result(stdout=b"world\n"),
        )
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main(["--seed", "13"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "seed=13" in err


# ------------------------------------------------------------------ subprocess failure
class TestSubprocessFailure:
    def test_first_run_failure_exits_one_and_surfaces_stderr(
        self, script_module, monkeypatch, capsys
    ):
        fake_run = _make_run_sequence(
            _result(stdout=b"", stderr=b"boom: missing module\n", returncode=2),
        )
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main([])
        assert rc == 1
        # The script must NOT have called subprocess.run a second time.
        assert len(fake_run.calls) == 1  # type: ignore[attr-defined]
        err = capsys.readouterr().err
        assert "FAIL" in err
        assert "first run" in err
        assert "exited 2" in err
        assert "boom: missing module" in err

    def test_second_run_failure_exits_one(self, script_module, monkeypatch, capsys):
        fake_run = _make_run_sequence(
            _result(stdout=b'{"seed": 42}\n', returncode=0),
            _result(stdout=b"", stderr=b"second blew up\n", returncode=1),
        )
        monkeypatch.setattr(script_module.subprocess, "run", fake_run)

        rc = script_module.main([])
        assert rc == 1
        err = capsys.readouterr().err
        assert "second run" in err
        assert "second blew up" in err
