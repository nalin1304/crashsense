"""
Onset_Detector — replaces the legacy 3-of-4 sliding-window vote debouncer.

Implements R8.1, R8.2, R8.3:

  * R8.1 — Lives at backend/audio_model/onset.py and replaces the 3-of-4 vote.
  * R8.2 — Emits at most one CRASH per onset and suppresses re-emission for
           a configurable refractory window (default 500 ms).
  * R8.3 — Exposes `refractory_ms` (int, 100–2000) and `threshold`
           (float, 0.0–1.0) parameters.

Design note (R8.4):
  * Two impacts spaced 200 ms apart with the default 500 ms refractory yield
    exactly one emission. With `refractory_ms=100` they yield two emissions.
    Behaviour is deterministic across runs.

Refractory monotonicity (R8.5, P5):
  * For any input score-stream S and any r1 < r2,
    `count(emissions(S, refractory=r1)) >= count(emissions(S, refractory=r2))`.
    A larger refractory can never produce *more* emissions than a smaller one
    because every score that the smaller refractory suppresses is also
    suppressed by the larger one (suppression windows are nested).
  * The corresponding property test lives in tests/pbt/test_onset_refractory.py
    (task 4.16).

Inputs are per-window scores already computed by the AST/ResNet head — the
"onset" is the first window whose score crosses `threshold` outside the
current refractory window. We do not run a separate spectral onset stage;
the model score already encodes the time-frequency evidence we care about.
"""

from __future__ import annotations

from typing import Optional


class OnsetDetector:
    """Score-driven onset detector with a configurable refractory window.

    Parameters
    ----------
    refractory_ms:
        Milliseconds during which any score above ``threshold`` is suppressed
        after an emission. Must be in the inclusive range ``[100, 2000]``.
    threshold:
        Minimum score required to consider a window as a candidate onset.
        Must be in the inclusive range ``[0.0, 1.0]``.
    """

    REFRACTORY_MIN_MS = 100
    REFRACTORY_MAX_MS = 2000
    THRESHOLD_MIN = 0.0
    THRESHOLD_MAX = 1.0

    def __init__(self, refractory_ms: int = 500, threshold: float = 0.5) -> None:
        if not isinstance(refractory_ms, int) or isinstance(refractory_ms, bool):
            raise TypeError(
                f"refractory_ms must be int, got {type(refractory_ms).__name__}"
            )
        if not (self.REFRACTORY_MIN_MS <= refractory_ms <= self.REFRACTORY_MAX_MS):
            raise ValueError(
                f"refractory_ms must be in [{self.REFRACTORY_MIN_MS}, "
                f"{self.REFRACTORY_MAX_MS}] ms, got {refractory_ms}"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(
                f"threshold must be float, got {type(threshold).__name__}"
            )
        threshold = float(threshold)
        if not (self.THRESHOLD_MIN <= threshold <= self.THRESHOLD_MAX):
            raise ValueError(
                f"threshold must be in [{self.THRESHOLD_MIN}, "
                f"{self.THRESHOLD_MAX}], got {threshold}"
            )

        self.refractory_ms: int = refractory_ms
        self.threshold: float = threshold
        self._last_emit_t_ms: Optional[int] = None

    @property
    def last_emit_t_ms(self) -> Optional[int]:
        """Timestamp (ms) of the most recent emission, or None."""
        return self._last_emit_t_ms

    def process(self, score: float, t_ms: int) -> bool:
        """Decide whether this (score, t_ms) pair should fire a CRASH emission.

        Logic:
          1. If ``score < threshold`` -> no emission, return False.
          2. If never emitted before, or
             ``t_ms - last_emit_t_ms >= refractory_ms`` -> emit, advance state,
             return True.
          3. Otherwise (within refractory window) -> return False.

        The refractory boundary is inclusive of ``t_ms == last + refractory_ms``
        so a score arriving exactly one refractory window later is allowed to
        emit.
        """
        if score < self.threshold:
            return False
        if (
            self._last_emit_t_ms is None
            or (t_ms - self._last_emit_t_ms) >= self.refractory_ms
        ):
            self._last_emit_t_ms = t_ms
            return True
        return False

    def reset(self) -> None:
        """Clear refractory state so the next above-threshold score emits."""
        self._last_emit_t_ms = None
