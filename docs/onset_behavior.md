# Onset Detector Behaviour

**Status:** Implemented (task 4.14, OnsetDetector live in `backend/audio_model/onset.py`)
**Owner:** Audio Model team
**Last updated:** 2025-01-15
**Related spec:** `.kiro/specs/crashsense-hardening/` — R8.1, R8.2, R8.3, R8.4, R8.5
**Related design section:** `design.md` §3.4 (Parallel tracks sketch — Onset detector paragraph) and `tasks.md` 4.14 (OnsetDetector implementation)

This document is the deliverable for **R8.4** of the CrashSense Hardening spec. It records the contract, the parameters, the behaviour on closely-spaced double impacts, and tuning guidance for field deployments.

---

## 1. Background

The legacy debouncer in `backend/audio_model/inference.py` was a 3-of-4 sliding-window vote: a CRASH was emitted only when 3 of the most recent 4 windows scored above the print threshold. This was easy to reason about on paper but had two field problems:

- **Latency was unbounded in the worst case.** A crash could need up to four window hops (≥ 1.5 s with 0.5 s hops) before any emission fired.
- **Refractory behaviour was implicit.** Re-fires after a true positive depended on how the next four windows happened to score, not on a configurable knob.

Per R8.1, the vote was replaced by an `OnsetDetector` keyed off the per-window model score. The first window whose score crosses `threshold` outside the current refractory window is the onset; the detector emits one CRASH at that window and suppresses further emissions until `refractory_ms` has elapsed.

---

## 2. The OnsetDetector contract

Per R8.2, the detector guarantees:

- **At most one CRASH per onset.** The first above-threshold window emits; all subsequent above-threshold windows within the refractory window are suppressed.
- **Refractory boundary is inclusive.** A score arriving exactly `refractory_ms` after the last emission *is* allowed to fire (the comparison is `>=`).
- **State is local to one detector instance.** Each `predict()` call constructs a fresh `OnsetDetector` so state never leaks between independent audio sessions.

The detector consumes scores already produced by the AST/ResNet head — it is not a separate spectral onset stage. The model score already encodes the time-frequency evidence we care about.

---

## 3. Parameters

Both knobs are env-overridable so deployments can tune without a code change.

| Parameter | Range | Default | Source |
| --- | --- | --- | --- |
| `refractory_ms` | `[100, 2000]` (int, ms) | `500` | `CRASHSENSE_ONSET_REFRACTORY_MS` |
| `threshold` | `[0.0, 1.0]` (float) | `0.85` (`PRINT_THRESHOLD`) | `CRASHSENSE_ONSET_THRESHOLD` |

Streaming inference in `backend/audio_model/inference.py` reads both env vars at module import. Out-of-range values raise `ValueError` at `OnsetDetector` construction time.

---

## 4. 200 ms-spaced double-impact behaviour (R8.4)

This is the canonical example R8.4 calls out: two true crash impacts in the same audio stream separated by 200 ms.

| `refractory_ms` | Emissions on a 200 ms-spaced double impact | Why |
| --- | --- | --- |
| `500` (default) | **Exactly one** | `200 < 500`, so the second impact falls inside the refractory window and is suppressed. |
| `200` | **Exactly two** | `200 >= 200`, so the second impact lands on the inclusive boundary and is allowed to fire. |
| `100` | **Exactly two** | `200 >= 100`, so the second impact is well outside the refractory window. |

The behaviour is **deterministic across runs**: the detector keeps no random state and the refractory comparison is a simple integer subtraction, so the same `(score_stream, refractory_ms, threshold)` tuple always yields the same emission count and the same emission timestamps.

The decision baked into the default config is the first row: at `refractory_ms = 500` a 200 ms double impact yields **one** emission, treated as a single physical crash. Tuning guidance for the alternative is in §6 below.

---

## 5. Refractory monotonicity (P5, R8.5)

For any input score-stream `S` and any two refractory values `r1 < r2`:

```
count(emissions(S, refractory=r1)) >= count(emissions(S, refractory=r2))
```

A larger refractory can never produce *more* emissions than a smaller one because every score the smaller refractory suppresses is also suppressed by the larger one — the suppression windows are nested. The corresponding Hypothesis property test lives in `tests/pbt/test_onset_refractory.py` (task 4.16).

This monotonicity is what makes refractory tuning safe: increasing `refractory_ms` to suppress nuisance re-fires never causes a real second crash to appear that was hidden at the lower setting.

---

## 6. Tuning guidance

The default of `500` ms is a balance suitable for highway-scenario crashes, which typically last 1–3 seconds (initial impact, secondary impact, debris). Adjust away from the default only with a measured reason.

| Scenario | Suggested `refractory_ms` | Trade-off |
| --- | --- | --- |
| Sustained tire screech or air brake noise causing repeated above-threshold windows | `1000`–`2000` | Fewer false re-fires on one sustained event. May miss a genuine second crash that arrives close behind the first. |
| Two close-by but distinct crashes (e.g. multi-car pile-up where impacts are 300–800 ms apart) | `100`–`200` | Catches the second impact as a separate event. Risk: a single sustained event may now produce two emissions. |
| Default (highway, mixed traffic) | `500` | Empirically suppresses re-fires on tire screeches and air brakes (typical duration ≤ 500 ms above threshold) while still catching distinct secondary crashes spaced ≥ 500 ms. |

Threshold tuning is independent. `0.85` (the streaming `PRINT_THRESHOLD`) was chosen so the onset detector fires only on high-confidence windows; lowering it increases recall on quiet crashes at the cost of false positives. We do not recommend running below `0.7` without the calibrator (R5) active.

---

## 7. Env var reference

```bash
# Suppress re-fires within 1 s of an emission (less sensitive to multi-impact crashes)
export CRASHSENSE_ONSET_REFRACTORY_MS=1000

# Lower the score threshold (more recall, more false positives)
export CRASHSENSE_ONSET_THRESHOLD=0.75
```

Both variables are read at module import time; restart the backend after changing them.

---

## 8. References

- `backend/audio_model/onset.py` — implementation
- `backend/audio_model/inference.py` — streaming integration (`ONSET_REFRACTORY_MS`, `ONSET_THRESHOLD`)
- `tests/pbt/test_onset_refractory.py` — refractory monotonicity property test (task 4.16)
- `.kiro/specs/crashsense-hardening/requirements.md` — R8.1, R8.2, R8.3, R8.4, R8.5
- `.kiro/specs/crashsense-hardening/design.md` — §3.4 onset detector paragraph
