# Implementation Plan: CrashSense Hardening

## Overview

Phase 1 (epic 1) establishes the real-world measurement baseline and the preservation gate. Phase 2 (epic 2) adds SNR-mixed training and re-evaluates against the Phase 1 number. Phase 3 (epic 3) brings up the production runtime in strict order Event_Store → dedup → dispatcher → backpressure → rate limiter / auth → metrics → model registry. Parallel tracks (epic 4 audio, 5 triangulation, 6 frontend, 7 testing) start once Phase 1 is green and run concurrently against the same preservation gate. Epic 8 is the final clean-checkout integration check.

Implementation languages: Python 3.11+ (backend, tooling, Hypothesis PBT) and TypeScript/JavaScript (frontend React, Vitest, fast-check). Property numbering P1–P16 follows the design's Correctness Properties section. Each task carries an `_Implements: R..., P..._` line and a `Depends on:` hint so the wave-based scheduler can parallelise correctly.

## Tasks

- [x] 1. Phase 1 — Real-world baseline + preservation gate
  - [x] 1.1 Pydantic schema for real-world label rows
    - Create `backend/audio_model/real_world_schema.py` with `RealWorldLabelRow` (filename regex, `crash`/`noise` literal, ISO 8601 timestamp, `annotator_id` regex, `source_uri` non-empty)
    - Expose `parse_labels_csv(path) -> tuple[list[RealWorldLabelRow], list[ErrorRow]]` that never raises on per-row failure
    - Enforce `source_uri` uniqueness at manifest level after parsing
    - Depends on: (none)
    - _Implements: R1.2, P14_

  - [x] 1.2 YouTube clip-curation tool
    - Create `scripts/curate_youtube_clips.py` recording `video_id`, `start_s`, `end_s`, `license`, `annotator_id`, `label`
    - Hand-cut 3–30 s windows; append rows to `data/real_world_test/labels.csv`
    - Reject clips whose `video_id` already appears in the AudioSet manifest
    - Depends on: 1.1
    - _Implements: R1.1, R1.2, R1.3_

  - [x] 1.3 Disjointness checker
    - Create `scripts/check_real_world_disjoint.py` computing set-intersection on `filename` and `source_uri` against `data/raw_audio/_manifest.json`
    - Also intersect on extracted YouTube `video_id` against the AudioSet `video_id` manifest
    - Exit non-zero with the offending list on any overlap
    - Depends on: 1.1, 1.4
    - _Implements: R1.3, R31.5_

  - [x] 1.4 Extend `dataset_prep.py` with `--dry-run` and source manifest emission
    - Add `--dry-run` flag emitting deterministic split JSON (sorted keys) without copying files
    - Generate `data/raw_audio/_manifest.json` listing every training/validation source URI
    - Preserve seed-42 byte-identical output across two consecutive dry runs
    - Depends on: (none)
    - _Implements: R31.4, P16_

  - [x] 1.5 JSON Schema for real-world eval report
    - Author `schemas/real_world_eval.schema.json` (draft 2020-12) with required fields per design §4.3
    - Include patterns for `git_sha`, `checkpoint_sha256`, `schema_version`
    - Validate via `jsonschema` in unit tests
    - Depends on: (none)
    - _Implements: R1.4_

  - [x] 1.6 `evaluate_real_world.py` CLI
    - Create `backend/audio_model/evaluate_real_world.py` with the CLI surface in design §3.1.3
    - Pin determinism: `torch.use_deterministic_algorithms(True)`, sorted iteration, `batch_size=1`, fixed seed, `json.dumps(..., sort_keys=True, indent=2)`
    - Exit codes 0 / 1 (fatal pre-flight) / 2 (>50% clips excluded)
    - Depends on: 1.1, 1.4, 1.5, 1.7
    - _Implements: R1.4, R1.5, R1.6, P13_

  - [x] 1.7 Eval helpers — errors CSV writer + report builder
    - Create `backend/audio_model/_eval_helpers.py` with `ErrorCsvWriter` (incremental append, R1.6 columns) and `ReportBuilder` (sklearn metrics, sorted-key JSON)
    - Compute `checkpoint_sha256` from the loaded checkpoint bytes
    - Depends on: 1.1
    - _Implements: R1.4, R1.6_

  - [x] 1.8 Fixture corpus for CI smoke
    - Add 4 short WAVs under `tests/fixtures/real_world_test/{crash,noise}/` (2 each, 3–10 s, mono 22050 Hz 16-bit)
    - Add `tests/fixtures/real_world_test/labels.csv` with valid plus intentionally-invalid rows for error-path coverage
    - Add `tests/fixtures/real_world_test/README.md` explaining purpose and regeneration
    - Depends on: (none)
    - _Implements: R1.1, R1.2, R1.6_

  - [ ] 1.9 Unit + property tests for `evaluate_real_world.py`
    - Create `tests/test_evaluate_real_world.py` with the six cases in design §6.1 (missing / short / long / non-wav / decode-error / csv-validation-error)
    - Hypothesis test for P14 `RealWorldLabelRow` CSV round-trip
    - Determinism test for P13 (run twice on fixture corpus, compare R1.5 fields byte-for-byte)
    - Depends on: 1.6, 1.8
    - _Implements: R1.5, R1.6, P13, P14_

  - [x] 1.10 `assert_metric.py` helper
    - Create `scripts/assert_metric.py` reading a JSON report, walking a dotted `--field` path, exiting non-zero when `value < --min`
    - Pure stdlib; emit actual / required to stderr on failure
    - Depends on: (none)
    - _Implements: R31.3, R31.5_

  - [x] 1.11 `check_seed42_reproducibility.py` helper
    - Create `scripts/check_seed42_reproducibility.py` running `dataset_prep.py --dry-run` twice and diffing JSON byte-for-byte
    - Print the unified diff to stderr on mismatch and exit non-zero
    - Depends on: 1.4
    - _Implements: R31.4, P16_

  - [x] 1.12 `make preservation-gate` target
    - Add the Make target in design §3.1.4 chaining: TDOA noise budget pytest, frontend smoke, WS round-trip, persistence-restart, held-out ≥97.0%, AudioSet ≥89.0%, seed-42 reproducibility, baseline-exists guard, disjointness check
    - Fail-fast on first non-zero sub-check (Make default)
    - Depends on: 1.3, 1.6, 1.10, 1.11
    - _Implements: R31.1, R31.2, R31.3, R31.4, R31.5_

  - [x] 1.13 CI workflow wiring
    - Add `.github/workflows/preservation-gate.yml` invoking `make preservation-gate` on every PR
    - Document `required-status-checks: ["preservation-gate"]` branch-protection note in `docs/ci.md`
    - Cache the model checkpoints + corpus fetch step
    - Depends on: 1.12
    - _Implements: R31.5_

  - [x] 1.14 Phase 1 checkpoint
    - Run `make preservation-gate` end-to-end on the fixture corpus and on the full corpus once it lands
    - Confirm `reports/real_world_eval_<sha>.json` written and validates against `schemas/real_world_eval.schema.json`
    - Ensure all tests pass, ask the user if questions arise.
    - _Implements: R1.4, R31.1, R31.2, R31.3, R31.4, R31.5_

- [ ] 2. Phase 2 — SNR-mixed training
  - [x] 2.1 SNR mixer
    - Create `backend/audio_model/noise_mixing.py::mix_clip(clean_signal, noise_signal, target_snr_db)` with power-based scaling within ±0.5 dB
    - Refactor existing `mix_with_highway_noise` and `mix_at_snr` into private helpers behind `mix_clip`
    - Same dtype, sample rate, channel count, length as `clean_signal`
    - Depends on: 1.14
    - _Implements: R2.1, R2.2_

  - [ ] 2.2 SNR mixer unit + property tests
    - Add `tests/test_noise_mixing.py` asserting measured SNR is within ±0.5 dB across `{0, 10, 20}` dB on synthetic and real noise stems
    - Hypothesis property for general `target_snr_db ∈ [-10, 30]` tolerance
    - Depends on: 2.1
    - _Implements: R2.2_

  - [x] 2.3 Trainer `--snr-mix` flag
    - Add `--snr-mix` to `backend/audio_model/train.py`; per-sample probability 0.5; uniform random `target_snr_db` from `{0, 10, 20}`
    - On noise-load failure: log filename at WARNING and skip that sample without aborting
    - Looped/cropped noise to match clean clip length
    - Depends on: 2.1
    - _Implements: R2.3, R2.4_

  - [x] 2.4 SNR-mixed checkpoint persistence
    - Persist trained weights to `backend/audio_model/checkpoints/<arch>_snr_<git_sha>.pth`
    - Refuse to overwrite Phase 1 baseline checkpoint (assert distinct path)
    - Emit checkpoint sha256 to training log
    - Depends on: 2.3
    - _Implements: R2.6_

  - [x] 2.5 Phase-2 evaluation gate
    - Re-run `evaluate_real_world.py` on the SNR checkpoint and emit `reports/real_world_eval_snr_<sha>.json`
    - Add `scripts/assert_phase2_gate.py` asserting `accuracy >= phase1_baseline - 0.01`
    - Add `make phase2-gate` Make target invoking the assertion
    - Depends on: 1.6, 2.4
    - _Implements: R2.5_

  - [x] 2.6 Phase 2 checkpoint
    - Run `make preservation-gate` followed by `make phase2-gate`
    - Verify SNR checkpoint did not overwrite the baseline; verify Phase 1 metrics still green
    - Ensure all tests pass, ask the user if questions arise.
    - _Implements: R2.5, R2.6, R31.5_

- [ ] 3. Phase 3 — Production runtime
  - [x] 3.1 `EventStoreBackend` interface
    - Create `backend/api/store.py::EventStoreBackend` abstract base (`write_batch`, `read_one`, `read_all`, `close`)
    - Add `make_event_store()` factory dispatching on `CRASHSENSE_EVENT_STORE_BACKEND` (`sqlite` default, `external` pluggable)
    - Depends on: 1.14
    - _Implements: R18.4_

  - [x] 3.2 SQLite WAL Event_Store with batched writer
    - Implement `SqliteEventStore(EventStoreBackend)` with `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`
    - Batched writer flushing on `write_batch_size` (default 50) or `write_batch_window_ms` (default 100), whichever first
    - Retry on `BUSY`/`LOCKED` up to 5 times with 10 ms exponential backoff; structured error log on final failure
    - Depends on: 3.1
    - _Implements: R18.1, R18.2, R18.3, R18.6_

  - [ ] 3.3 Event_Store concurrency + restart tests
    - Add `tests/test_event_store.py` with 1000 events / 10 producers / 5 s wall-clock budget asserting 1000 distinct rows
    - `test_survives_restart` re-opens DB after process-restart and asserts all rows present (preservation gate hook)
    - Depends on: 3.2
    - _Implements: R18.5, R31.2_

  - [x] 3.4 Event_Deduplicator
    - Create `backend/api/dedup.py` keyed on `event_id` with sliding `dedup_window_seconds` (default 300)
    - Wire into `POST /detect-audio` and `/simulate-crash`: persist plus broadcast on first sight; return `{"deduplicated": true}` on repeat
    - Reject HTTP 400 on missing `event_id`
    - Depends on: 3.2
    - _Implements: R15.1, R15.2, R15.3, R15.5_

  - [ ] 3.5 Dedup property tests (Hypothesis)
    - `tests/pbt/test_dedup_properties.py`: P3 idempotency over N submissions, P4 non-duplication across M distinct ids
    - Use 100+ Hypothesis iterations; minimal counterexample on failure
    - Depends on: 3.4, 7.8
    - _Implements: R15.4, R30.4, R30.5, P3, P4_

  - [ ] 3.6 Per_Event_Dispatcher
    - Create `backend/api/dispatcher.py` with per-`event_id` state machine, `max_concurrent_dispatches` (default 8), FIFO queue when full
    - Allocate dedicated drone slot per event with origin equal to the nearest Toll_Plaza
    - Transitions `DETECTED → DRONE_DISPATCHED → DRONE_ARRIVED | DISPATCH_FAILED`; release slot on terminal state without affecting other events
    - Depends on: 3.4
    - _Implements: R16.1, R16.2, R16.3, R16.5, R16.6_

  - [ ] 3.7 Dispatcher slot-exclusivity property test
    - `tests/pbt/test_dispatch_slot_exclusivity.py` Hypothesis stateful test driving submit/transition steps and asserting no two active dispatches share a slot id
    - Depends on: 3.6, 7.8
    - _Implements: R16.4, R30.9, P8_

  - [ ] 3.8 Backpressure_Controller
    - Update `backend/api/ws_manager.py` with per-client bounded queue (`client_queue_max` default 256), drop-oldest on overflow, `ws_messages_dropped_total{client_id}` counter
    - Per-client send loop; `slow_client_timeout_ms` (default 5000) closes the connection and increments `ws_clients_dropped_total`
    - Fast-client send latency independent of slow-client behaviour
    - Depends on: 3.6
    - _Implements: R17.1, R17.2, R17.3, R17.4_

  - [ ] 3.9 Backpressure fast-client property test
    - `tests/pbt/test_backpressure_fast_client.py` with mixed fast/slow clients; assert fast-client latency bound holds under any slow-client behaviour
    - Depends on: 3.8, 7.8
    - _Implements: R17.5, R30.11, P10_

  - [ ] 3.10 Rate_Limiter middleware
    - Create `backend/api/rate_limit.py` token-bucket per route with `rate_limit_burst` and `rate_limit_sustained_per_minute`
    - Mount on `/simulate-crash` and `/detect-audio`; respond 429 with `Retry-After` seconds
    - Log every 429 at WARNING with route + client id + correlation id
    - Depends on: 3.8
    - _Implements: R19.1, R19.2, R19.6_

  - [ ] 3.11 Bearer auth on `/simulate-crash` and `/detect-audio`
    - Create `backend/api/auth.py` validating `Authorization: Bearer <token>` against the configured backend
    - Respond 401 plus WARNING log on missing or invalid token; do not execute pipeline
    - Update `backend/api/routes.py` to require the dependency on both routes
    - Depends on: 3.10
    - _Implements: R19.4, R19.5, R19.6_

  - [ ] 3.12 Rate limiter budget-bound property test
    - `tests/pbt/test_rate_limit_budget.py` Hypothesis: across any rolling 60 s window, accepted ≤ `burst + sustained_per_minute`
    - Depends on: 3.10, 7.8
    - _Implements: R19.3, R30.10, P9_

  - [x] 3.13 SensorDeployment Pydantic schema + parser
    - Create `backend/triangulation/schema.py` with `SensorDeployment` and `SensorRecord` per design §4.1
    - Create `backend/triangulation/deployment_io.py` with `load_deployment(path)` / `dump_deployment(d, path)` supporting YAML and JSON
    - Production-mode validator rejects any sensor with `clock_source == "NONE"`
    - Depends on: 1.14
    - _Implements: R10.1, R10.2, R10.3, R10.5_

  - [x] 3.14 Bundled demo deployment + loader fallback
    - Generate `backend/triangulation/deployments/demo_3sensor.yaml` from the existing `MINIMAL` tuple in `sensor_config.py`
    - Backend loads `CRASHSENSE_DEPLOYMENT_PATH` at startup, falling back to the bundled file
    - Preserve R31.1 coordinates exactly
    - Depends on: 3.13
    - _Implements: R10.6_

  - [x] 3.15 `GET /sensors` returns deployment mode
    - Update `backend/api/routes.py` to read the active deployment and append `mode` to the per-sensor response
    - Preserve every base-spec response field for backward compatibility
    - Depends on: 3.13, 3.14
    - _Implements: R10.7_

  - [ ] 3.16 SensorDeployment round-trip property test
    - `tests/pbt/test_sensor_deployment_roundtrip.py` Hypothesis: `parse(dump(D)) == D` field-wise, in both YAML and JSON
    - Depends on: 3.13, 7.8
    - _Implements: R10.4, R30.12, P11_

  - [ ] 3.17 Latency_Recorder + `deadline_ms` parameter
    - Create `backend/api/metrics.py` with `audio_inference_latency_ms` (Prometheus histogram) and `tdoa_localize_latency_ms`
    - Update `backend/audio_model/inference.py::predict` to accept `deadline_ms` (50–5000, default 500); return `event="DEADLINE"` plus `deadline_exceeded=true` on overrun
    - EMA of p99 latency; emit `latency_degraded` WARNING when EMA p99 > 250 ms for 60 consecutive seconds
    - Depends on: 3.8
    - _Implements: R9.1, R9.2, R9.3, R9.4, R9.5_

  - [ ] 3.18 `GET /metrics` Prometheus endpoint
    - Mount Prometheus client in `backend/api/main.py` exposing histograms + counters from R21.3
    - Add `crash_events_total{outcome}`, `crash_events_deduplicated_total`, `model_swap_failures_total`, `ws_messages_dropped_total`, `ws_clients_dropped_total`
    - Respond within 1 s under load
    - Depends on: 3.17
    - _Implements: R21.3_

  - [ ] 3.19 JSON-Lines logging + correlation ID middleware
    - Create `backend/api/logging_config.py` emitting JSON Lines with `timestamp_iso8601`, `level`, `logger`, `message`, `correlation_id`, optional `event_id`
    - Create `backend/api/middleware.py::CorrelationIdMiddleware` generating UUID v4; adopt `X-Correlation-ID` header when valid
    - Set correlation id on every downstream log line for that request
    - Depends on: 3.18
    - _Implements: R21.1, R21.2, R21.5_

  - [ ] 3.20 Correlation-id propagation test
    - `tests/test_correlation_id.py` asserting all log lines for a single CrashEvent share the same `correlation_id`
    - Verify `X-Correlation-ID` adoption when caller supplies a valid UUID v4
    - Depends on: 3.19
    - _Implements: R21.4_

  - [ ] 3.21 Model_Registry + `ACTIVE.json` hot-swap
    - Create `backend/audio_model/registry.py` loading primary + secondary checkpoints with `ChannelTransition(primary_id, secondary_id, ab_split_percent, swap_started_at)` under an `RWLock`
    - Watcher thread polls `ACTIVE.json` every 30 s; per-`predict()` routing captures the decision at call start
    - Hold previous primary for `swap_drain_seconds` (default 30); fall back plus increment `model_swap_failures_total` on missing/unloadable checkpoint
    - Depends on: 3.18
    - _Implements: R20.1, R20.2, R20.3, R20.4, R20.5_

  - [ ] 3.22 `GET /model/active` endpoint
    - Return the contents of `ACTIVE.json` plus the loaded checkpoint's most recent `accuracy` from `reports/real_world_eval_*.json`
    - Respond within 1 s
    - Depends on: 3.21
    - _Implements: R20.6_

  - [ ] 3.23 Model registry hot-swap tests
    - `tests/test_model_registry.py`: swap mid-prediction does not drop in-flight, drain hold honoured, fallback on missing checkpoint
    - Assert `model_swap_failures_total` increments on broken pointer
    - Depends on: 3.21
    - _Implements: R20.4, R20.5_

  - [ ] 3.24 Phase 3 checkpoint
    - Run `make preservation-gate`, `make load-test`, `make pbt`
    - Verify Event_Store + dedup + dispatcher + backpressure landed in order; verify `/metrics` and `/model/active` reachable
    - Ensure all tests pass, ask the user if questions arise.
    - _Implements: R9, R10, R15, R16, R17, R18, R19, R20, R21_

- [ ] 4. Parallel track — Audio
  - [x] 4.1 Severity_Classifier
    - Implement `backend/audio_model/severity.py::grade(audio_signal_or_path)` returning `{severity, severity_confidence}`
    - Wire into `Audio_Detector` so every `event="CRASH"` window calls `grade` before returning
    - Fall back to `("moderate", 0.0)` and WARNING log on exception or out-of-range label
    - Depends on: 1.14
    - _Implements: R3.1, R3.2, R3.4_

  - [x] 4.2 Severity head training + macro-F1 ≥0.60 gate
    - Add `backend/audio_model/train_severity.py` training the severity head on the labelled crash subset of the real-world set
    - `scripts/assert_severity_gate.py` asserting macro-F1 ≥ 0.60 across `{minor, moderate, severe}`
    - Persist head to `backend/audio_model/checkpoints/severity_<git_sha>.pth`
    - Depends on: 4.1
    - _Implements: R3.5_

  - [ ] 4.3 Extend `CrashEvent` schema with severity fields
    - Add `severity` (literal) and `severity_confidence` (0.0–1.0) to `backend/api/schemas.py::CrashEvent`
    - Update broadcast and persistence paths to include both fields
    - Depends on: 4.1, 3.4
    - _Implements: R3.3_

  - [x] 4.4 Spatial audio decision doc
    - Author `docs/spatial_audio_decision.md` with `Hypothesis / Experimental Setup / Results / Decision (ship or defer)`
    - Default decision: defer with measured lift, cost estimate, revisit conditions per design §3.4
    - Depends on: 1.14
    - _Implements: R4.1, R4.3_

  - [x] 4.5 Spatial audio downmix path
    - Update `backend/audio_model/inference.py` to downmix multi-channel input to mono when channel count is outside the supported range
    - INFO log with correlation id on every downmix
    - Depends on: 4.4
    - _Implements: R4.4_

  - [x] 4.6 Calibrator (temperature scaling + isotonic)
    - Create `backend/audio_model/calibrate.py::fit(method, validation_logits, validation_labels)` returning an object with `transform(logits)`
    - Persist to `backend/audio_model/checkpoints/<arch>_calibrator_<git_sha>.json` in plain text (sorted breakpoints for isotonic)
    - Audio_Detector applies `transform` before returning `confidence`; falls back to softmax + `calibration_active=false` + WARNING on missing/corrupt file
    - Depends on: 1.14
    - _Implements: R5.1, R5.3, R5.4, R5.5_

  - [ ] 4.7 Calibrator round-trip + ECE tests
    - `tests/test_calibrate.py` asserting ECE < 0.01 on the real-world set across 15 equal-width bins
    - `tests/pbt/test_calibrator_roundtrip.py` Hypothesis: `decode(encode(C)).transform(x)` within 1e-6 of `C.transform(x)`
    - Depends on: 4.6, 7.8
    - _Implements: R5.2, R5.6, R30.13, P12_

  - [x] 4.8 Ambient capture corpus + manifest
    - Populate `data/ambient_long/` with ≥3 sessions totalling ≥24 cumulative hours
    - Author `data/ambient_long/manifest.json` per session: `filename`, `duration_seconds`, `sample_rate_hz`, `recorded_at_iso8601`, `location_label`
    - Add `scripts/curate_ambient.py` ingestion helper recording provenance
    - Depends on: 1.14
    - _Implements: R6.1_

  - [x] 4.9 Ambient analysis script
    - Create `backend/audio_model/ambient_stats.py` computing per-session and aggregate mean RMS, A-weighted SPL, spectral centroid distribution, zero-crossing rate distribution
    - Write `reports/ambient_stats.json` and `reports/ambient_stats.errors.csv` for missing/invalid sessions
    - Depends on: 4.8
    - _Implements: R6.2, R6.4_

  - [x] 4.10 Ambient stats summary doc
    - Author `docs/ambient_stats_summary.md` narrating distribution highlights from `reports/ambient_stats.json`
    - Reference deployment-relevant SPL ranges and spectral centroid behaviour
    - Depends on: 4.9
    - _Implements: R6.2_

  - [x] 4.11 Trainer `--ambient-mix` flag
    - Add `--ambient-mix` to `train.py`; sample negatives from `data/ambient_long/` at per-batch probability in `[0.2, 0.5]`
    - Compose with `--snr-mix` cleanly when both are passed
    - Depends on: 4.8, 2.3
    - _Implements: R6.3_

  - [x] 4.12 Adversarial_Suite corpus
    - Populate `data/adversarial/{horns,fireworks,tire_blowouts,airbrakes}/` with ≥50 WAV clips each (3–30 s, mono 22050 Hz)
    - Add `scripts/curate_adversarial.py` with provenance manifest
    - Confirm zero overlap with training corpora via `check_real_world_disjoint` extension
    - Depends on: 1.14
    - _Implements: R7.1_

  - [x] 4.13 `evaluate_adversarial.py`
    - Create `backend/audio_model/evaluate_adversarial.py` running the active checkpoint and emitting per-category FPR to `reports/adversarial_<git_sha>.json`
    - Exit non-zero with offending categories on stderr if any per-category FPR > 0.10
    - Depends on: 4.12
    - _Implements: R7.2, R7.3, R7.4_

  - [x] 4.14 Onset_Detector replacing 3-of-4 vote
    - Create `backend/audio_model/onset.py` with `refractory_ms` (100–2000) and `threshold` (0.0–1.0) parameters
    - Replace the 3-of-4 sliding-window vote in `backend/audio_model/inference.py` with onset-driven emission and 500 ms refractory
    - Emit at most one CRASH per onset; suppress within refractory window
    - Depends on: 1.14
    - _Implements: R8.1, R8.2, R8.3_

  - [x] 4.15 Onset behaviour doc
    - Author `docs/onset_behavior.md` documenting the 200 ms-spaced double-impact behaviour (one or two emissions, consistent across runs)
    - Reference `refractory_ms` semantics and tuning guidance
    - Depends on: 4.14
    - _Implements: R8.4_

  - [ ] 4.16 Onset refractory monotonicity property test
    - `tests/pbt/test_onset_refractory.py` Hypothesis: for `r1 < r2`, emissions(S, r1) ≥ emissions(S, r2) over generated audio streams
    - Depends on: 4.14, 7.8
    - _Implements: R8.5, R30.6, P5_

- [ ] 5. Parallel track — Triangulation
  - [x] 5.1 Atmospheric_Corrector
    - Create `backend/triangulation/atmospheric.py::effective_speed_of_sound(temperature_c, humidity_pct, wind_vector_mps, propagation_unit_vector)`
    - Validate input ranges per R11.2; reuse the existing `speed_of_sound_at(T, RH)` baseline; add wind-projection term `c_eff = c_still + dot(wind_vec, prop_vec)`
    - INFO log with correlation id once per CrashEvent when defaults substituted
    - Depends on: 1.14
    - _Implements: R11.1, R11.2, R11.6_

  - [x] 5.2 Atmospheric integration into `tdoa_localize`
    - Update `backend/triangulation/tdoa_solver.py` to use `effective_speed_of_sound` in place of `SPEED_OF_SOUND` when active deployment `mode == "production"`
    - Pass through measured atmospheric inputs from the request context
    - Depends on: 5.1, 3.13
    - _Implements: R11.5_

  - [ ] 5.3 Atmospheric monotonicity + sign-correctness PBTs
    - `tests/pbt/test_atmospheric_properties.py` Hypothesis: P6 temperature monotonicity at fixed humidity + wind; P7 wind sign-correctness given prop · wind dot product
    - Depends on: 5.1, 7.8
    - _Implements: R11.3, R11.4, R30.7, R30.8, P6, P7_

  - [x] 5.4 Multipath_Filter
    - Integrate residual-L2 filter into `tdoa_localize`: reject candidates whose residual L2 > `multipath_residual_threshold_s` (default 0.005)
    - Select smallest residual on multiple local minima; tie-break by distance to active sensor centroid
    - Signal `"multipath_rejected"` when no candidate qualifies; do not return `lat`/`lon`
    - Depends on: 5.2
    - _Implements: R12.1, R12.2, R12.3, R12.4_

  - [ ] 5.5 Multipath simulation test
    - `tests/test_multipath_filter.py` injecting reflected paths with delay 5–50 ms over 100 trials
    - Assert ≥90/100 trials reject the phantom and accept the direct-path solution
    - Depends on: 5.4
    - _Implements: R12.5_

  - [x] 5.6 Clock_Skew_Estimator
    - Create `backend/triangulation/clock_skew.py::estimate_offset(sensor_id, ntp_or_ptp_samples)` returning `{offset_us, offset_us_bound, last_measured_at_iso8601}`
    - Compute `offset_us_bound = max(2 * stdev(samples), measurement_floor_us)` so it dominates noise
    - In-memory cache per `sensor_id`
    - Depends on: 5.2
    - _Implements: R13.1_

  - [ ] 5.7 Clock skew enforcement in solver
    - Update `tdoa_localize` to subtract `offset_us` from each sensor's arrival time before residual computation
    - Signal `"clock_skew_exceeded"` when any participating sensor's `offset_us_bound` > its `SensorRecord.clock_offset_us_bound`
    - Identify the offending sensor in the failure payload
    - Depends on: 5.6
    - _Implements: R13.2, R13.3_

  - [ ] 5.8 `GET /sensors/clock-status` endpoint
    - Add the route to `backend/api/routes.py` returning per-sensor `{sensor_id, offset_us, offset_us_bound, last_measured_at_iso8601, clock_source}` within 1 s
    - Source values from the Clock_Skew_Estimator cache + active deployment metadata
    - Depends on: 5.6, 3.15
    - _Implements: R13.5_

  - [ ] 5.9 Clock skew bound property test
    - `tests/pbt/test_clock_skew_bound.py` Hypothesis: for σ_us < 100, returned `offset_us_bound ≥ 2 · σ_us`
    - Depends on: 5.6, 7.8
    - _Implements: R13.4, P15_

  - [ ] 5.10 Overdetermined_Solver with single-sensor-failure tolerance
    - Extend `backend/triangulation/tdoa_solver.py` from 3-sensor closed form to N-sensor least-squares over arrival residuals (`N ∈ [3, 16]`)
    - Drop sensors that miss `sensor_timeout_ms` (default 1000); signal `"insufficient_sensors"` when N=3 and one fails or N≥4 and ≥2 fail
    - Permutation-symmetric by construction (sum-of-squares is permutation-invariant)
    - Depends on: 5.4, 5.7
    - _Implements: R14.1, R14.2, R14.3, R14.4_

  - [ ] 5.11 TDOA noise-budget bound doc + property test
    - Author `docs/tdoa_error_bound.md` deriving the noise-budget bound as a function of geometry and σ
    - `tests/pbt/test_tdoa_noise_budget.py` Hypothesis: localization error ≤ documented bound under σ ∈ [0, 0.005] s, also satisfies R14.6 (≥95/100 within 30 m at σ=2 ms)
    - Depends on: 5.10, 7.8
    - _Implements: R14.6, R30.2, R31.1, P1_

  - [ ] 5.12 TDOA permutation symmetry property test (all N)
    - `tests/pbt/test_tdoa_permutation.py` Hypothesis over `N ∈ [3, 16]`: applying any permutation π to sensors and arrival times produces `(lat, lon)` within 1e-6 degrees of the unpermuted result
    - Depends on: 5.10, 7.8
    - _Implements: R14.5, R30.3, P2_

- [ ] 6. Parallel track — Frontend
  - [x] 6.1 Mapbox optional dependency + `VITE_MAP_PROVIDER`
    - Move `mapbox-gl` and `@mapbox/mapbox-gl-geocoder` from `dependencies` to `optionalDependencies` in `frontend/package.json`
    - Add `VITE_MAP_PROVIDER` (`"leaflet"` default, `"mapbox"`) and `VITE_MAPBOX_API_KEY` env handling
    - Document env vars in `frontend/README.md`
    - Depends on: 1.14
    - _Implements: R22.5_

  - [x] 6.2 MapAdapter component
    - Create `frontend/src/components/MapAdapter.jsx` selecting Leaflet or Mapbox at build time based on `VITE_MAP_PROVIDER` and key presence
    - Preserve all base-spec sensor and CrashEvent rendering behaviour; emit no console errors when Mapbox key missing
    - Both code paths share a single `<MapAdapter>` interface
    - Depends on: 6.1
    - _Implements: R22.1, R22.2, R22.3, R22.4_

  - [x] 6.3 `GET /events/{event_id}` endpoint
    - Add the route to `backend/api/routes.py` returning the persisted CrashEvent (or 404)
    - Backed by the `Event_Store` query path
    - Depends on: 3.15, 3.2
    - _Implements: R23.4_

  - [ ] 6.4 ReplayController.jsx
    - Create `frontend/src/components/ReplayController.jsx` fetching `GET /events/{event_id}` within 1 s
    - Re-run `<DroneTracker>` animation in replay mode with `REPLAY (<original_timestamp>)` non-modal banner; suppress `DRONE_ARRIVED` broadcast on completion
    - Render concurrent live events without interrupting the replay
    - Depends on: 6.2, 6.3
    - _Implements: R23.1, R23.2, R23.3, R23.5_

  - [ ] 6.5 AlertPanel filter + sort
    - Update `frontend/src/components/AlertPanel.jsx` with status multi-select (`active`/`resolved`), severity multi-select (`minor`/`moderate`/`severe`), and sort options (`newest_first`/`oldest_first`/`severity_high_to_low`/`severity_low_to_high`)
    - Local re-render < 100 ms; empty-state message + `Clear filters` action when no card matches
    - Status classification: `active` for `DETECTED`/`DRONE_DISPATCHED`, `resolved` for `DRONE_ARRIVED`/`DISPATCH_FAILED`
    - Depends on: 6.2, 4.3
    - _Implements: R24.1, R24.2, R24.3, R24.4, R24.5, R24.6_

  - [ ] 6.6 axe-core CI a11y test
    - Add `frontend/src/__tests__/a11y.spec.jsx` using `jest-axe`/`@axe-core/react` asserting zero `serious` or `critical` violations on every primary route
    - Wire into Vitest CI run; document failure triage in `docs/accessibility_audit.md`
    - Depends on: 6.5
    - _Implements: R25.1_

  - [x] 6.7 Reduced-motion + contrast/focus styles
    - Add `frontend/src/styles/reduced_motion.css` with `prefers-reduced-motion` media query collapsing drone-flight, alert-slide, and wavefront animations to ≤200 ms cross-fades
    - Bake focus-indicator (≥3:1) and body-text contrast (≥4.5:1) into Tailwind theme tokens
    - Depends on: 6.5
    - _Implements: R25.2, R25.4, R25.5_

  - [ ] 6.8 Keyboard map + accessibility audit doc
    - Author `docs/keyboard_map.md` enumerating every interactive element and its keyboard activation (Tab, Shift+Tab, Enter, Space, Escape, Arrow keys)
    - Author `docs/accessibility_audit.md` tracking manual screen-reader walkthroughs, cognitive-load review, expert WCAG audit as open work
    - Depends on: 6.7
    - _Implements: R25.3, R25.6_

  - [x] 6.9 Tailwind responsive breakpoints + 44 px hit targets
    - Update `frontend/tailwind.config.js` with breakpoints: `<768` mobile drawer (60% viewport when active), `768–1023` 70/30 split, `≥1024` 80/20 split
    - Add Tailwind plugin enforcing 44 × 44 px hit targets on touch viewports
    - Update `App.jsx` layout to honour all four breakpoints
    - Depends on: 6.5
    - _Implements: R26.2, R26.3, R26.4, R26.5_

  - [ ] 6.10 Visual-regression suite at 375/768/1024/1440
    - Add Playwright config + snapshots under `frontend/tests/visual/` for each viewport width
    - Wire into CI; assert no horizontal scroll at any width
    - Depends on: 6.9
    - _Implements: R26.1_

- [ ] 7. Parallel track — Testing infrastructure
  - [ ] 7.1 Vitest unit-test scaffold
    - Add `frontend/src/__tests__/{Map,AlertPanel,DroneTracker,AudioMonitor,useWebSocket}.test.{jsx,js}` mocking `fetch` and `WebSocket`
    - Configure `c8` thresholds: 70% line / 60% branch
    - Update `npm test` to run non-watch and exit non-zero on failure
    - Depends on: 6.5
    - _Implements: R27.1, R27.2, R27.3, R27.4_

  - [ ] 7.2 Load test — WebSocket fan-out
    - Create `tests/load/test_ws_concurrent.py` opening 100 concurrent clients and asserting every client receives every broadcast within 2 s, 0% drop
    - Emit `reports/load_ws_concurrent_<timestamp>.json` with throughput + p50/p99 latency + drop counts
    - Depends on: 3.8
    - _Implements: R28.1, R28.3_

  - [ ] 7.3 Load test — crash throughput
    - Create `tests/load/test_crash_throughput.py` submitting 1000 events within 60 s; assert 1000 distinct rows + ≥99% HTTP 200
    - Emit `reports/load_crash_throughput_<timestamp>.json` summary
    - Depends on: 3.4, 3.2
    - _Implements: R28.2, R28.3_

  - [ ] 7.4 `make load-test` target + perf baseline doc
    - Add `make load-test` Makefile target invoking both load tests on the reference hardware
    - Author `docs/perf_baseline.md` documenting the reference SKU (e.g., `g4dn.xlarge`), thermal envelope, and reproducibility notes
    - Depends on: 7.2, 7.3
    - _Implements: R28.4, R9.2_

  - [ ] 7.5 mutmut config (Python)
    - Add `mutmut_config.py` covering `backend/triangulation/tdoa_solver.py`, `backend/triangulation/atmospheric.py`, `backend/api/dedup.py`, `backend/api/rate_limit.py`, `backend/audio_model/onset.py`
    - Restrict to behaviour-mutations (skip docstrings/imports)
    - Depends on: 5.10, 5.1, 3.4, 3.10, 4.14
    - _Implements: R29.1_

  - [ ] 7.6 Stryker config (JS)
    - Add `frontend/stryker.conf.json` covering `frontend/src/components/AlertPanel.jsx`, `DroneTracker.jsx`, `frontend/src/hooks/useWebSocket.js`
    - Mocha/Vitest test runner integration
    - Depends on: 6.5, 7.1
    - _Implements: R29.2_

  - [ ] 7.7 `make mutation-test` target + threshold gate
    - Aggregate mutmut + Stryker into a single `make mutation-test` target with thresholds 70% Py / 60% JS
    - Emit per-module breakdown to `reports/mutation_<timestamp>.json`; exit non-zero on threshold miss
    - Depends on: 7.5, 7.6
    - _Implements: R29.3, R29.4_

  - [ ] 7.8 Hypothesis PBT harness (Python)
    - Create `tests/pbt/conftest.py` with shared strategies (sensor configs, arrival times, event ids, refractory ranges)
    - Set ≥100 iterations, deterministic seed for CI, minimal-counterexample shrinking on failure
    - Add `tests/pbt/__init__.py` + pytest markers
    - Depends on: 1.14
    - _Implements: R30.1, R30.14_

  - [ ] 7.9 fast-check PBT harness (JS)
    - Create `frontend/src/__tests__/pbt/setup.ts` with shared `fc` arbitraries for AlertPanel filter inputs and useWebSocket message sequences
    - ≥100 runs per property; fast-check shrinking enabled
    - Depends on: 6.5, 7.1
    - _Implements: R30.1, R30.14_

  - [ ] 7.10 Wire all 16 properties into `make pbt`
    - Add `make pbt` target running every Hypothesis test under `tests/pbt/` and every fast-check property under `frontend/src/__tests__/pbt/`
    - Aggregate exit code; print minimal counterexamples; emit `reports/pbt_<timestamp>.json` summary
    - Verify coverage of P1–P16 by parsing test markers
    - Depends on: 1.9, 2.2, 3.5, 3.7, 3.9, 3.12, 3.16, 4.7, 4.16, 5.3, 5.9, 5.11, 5.12, 7.8, 7.9
    - _Implements: R30.1, R30.14, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16_

- [ ] 8. Final integration
  - [ ] 8.1 End-to-end preservation gate on clean checkout
    - Run `git clone --depth 1 . /tmp/crashsense-fresh && cd /tmp/crashsense-fresh && make preservation-gate && make pbt && make load-test && make mutation-test`
    - Verify Phase 1 baseline JSON written, all 16 PBTs green, mutation thresholds met, load-test summaries emitted, axe-core a11y green
    - Document the procedure in `README.md` and add a CI job `clean-checkout-integration.yml` running on the `main` branch nightly
    - Depends on: 1.13, 2.6, 3.24, 4.7, 5.11, 6.10, 7.10
    - _Implements: R1, R2, R3, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16_

## Notes

- Tasks marked with `*` are optional test/property-test sub-tasks and may be skipped for a faster MVP. Core implementation sub-tasks are not marked optional.
- Checkpoint sub-tasks (1.14, 2.6, 3.24) are top-level epic checkpoints surfacing "Ensure all tests pass, ask the user if questions arise."; they are excluded from the dependency graph per workflow rules.
- Phase ordering: every Phase 2, Phase 3, and parallel-track task transitively depends on 1.14 (Phase 1 checkpoint), so the Phase 1 baseline lands before anything that needs to compare against it.
- Phase 3 production runtime ordering is enforced by the dependency chain `3.1 → 3.2 → 3.4 → 3.6 → 3.8`, then rate-limit/auth, metrics, logging, and registry layered on top.
- Each property test references its property number explicitly so the `make pbt` aggregator can grep for marker coverage of P1–P16.
- Parallel-track grouping (epics 4, 5, 6, 7) is advisory; the dependency graph is the authoritative scheduler input.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.4", "1.5", "1.7", "1.8", "1.10"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.6", "1.11"] },
    { "id": 2, "tasks": ["1.9", "1.12"] },
    { "id": 3, "tasks": ["1.13"] },
    { "id": 4, "tasks": ["2.1", "3.1", "3.13", "4.1", "4.4", "4.6", "4.8", "4.12", "4.14", "5.1", "6.1", "7.8"] },
    { "id": 5, "tasks": ["2.2", "2.3", "3.2", "3.14", "4.2", "4.5", "4.9", "4.13", "4.15", "5.2", "6.2"] },
    { "id": 6, "tasks": ["2.4", "3.3", "3.4", "3.15", "3.16", "4.3", "4.10", "4.11", "5.4", "5.6", "6.3"] },
    { "id": 7, "tasks": ["2.5", "3.5", "3.6", "4.7", "4.16", "5.3", "5.5", "5.7", "6.4", "6.5"] },
    { "id": 8, "tasks": ["3.7", "3.8", "5.8", "5.9", "5.10", "6.6", "6.7", "6.9", "7.1"] },
    { "id": 9, "tasks": ["3.9", "3.10", "3.17", "5.11", "5.12", "6.8", "6.10", "7.2", "7.6", "7.9"] },
    { "id": 10, "tasks": ["3.11", "3.12", "3.18", "7.3", "7.5"] },
    { "id": 11, "tasks": ["3.19", "3.21", "7.4", "7.7"] },
    { "id": 12, "tasks": ["3.20", "3.22", "3.23"] },
    { "id": 13, "tasks": ["7.10"] },
    { "id": 14, "tasks": ["8.1"] }
  ]
}
```
