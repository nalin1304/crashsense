# Requirements Document

## Introduction

CrashSense Hardening is a follow-on production-readiness effort that builds on the existing CrashSense system (see `.kiro/specs/crashsense/requirements.md`). The original system already delivers sub-meter TDOA accuracy under 2 ms Gaussian noise, a 97.31% audio classifier, a working WebSocket Dashboard, and a reproducible (seed 42) data pipeline. This spec preserves those capabilities and addresses the gaps that block field deployment: real-world audio evaluation, noise-robust training, severity grading, atmospheric and clock-skew compensation in TDOA, multi-event dispatch, durable persistence, rate limiting, model versioning, accessibility, mobile responsiveness, and a property-based test suite.

Requirements are grouped by subsystem and each requirement is tagged with a delivery phase so the work can be sequenced without re-planning:

- **[Phase 1]** Real-world test set — establish baseline measurement before any change
- **[Phase 2]** SNR-mixed training — robustness lift, evaluated against the Phase 1 baseline
- **[Phase 3]** Production runtime — rate limits, model versioning, structured logging with correlation IDs, latency metrics, concurrent-crash dispatcher
- **[Parallel]** Tracks that may proceed concurrently with phases 1–3: severity grading, atmospheric / multipath / clock-skew, frontend mobile and accessibility, testing infrastructure

This document specifies *what* the hardened system must do. Implementation choices live in `design.md`.

## Glossary

Terms inherited from the base spec retain their definitions. New or refined terms used in this document:

- **Real_World_Test_Set**: A held-out evaluation corpus of manually labeled dashcam audio clips, disjoint from any training or validation set.
- **SNR_Mixer**: The component that mixes clean crash audio with highway road-noise stems at controlled signal-to-noise ratios.
- **Severity_Classifier**: The component that assigns each detected crash a severity label of `minor`, `moderate`, or `severe`.
- **Calibrator**: The post-hoc probability calibration component (temperature scaling and isotonic regression variants) applied to Audio_Detector logits.
- **ECE**: Expected Calibration Error, computed with 15 equal-width probability bins.
- **Onset_Detector**: The time-frequency onset detection component that replaces the legacy 3-of-4 sliding-window vote debouncer.
- **Adversarial_Suite**: The labeled corpus of crash-adjacent non-crash sounds (vehicle horns, fireworks, tire blowouts, air brakes) used for negative-class robustness evaluation.
- **Latency_Recorder**: The instrumentation component that records per-stage processing time and exposes p50 and p99 metrics.
- **Sensor_Deployment_Schema**: The versioned schema describing real-deployment sensor configurations (location, altitude, clock source, microphone model).
- **Atmospheric_Corrector**: The component that adjusts the effective speed of sound for measured temperature, humidity, and wind.
- **Multipath_Filter**: The component that detects and rejects TDOA solutions consistent with reflected (phantom) acoustic paths.
- **Clock_Skew_Estimator**: The component that estimates and bounds per-sensor clock offset and drift using NTP or PTP measurements.
- **Overdetermined_Solver**: The TDOA solver that operates on four or more sensors using least-squares residual minimization and tolerates the failure of one sensor.
- **Event_Deduplicator**: The component that ensures any single physical crash produces at most one broadcast CrashEvent regardless of consensus re-fires or duplicate POSTs.
- **Per_Event_Dispatcher**: The dispatch state machine that allocates one drone slot per active CrashEvent without sharing global mutable dispatch state.
- **Backpressure_Controller**: The WebSocket_Manager subcomponent that enforces a per-client bounded outbound queue and drops slow clients without stalling fast clients.
- **Event_Store**: The durable storage backend for CrashEvent records, replacing the unhardened SQLite usage.
- **Rate_Limiter**: The middleware that enforces per-route burst and sustained request budgets.
- **Model_Registry**: The versioned checkpoint registry exposing an active-version pointer, hot-swap capability, and A/B routing.
- **Replay_Controller**: The frontend component that re-runs the dispatch animation for any past `event_id`.
- **Alert_Filter**: The frontend control that filters and sorts the Alert_Panel by status (active or resolved) and severity.
- **Responsive_Layout**: Dashboard layouts validated at 375 px, 768 px, 1024 px, and 1440 px viewport widths.
- **PBT_Suite**: The property-based test suite running against the components named in the Testing subsystem.
- **Correlation_ID**: A UUID v4 string attached to a single crash detection and propagated through every log line, metric, and downstream event for that detection.

## Requirements

Requirements are grouped into six subsystems (Audio Model, Triangulation, Backend / System, Frontend / UX, Testing, Cross-Cutting). Each requirement carries a delivery-phase tag in its title.

---

**Subsystem A: Audio Model**

### Requirement 1: [Audio] [Phase 1] Real-World Test Set

**User Story:** As a machine learning engineer, I want a held-out test set of manually labeled real dashcam audio, so that I can measure the Audio_Detector's true field performance and detect regressions when the model is changed.

#### Acceptance Criteria

1. THE CrashSense SHALL store the Real_World_Test_Set as WAV files under `data/real_world_test/crash/` and `data/real_world_test/noise/` with each clip having a duration between 3.0 seconds and 30.0 seconds inclusive.
2. THE Real_World_Test_Set SHALL contain at least 100 crash clips and at least 100 non-crash clips, each labeled by a human annotator and recorded with a label-provenance file at `data/real_world_test/labels.csv` containing the columns `filename`, `label`, `annotator_id`, `annotated_at_iso8601`, and `source_uri`.
3. THE Real_World_Test_Set SHALL share zero filenames and zero source URIs with the training set, the validation set, and the AudioSet-derived corpus referenced in the base spec.
4. THE CrashSense SHALL provide an evaluation script `backend/audio_model/evaluate_real_world.py` that loads the Real_World_Test_Set, runs the Audio_Detector active checkpoint, and writes a JSON report to `reports/real_world_eval_<git_sha>.json` containing `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `confusion_matrix`, `n_clips`, `checkpoint_id`, and `evaluated_at_iso8601`.
5. WHEN `evaluate_real_world.py` is invoked twice in succession on the same checkpoint and the same Real_World_Test_Set on the same host, THE CrashSense SHALL produce two reports whose `accuracy`, `precision`, `recall`, `f1`, and `confusion_matrix` fields are identical.
6. IF a clip listed in `labels.csv` is missing from disk, has a non-WAV extension, or has a duration outside the 3.0–30.0 second range, THEN THE evaluation script SHALL exclude that clip from the report, append a record to `reports/real_world_eval_<git_sha>.errors.csv` identifying the clip and reason, and continue processing the remaining clips.

### Requirement 2: [Audio] [Phase 2] SNR-Mixed Training

**User Story:** As a machine learning engineer, I want training data mixed with highway road noise at controlled signal-to-noise ratios, so that the Audio_Detector remains accurate in noisy real-world conditions.

#### Acceptance Criteria

1. THE SNR_Mixer SHALL be implemented at `backend/audio_model/noise_mixing.py` and SHALL expose a function `mix_clip(clean_signal, noise_signal, target_snr_db)` that returns a mixed signal of the same sample rate, channel count, and length as `clean_signal`.
2. WHEN `mix_clip` is called with a `target_snr_db` value in the set `{0, 10, 20}` and `clean_signal` and `noise_signal` are both non-silent, THE SNR_Mixer SHALL produce an output whose measured SNR (computed as 10·log10(power(clean) / power(scaled_noise))) is within ±0.5 dB of `target_snr_db`.
3. WHEN the Trainer is invoked with the flag `--snr-mix`, THE Trainer SHALL augment each crash training sample with a uniformly random highway-noise stem mixed at a uniformly random SNR drawn from the set `{0, 10, 20}` dB, applied with probability 0.5 per sample.
4. IF a noise stem fails to load or is shorter than the target clip duration after looping, THEN THE Trainer SHALL skip noise mixing for that sample, log the failure with the noise filename, and continue training without raising an error.
5. WHEN the SNR-mixed Trainer run completes, THE CrashSense SHALL report Real_World_Test_Set accuracy that meets or exceeds the Phase 1 baseline accuracy minus 1.0 percentage point, measured by the script defined in Requirement 1.4.
6. THE Trainer SHALL persist the SNR-mixed checkpoint to `backend/audio_model/checkpoints/<arch>_snr_<git_sha>.pth` and SHALL NOT overwrite the Phase 1 baseline checkpoint.

### Requirement 3: [Audio] [Parallel] Severity Grading

**User Story:** As an emergency dispatcher, I want each detected crash assigned a severity grade, so that I can prioritize response based on likely impact intensity.

#### Acceptance Criteria

1. THE Severity_Classifier SHALL be implemented at `backend/audio_model/severity.py` and SHALL expose a function `grade(audio_signal_or_path)` that returns a dictionary with keys `severity` (one of `"minor"`, `"moderate"`, `"severe"`) and `severity_confidence` (float in the inclusive range 0.0 to 1.0).
2. WHEN the Audio_Detector returns `event = "CRASH"` for a window, THE Audio_Detector SHALL invoke the Severity_Classifier on the same window before returning to the caller and SHALL include the `severity` and `severity_confidence` fields in the prediction dictionary defined in base-spec Requirement 4.6.
3. THE CrashEvent schema SHALL be extended to include the required field `severity` (string, one of `"minor"`, `"moderate"`, `"severe"`) and the required field `severity_confidence` (float, range 0.0 to 1.0 inclusive).
4. IF the Severity_Classifier raises an exception or returns a `severity` value outside the three permitted labels, THEN THE Audio_Detector SHALL set `severity = "moderate"`, set `severity_confidence = 0.0`, and log the failure at WARNING level with the Correlation_ID.
5. WHEN the Severity_Classifier is evaluated against the Real_World_Test_Set crash subset annotated with severity labels, THE Severity_Classifier SHALL achieve macro-averaged F1 of at least 0.60 across the three severity classes.

### Requirement 4: [Audio] [Parallel] Spatial Audio Features

**User Story:** As a machine learning engineer, I want a documented decision on spatial audio features, so that the team either ships binaural or array features with a measurable lift or records why the work is deferred.

#### Acceptance Criteria

1. THE CrashSense SHALL produce a decision document at `docs/spatial_audio_decision.md` containing the sections `Hypothesis`, `Experimental Setup`, `Results`, and `Decision (ship or defer)`.
2. WHERE the decision is to ship spatial features, THE Audio_Detector SHALL accept multi-channel input with channel count in the inclusive range 2 to 8, SHALL compute spatial features (interaural time difference and interaural level difference for binaural input, or beamformed features for array input), and SHALL achieve Real_World_Test_Set accuracy at least 1.0 percentage point above the SNR-mixed mono baseline.
3. WHERE the decision is to defer spatial features, THE decision document SHALL record the measured lift on a representative spatial test set, the cost estimate in engineer-weeks, and the conditions under which the work will be revisited.
4. WHEN the Audio_Detector is invoked with a number of input channels outside the supported range, THE Audio_Detector SHALL downmix the input to mono and SHALL log the downmix at INFO level with the Correlation_ID.

### Requirement 5: [Audio] [Parallel] Confidence Calibration

**User Story:** As a downstream consumer of crash confidence scores, I want calibrated probabilities, so that a confidence value of 0.9 corresponds to a 90% empirical accuracy.

#### Acceptance Criteria

1. THE Calibrator SHALL be implemented at `backend/audio_model/calibrate.py` and SHALL provide two calibration methods, `temperature_scaling` and `isotonic_regression`, exposed through the function `fit(method, validation_logits, validation_labels)` returning a calibration object with a `transform(logits)` method.
2. WHEN the Calibrator is fitted on the Phase 1 validation set and evaluated on the Real_World_Test_Set, THE selected calibration method SHALL achieve Expected Calibration Error of less than 0.01 (1.0 percent) computed with 15 equal-width probability bins.
3. THE CrashSense SHALL persist the fitted calibration object to `backend/audio_model/checkpoints/<arch>_calibrator_<git_sha>.json` with all parameters serialized in plain text and SHALL load that file at inference initialization.
4. WHEN the Audio_Detector returns a `confidence` field, THE Audio_Detector SHALL apply the active calibration object's `transform` before returning, so that the returned confidence is the calibrated probability.
5. IF the calibration object file is missing or fails to deserialize, THEN THE Audio_Detector SHALL fall back to uncalibrated softmax probabilities, set the prediction field `calibration_active` to `false`, and log the fallback at WARNING level once per process.
6. THE Calibrator SHALL provide a round-trip property: for every fitted calibrator C and every input logit vector x, `decode(encode(C)).transform(x)` SHALL produce values within 1e-6 of `C.transform(x)`, where `encode` and `decode` are the JSON serializer and parser.

### Requirement 6: [Audio] [Parallel] Long-Stretch Ambient Capture

**User Story:** As a machine learning engineer, I want statistics on long ambient highway recordings, so that training and evaluation reflect the actual distribution of background sound the deployed system will encounter.

#### Acceptance Criteria

1. THE CrashSense SHALL record at least 24 cumulative hours of ambient highway audio across at least 3 distinct recording sessions, stored under `data/ambient_long/` with one WAV file per session and a manifest at `data/ambient_long/manifest.json` listing each session's `filename`, `duration_seconds`, `sample_rate_hz`, `recorded_at_iso8601`, and `location_label`.
2. THE CrashSense SHALL provide an analysis script `backend/audio_model/ambient_stats.py` that computes per-session and aggregate statistics including mean RMS, A-weighted SPL estimate, spectral centroid distribution, and zero-crossing rate distribution, written to `reports/ambient_stats.json`.
3. WHEN the Trainer is invoked with the flag `--ambient-mix`, THE Trainer SHALL draw negative-class samples from the ambient recordings in addition to AudioSet-derived noise, with the per-batch ambient sampling probability in the inclusive range 0.2 to 0.5.
4. IF a session manifest entry references a missing or non-WAV file, THEN THE analysis script SHALL exclude that entry from the report, append the error to `reports/ambient_stats.errors.csv`, and continue processing the remaining sessions.

### Requirement 7: [Audio] [Parallel] Adversarial Robustness Suite

**User Story:** As a machine learning engineer, I want an adversarial test set of crash-adjacent sounds, so that the false-positive rate on horns, fireworks, tire blowouts, and air brakes is bounded.

#### Acceptance Criteria

1. THE Adversarial_Suite SHALL be stored under `data/adversarial/` with subdirectories `horns/`, `fireworks/`, `tire_blowouts/`, and `airbrakes/`, each containing at least 50 WAV clips of duration 3.0 to 30.0 seconds, all labeled non-crash.
2. THE CrashSense SHALL provide an evaluation script `backend/audio_model/evaluate_adversarial.py` that runs the Audio_Detector active checkpoint on the Adversarial_Suite and writes a per-category false-positive rate report to `reports/adversarial_<git_sha>.json`.
3. WHEN `evaluate_adversarial.py` is run on the active checkpoint, THE Audio_Detector SHALL produce a per-category false-positive rate of less than 0.10 (10 percent) for each of the four categories.
4. IF any per-category false-positive rate exceeds 0.10, THEN the evaluation script SHALL exit with a non-zero status code and SHALL print the offending categories and their false-positive rates to standard error.

### Requirement 8: [Audio] [Parallel] Time-Frequency Onset Detector

**User Story:** As a system integrator, I want crash detection triggered by a time-frequency onset detector instead of the legacy 3-of-4 sliding-window vote, so that the trigger latency is bounded and refractory behavior is well-defined.

#### Acceptance Criteria

1. THE Onset_Detector SHALL be implemented at `backend/audio_model/onset.py` and SHALL replace the 3-of-4 window-vote debouncer in `backend/audio_model/inference.py`.
2. WHEN the Audio_Detector processes a continuous stream and the Onset_Detector identifies an onset at time t with peak score above the configured threshold, THE Audio_Detector SHALL emit at most one CRASH prediction associated with that onset and SHALL suppress further CRASH predictions for a refractory window of 500 milliseconds following t.
3. THE Onset_Detector SHALL expose a `refractory_ms` parameter (integer, range 100 to 2000 inclusive) and a `threshold` parameter (float, range 0.0 to 1.0 inclusive).
4. WHEN the Onset_Detector receives two true crash impacts separated by 200 milliseconds in the same audio stream, THE Onset_Detector SHALL emit either one CRASH prediction (if 200 ms is less than `refractory_ms`) or two CRASH predictions in chronological order (if 200 ms is greater than or equal to `refractory_ms`), with the chosen behavior documented in the design and consistent across runs.
5. THE Onset_Detector SHALL satisfy a refractory monotonicity property: for any input stream S and any two refractory values r1 < r2, the count of emitted predictions on S with refractory r1 SHALL be greater than or equal to the count of emitted predictions on S with refractory r2.

### Requirement 9: [Audio] [Phase 3] Inference Latency

**User Story:** As a system operator, I want measured and bounded inference latency, so that downstream dispatch happens within the operational deadline.

#### Acceptance Criteria

1. THE Latency_Recorder SHALL record per-prediction wall-clock time from audio-window-available to prediction-returned and SHALL expose `audio_inference_latency_ms_p50` and `audio_inference_latency_ms_p99` metrics through the Prometheus metrics endpoint defined in Requirement 21.
2. WHEN the Audio_Detector is benchmarked on the Real_World_Test_Set on the reference hardware documented in `docs/perf_baseline.md`, THE Audio_Detector SHALL produce a p50 latency of at most 100 milliseconds and a p99 latency of at most 250 milliseconds.
3. THE Audio_Detector SHALL accept a `deadline_ms` parameter on `predict` (integer, range 50 to 5000 inclusive, default 500) and SHALL return a prediction dictionary with field `deadline_exceeded = true` and `event = "DEADLINE"` when inference cannot complete within `deadline_ms`.
4. WHEN two crash impacts arrive 200 milliseconds apart in the same stream, THE Audio_Detector SHALL produce defined behavior consistent with Requirement 8.4 and SHALL NOT drop or merge predictions silently.
5. IF the Latency_Recorder's exponential moving average of p99 latency exceeds 250 milliseconds for 60 consecutive seconds, THEN THE Backend_API SHALL emit a `latency_degraded` log event at WARNING level with the current p50 and p99 values.

---

**Subsystem B: Triangulation**

### Requirement 10: [Triangulation] [Phase 3] Sensor Deployment Schema and Mode Separation

**User Story:** As a deployment engineer, I want a versioned sensor-configuration schema with a clear demo/production split, so that the same code runs both the seed-42 demo and a real field deployment without ambiguity.

#### Acceptance Criteria

1. THE Sensor_Deployment_Schema SHALL be defined at `backend/triangulation/schema.py` as a Pydantic model named `SensorDeployment` with fields `schema_version` (string, semver format), `mode` (string, one of `"demo"` or `"production"`), `sensors` (list of `SensorRecord` items, length in the inclusive range 3 to 16), and `metadata` (object with `name`, `created_at_iso8601`, `notes`).
2. THE `SensorRecord` model SHALL include the fields `sensor_id` (string, 1 to 32 characters, matching the regex `^[A-Za-z0-9_-]+$`), `display_name` (string, 1 to 64 characters), `lat` (float, range -90.0 to 90.0), `lon` (float, range -180.0 to 180.0), `altitude_m` (float, range -500.0 to 5000.0), `is_toll_plaza` (boolean), `clock_source` (string, one of `"NTP"`, `"PTP"`, `"GPS"`, `"NONE"`), `clock_offset_us_bound` (float, range 0.0 to 1000000.0), and `microphone_model` (string, 1 to 64 characters).
3. THE CrashSense SHALL provide a parser `load_deployment(path)` and a pretty printer `dump_deployment(deployment, path)` for the Sensor_Deployment_Schema, with both YAML and JSON serializations supported.
4. FOR every valid `SensorDeployment` instance D, parsing the output of `dump_deployment(D, ...)` SHALL produce an instance equal to D under field-wise comparison (round-trip property).
5. IF the loaded deployment has `mode = "production"` and any sensor has `clock_source = "NONE"`, THEN THE parser SHALL raise a validation error indicating that production mode requires a synchronized clock source on every sensor.
6. WHEN the Backend_API starts, THE Backend_API SHALL load the deployment file referenced by the environment variable `CRASHSENSE_DEPLOYMENT_PATH`, falling back to the bundled demo deployment at `backend/triangulation/deployments/demo_3sensor.yaml` if the variable is unset.
7. WHEN the Backend_API exposes `GET /sensors`, THE Backend_API SHALL return the active deployment's sensors with the additional field `mode` set to the deployment's mode value, preserving backward compatibility with the base spec's response fields.

### Requirement 11: [Triangulation] [Parallel] Atmospheric Correction

**User Story:** As a deployment engineer, I want the speed of sound corrected for measured atmospheric conditions, so that TDOA localization remains accurate across temperature, humidity, and wind variation.

#### Acceptance Criteria

1. THE Atmospheric_Corrector SHALL be implemented at `backend/triangulation/atmospheric.py` and SHALL expose `effective_speed_of_sound(temperature_c, humidity_pct, wind_vector_mps, propagation_unit_vector)` returning a positive float in meters per second.
2. THE Atmospheric_Corrector SHALL accept `temperature_c` in the inclusive range -40.0 to 60.0, `humidity_pct` in the inclusive range 0.0 to 100.0, `wind_vector_mps` as a 3-tuple of floats with each component in the inclusive range -50.0 to 50.0, and `propagation_unit_vector` as a 3-tuple of floats with L2 norm in the inclusive range 0.999 to 1.001.
3. THE Atmospheric_Corrector SHALL satisfy a temperature monotonicity property: WHEN `humidity_pct` and `wind_vector_mps` are held fixed and `temperature_c` increases, THE returned effective speed of sound SHALL strictly increase.
4. THE Atmospheric_Corrector SHALL satisfy a wind sign-correctness property: WHEN `propagation_unit_vector` and `wind_vector_mps` have positive dot product, THE returned effective speed of sound SHALL be greater than the still-air speed at the same temperature and humidity, and WHEN the dot product is negative, THE returned effective speed SHALL be less than the still-air speed.
5. WHEN `tdoa_localize` is invoked with an active deployment in `production` mode, THE Triangulation_Solver SHALL use the Atmospheric_Corrector's effective speed of sound in place of the constant `SPEED_OF_SOUND`.
6. IF measured atmospheric inputs are unavailable, THEN THE Triangulation_Solver SHALL use `temperature_c = 20.0`, `humidity_pct = 50.0`, `wind_vector_mps = (0.0, 0.0, 0.0)`, and SHALL log the substitution at INFO level with the Correlation_ID once per CrashEvent.

### Requirement 12: [Triangulation] [Parallel] Multipath Rejection

**User Story:** As a deployment engineer, I want the Triangulation_Solver to detect and reject phantom solutions caused by acoustic reflections, so that the dispatched drone is not sent to a non-existent crash.

#### Acceptance Criteria

1. THE Multipath_Filter SHALL be integrated into `tdoa_localize` and SHALL evaluate every candidate solution returned by the optimizer.
2. WHEN `tdoa_localize` is invoked, THE Multipath_Filter SHALL compute the residual vector between observed time delays and time delays predicted by the candidate solution, and SHALL reject the candidate if the residual L2 norm exceeds a configured threshold `multipath_residual_threshold_s` (float, range 0.0001 to 0.1 inclusive, default 0.005).
3. WHEN the optimizer returns multiple local minima, THE Multipath_Filter SHALL select the candidate with the smallest residual L2 norm, with ties broken by selecting the candidate closest to the centroid of the active sensor set.
4. IF no candidate satisfies the residual threshold, THEN THE Triangulation_Solver SHALL signal localization failure with reason `"multipath_rejected"` and SHALL NOT return `lat` or `lon`.
5. WHEN simulated reflected paths with reflection delay in the inclusive range 5 ms to 50 ms are injected into 100 trials, THE Multipath_Filter SHALL reject the phantom solution and accept the direct-path solution in at least 90 of the 100 trials.

### Requirement 13: [Triangulation] [Parallel] Clock Skew Handling

**User Story:** As a deployment engineer, I want per-sensor clock offsets estimated and bounded, so that TDOA solutions account for synchronization error rather than treating sensor timestamps as ground truth.

#### Acceptance Criteria

1. THE Clock_Skew_Estimator SHALL be implemented at `backend/triangulation/clock_skew.py` and SHALL expose `estimate_offset(sensor_id, ntp_or_ptp_samples)` returning a dictionary with keys `offset_us` (float), `offset_us_bound` (float, non-negative), and `last_measured_at_iso8601` (string).
2. WHEN `tdoa_localize` is invoked with an active deployment, THE Triangulation_Solver SHALL subtract each sensor's `offset_us` from its observed arrival time before computing residuals.
3. IF any participating sensor has `offset_us_bound` greater than `clock_offset_us_bound` from its `SensorRecord`, THEN THE Triangulation_Solver SHALL signal localization failure with reason `"clock_skew_exceeded"`, identify the offending sensor, and SHALL NOT return `lat` or `lon`.
4. THE Clock_Skew_Estimator SHALL satisfy the property: WHEN `ntp_or_ptp_samples` are perturbed by zero-mean Gaussian noise with standard deviation σ_us less than 100, THE returned `offset_us_bound` SHALL be greater than or equal to 2·σ_us.
5. THE Backend_API SHALL expose `GET /sensors/clock-status` returning, within 1 second, a JSON array with one entry per sensor containing `sensor_id`, `offset_us`, `offset_us_bound`, `last_measured_at_iso8601`, and `clock_source`.

### Requirement 14: [Triangulation] [Parallel] Overdetermined Solver With Single-Sensor Failure Tolerance

**User Story:** As a deployment engineer, I want the Triangulation_Solver to accept four or more sensors and tolerate one sensor failure, so that field deployments remain available when a single sensor goes offline.

#### Acceptance Criteria

1. THE Overdetermined_Solver SHALL accept any sensor count in the inclusive range 3 to 16 and SHALL minimize a least-squares residual over all available sensors.
2. WHEN the active deployment contains 4 or more sensors and exactly one sensor has not reported an arrival time within `sensor_timeout_ms` (integer, range 100 to 5000 inclusive, default 1000), THE Overdetermined_Solver SHALL exclude that sensor and solve with the remaining sensors.
3. WHEN the active deployment contains 3 sensors and any sensor fails to report within `sensor_timeout_ms`, THE Overdetermined_Solver SHALL signal localization failure with reason `"insufficient_sensors"`.
4. WHEN the active deployment contains 4 or more sensors and 2 or more sensors fail to report within `sensor_timeout_ms`, THE Overdetermined_Solver SHALL signal localization failure with reason `"insufficient_sensors"`.
5. THE Overdetermined_Solver SHALL satisfy a permutation symmetry property: for any sensor permutation π applied to the input arrival times and the input sensor list, the returned `lat` and `lon` SHALL match the unpermuted result within 1e-6 degrees.
6. THE Overdetermined_Solver SHALL satisfy a noise-budget property: for arrival times perturbed by independent Gaussian noise with standard deviation 2 milliseconds, the localization error SHALL be less than or equal to 30 meters in at least 95 of 100 trials, preserving the base-spec accuracy contract.

---

**Subsystem C: Backend / System**

### Requirement 15: [Backend] [Phase 3] Event Deduplication

**User Story:** As a backend engineer, I want every physical crash to produce at most one broadcast CrashEvent, so that consensus re-fires and duplicate POSTs do not result in duplicate dispatches.

#### Acceptance Criteria

1. THE Event_Deduplicator SHALL be implemented at `backend/api/dedup.py` and SHALL key on `event_id` (UUID v4, 36 characters).
2. WHEN a CrashEvent with `event_id` E is submitted to the Backend_API and no event with `event_id` E has been recorded in the dedup store within the configured window `dedup_window_seconds` (integer, range 1 to 3600 inclusive, default 300), THE Backend_API SHALL persist the event, broadcast it to WebSocket clients, and record E in the dedup store.
3. WHEN a CrashEvent with `event_id` E is submitted and an event with `event_id` E has already been recorded within the dedup window, THE Backend_API SHALL return HTTP 200 with a body indicating `deduplicated = true`, SHALL NOT persist the event a second time, and SHALL NOT broadcast the event a second time.
4. THE Event_Deduplicator SHALL satisfy an idempotency property: for any sequence of N submissions with the same `event_id`, the count of broadcast messages with that `event_id` SHALL equal exactly 1.
5. IF a CrashEvent is submitted without an `event_id` field, THEN THE Backend_API SHALL reject the request with HTTP 400 and a body identifying the missing field.

### Requirement 16: [Backend] [Phase 3] Per-Event Drone Dispatch

**User Story:** As a backend engineer, I want each CrashEvent allocated its own drone slot, so that two crashes never share or hijack a single global drone state.

#### Acceptance Criteria

1. THE Per_Event_Dispatcher SHALL maintain a map keyed by `event_id` of dispatch state machines, with a configured maximum concurrent capacity `max_concurrent_dispatches` (integer, range 1 to 100 inclusive, default 8).
2. WHEN a new CrashEvent with `event_id` E is broadcast and the count of active dispatches is below `max_concurrent_dispatches`, THE Per_Event_Dispatcher SHALL allocate a dedicated drone slot for E with origin equal to the nearest Toll_Plaza and SHALL NOT mutate the dispatch state of any other active event.
3. WHEN the count of active dispatches has reached `max_concurrent_dispatches`, THE Per_Event_Dispatcher SHALL queue the new CrashEvent in FIFO order and SHALL begin its dispatch when an existing dispatch transitions to `DRONE_ARRIVED`.
4. THE Per_Event_Dispatcher SHALL satisfy a slot-exclusivity property: at any instant, no two active dispatches SHALL share the same drone slot identifier.
5. WHEN a dispatch reaches `DRONE_ARRIVED` for `event_id` E, THE Per_Event_Dispatcher SHALL release E's slot, broadcast the `DRONE_ARRIVED` update, and SHALL NOT affect any other active dispatch.
6. IF a dispatch fails (drone error, animation timeout, or origin unresolvable), THEN THE Per_Event_Dispatcher SHALL transition that event to a `DISPATCH_FAILED` status, release the slot, and broadcast a status update without altering other active dispatches.

### Requirement 17: [Backend] [Phase 3] WebSocket Backpressure

**User Story:** As a backend engineer, I want per-client bounded outbound queues, so that one slow Dashboard cannot stall the whole broadcast loop.

#### Acceptance Criteria

1. THE Backpressure_Controller SHALL maintain a per-client outbound queue with maximum depth `client_queue_max` (integer, range 16 to 4096 inclusive, default 256) measured in messages.
2. WHEN a broadcast is issued, THE Backpressure_Controller SHALL enqueue the message on each client's queue and SHALL deliver it through that client's dedicated send loop.
3. WHEN a client's outbound queue is at `client_queue_max` and a new message arrives for that client, THE Backpressure_Controller SHALL drop the oldest queued message for that client, increment the metric `ws_messages_dropped_total{client_id}` by 1, and continue serving other clients without blocking.
4. WHEN a client's send loop fails to drain a single message within `slow_client_timeout_ms` (integer, range 1000 to 60000 inclusive, default 5000), THE Backpressure_Controller SHALL close that client's connection, remove it from the active connection collection, and increment `ws_clients_dropped_total` by 1.
5. THE Backpressure_Controller SHALL satisfy a fast-client-non-stalling property: for any mix of fast and slow clients, the wall-clock delivery latency to a fast client SHALL be bounded by the broadcast cost plus the fast client's own send time, independent of any slow client's behavior.

### Requirement 18: [Backend] [Phase 3] SQLite Hardening and Event Store

**User Story:** As a backend engineer, I want durable, low-contention persistence for CrashEvents, so that the system survives restarts and concurrent writes without data loss.

#### Acceptance Criteria

1. THE Event_Store SHALL be implemented at `backend/api/store.py` and SHALL persist CrashEvents using SQLite with WAL journaling (`PRAGMA journal_mode = WAL`) and `synchronous = NORMAL`.
2. THE Event_Store SHALL batch writes using a transaction window of up to `write_batch_size` (integer, range 1 to 1000 inclusive, default 50) events or `write_batch_window_ms` (integer, range 1 to 1000 inclusive, default 100) milliseconds, whichever is reached first.
3. WHEN the Backend_API restarts, THE Event_Store SHALL recover all CrashEvents committed before the restart and SHALL make them queryable through `GET /events`.
4. THE Event_Store SHALL expose a configuration flag `CRASHSENSE_EVENT_STORE_BACKEND` accepting values `"sqlite"` (default) and `"external"`, where `"external"` routes writes to a pluggable backend via the interface defined in `backend/api/store.py::EventStoreBackend`.
5. WHEN 1000 CrashEvents are written concurrently from 10 producer tasks, THE Event_Store SHALL persist exactly 1000 distinct rows by `event_id`, SHALL preserve insertion order within each producer, and SHALL complete within 5 seconds on the reference hardware.
6. IF a write fails due to a transient SQLite `BUSY` or `LOCKED` error, THEN THE Event_Store SHALL retry up to 5 times with exponential backoff starting at 10 ms, and on final failure SHALL emit a structured error log and propagate the error to the caller.

### Requirement 19: [Backend] [Phase 3] Rate Limiting and Authentication

**User Story:** As a backend engineer, I want rate limits and stronger authentication on `/simulate-crash` and `/detect-audio`, so that the system cannot be abused or brought down by an unauthenticated client.

#### Acceptance Criteria

1. THE Rate_Limiter SHALL enforce per-route token-bucket limits configured by `rate_limit_burst` (integer, range 1 to 1000 inclusive) and `rate_limit_sustained_per_minute` (integer, range 1 to 100000 inclusive).
2. WHEN a client exceeds the configured burst or sustained budget on `/simulate-crash` or `/detect-audio`, THE Backend_API SHALL respond with HTTP 429 and a `Retry-After` header indicating the seconds until the next allowed request.
3. THE Rate_Limiter SHALL satisfy a budget-bound property: across any rolling 60-second window, the count of accepted requests for a single client on a rate-limited route SHALL be less than or equal to `rate_limit_burst + rate_limit_sustained_per_minute`.
4. THE Backend_API SHALL require requests to `/simulate-crash` and `/detect-audio` to carry an `Authorization: Bearer <token>` header where `<token>` is validated against the configured authentication backend.
5. IF a request to `/simulate-crash` or `/detect-audio` is missing the `Authorization` header or carries an invalid token, THEN THE Backend_API SHALL respond with HTTP 401 and SHALL NOT execute the pipeline.
6. THE Backend_API SHALL log every 401 and 429 response with the route, the client identifier, and the reason at WARNING level with a Correlation_ID.

### Requirement 20: [Backend] [Phase 3] Model Versioning and Hot-Swap

**User Story:** As a machine learning engineer, I want versioned checkpoints with an active-version pointer, hot-swap, and A/B routing, so that models can be promoted or rolled back without restarting the Backend_API.

#### Acceptance Criteria

1. THE Model_Registry SHALL be implemented at `backend/audio_model/registry.py` and SHALL store checkpoints under `backend/audio_model/checkpoints/` with filenames matching `^[a-z0-9_]+_v\\d+_[a-z0-9]+\\.pth$` (architecture, version, short hash).
2. THE Model_Registry SHALL maintain an active-version pointer at `backend/audio_model/checkpoints/ACTIVE.json` containing fields `primary_checkpoint_id` (string), `secondary_checkpoint_id` (string or null), `ab_split_percent` (integer, range 0 to 100 inclusive, default 0), and `updated_at_iso8601`.
3. WHEN the Audio_Detector is invoked, THE Audio_Detector SHALL select the primary checkpoint with probability `(100 - ab_split_percent) / 100` and the secondary checkpoint with probability `ab_split_percent / 100`, recording the selected checkpoint id in the prediction dictionary as `checkpoint_id`.
4. WHEN the active-version pointer file is updated on disk, THE Audio_Detector SHALL hot-swap to the new pointer within 30 seconds and SHALL NOT drop in-flight predictions.
5. IF the active-version pointer references a missing or unloadable checkpoint, THEN THE Audio_Detector SHALL retain the previously loaded checkpoint, log the failure at ERROR level, and increment the metric `model_swap_failures_total` by 1.
6. THE Backend_API SHALL expose `GET /model/active` returning, within 1 second, the contents of `ACTIVE.json` plus the loaded checkpoint's `accuracy` from the most recent evaluation report.

### Requirement 21: [Backend] [Phase 3] Structured Logging With Correlation IDs and Metrics

**User Story:** As a system operator, I want structured logs with correlation IDs and Prometheus metrics, so that I can trace any single crash detection from audio through dispatch and observe system health.

#### Acceptance Criteria

1. THE Backend_API SHALL emit logs in JSON Lines format to standard output, with each log line containing the fields `timestamp_iso8601`, `level`, `logger`, `message`, `correlation_id`, and `event_id` where applicable.
2. WHEN a crash detection begins, THE Backend_API SHALL generate a Correlation_ID (UUID v4) and SHALL propagate it through every subsequent log line, every metric label, and every CrashEvent broadcast for that detection.
3. THE Backend_API SHALL expose a Prometheus-compatible `GET /metrics` endpoint on port 8000 returning, within 1 second, at minimum the metrics `audio_inference_latency_ms` (histogram), `tdoa_localize_latency_ms` (histogram), `ws_messages_dropped_total` (counter), `ws_clients_dropped_total` (counter), `model_swap_failures_total` (counter), `crash_events_total` (counter), and `crash_events_deduplicated_total` (counter).
4. THE Backend_API SHALL satisfy a correlation-id propagation property: for any single CrashEvent, all log lines emitted in handling that event SHALL share the same `correlation_id` value.
5. IF a request arrives with an existing `X-Correlation-ID` header containing a valid UUID v4, THEN THE Backend_API SHALL adopt that value as the Correlation_ID instead of generating a new one.

---

**Subsystem D: Frontend / UX**

### Requirement 22: [Frontend] [Parallel] Mapbox Removal or Feature Flag

**User Story:** As a frontend developer, I want Mapbox removed or hidden behind a feature flag with Leaflet as the primary map, so that the Dashboard does not depend on a third-party API key in production.

#### Acceptance Criteria

1. THE Dashboard SHALL render the Map_View using Leaflet over OpenStreetMap tiles by default, with all sensor and CrashEvent rendering behavior from base-spec Requirements 14 and 15 preserved.
2. THE Dashboard SHALL expose a build-time feature flag `VITE_MAP_PROVIDER` accepting values `"leaflet"` (default) and `"mapbox"`.
3. WHEN `VITE_MAP_PROVIDER` is set to `"mapbox"` at build time and a valid Mapbox API key is provided through `VITE_MAPBOX_API_KEY`, THE Dashboard SHALL render the Map_View using Mapbox GL JS, preserving all sensor and CrashEvent rendering behavior.
4. WHEN `VITE_MAP_PROVIDER` is unset, set to `"leaflet"`, or set to `"mapbox"` without a valid API key, THE Dashboard SHALL render the Map_View using Leaflet without emitting any console errors related to a missing Mapbox key.
5. THE `frontend/package.json` SHALL declare `mapbox-gl` and `@mapbox/mapbox-gl-geocoder` under `optionalDependencies` rather than `dependencies`, so that a default install (Leaflet path) does not require either package.

### Requirement 23: [Frontend] [Parallel] Replay Past Events

**User Story:** As an operator, I want to replay the dispatch animation for any past CrashEvent, so that I can review historical incidents during after-action analysis.

#### Acceptance Criteria

1. THE Replay_Controller SHALL be implemented in `frontend/src/components/ReplayController.jsx` and SHALL accept any `event_id` resolvable through `GET /events/{event_id}`.
2. WHEN the operator selects a past CrashEvent and activates the `Replay` action, THE Replay_Controller SHALL fetch that event from `GET /events/{event_id}` within 1 second and SHALL re-run the full Drone_Tracker animation from the original `drone_origin` to the original crash coordinate using the original `event_id` and timestamp.
3. WHILE a replay animation is active, THE Replay_Controller SHALL display a non-modal banner with text `REPLAY (<original_timestamp>)` and SHALL NOT broadcast a `DRONE_ARRIVED` update to the WebSocket channel.
4. WHEN the operator activates `Replay` for an `event_id` that does not exist, THE Backend_API SHALL respond with HTTP 404 and the Dashboard SHALL display a non-blocking error indicator stating that the event was not found.
5. WHILE a live CrashEvent arrives during an active replay, THE Dashboard SHALL render the live event on the Map_View and the Alert_Panel without interrupting the replay animation.

### Requirement 24: [Frontend] [Parallel] Alert Panel Filter and Sort

**User Story:** As an operator, I want to filter and sort the Alert_Panel by status and severity, so that I can focus on what needs attention.

#### Acceptance Criteria

1. THE Alert_Filter SHALL be rendered at the top of the Alert_Panel and SHALL expose a status filter (multi-select among `active`, `resolved`) and a severity filter (multi-select among `minor`, `moderate`, `severe`).
2. WHEN the operator changes any filter selection, THE Alert_Panel SHALL re-render the visible cards within 100 milliseconds, showing only cards matching the active filter combination.
3. THE Alert_Filter SHALL expose a sort control with options `newest_first` (default), `oldest_first`, `severity_high_to_low`, and `severity_low_to_high`.
4. WHEN the operator changes the sort selection, THE Alert_Panel SHALL re-order the visible cards within 100 milliseconds without reloading data from the Backend_API.
5. WHILE no card matches the active filter combination, THE Alert_Panel SHALL display an empty-state message identifying the active filters and offering a one-click `Clear filters` action.
6. THE Alert_Panel SHALL classify a CrashEvent as `active` while its status is one of `DETECTED`, `DRONE_DISPATCHED` and as `resolved` while its status is `DRONE_ARRIVED` or `DISPATCH_FAILED`.

### Requirement 25: [Frontend] [Parallel] Accessibility Audit

**User Story:** As an operator using assistive technology, I want the Dashboard to follow WCAG 2.1 AA guidelines, so that the system is usable by people with disabilities. (Note: full WCAG conformance requires manual testing with assistive technologies and an accessibility expert review; this requirement specifies what can be verified in code and automated tooling.)

#### Acceptance Criteria

1. THE Dashboard SHALL achieve zero `axe-core` violations at impact level `serious` or `critical` on every primary route, verified by an automated test in the CI pipeline.
2. THE Dashboard SHALL provide a visible keyboard focus indicator with a contrast ratio of at least 3:1 against adjacent colors on every interactive element.
3. WHEN the operator navigates the Dashboard using only the keyboard (Tab, Shift+Tab, Enter, Space, Escape, Arrow keys), THE Dashboard SHALL allow reaching and activating every interactive element listed in `docs/keyboard_map.md` without a pointer device.
4. THE Dashboard SHALL render all body text with a contrast ratio of at least 4.5:1 against its background and all non-text indicators with a contrast ratio of at least 3:1.
5. WHEN `prefers-reduced-motion` is set in the operator's user agent, THE Dashboard SHALL replace the drone-flight animation, alert-panel slide, and wavefront expansion with cross-fade transitions of 200 milliseconds or less.
6. THE CrashSense SHALL document in `docs/accessibility_audit.md` the manual-testing items that automated tooling cannot verify (screen-reader walkthroughs, cognitive-load review, expert WCAG audit) and SHALL track those items as open work.

### Requirement 26: [Frontend] [Parallel] Responsive Layouts

**User Story:** As an operator, I want the Dashboard usable on phones, tablets, laptops, and large desktops, so that the system can be monitored in the field.

#### Acceptance Criteria

1. THE Dashboard SHALL render without horizontal scroll at viewport widths of 375 px, 768 px, 1024 px, and 1440 px, verified by an automated visual-regression test for each width.
2. WHILE the viewport width is below 768 px, THE Dashboard SHALL render the Map_View at 100 percent width and the Alert_Panel as a bottom drawer that opens to occupy 60 percent (±2 percent) of the viewport height when a CrashEvent is active.
3. WHILE the viewport width is in the inclusive range 768 px to 1023 px, THE Dashboard SHALL render the Map_View at 70 percent (±2 percent) width and the Alert_Panel at 30 percent (±2 percent) width.
4. WHILE the viewport width is 1024 px or greater, THE Dashboard SHALL render the layout specified in base-spec Requirement 13 (Map_View 80 percent, Alert_Panel 20 percent on event).
5. THE Dashboard SHALL render every interactive element with a hit target of at least 44 px by 44 px on viewports below 768 px width.

---

**Subsystem E: Testing**

### Requirement 27: [Testing] [Parallel] Unit Tests for React Components

**User Story:** As a frontend developer, I want unit tests for the core React components, so that regressions are caught before merge.

#### Acceptance Criteria

1. THE CrashSense SHALL provide Vitest test files under `frontend/src/__tests__/` covering the components `Map.jsx`, `AlertPanel.jsx`, `DroneTracker.jsx`, `AudioMonitor.jsx`, and the hook `useWebSocket.js`.
2. THE Vitest suite SHALL achieve at least 70 percent line coverage and at least 60 percent branch coverage across the files listed in E1.1, measured by `c8` or `istanbul`.
3. WHEN `npm test` is run in the `frontend/` directory, THE Vitest suite SHALL execute in non-watch mode and SHALL exit with a non-zero status code on any failing test.
4. THE Vitest suite SHALL mock all `fetch` and `WebSocket` interactions and SHALL NOT make outbound network calls during test execution.

### Requirement 28: [Testing] [Parallel] Load Tests

**User Story:** As a backend engineer, I want load tests for WebSocket fan-out and crash throughput, so that capacity is measured and regressions are detected.

#### Acceptance Criteria

1. THE CrashSense SHALL provide a load test at `tests/load/test_ws_concurrent.py` that opens 100 concurrent WebSocket clients and asserts that every client receives every broadcast CrashEvent within 2 seconds, with a tolerated drop rate of 0 percent.
2. THE CrashSense SHALL provide a load test at `tests/load/test_crash_throughput.py` that submits 1000 CrashEvents within 60 seconds and asserts that the Event_Store persists 1000 distinct rows by `event_id` and the Backend_API returns HTTP 200 for at least 99 percent of submissions.
3. WHEN a load test fails its assertions, THE test SHALL exit with a non-zero status code and SHALL emit a JSON summary at `reports/load_<test_name>_<timestamp>.json` containing observed throughput, p50 and p99 latency, and the count of dropped or rejected events.
4. THE load tests SHALL run on the reference hardware documented in `docs/perf_baseline.md` and SHALL be invocable through `make load-test`.

### Requirement 29: [Testing] [Parallel] Mutation Testing

**User Story:** As a quality engineer, I want mutation testing on the core invariants, so that the test suite is shown to detect injected faults.

#### Acceptance Criteria

1. THE CrashSense SHALL configure `mutmut` for the Python modules `backend/triangulation/tdoa_solver.py`, `backend/triangulation/atmospheric.py`, `backend/api/dedup.py`, `backend/api/rate_limit.py`, and `backend/audio_model/onset.py`.
2. THE CrashSense SHALL configure `Stryker` for the JavaScript files `frontend/src/components/AlertPanel.jsx`, `frontend/src/components/DroneTracker.jsx`, and `frontend/src/hooks/useWebSocket.js`.
3. WHEN `make mutation-test` is run, THE CrashSense SHALL achieve a mutation score of at least 70 percent across the configured Python modules and at least 60 percent across the configured JavaScript files.
4. IF the mutation score falls below the configured thresholds, THEN `make mutation-test` SHALL exit with a non-zero status code and SHALL emit a per-module breakdown to `reports/mutation_<timestamp>.json`.

### Requirement 30: [Testing] [Parallel] Property-Based Tests

**User Story:** As a quality engineer, I want property-based tests covering the core algorithmic invariants, so that subtle bugs in TDOA, dedup, onset detection, atmospheric correction, dispatch, rate limiting, and backpressure are caught by random exploration rather than handwritten examples.

#### Acceptance Criteria

1. THE PBT_Suite SHALL be implemented using Hypothesis (Python) and fast-check (JavaScript) and SHALL run as part of the `make pbt` target.
2. THE PBT_Suite SHALL include a TDOA solver noise-budget property: FOR any sensor configuration with 3 to 16 sensors and any crash coordinate strictly inside the convex hull of the sensors, perturbing arrival times with independent Gaussian noise of standard deviation in the inclusive range 0.0 to 0.005 seconds SHALL produce a localization error less than or equal to a noise-budget bound documented in `docs/tdoa_error_bound.md`.
3. THE PBT_Suite SHALL include a TDOA permutation symmetry property: FOR any permutation π applied to both the sensor list and arrival times, the returned `lat` and `lon` SHALL match the unpermuted result within 1e-6 degrees.
4. THE PBT_Suite SHALL include an event dedup idempotency property: FOR any sequence of N submissions with the same `event_id`, the count of broadcast messages with that `event_id` SHALL equal exactly 1.
5. THE PBT_Suite SHALL include an event-id non-duplication broadcast property: FOR any interleaving of M distinct `event_id` submissions, no `event_id` SHALL appear in the broadcast log more than once.
6. THE PBT_Suite SHALL include an onset detector refractory monotonicity property as defined in Requirement 8.5.
7. THE PBT_Suite SHALL include an atmospheric correction temperature monotonicity property as defined in Requirement 11.3.
8. THE PBT_Suite SHALL include an atmospheric correction wind sign-correctness property as defined in Requirement 11.4.
9. THE PBT_Suite SHALL include a dispatch slot-exclusivity property as defined in Requirement 16.4.
10. THE PBT_Suite SHALL include a rate limiter budget-bound property as defined in Requirement 19.3.
11. THE PBT_Suite SHALL include a backpressure fast-client-non-stalling property as defined in Requirement 17.5.
12. THE PBT_Suite SHALL include a Sensor_Deployment_Schema round-trip property as defined in Requirement 10.4.
13. THE PBT_Suite SHALL include a Calibrator round-trip property as defined in Requirement 5.6.
14. WHEN any property fails, THE PBT_Suite SHALL emit the minimal counterexample produced by the framework's shrinking and SHALL exit with a non-zero status code.

---

**Subsystem F: Cross-Cutting Constraints**

### Requirement 31: [Cross-Cutting] [Phase 1] Backward Compatibility With Base CrashSense

**User Story:** As a project owner, I want every preserved capability from the base spec to continue working, so that hardening does not regress what already works.

#### Acceptance Criteria

1. THE CrashSense SHALL preserve TDOA solver accuracy as defined in base-spec Requirement 7.6 (95 of 100 trials within 30 meters under 2 ms Gaussian noise) on the 3-sensor minimal deployment.
2. THE CrashSense SHALL preserve the existing frontend smoke test, the WebSocket round-trip test, and the persistence-survives-restart test, with all three passing on the hardened build.
3. THE CrashSense SHALL preserve the audio model held-out test set accuracy of at least 97.0 percent and the AudioSet real-world subset accuracy of at least 89.0 percent through any model change made under this spec.
4. THE CrashSense SHALL preserve the seed-42 reproducibility of the data pipeline, with two `dataset_prep.py` runs on the same source archive producing byte-identical splits.
5. IF any preserved-capability assertion in X1.1 through X1.4 fails on a candidate hardened build, THEN THE CrashSense SHALL block promotion of that build to the active model or active deployment.
