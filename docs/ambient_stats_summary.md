# Ambient Stats Summary

**Status:** Tooling live, real numbers pending real recordings (task 4.10, R6.2)
**Owner:** Audio Model team
**Last updated:** 2025-01-15
**Related spec:** `.kiro/specs/crashsense-hardening/` — R6.1, R6.2, R6.4
**Related design section:** `design.md` §3.4 (Parallel tracks sketch — long-stretch ambient capture is the bullet that funnels into the `--ambient-mix` trainer flag from R6.3)

This document is the deliverable for **R6.2** of the CrashSense Hardening spec. It narrates the highlights of `reports/ambient_stats.json` and records the deployment-relevant numbers we expect ambient highway audio to land in once real recordings replace the committed placeholders.

---

## 1. What the ambient corpus is for

`data/ambient_long/` holds the **long-stretch ambient highway** corpus required by R6.1: ≥3 distinct recording sessions totalling ≥24 cumulative hours, with one WAV per session and a manifest at `data/ambient_long/manifest.json` describing each session's `filename`, `duration_seconds`, `sample_rate_hz`, `recorded_at_iso8601`, and `location_label`. See `data/ambient_long/README.md` for the schema and the placeholder policy.

The corpus exists for two downstream consumers:

- **R6.3 / `--ambient-mix` trainer flag.** Negative-class samples are drawn from these recordings (per-batch probability 0.2–0.5) so the classifier sees the same ambient distribution at training time that it will see in the field.
- **R6.2 / this report.** A statistical fingerprint of that ambient distribution, written to `reports/ambient_stats.json`. The fingerprint is what we point at when asking "is the deployed model's training noise representative of where it will run?" — without it the SNR-mixed checkpoints from Phase 2 are calibrated against a noise distribution we never measured.

---

## 2. What `python -m backend.audio_model.ambient_stats` outputs

Per session (and as a duration-weighted aggregate), `reports/ambient_stats.json` carries four metric blocks:

- **`mean_rms`** — root-mean-square of the waveform in `[0, 1]` linear amplitude. The headline loudness number before any frequency weighting.
- **`a_weighted_spl_dbfs`** — mean RMS in dBFS *after* the IEC 61672 A-weighting IIR filter. dBFS, not dB SPL — converting to absolute SPL requires the recording chain's microphone sensitivity and gain, which the corpus manifest does not carry. See §3 below for what a dBFS number maps to in the field.
- **`spectral_centroid_hz`** — distribution `{mean, p50, p95, max}` of the per-frame spectral centroid (`librosa.feature.spectral_centroid`). The "where is the energy?" indicator.
- **`zero_crossing_rate`** — distribution `{mean, p50, p95, max}` of per-frame ZCR (`librosa.feature.zero_crossing_rate`). A cheap noisiness / aperiodicity proxy that complements the centroid.

Sessions with `is_placeholder: true` are counted in `n_sessions_skipped_placeholder` and excluded from every metric — the placeholder bytes are synthetic and would corrupt the distribution. R6.4 fail-open: missing-on-disk, non-WAV, or decode-error sessions land in `reports/ambient_stats.errors.csv` and the loop continues.

---

## 3. Deployment-relevant SPL ranges

A-weighted dB(A) at the source maps roughly to the dBFS the mics record after the gain chain. The ranges below are field expectations from highway-audio literature; numbers in the eventual `reports/ambient_stats.json` should fall inside one of these bands once real recordings land.

| Scenario | Expected dB(A) at source | Notes |
| --- | --- | --- |
| **Highway interior cabin / dashcam at speed** | ~80–90 dB(A) | A typical dashcam capture chain at unity gain lands around `-25` dBFS A-weighted RMS for this band (≈ 75 dBFS reference). The dominant content is low-frequency tire/wind roar. |
| **Toll plaza ambient** | ~70–78 dB(A) | Mix of idling diesels, light braking, and reflected highway hum. We expect a centroid pulled up by transient mechanical noise (gates, light brakes). |
| **Sound-wall isolated stretch** | ~60–70 dB(A) | Sound walls drop the dominant LF roar by 10–15 dB(A). Useful as a "quiet" anchor session — if a real recording lands here, it should be the lowest A-weighted SPL in the corpus. |

Practical reading: a session reporting `a_weighted_spl_dbfs` near `-30` to `-25` is plausibly highway interior; near `-35` to `-30` is plausibly toll plaza; below `-40` is plausibly sound-wall isolated. Numbers far outside these bands warrant looking at the recording chain (mic sensitivity, gain, clipping) before trusting downstream `--ambient-mix` training.

---

## 4. Spectral centroid behaviour

Spectral centroid is the moment that separates "highway roar" from "crash-adjacent transient" in this corpus.

| Sound | Centroid range | Why |
| --- | --- | --- |
| Pure highway road roar | **~200–500 Hz** | Tire/road interaction and wind noise concentrate energy in the low band. A long ambient session should have a `mean` and `p50` centroid in this band; `p95` typically still under 1 kHz. |
| Tire screech | ~2–4 kHz | Stick-slip squeal pushes energy into the mid-treble. Will appear as outliers in `p95` / `max` if a session captures braking events. |
| Crash impact | broadband, centroid 1–3 kHz | Wide-band transient. Not expected in *ambient* sessions; if `max` centroid sits in this band on a long session, the recording likely includes a crash and should be relabelled. |
| Air brake | ~500–1500 Hz | Mid-band whoosh. Plausible in toll-plaza sessions; pulls the `p95` centroid up without pushing the `mean`. |

The `p50` of `spectral_centroid_hz` in the aggregate is the single most useful number to glance at. If real sessions land at `~200–500 Hz`, the corpus is doing what R6.1 describes — capturing the ambient LF distribution. If `p50` rides above 1 kHz on a long session, the recording almost certainly contains transients (sirens, screeches, mechanical work) that R6 ambient is supposed to *exclude*.

---

## 5. Current state of the report

The committed manifest is **placeholder-only** (three synthetic sessions, each flagged `is_placeholder: true`). `python -m backend.audio_model.ambient_stats` runs cleanly against it, but every session is skipped, `n_sessions_processed` is `0`, and the `aggregate` block is the all-zeros default. There is no deployable ambient fingerprint until real recordings land.

When real recordings arrive:

1. Operator runs `python scripts/curate_ambient.py add --audio-file ... --location-label ... --recorded-at ... --copy-as-managed` for each session. The helper probes the WAV and appends a real (non-placeholder) entry to the manifest — see `scripts/curate_ambient.py` and `data/ambient_long/README.md` for the ingestion workflow.
2. CI re-runs `python -m backend.audio_model.ambient_stats`, populating `reports/ambient_stats.json` with the first non-zero distribution.
3. This document is updated with measured `mean_rms`, `a_weighted_spl_dbfs`, and centroid distributions, and the §3 / §4 expectations are confirmed or adjusted against the real numbers.

Until that flow runs end-to-end, treat every concrete dBFS or centroid value in this document as a **field expectation**, not a measured corpus statistic.

---

## 6. References

- `backend/audio_model/ambient_stats.py` — analysis script, JSON / errors-CSV writer (task 4.9)
- `data/ambient_long/manifest.json` — session manifest (R6.1)
- `data/ambient_long/README.md` — corpus contract, placeholder policy, ingestion workflow
- `scripts/curate_ambient.py` — provenance-recording ingestion helper (task 4.8)
- `reports/ambient_stats.json` — output of the analysis script (R6.2)
- `reports/ambient_stats.errors.csv` — fail-open per-session diagnostics (R6.4)
- `.kiro/specs/crashsense-hardening/requirements.md` — R6.1, R6.2, R6.3, R6.4
- `.kiro/specs/crashsense-hardening/design.md` — §3.4 long-stretch ambient capture paragraph
