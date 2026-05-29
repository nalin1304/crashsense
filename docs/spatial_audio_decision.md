# Spatial Audio Decision

**Status:** Defer (fairly confident, not absolute)
**Owner:** Audio Model team
**Last updated:** 2025-01-15
**Revisit triggers:** see [§5 Revisit Conditions](#5-revisit-conditions)
**Related spec:** `.kiro/specs/crashsense-hardening/` — R4.1, R4.2, R4.3, R4.4
**Related design section:** `design.md` §3.4 (Parallel Tracks Sketch — Spatial audio decision doc)

This document is the single deliverable for **R4** of the CrashSense Hardening spec. Per R4.1 it has the four required sections (`Hypothesis`, `Experimental Setup`, `Results`, `Decision`) and per R4.3 the deferral records the measured lift, the cost estimate in engineer-weeks, and the conditions under which the work will be revisited.

---

## 1. Hypothesis

Would adding spatial-audio features to `Audio_Detector` lift accuracy on `Real_World_Test_Set` enough to justify the engineering and data-acquisition cost?

Concretely, two candidate input shapes are on the table:

- **Binaural input** (channel count = 2). Features added on top of the existing log-mel pipeline: interaural time difference (ITD) and interaural level difference (ILD).
- **Microphone-array input** (channel count in `[3, 8]`). Features added: delay-and-sum or MVDR beamformed magnitude spectrograms, with per-direction-of-arrival channel stacking.

The hypothesis we want to falsify is:

> Spatial features yield ≥ 1.0 percentage point of accuracy lift on `Real_World_Test_Set` over the SNR-mixed mono baseline (R4.2 threshold), measured by `evaluate_real_world.py` on a checkpoint trained with the same recipe except for the spatial channels.

If true → ship (R4.2 path). If false or unmeasurable → defer (R4.3 path, this document).

---

## 2. Experimental Setup

### 2.1 What we would measure

| Item | Value |
| --- | --- |
| Test set | `data/real_world_test/` — same corpus as Phase 1 baseline |
| Baseline | SNR-mixed mono checkpoint from Phase 2 (`<arch>_snr_<git_sha>.pth`) |
| Candidate | Same architecture and training recipe, additional spatial input channels in `[2, 8]` |
| Metric | `accuracy` field of `reports/real_world_eval_<sha>.json` (per R1.4) |
| Decision threshold | candidate − baseline ≥ 0.010 (1.0 percentage point, per R4.2) |
| Runs | 5 seeds; report mean and 95% bootstrap CI of the lift |

The disjointness checker in `scripts/check_real_world_disjoint.py` would still apply — any spatial test set must be disjoint with the spatial training set, by `filename` and by `source_uri`.

### 2.2 What we cannot measure today

The current corpus is **mono throughout**: `data/real_world_test/`, `data/raw_audio/crash/`, `data/raw_audio/noise/`, and the AudioSet subset all hold single-channel WAVs at 22050 Hz / 16-bit (this is the format invariant in design §3.1.1 and `spectrogram_gen.SAMPLE_RATE`). There is no multi-channel material in the repo, the held-out set, the AudioSet pull, or the ambient-long capture (R6).

That means the experiment described in §2.1 cannot run yet. To run it we would need:

1. A **multi-channel training corpus** — at least a few hundred crash clips and a comparable noise pool, captured with a known sensor geometry (binaural headworn or a small array), with provenance recorded the same way `labels.csv` already does for the mono real-world set.
2. A **multi-channel real-world test set** disjoint from the training corpus, ≥ 100 crash + 100 non-crash, same duration constraints as R1.1, captured on hardware representative of the eventual deployment.
3. **Hardware identification** — what sensor will actually be deployed? A consumer dashcam? A binaural headworn unit? A roadside array? The right corpus depends on the answer, and recording the wrong format wastes the budget.

Until all three exist, this section is design-time analysis, not measured science. The estimated lift in §3 is from public literature, not from running this system.

### 2.3 Cost estimate

If we did pursue ship-rather-than-defer:

| Workstream | Estimate |
| --- | --- |
| Multi-channel corpus acquisition (capture + annotation + provenance + disjointness audit) | ~6 engineer-weeks |
| Model retraining + spatial-feature extraction code + `evaluate_real_world.py` extension to multi-channel | ~4 engineer-weeks |
| Hardware procurement and field validation (separate from the model work) | unbounded — depends on vendor |
| **Total model-side** | **~10 engineer-weeks** |

For comparison, the Phase 2 SNR-mixing track in this same spec is ~2 engineer-weeks of model work and addresses the larger near-term gap (in-the-wild mono robustness).

---

## 3. Results

No measured lift on this codebase yet — the corpus is mono, so the candidate cannot be trained or evaluated.

Literature reference (paraphrased, not quoted): on similar binary acoustic-event-detection tasks with non-stationary background noise, beamforming and blind source separation typically deliver around **0.5 to 2.0 percentage points** of accuracy lift over the mono baseline. Lift is correlated with SNR — bigger gains under heavier interference, smaller gains in already-clean conditions. Headworn binaural ITD/ILD features tend to land at the lower end of that range for non-localization tasks (i.e. when the model only needs to say *crash or not*, not *where*). Sources include the CHiME challenges and array-processing surveys. Content was rephrased for licensing compliance.

Translated to our setting: we expect the actual lift on a deployment that captures highway-quality mono dashcam audio to fall **near or below the 1.0 percentage point threshold in R4.2**, because:

1. The interferer profile (highway road noise) is largely stationary in the bands relevant to the crash signature, which limits beamforming gain.
2. CrashSense already routes spatial reasoning to the **TDOA solver** (R10–R14) — the audio model does not need to localize, only to classify. Most of the literature's spatial lift comes from the localization sub-task, not the detection sub-task.
3. The Phase 2 SNR-mixed checkpoint will already see noise variability in training, so part of the lift spatial features would have provided is already captured by data augmentation.

This is not a proof. It is the reason we expect the experiment, when it runs, to come out below the ship threshold. Honest framing: we are **fairly confident** in deferring, not certain.

---

## 4. Decision (ship or defer)

**Defer.**

Rationales, in priority order:

1. **The deployment hardware is mono.** Real-world dashcam, smartphone, and roadside-microphone hardware overwhelmingly captures single-channel audio. A spatial model has nothing to consume on the data path that pays the bills, and would require parallel mono fallback (R4.4 — already specified for out-of-range channel counts).
2. **The cost is high relative to the expected lift.** ~10 engineer-weeks of model work plus hardware work for a literature-estimated lift of 0.5–2.0 pp, with a real chance of landing below the R4.2 threshold once the localization-vs-detection split is accounted for.
3. **Phase 2 SNR mixing addresses the bigger near-term gap.** In-the-wild robustness on mono is the headline failure mode the Phase 1 baseline is exposing. SNR augmentation is < 2 weeks and addresses it directly. We should ship that first and reread the baseline before committing to spatial.
4. **The downmix path is already required.** R4.4 mandates that `Audio_Detector` downmix multi-channel input to mono and INFO-log with a correlation id. That work (task 4.5 in `tasks.md`) lands regardless of this decision and keeps the door open: if a vendor sends us stereo or four-channel audio tomorrow, the pipeline already accepts it.

What "defer" means here:

- We do **not** add spatial-feature code to `inference.py` or the training loop.
- We do **not** acquire multi-channel training data.
- We **do** keep the multi-channel-to-mono downmix path (R4.4) as a load-bearing input adapter.
- We **do** keep this document live so the next reader sees the reasoning, not just the verdict.

---

## 5. Revisit Conditions

This decision is reopened if **any** of the following becomes true:

1. **A deployment vendor commits to binaural or array sensors.** First milestone in that case is a 200-clip stereo or 4-channel test subset from the vendor's hardware, evaluated against a stereo-aware checkpoint trained from the existing mono crashes upmixed with HRTF augmentation as a stopgap. If that subset shows ≥ 0.5 pp lift over downmix-to-mono, we open a full corpus acquisition workstream.
2. **Phase 1 baseline accuracy plateaus below 85% on `Real_World_Test_Set` after Phase 2 SNR-mixed training lands** (R2 fully shipped). At that point the model is signal-starved and adding input channels becomes a more attractive lever than further data augmentation. The trigger is mechanical: `make preservation-gate` green plus a Phase 2 baseline number under 0.85 on `Real_World_Test_Set`.
3. **Independent published results show ≥ 3 pp lift** from spatial features on a directly comparable highway-acoustic detection task, on a corpus and hardware comparable to ours. That changes the cost-benefit math regardless of points 1 and 2.

If none of the above triggers within four quarters, this document gets a `Last reviewed` bump and the same decision, with a fresh cost estimate.

---

## 6. References

- `requirements.md` R4 (this document is the literal R4.1 deliverable)
- `design.md` §3.4 — Parallel Tracks Sketch — Spatial audio decision doc
- `tasks.md` 4.4 — task that authors this document
- `tasks.md` 4.5 — the multi-channel downmix path (R4.4), which lands regardless of this decision
- Phase 2 design — `design.md` §3.2 (SNR_Mixer), the alternative track that addresses the larger near-term gap
