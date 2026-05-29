# Real-World Test Fixture Corpus (CI smoke only)

This directory holds a tiny synthetic corpus used to exercise the
`backend/audio_model/evaluate_real_world.py` I/O path under CI. It is **not**
an evaluation corpus and **must not** appear in any accuracy / preservation
report. The real corpus lives under `data/real_world_test/` and is governed by
R1.1, R1.2, R1.3.

The fixtures are committed to the repo because they are small (≈0.8 MB total)
and the synthetic generator is deterministic, so reviewers can diff regenerated
bytes against the committed bytes during PR review.

## Files

All audio is **mono, 22050 Hz, 16-bit PCM** (matches
`backend/audio_model/spectrogram_gen.SAMPLE_RATE`).

| Path | Duration | Content |
| --- | --- | --- |
| `crash/rw_crash_0001.wav` | 5.0 s | 200→4000 Hz sine sweep with attack-decay envelope |
| `crash/rw_crash_0002.wav` | 3.0 s | seeded white-noise burst, peak ≈ 0.5 |
| `crash/rw_crash_short.wav` | 1.5 s | 880 Hz sine — **intentionally below the 3.0 s minimum** (drives the `duration_out_of_range` error path) |
| `noise/rw_noise_0001.wav` | 4.0 s | pink noise from seeded white-noise via 1/√f spectral shaping |
| `noise/rw_noise_0002.wav` | 6.0 s | near-silence with two 0.05-amplitude tone bursts at 1.5 s and 4.0 s |

## `labels.csv`

The manifest mixes **valid** and **intentionally invalid** rows. The invalid
rows exist so `tests/test_evaluate_real_world.py` can verify each branch of
the R1.6 error-handling table without having to construct fragile temp
fixtures inside the test.

Valid rows — one per file (annotator_id `testbot`, `source_uri` `fixture:rw_<class>_<id>`):

- `rw_crash_0001.wav`
- `rw_crash_0002.wav`
- `rw_noise_0001.wav`
- `rw_noise_0002.wav`

Intentionally invalid rows — each exercises one R1.6 / §5.1 failure mode:

| Row | Failure mode | Expected `reason` |
| --- | --- | --- |
| `rw_crash_0099.wav` (file does not exist on disk) | missing file referenced by manifest | `missing_on_disk` |
| `rw_crash_0100.mp3` (`.mp3` extension) | non-WAV extension caught by filename regex / extension check | `non_wav_extension` |
| `rw_crash_short.wav` (1.5 s real WAV) | duration outside 3.0–30.0 s range | `duration_out_of_range` |
| `rw_crash_0001.wav` with `annotator_id = "bot!@#"` | malformed annotator id (fails `^[A-Za-z0-9_-]+$`) | `csv_validation_error` |

The "malformed annotator_id" row reuses an existing real filename so the row
fails Pydantic validation **before** any disk I/O — it isolates the
CSV-validation branch from the missing-file branch.

## Regeneration

```bash
python tests/fixtures/real_world_test/_regenerate.py
```

The generator pins `np.random.default_rng(42)` for any stochastic content and
uses pure formulae over `np.arange` for the deterministic content, so two
back-to-back runs on the same numpy/scipy versions produce **byte-identical**
WAVs. If a regeneration changes any committed byte, the diff is signal — either
the generator changed, the dependency versions drifted, or the fixtures were
hand-edited.

`labels.csv` and this README are hand-maintained; the regeneration script
deliberately does not touch them so reviewers can see provenance changes in
diff form.

## Scope guarantee

These fixtures are CI-smoke-only:

- They are **synthetic** and contain no real crash audio.
- They live under `tests/fixtures/`, not `data/`.
- They must not be referenced by any preservation-gate metric (`make
  preservation-gate`), evaluation report (`reports/real_world_eval_*.json`),
  or training run.
