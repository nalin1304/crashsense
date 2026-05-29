# Severity Labels — `data/real_world_test/severity_labels.csv`

This file documents the label format consumed by
`backend/audio_model/train_severity.py` and `scripts/assert_severity_gate.py`
(task 4.2, R3.5). The severity-labelled corpus is a strict subset of the
Phase 1 Real_World_Test_Set crash clips: every entry here has a
corresponding row with `label=crash` in `data/real_world_test/labels.csv`.

## Status

Not yet populated. The Phase 1 corpus does not ship severity-grade
ground-truth labels — those require a clinician-style annotation pass
that has not been scoped into Phase 1. Until this file lands the
heuristic placeholder in `backend/audio_model/severity.py` (task 4.1)
remains the active classifier and `make severity-gate` is not part of
the preservation gate.

## File layout

```
data/real_world_test/
├── crash/                       # ≥100 WAV files, 3.0–30.0 s each (Phase 1)
│   ├── rw_crash_0001.wav
│   ├── rw_crash_0002.wav
│   └── ...
├── labels.csv                   # crash/noise binary manifest (Phase 1)
└── severity_labels.csv          # severity manifest (this document)
```

`severity_labels.csv` lives alongside `labels.csv` in the same root and
references the same `crash/<filename>.wav` clips.

## CSV schema

CSV with header. Two columns, in order:

| Column | Type | Constraint |
| --- | --- | --- |
| `filename` | string | Must match the regex `^[A-Za-z0-9_\-]+\.wav$`; resolves to `data/real_world_test/crash/<filename>` |
| `severity` | string | Exactly one of `minor`, `moderate`, `severe` |

Rules baked into the parser
(`backend.audio_model.train_severity.parse_severity_labels_csv`):

- Every `filename` must be unique within the manifest.
- Every `filename` should also exist as a row in `labels.csv` with
  `label=crash`. The trainer does not enforce this hard (it only needs
  the WAV on disk), but the Phase 1 disjointness checks already require
  it implicitly through the binary manifest.
- Rows that fail validation are skipped, not fatal — the trainer logs
  them at WARNING and continues with the surviving rows. If every row
  is invalid the trainer exits 1.

## Label semantics

The three classes are mutually exclusive and exhaustive:

| Label | Description |
| --- | --- |
| `minor` | Light contact, scrape, or low-energy fender bender. Audible impact but no obvious metal deformation; passenger compartment intact. |
| `moderate` | Mid-energy collision with clear metal deformation, glass breakage, or airbag deployment audible. Vehicles likely drivable to the shoulder but not under their own power. |
| `severe` | High-energy collision with multiple distinct impact phases (primary impact, secondary scrape, debris settling). Likely incapacitated occupants, immediate dispatch warranted. |

Annotators should label based on the **acoustic signal**, not the
visual context of the source video, because the deployed classifier
only sees audio.

## Example

```csv
filename,severity
rw_crash_0001.wav,minor
rw_crash_0002.wav,severe
rw_crash_0003.wav,moderate
rw_crash_0004.wav,minor
rw_crash_0005.wav,moderate
```

## How to label new clips

1. Annotate every clip in `data/real_world_test/crash/` until at least
   30 clips per class are present (R3.5 macro-F1 ≥ 0.60 needs roughly
   balanced support to be meaningful).
2. Append rows to `severity_labels.csv` with the schema above.
3. Train the head:
   ```bash
   python -m backend.audio_model.train_severity
   ```
   This writes `backend/audio_model/checkpoints/severity_<git_sha>.pth`.
4. Validate the gate:
   ```bash
   python scripts/assert_severity_gate.py \
     --checkpoint backend/audio_model/checkpoints/severity_<git_sha>.pth \
     --labels data/real_world_test/severity_labels.csv
   ```
   The script exits 0 with `OK: macro_f1=X.XX >= 0.60` on success and
   1 with the actual macro-F1 in the FAIL message on regression.

Once a checkpoint is committed, `severity.grade()` automatically prefers
the trained head over the heuristic; the heuristic remains the fallback
when no checkpoint is present or one fails to load.
