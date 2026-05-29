# Long-Stretch Ambient Capture Corpus

This directory holds the **ambient highway** corpus referenced by R6.1
(`.kiro/specs/crashsense-hardening/requirements.md`). The deployed
contract is:

> ≥3 distinct recording sessions, ≥24 cumulative hours of ambient
> highway audio, with one WAV file per session and a manifest at
> `data/ambient_long/manifest.json` listing each session's `filename`,
> `duration_seconds`, `sample_rate_hz`, `recorded_at_iso8601`, and
> `location_label`.

## Status: stub corpus — real recordings pending

The three WAVs currently committed (`ambient_001.wav`, `ambient_002.wav`,
`ambient_003.wav`) are **synthetic placeholders**, not real ambient
recordings. They exist so the rest of the toolchain
(`backend/audio_model/ambient_stats.py`, `scripts/curate_ambient.py`,
the `--ambient-mix` trainer flag) has something to read in CI and so
reviewers can audit the manifest schema. They are 10 s of pink noise +
low-frequency rumble each, generated deterministically by
`_regenerate_placeholders.py`.

Every placeholder session in `manifest.json` carries the flag
`"is_placeholder": true`. **Analysis scripts (`ambient_stats.py`,
trainer ambient mixing) MUST skip any session where this flag is true**
so the placeholder bytes never feed into a metric or a model.
The manifest also reports `duration_seconds: 36000.0` (10 hours) per
placeholder so the eventual ≥24-hour total invariant is signalled in
the file shape today, but again — those numbers are aspirational, not
measured.

When the real recordings land, the curator runs `curate_ambient.py add`
(see below) for each session; the helper writes the real
`duration_seconds` and `sample_rate_hz` it probes off disk and **does
not set `is_placeholder`**, so analysis scripts treat the session as
real automatically.

## Manifest schema

`manifest.json` is a JSON document with this shape:

```json
{
  "schema_version": "1.0.0",
  "generated_at_iso8601": "<UTC ISO 8601>",
  "sessions": [
    {
      "filename": "ambient_001.wav",
      "duration_seconds": 36000.0,
      "sample_rate_hz": 22050,
      "recorded_at_iso8601": "2025-01-15T06:00:00Z",
      "location_label": "highway_north_bound",
      "is_placeholder": true
    }
  ]
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `filename` | string | Relative to `data/ambient_long/`, must end in `.wav` |
| `duration_seconds` | float > 0 | Probed via `soundfile.info()` at ingest time; sanity-checked > 60 s for real sessions |
| `sample_rate_hz` | int | Must be one of `16000`, `22050`, `44100`, `48000` |
| `recorded_at_iso8601` | string | RFC 3339 / ISO 8601 UTC, e.g. `2025-01-15T06:00:00Z` |
| `location_label` | string | Short human-readable tag; `PLACEHOLDER:` prefix flags stub sessions |
| `is_placeholder` | bool, optional | `true` for stub sessions; absent / `false` for real ingests |

The placeholder rows use a `PLACEHOLDER:` prefix in `location_label`
in addition to `is_placeholder: true` so the stub status is visible
both to schema-aware code (the flag) and to a human grepping the JSON
(the prefix).

## Ingesting real recordings

Use `scripts/curate_ambient.py add` to register a recording. The helper
probes the WAV's actual `duration_seconds` and `sample_rate_hz` off
disk (so the manifest cannot drift from the file), then appends a
session entry to `manifest.json`:

```bash
python scripts/curate_ambient.py add \
    --audio-file /path/to/raw_session.wav \
    --location-label "I-5_north_milepost_142" \
    --recorded-at 2025-03-10T22:00:00Z \
    --copy-as-managed
```

`--copy-as-managed` copies the file to `data/ambient_long/ambient_NNN.wav`
using the next available 3-digit sequence number; without it, the
helper records the session under the file's existing basename.

To check the manifest after edits:

```bash
python scripts/curate_ambient.py validate
```

`validate` walks every entry, confirms the file exists, and asserts
the manifest's `duration_seconds` matches the on-disk value within
1 second (placeholder rows are skipped because their durations are
intentionally aspirational).

## Regenerating the stub WAVs

```bash
python data/ambient_long/_regenerate_placeholders.py
```

The generator pins `np.random.default_rng(42)` for any stochastic
content, so two back-to-back runs produce byte-identical bytes.

## Scope

- Stub WAVs are synthetic. They contain no real highway audio.
- Stub sessions in `manifest.json` carry `is_placeholder: true`.
- Analysis scripts must skip placeholder sessions; trainer
  `--ambient-mix` must skip placeholder sessions; the R6.1 24-hour
  invariant is **not** satisfied by the committed stubs and is checked
  separately against real-only sessions when the corpus lands.
