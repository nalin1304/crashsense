# Adversarial_Suite Corpus (R7.1)

This directory holds the `Adversarial_Suite` — labelled non-crash audio in
four crash-adjacent categories used to bound the Audio_Detector false-positive
rate (R7.2, R7.3). The four categories are required by R7.1 and must not
change without a spec update:

```
data/adversarial/
├── horns/                # vehicle horns
├── fireworks/            # firecrackers, mortars, aerial shells
├── tire_blowouts/        # blowout / tire pop / explosive deflation
├── airbrakes/            # truck / bus air brake whoosh
├── manifest.json         # per-category provenance manifest
├── _regenerate_placeholders.py
└── README.md             # this file
```

## Status: stubbed (placeholders only)

This corpus is currently **stubbed**. Each category contains 5 short
synthetic placeholder WAVs (`<category>_placeholder_<NN>.wav`, 4.0 s each,
mono PCM 22050 Hz 16-bit). The production target is **≥50 real WAV clips
per category** (R7.1); placeholders exist solely so

* `evaluate_adversarial.py` (task 4.13) can be exercised end-to-end,
* `scripts/curate_adversarial.py` (this task) has a real manifest to
  append to, and
* `scripts/check_real_world_disjoint.py` (extended in this task) has
  something to cross-check against the training manifest.

Real audio replaces the placeholders one clip at a time via
`scripts/curate_adversarial.py add` (see below). Placeholder rows are
distinguishable in the manifest by `is_placeholder: true` so any consumer
that wants to exclude synthetic data can do so trivially.

## Audio format invariants

Every clip — placeholder and real — must satisfy:

| Property | Required value | Source |
| --- | --- | --- |
| File extension | `.wav` | R7.1 |
| Container | RIFF WAV | R7.1 |
| Sample rate | 22050 Hz | matches `spectrogram_gen.SAMPLE_RATE` |
| Channels | 1 (mono) | matches Audio_Detector input |
| Subtype | `PCM_16` (16-bit signed PCM) | matches existing fixtures |
| Duration | 3.0 ≤ duration ≤ 30.0 s | R7.1 |

`scripts/curate_adversarial.py add` enforces every row in this table at
ingest time. Real recordings rarely arrive in this exact format; resample
with `sox`, `ffmpeg`, or `librosa` before ingest, e.g.:

```bash
sox input.mp3 -r 22050 -c 1 -b 16 cleaned.wav trim 0 10
python scripts/curate_adversarial.py add \
    --category horns \
    --audio-file cleaned.wav \
    --license "CC-BY-4.0" \
    --source-uri "freesound:1234567"
```

`add` will

1. validate the format and duration,
2. allocate the next `<category>_<NNNN>.wav` filename,
3. copy the WAV into `data/adversarial/<category>/`,
4. append a row to `manifest.json` with the licence, source URI, measured
   duration, and an ISO-8601 ingest timestamp.

The `--source-uri` is enforced unique across the entire manifest — the same
clip cannot be ingested into two categories.

## `manifest.json` schema

```json
{
  "schema_version": "1.0.0",
  "generated_at_iso8601": "...",
  "last_updated_iso8601": "...",
  "categories": {
    "horns": [
      {
        "filename": "horns_0001.wav",
        "source_uri": "freesound:1234567",
        "license": "CC-BY-4.0",
        "duration_s": 7.42,
        "recorded_at_iso8601": "2025-01-15T14:30:00Z",
        "is_placeholder": false
      }
    ],
    "fireworks": [...],
    "tire_blowouts": [...],
    "airbrakes": [...]
  }
}
```

`scripts/curate_adversarial.py validate` walks the manifest and asserts
every entry resolves to a WAV that still satisfies the format invariants
and whose recorded `duration_s` matches the actual file within 50 ms.

## Disjointness with training corpora (R1.3 extension)

`scripts/check_real_world_disjoint.py` is extended in this task to also
cross-check `data/adversarial/manifest.json` against the training source
manifest at `data/raw_audio/_manifest.json`. Any adversarial `source_uri`
present in the training set is reported and the script exits non-zero, so
`make preservation-gate` will fail. Synthetic placeholder rows are skipped
because their `synthetic:...` URIs cannot collide with anything in the
training set.

## Regenerating the placeholders

```bash
python data/adversarial/_regenerate_placeholders.py
```

The generator pins `np.random.default_rng(42 + offset)` per category so
repeated runs produce **byte-identical** WAVs. The script is also
*idempotent*: it preserves any real (non-placeholder) entries already in
`manifest.json` and only rewrites placeholder rows. Safe to run after
real clips have been ingested.

## Scope

| Use | Allowed |
| --- | --- |
| Negative-class evaluation (`evaluate_adversarial.py`, R7.2 / R7.3) | Yes, once real clips replace placeholders |
| Negative-class augmentation in training | No — adversarial corpus is held out |
| Replacing the real-world test set | No — different invariants and licence requirements |

The placeholder clips are synthetic and are **not** representative of real
horns / fireworks / blowouts / airbrakes; do not use them to draw any
performance conclusion.
