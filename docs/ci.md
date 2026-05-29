# CI: Preservation Gate

The preservation gate (`make preservation-gate`) is the single CI-blocking job
that asserts every preserved capability from the base CrashSense spec still
holds on the candidate hardened build. It is the merge-blocking signal for
PRs targeting `main` (R31.5) and the same gate is consulted by the Phase 3
model registry promotion script before it rewrites `ACTIVE.json`.

This document covers:

- What the gate checks and where each sub-check is defined.
- How to run the gate locally before opening a PR.
- How to set up branch protection so the gate is required to merge.
- Common failures and how to fix them.

## What the gate checks

The single entry point is the Make target `preservation-gate` in the
repository root `Makefile`. Sub-checks run in order and the gate exits
non-zero on the first red (Make's default fail-fast). Authoritative source:
`design.md` section 3.1.4 and `requirements.md` R31.

| Sub-check | Requirement | Backing test or script |
| --- | --- | --- |
| TDOA noise budget (3-sensor, 95/100 within 30 m at σ=2 ms) | R31.1 | `pytest tests/test_tdoa_solver.py` |
| Frontend smoke test | R31.2 | `frontend/src/__tests__/smoke.test.jsx` (skip-with-TODO until task 7.1 lands) |
| WebSocket round-trip | R31.2 | `pytest tests/test_ws_manager.py` |
| Persistence-restart | R31.2 | `pytest tests/test_event_store.py` (skip-with-TODO until task 3.3 lands) |
| Audio held-out accuracy ≥ 97.0% | R31.3 | `python -m backend.audio_model.evaluate` + `scripts/assert_metric.py` |
| AudioSet subset accuracy ≥ 89.0% | R31.3 | `scripts/assert_metric.py` (conditional on `per_source.audioset` being present) |
| Seed-42 byte-identical splits | R31.4 | `scripts/check_seed42_reproducibility.py` |
| Phase 1 baseline report exists | R1, R31.5 | `test -f reports/real_world_eval_<sha>.json` |
| Real-world corpus disjoint with training corpora | R1.3, R31.5 | `scripts/check_real_world_disjoint.py` |

Sub-checks whose backing test does not yet exist (frontend smoke until
task 7.1, event-store tests until task 3.3) print a `SKIP: TODO` line and
continue. They never fail the gate. The CI workflow runs the same Makefile
and inherits the same skip paths, so a cold run on a fresh checkout is a
warning, not a failure.

## Run the gate locally

From the repo root, with the Python virtualenv and `npm install` already
done in `frontend/`:

```bash
make preservation-gate
```

Phase 1 baseline guard: the gate refuses to pass until
`reports/real_world_eval_<git_sha>.json` exists for the current commit.
Generate it before opening the PR:

```bash
python -m backend.audio_model.evaluate_real_world
```

This writes the report and a sibling `*.errors.csv` describing any clips
excluded by the determinism contract in `evaluate_real_world.py`.

## Branch protection setup

The CI job published by `.github/workflows/preservation-gate.yml` is named
`preservation-gate`. Wire it as a required status check on `main`:

```
Settings → Branches → Branch protection rules → Add rule
  Branch name pattern: main
  ✓ Require a pull request before merging
  ✓ Require status checks to pass before merging
      Required status checks:
        - preservation-gate
  ✓ Require branches to be up to date before merging
```

Equivalent API call (`gh` CLI):

```bash
gh api -X PUT \
  repos/:owner/:repo/branches/main/protection \
  -f required_status_checks.strict=true \
  -F 'required_status_checks.contexts[]=preservation-gate' \
  -f enforce_admins=true \
  -f required_pull_request_reviews.required_approving_review_count=1 \
  -f restrictions= 
```

## Common failures and how to fix them

### `FAIL: reports/real_world_eval_<sha>.json not found.`

Phase 1 baseline guard tripped. Run the real-world evaluator on the active
checkpoint and commit any data manifest changes the run depends on:

```bash
python -m backend.audio_model.evaluate_real_world
```

The script writes a deterministic report keyed on the current `git_sha` and
exits with a non-zero status if more than 50% of clips are excluded (signals
corpus rot, see R1.6). The error CSV at
`reports/real_world_eval_<sha>.errors.csv` lists every excluded clip and the
reason.

### `R31.3 Audio held-out accuracy >= 97.0% — FAIL`

The active audio checkpoint regressed on the held-out test split. Either fix
the regression (revert the change, retrain) or, if the regression is
intentional, raise the issue with the model registry process before merging.
The gate is also the promotion gate, so a sub-check that is red here will
also block any update to `ACTIVE.json`.

### `Seed-42 byte-identical splits — FAIL`

`scripts/check_seed42_reproducibility.py` runs `dataset_prep.py --dry-run`
twice and diffs the output JSON byte-for-byte. A non-empty diff prints to
stderr. Common causes:

- A new non-deterministic call site in `dataset_prep.py` (e.g., `set` order,
  dict ordering, unsorted directory listings).
- A new file in `data/raw_audio/` that the deterministic ordering does not
  cover. Re-run `dataset_prep.py` once locally and check the manifest.

### `Real-world corpus disjoint with training corpora — FAIL`

`scripts/check_real_world_disjoint.py` found shared filenames or
`source_uri` values between `data/real_world_test/labels.csv` and the
training manifest at `data/raw_audio/_manifest.json`. Remove the offending
clips from the real-world set and re-run.

### Heavy-artifact cache miss in CI

The CI workflow caches three heavy artifact directories
(`backend/audio_model/checkpoints/`, `data/spectrograms/`, `data/raw_audio/`)
keyed on the hash of the trainer and dataset-prep sources. On a cold run
(first build after a trainer or dataset-prep change) the cache misses and
the sub-checks that need those artifacts emit `SKIP: TODO` lines. The
workflow surfaces a `::warning::` line in the build log so this is visible.
The gate still passes; it just covers fewer contracts on that one run.

To prime the cache locally and push the artifacts via a follow-up commit
that updates only the trainer hash inputs, regenerate the data:

```bash
# 1. Rebuild raw audio + manifest
python -m backend.audio_model.dataset_prep

# 2. Regenerate spectrograms
python -m backend.audio_model.spectrogram_gen

# 3. Retrain (or pull from a release artifact)
python -m backend.audio_model.train
```

The next CI run on a branch whose hashes match what produced the artifacts
will hit the cache and run the full gate.

## Why the gate is one job, not seven

Single command, single failure surface. The gate's job is to be the merge
contract, not a dashboard. Ten green sub-checks and one red one is still a
red gate. CI calls one thing; humans run one thing locally. See `design.md`
section 3.1.4 (design principle 2: "the preservation gate is one job, not
seven").
