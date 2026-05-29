.PHONY: install backend frontend demo dev clean preservation-gate phase2-gate

# git short-sha is used to name per-build artifact reports under reports/.
# Resolved at parse time so every recipe line in this file sees the same value.
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

install:
	python3 -m pip install -r backend/requirements.txt
	cd frontend && npm install

backend:
	uvicorn backend.api.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

demo:
	python3 backend/demo_runner.py --backend http://127.0.0.1:8000

tdoa:
	python3 tdoa_demo.py --trials 5

dev:
	./start.sh

clean:
	rm -rf data/spectrograms backend/audio_model/crash_detector.pth

# ============================================================================
# Preservation gate (R31.1 - R31.5)
# ----------------------------------------------------------------------------
# Single CI-blocking job that asserts every preserved capability from the base
# CrashSense spec still holds on the candidate hardened build. Sub-checks run
# in order; Make's default fail-fast semantics turn the first red check into
# the stop reason. A non-zero exit here is the merge-blocking signal in CI
# (R31.5) and is also consulted by the Phase-3 model registry promotion
# script before it rewrites ACTIVE.json.
#
#   R31.1  TDOA solver: 95/100 trials within 30 m under sigma=2 ms noise
#   R31.2  Frontend smoke + WebSocket round-trip + persistence-restart
#   R31.3  Audio held-out >= 97.0%, AudioSet subset >= 89.0%
#   R31.4  Seed-42 byte-identical dataset splits
#   R31.5  Block promotion / merge on any sub-check failure;
#          plus real-world corpus disjointness (R1.3) and baseline existence.
#
# Reference: design.md section 3.1.4.
#
# Notes for maintainers:
#   * The design names specific test functions (test_noise_budget_gate,
#     test_round_trip, test_survives_restart). Until those names land we
#     invoke the whole test files - they cover the same contracts.
#   * Sub-checks whose backing test/script does not exist yet (frontend
#     smoke until task 7.1, etc.) print a "SKIP: TODO" line and continue;
#     they do not fail the gate. The CI workflow (task 1.13) will add full
#     coverage once those tasks land.
#   * The AudioSet sub-check is conditional: backend/audio_model/evaluate.py
#     emits per_source.<collection> only for collections that actually
#     contributed clips to the held-out split. When the split happens to
#     contain zero AudioSet clips the per_source.audioset key is legitimately
#     absent, so we probe with a stdlib json one-liner first and skip rather
#     than fail when the field is missing.
# ============================================================================
preservation-gate:
	@echo "==> R31.1 TDOA noise budget gate (3-sensor, 95/100 within 30m, sigma=2ms)"
	pytest tests/test_tdoa_solver.py -q --no-header

	@echo "==> R31.2 Frontend smoke test"
	@if [ -f frontend/src/__tests__/smoke.test.jsx ]; then \
		cd frontend && npm test -- --run --reporter=dot src/__tests__/smoke.test.jsx; \
	else \
		echo "  SKIP: TODO smoke test pending task 7.1 (frontend/src/__tests__/smoke.test.jsx absent)"; \
	fi

	@echo "==> R31.2 WebSocket round-trip test"
	@if [ -f tests/test_ws_manager.py ]; then \
		pytest tests/test_ws_manager.py -q --no-header; \
	else \
		echo "  SKIP: TODO ws round-trip pending (tests/test_ws_manager.py absent)"; \
	fi

	@echo "==> R31.2 Persistence-restart test"
	@if [ -f tests/test_event_store.py ]; then \
		pytest tests/test_event_store.py -q --no-header; \
	else \
		echo "  SKIP: TODO persistence-restart pending task 3.3 (tests/test_event_store.py absent)"; \
	fi

	@echo "==> R31.3 Audio held-out accuracy >= 97.0%"
	@mkdir -p reports
	python -m backend.audio_model.evaluate \
		--json reports/heldout_$(GIT_SHA).json
	python scripts/assert_metric.py \
		reports/heldout_$(GIT_SHA).json \
		--field metrics.accuracy --min 0.970

	@echo "==> R31.3 AudioSet subset accuracy >= 89.0% (conditional)"
	@if python -c "import json,sys; r=json.load(open('reports/heldout_$(GIT_SHA).json')); sys.exit(0 if 'audioset' in r.get('per_source',{}) else 1)"; then \
		python scripts/assert_metric.py \
			reports/heldout_$(GIT_SHA).json \
			--field per_source.audioset.accuracy --min 0.890; \
	else \
		echo "  SKIP: per_source.audioset absent in held-out report (no AudioSet clips in split)"; \
	fi

	@echo "==> R31.4 Seed-42 byte-identical splits"
	python scripts/check_seed42_reproducibility.py

	@echo "==> R31.5 Phase 1 baseline must exist (R1)"
	@if [ ! -f reports/real_world_eval_$(GIT_SHA).json ]; then \
		echo "FAIL: reports/real_world_eval_$(GIT_SHA).json not found." >&2; \
		echo "Run \`python -m backend.audio_model.evaluate_real_world\` before opening this PR." >&2; \
		exit 1; \
	fi

	@echo "==> R31.5 Real-world corpus disjoint with training corpora"
	python scripts/check_real_world_disjoint.py

	@echo "PRESERVATION GATE: all checks passed"

# ============================================================================
# Phase 2 SNR-mixed evaluation gate (R2.5)
# ----------------------------------------------------------------------------
# Asserts that the SNR-mixed checkpoint produced by `train.py --snr-mix`
# (task 2.3 / 2.4) does not regress real-world accuracy by more than 1.0
# percentage point versus the Phase 1 baseline. The flow is:
#
#   1. Locate the SNR checkpoint at
#      backend/audio_model/checkpoints/resnet18_snr_$(GIT_SHA).pth.
#      If it is missing, print an actionable message pointing at the
#      training command and fail. The Phase 2 trainer (R2.6) writes this
#      filename and is forbidden from overwriting the baseline checkpoint.
#
#   2. Re-run evaluate_real_world.py against that checkpoint and emit
#      reports/real_world_eval_snr_$(GIT_SHA).json (plus its sibling
#      .errors.csv) -- same script, same JSON schema, same determinism
#      contract as the Phase 1 baseline (R1.4 / R1.5).
#
#   3. Call scripts/assert_phase2_gate.py to compare candidate.accuracy
#      against baseline.accuracy with the default 0.01 tolerance per R2.5.
#      Non-zero exit blocks promotion of the SNR-mixed checkpoint.
#
# This target is independent of `preservation-gate`; the recommended Phase 2
# checkpoint flow (task 2.6) is `make preservation-gate && make phase2-gate`.
# ============================================================================
phase2-gate:
	@echo "==> R2.5 Locate SNR-mixed checkpoint"
	@mkdir -p reports
	@if [ ! -f backend/audio_model/checkpoints/resnet18_snr_$(GIT_SHA).pth ]; then \
		echo "FAIL: SNR-mixed checkpoint not found at backend/audio_model/checkpoints/resnet18_snr_$(GIT_SHA).pth" >&2; \
		echo "      Run \`python -m backend.audio_model.train --snr-mix\` (task 2.3 / 2.4) before invoking phase2-gate." >&2; \
		exit 1; \
	fi

	@echo "==> R2.5 Phase 1 baseline must exist"
	@if [ ! -f reports/real_world_eval_$(GIT_SHA).json ]; then \
		echo "FAIL: Phase 1 baseline reports/real_world_eval_$(GIT_SHA).json not found." >&2; \
		echo "      Run \`python -m backend.audio_model.evaluate_real_world\` first." >&2; \
		exit 1; \
	fi

	@echo "==> R2.5 Re-run evaluate_real_world.py on the SNR checkpoint"
	python -m backend.audio_model.evaluate_real_world \
		--checkpoint backend/audio_model/checkpoints/resnet18_snr_$(GIT_SHA).pth \
		--out reports/real_world_eval_snr_$(GIT_SHA).json \
		--errors reports/real_world_eval_snr_$(GIT_SHA).errors.csv

	@echo "==> R2.5 Assert candidate accuracy >= baseline - 0.01"
	python scripts/assert_phase2_gate.py \
		--baseline reports/real_world_eval_$(GIT_SHA).json \
		--candidate reports/real_world_eval_snr_$(GIT_SHA).json

	@echo "PHASE 2 GATE: SNR-mixed checkpoint within tolerance of baseline"
