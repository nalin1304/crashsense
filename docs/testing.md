# Testing

CrashSense ships with a pytest suite (87 tests) plus a Playwright-based dashboard smoke test.

## Running everything

```bash
# Backend unit + integration tests
.venv/bin/python -m pytest tests/

# Frontend dashboard smoke test (requires backend + dev server running)
./.venv/bin/python -m uvicorn backend.api.main:app --port 8000 &
( cd frontend && npm run dev ) &
.venv/bin/python scripts/smoke_dashboard.py
```

## What's covered

| Test file | What it verifies |
|---|---|
| `tests/test_sensor_config.py` | Sensor coordinates, IDs, toll-plaza membership match the spec |
| `tests/test_geo_utils.py` | meters↔latlon conversion (within 3% / 1 km), bearings, nearest-sensor selection, error handling |
| `tests/test_tdoa_solver.py` | Forward simulation correctness, noiseless inverse roundtrip, **noisy 50-trial localization with ≥45 within 30 m** |
| `tests/test_schemas.py` | CrashEvent field bounds, status enum, JSON serialization |
| `tests/test_ws_manager.py` | Connection add/remove, broadcast serialization, drop on send failure, empty-pool no-op |
| `tests/test_spectrogram.py` | Pad/trim length, mel-db shape, RGB output, disk writing |
| `tests/test_inference.py` | Predict return contract, accepts path or array, accuracy on real ESC-50 samples (skipped if checkpoint missing) |
| `tests/test_api_routes.py` | All REST endpoints, payload validation, file-size and format limits |
| `scripts/smoke_dashboard.py` | Headless Chromium opens dashboard, clicks Simulate Crash, verifies DETECTING → DRONE DISPATCHED → DRONE ARRIVED transition |

## Latest results

```
tests/test_api_routes.py        12 passed
tests/test_geo_utils.py         22 passed
tests/test_inference.py          7 passed
tests/test_schemas.py           17 passed
tests/test_sensor_config.py      7 passed
tests/test_spectrogram.py        9 passed
tests/test_tdoa_solver.py        8 passed
tests/test_ws_manager.py         5 passed
============================= 87 passed in 12.73s
```

Dashboard smoke test:
```
Initial render: leaflet tiles loaded = 28, sensors = 3
Clicked Simulate Crash
After detection: badges visible = ['CRASH DETECTED', 'DRONE DISPATCHED']
DRONE DISPATCHED badge rendered
DRONE ARRIVED badge rendered
ALL CHECKS PASSED
```

## Adding new tests

The repo follows pytest conventions with shared `conftest.py` for path setup. Tests live at the repo root in `tests/` and use a `Test*` class wrapper for grouping. `pytest-asyncio` is configured in `auto` mode so plain `async def test_*` functions work without decorators.

For tests that depend on the trained model:

```python
@pytest.mark.skipif(not CHECKPOINT.exists(), reason="trained checkpoint missing")
def test_thing():
    ...
```

For tests that depend on the dataset:

```python
pytestmark = pytest.mark.skipif(not _have_dataset(), reason="dataset not prepared")
```
