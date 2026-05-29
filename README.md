# CrashSense

**Acoustic Triangulation & Autonomous Drone Dispatch.**
A live demo that detects highway crashes from audio, localizes them via TDOA across three roadside sensors, and dispatches a virtual drone from the nearest toll plaza on an interactive map.

```
crashsense/
├── backend/
│   ├── audio_model/         # dataset prep, spectrograms, ResNet-18, training, inference
│   ├── triangulation/       # sensor config, geo helpers, TDOA forward+inverse solver
│   ├── api/                 # FastAPI app, routes, schemas, WebSocket manager
│   ├── demo_runner.py       # end-to-end pipeline orchestrator
│   └── requirements.txt
├── frontend/                # React + Vite + Tailwind dashboard
│   └── src/components/      # Map, AlertPanel, DroneTracker, AudioMonitor, etc.
├── data/
│   ├── raw_audio/           # populated by dataset_prep.py
│   └── spectrograms/        # populated by spectrogram_gen.py
├── docs/                    # detailed component documentation
├── tdoa_demo.py             # TDOA proof-of-concept
└── start.sh                 # backend + frontend + demo launcher
```

## TL;DR

```bash
# 1. one-time setup (creates venv, installs deps, downloads data, trains model)
./scripts/setup.sh

# 2. run everything
./start.sh

# 3. (optional) run the test suite
./.venv/bin/python -m pytest tests/
```

Open <http://127.0.0.1:5173>. Click **Simulate Crash** in the alert panel and you'll see the red pin drop, the wavefront expand, the drone fly from the nearest toll plaza, and the alert card progress through DETECTING → DRONE DISPATCHED → DRONE ARRIVED.

## What's actually verified

This system has been run end-to-end on macOS / Apple Silicon with Python 3.14 and Node 25:

| Stage | Result |
|---|---|
| Dataset acquisition (ESC-50 + UrbanSound8K + Freesound + AudioSet) | **4,606 WAVs** (2,400 crash + 2,206 noise; 3,025 unique sources) |
| Spectrogram generation | 4,606 PNGs in ~80 s |
| Source-aware test split | **zero source overlap** between train / val / test |
| ResNet-18 + SpecAugment + Mixup + Focal | 94.10% held-out accuracy, ECE 1.54% |
| AST + linear head + temperature scaling | **95.74% held-out accuracy**, AudioSet 89.13%, ECE 0.86% |
| 5-fold CV (AST head) | **97.35% ± 0.35%** mean accuracy |
| Real Freesound crash classification | 18 / 18 collision recordings at 97–100 % CRASH |
| Streaming detection | **3-of-4 consensus**, debounced for `CONSENSUS_N` hops |
| Event persistence | SQLite-backed, hydrates dashboard on connect |
| Optional WebSocket auth | shared-secret token via `CRASHSENSE_WS_TOKEN` |
| Pytest suite | **103 / 103 tests passing in 4 s** |
| Headless dashboard smoke test | All 6 state transitions verified |
| End-to-end demo pipeline | < 5 s (ResNet) / ~18 s (AST CPU) audio → broadcast |

## Running each piece individually

| Command | What it does |
|---|---|
| `./.venv/bin/python -m uvicorn backend.api.main:app --reload --port 8000` | Backend only |
| `cd frontend && npm run dev` | Frontend only |
| `./.venv/bin/python tdoa_demo.py --trials 5` | TDOA proof-of-concept |
| `./.venv/bin/python -m backend.demo_runner` | End-to-end demo (needs backend running) |

## How the system works

### Phase 1 — Audio AI
1. **`dataset_prep.py`** downloads ESC-50 (a stable Creative Commons audio dataset) and copies six categories into `data/raw_audio/`:
   - **crash:** `car_horn`, `engine`, `breaking_glass`
   - **noise:** `wind`, `rain`, `crackling_fire`
   It then applies six augmentations (pitch ±2, stretch 0.9 / 1.1, additive noise at 18 / 12 dB SNR) to reach the >= 500-per-class minimum.
2. **`spectrogram_gen.py`** converts every WAV into a 224×224 PNG mel spectrogram (22.05 kHz, 128 mel bands, fmax 8 kHz, 3-second window). Pure NumPy + PIL, no matplotlib in the hot path.
3. **`train.py`** fits a torchvision ResNet-18 (ImageNet pretrained, 2-class head) with deterministic 80/20 split, RandomHorizontalFlip(0.5) + ColorJitter, Adam(1e-4, 1e-4), CrossEntropyLoss, batch size 32. Saves `crash_detector.pth` after every best-so-far epoch.
4. **`inference.py`** exposes `predict(audio)` and `predict_stream(audio)`. Loads the checkpoint exactly once per process. Falls back to a deterministic energy-based stub when no checkpoint is present (set `CRASHSENSE_REQUIRE_CHECKPOINT=1` to fail fast instead).

### Phase 2 — TDOA triangulation
- **`sensor_config.py`** pins three sensors at Mumbai highway coordinates. S1 and S3 are toll plazas (drone dispatch origins); S2 is a CCTV pole only.
- **`tdoa_solver.simulate_arrival_times(lat, lon, sensors)`** returns per-sensor arrival times using `geopy.distance.geodesic` divided by `SPEED_OF_SOUND = 343 m/s`.
- **`tdoa_solver.tdoa_localize(time_delays, sensors)`** picks the sensor with smallest arrival time as reference, runs `scipy.optimize.fsolve` from the triangle centroid with TDOA residuals, and rejects solutions outside the sensor-triangle bounding box.
- **`geo_utils.py`** provides `meters_to_latlon_offset`, `bearing_between_two_points`, and `nearest_sensor_to_point` shared by the backend and the drone tracker.

### Phase 3 — Backend API
- **`api/main.py`** — FastAPI app with permissive CORS.
- **`api/ws_manager.py`** — async connection manager (capped at 500 clients, 5 s send timeout, drops failing connections automatically).
- **`api/schemas.py`** — strongly typed `CrashEvent` Pydantic model with all the bounds from the spec (UUID v4 event_id, ISO 8601 timestamp, lat/lon ranges, confidence 0-1, status enum).
- **`api/routes.py`** exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/sensors` | Sensor coordinates |
| `POST` | `/simulate-crash` | TDOA pipeline + WS broadcast |
| `POST` | `/detect-audio` | Audio inference (≤ 10 MB WAV/MP3) |
| `POST` | `/drone-arrived` | Tracker reports drone arrival |
| `GET` | `/drone-status` | Current drone position |
| `WS` | `/ws/events` | Live CrashEvent stream |

### Phase 4 — Dashboard
- **`App.jsx`** — full-screen layout, top status bar that switches MONITORING ↔ CRASH DETECTED based on incoming events.
- **`Map.jsx`** — Leaflet + OpenStreetMap (no API key required) with pulsing-blue sensor dots, drone-dock SVG markers for toll plazas, red pin with bounce, dashed expanding wavefront, and a 50-pin cap.
- **`DroneTracker.jsx`** — `requestAnimationFrame` interpolation over 8 s, live ETA countdown, green arrival pulse, and a `POST /drone-arrived` so all dashboards stay in sync.
- **`AlertPanel.jsx`** — newest-first cards with status badges (yellow DETECTING / orange DRONE DISPATCHED / green DRONE ARRIVED), Simulate Crash button (samples uniformly inside the sensor triangle), Replay Last Event button.
- **`AudioMonitor.jsx`** — Web Audio API AnalyserNode rendering a live waveform of a synthesized dashcam clip.
- **`useWebSocket.js`** — auto-reconnect with 1 s → 30 s exponential backoff, 100-event ring buffer, swallows malformed JSON.

### Phase 5 — Demo orchestration
**`backend/demo_runner.py`** loads the trained checkpoint, plays a real ESC-50 crash clip (or falls back to a synthetic chirp), runs sliding-window inference, picks a random crash inside the sensor triangle, runs the TDOA simulate-then-solve roundtrip, and POSTs to `/simulate-crash`. Total wall time on Apple M-series: ~3 s.

## Configuration

Environment variable | Default | Effect
---|---|---
`CRASHSENSE_REQUIRE_CHECKPOINT` | unset | Set to `1` to refuse to start when `crash_detector.pth` is missing instead of falling back to the energy stub.
`VITE_BACKEND_URL` | `http://127.0.0.1:8000` | Override backend URL used by the dashboard.
`VITE_WS_URL` | derived from `VITE_BACKEND_URL` | Override WebSocket URL.

## Documentation index

- [docs/setup.md](docs/setup.md) — installation, dataset download, training
- [docs/architecture.md](docs/architecture.md) — system diagram + data flow
- [docs/api.md](docs/api.md) — REST + WebSocket reference with examples
- [docs/training.md](docs/training.md) — dataset schema, model details, augmentation, evaluation
- [docs/testing.md](docs/testing.md) — pytest suite + headless dashboard smoke test
- [docs/troubleshooting.md](docs/troubleshooting.md) — common failure modes
