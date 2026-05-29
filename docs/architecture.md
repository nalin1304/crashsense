# Architecture

```
                                                  ┌──────────────────────────┐
                                                  │  Browser (Dashboard)     │
                                                  │  React + Leaflet         │
                                                  └────────────┬─────────────┘
                                                               │ WS /ws/events
                                                               │ REST /sensors etc.
                                                               ▼
┌──────────────────────────┐    ┌────────────────────────────────────────────────┐
│  demo_runner.py          │───▶│  FastAPI Backend (port 8000)                   │
│   1. play crash audio    │ POST│  ┌────────────────────────────────────────┐  │
│   2. inference           │    │  │  routes.py    /simulate-crash, etc.    │  │
│   3. random crash point  │    │  │  ws_manager   broadcast(message)       │  │
│   4. TDOA simulate+solve │    │  │  schemas.py   CrashEvent (Pydantic)    │  │
│   5. POST CrashEvent     │    │  └────────────────────────────────────────┘  │
└──────────────────────────┘    │              ▲                                │
                                │              │                                │
                                │   ┌──────────┴────────────┐                  │
                                │   │ inference.predict()   │                  │
                                │   │ tdoa_solver.localize()│                  │
                                │   │ sensor_config         │                  │
                                │   └───────────────────────┘                  │
                                └────────────────────────────────────────────────┘
                                                ▲
                                                │  loads at process start
                                                │
                                  ┌─────────────┴────────────────┐
                                  │  audio_model/                │
                                  │  ├─ dataset_prep.py          │
                                  │  ├─ spectrogram_gen.py       │
                                  │  ├─ model.py (ResNet-18)     │
                                  │  ├─ train.py                 │
                                  │  ├─ inference.py             │
                                  │  └─ crash_detector.pth ⬤    │
                                  └──────────────────────────────┘
```

## End-to-end flow when a crash is simulated

1. **User clicks Simulate Crash** in the AlertPanel, or **demo_runner.py** generates a random point inside the sensor triangle.
2. **Frontend / runner** POSTs `{lat, lon}` to `POST /simulate-crash`.
3. **Backend**
   - Calls `simulate_arrival_times(lat, lon, SENSORS)` to compute per-sensor arrival times in seconds.
   - Feeds those times into `tdoa_localize(time_delays, SENSORS)` and recovers an estimated `(lat, lon)`. If the solver rejects the result (out of triangle bbox), the original input is used instead.
   - Builds a `CrashEvent` with `nearest_sensor`, `drone_origin_*` (nearest toll plaza), `eta_seconds=8`, and `status="DETECTED"`.
   - Broadcasts the event to every active `/ws/events` subscriber.
4. **Dashboard** receives the WebSocket message:
   - `Map.jsx` drops a red pin with a bounce animation and a 0→500 m dashed expanding circle.
   - `AlertPanel.jsx` prepends a card with the DETECTING badge.
   - `DroneTracker.jsx` interpolates the drone icon from the toll plaza to the crash pin over 8 seconds using `requestAnimationFrame`. When it arrives, it POSTs to `/drone-arrived`, which broadcasts a new event with `status="DRONE_ARRIVED"` so every connected dashboard updates the badge to green.

## Concurrency model

- **Backend** is asyncio-native (FastAPI / uvicorn). Inference is CPU-bound and runs synchronously in the route handler — fine for a single-operator demo, but a production deployment would offload to a worker pool.
- **WebSocket manager** uses an `asyncio.Lock` to guard the connection list and `asyncio.gather` with per-client timeouts so a single misbehaving client cannot stall the broadcast.
- **Frontend** uses a single WebSocket connection per browser tab. Reconnection backoff is handled in `useWebSocket.js`.

## Single source of truth

`backend/triangulation/sensor_config.py` is imported by:

- The TDOA solver (forward + inverse).
- The REST `/sensors` endpoint.
- `nearest_sensor_to_point`.
- The demo runner (random points inside the triangle).

This keeps the dashboard map markers, the inverse-solver bounding box, and the simulated arrival times in sync without any duplication.
