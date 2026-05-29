# API Reference

Base URL: `http://127.0.0.1:8000`

## REST Endpoints

### `GET /health`
Liveness probe.

```bash
$ curl http://127.0.0.1:8000/health
{"status":"ok"}
```

### `GET /sensors`
Returns all configured sensors.

```bash
$ curl http://127.0.0.1:8000/sensors
[
  {"id":"S1","name":"Toll Plaza Alpha","lat":19.1136,"lon":72.8697,"is_toll_plaza":true},
  {"id":"S2","name":"CCTV Pole B12",  "lat":19.1089,"lon":72.8812,"is_toll_plaza":false},
  {"id":"S3","name":"Toll Plaza Beta", "lat":19.1201,"lon":72.8754,"is_toll_plaza":true}
]
```

### `POST /simulate-crash`
Triggers the full TDOA pipeline and broadcasts a `CrashEvent`.

```bash
$ curl -X POST http://127.0.0.1:8000/simulate-crash \
    -H "Content-Type: application/json" \
    -d '{"lat":19.1145,"lon":72.876}'
{
  "event_id": "90991c7e-0616-432a-b11a-a1cb67073bda",
  "timestamp": "2026-05-27T22:38:39.896999+00:00",
  "crash_lat": 19.11450000066017,
  "crash_lon": 72.87600000044743,
  "confidence": 0.95,
  "nearest_sensor": "S3",
  "drone_origin_lat": 19.1201,
  "drone_origin_lon": 72.8754,
  "eta_seconds": 8,
  "status": "DETECTED"
}
```

Validation: `lat` ∈ [−90, 90], `lon` ∈ [−180, 180].

### `POST /detect-audio`
Runs the audio classifier on an uploaded WAV/MP3.

```bash
$ curl -X POST http://127.0.0.1:8000/detect-audio \
    -F "file=@data/raw_audio/crash/esc_2-144137-A-43.wav;type=audio/wav"
{"event":"CRASH","confidence":0.9999169111251831}
```

Limits: ≤ 10 MB, format must be `audio/wav` / `audio/mpeg` (or filename ending in `.wav` / `.mp3`).

### `POST /drone-arrived`
Called by the Drone_Tracker when the drone reaches the crash pin. Body is the original `CrashEvent`. Broadcasts an updated event with `status="DRONE_ARRIVED"`.

### `GET /drone-status`
Returns the current drone position.

```json
{"lat": 19.1201, "lon": 72.8754, "status": "in_transit"}
```

`status` is one of `idle`, `in_transit`, or `arrived`.

## WebSocket: `/ws/events`

Subscribers receive every `CrashEvent` (in either `DETECTED`, `DRONE_DISPATCHED`, or `DRONE_ARRIVED` state). One JSON object per message. Example using `websockets`:

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect('ws://127.0.0.1:8000/ws/events') as ws:
        async for msg in ws:
            event = json.loads(msg)
            print(event['event_id'], event['status'])

asyncio.run(main())
```

The manager:

- Caps active connections at 500.
- Removes a client automatically on a normal disconnect, an error, or a 5-second send timeout.
- Returns silently when the connection list is empty (no exceptions on broadcast).

## CrashEvent schema

| Field | Type | Constraints |
|---|---|---|
| `event_id` | string | UUID v4, exactly 36 characters |
| `timestamp` | string (ISO 8601) | UTC |
| `crash_lat` | float | [−90, 90] |
| `crash_lon` | float | [−180, 180] |
| `confidence` | float | [0, 1] |
| `nearest_sensor` | string | length 1–64 |
| `drone_origin_lat` | float | [−90, 90] |
| `drone_origin_lon` | float | [−180, 180] |
| `eta_seconds` | int | [0, 86400] |
| `status` | enum | `DETECTED` / `DRONE_DISPATCHED` / `DRONE_ARRIVED` |
