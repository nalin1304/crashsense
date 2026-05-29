# Troubleshooting

## `pip install -r backend/requirements.txt` fails on torch

The pinned `torch==2.10.0` ships pre-built wheels for Python 3.10 – 3.14 on macOS arm64 / x86_64 and Linux x86_64. If pip can't find a matching wheel:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
```

For NVIDIA GPUs, use the `cu124` index instead.

## `ModuleNotFoundError: No module named 'backend'`

Always run modules with `-m` from the repo root, **not** by file path:

```bash
# Good
python -m backend.audio_model.train
python -m backend.demo_runner

# Bad - the relative imports will break
python backend/audio_model/train.py
```

The exception is `tdoa_demo.py`, which is a top-level script and does its own `sys.path` setup.

## Training is too slow

Each ResNet-18 epoch takes ~7 minutes on CPU. Options:

1. **Stop early.** The trainer saves after every best-so-far epoch, so killing it at any point still leaves a usable checkpoint.
2. **Reduce epochs.** `python -m backend.audio_model.train --epochs 3` is plenty for the ESC-50 dataset.
3. **Use a GPU.** Install the CUDA torch wheels and a single epoch drops to ~30 s.

## `crash_detector.pth` not found at runtime

The backend and demo runner fall back to a deterministic energy-based stub when the checkpoint is missing. The stub is *not* reliable — it's only there so the visual demo can run before training is complete.

To force a hard failure instead:

```bash
export CRASHSENSE_REQUIRE_CHECKPOINT=1
python -m uvicorn backend.api.main:app --port 8000
```

## `/detect-audio` returns 415 / 413

- 415 Unsupported Media Type — the upload's `content-type` is not `audio/wav` / `audio/mpeg` and the filename doesn't end in `.wav` / `.mp3`. Force the type with curl:
  ```bash
  curl -F "file=@clip.wav;type=audio/wav" http://127.0.0.1:8000/detect-audio
  ```
- 413 Request Entity Too Large — the file exceeds 10 MB. Trim it or transcode to MP3.

## Frontend shows "Sensor data unavailable"

Means the backend isn't reachable on `http://127.0.0.1:8000`. Check `curl http://127.0.0.1:8000/health` from the same machine. The dashboard retries `GET /sensors` up to three times at 5-second intervals.

## Mapbox vs Leaflet

The dashboard uses Leaflet + OpenStreetMap by default — no API key required. Mapbox GL JS and the geocoder are installed via `package.json` so you can swap in a Mapbox token by extending `frontend/src/components/Map.jsx`.

## ESC-50 download fails

The download URL is `https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip`. If that's unreachable from your network, you can clone the repo manually and place its contents at `.cache/esc50/ESC-50-master/`:

```bash
git clone --depth 1 https://github.com/karoldvl/ESC-50.git .cache/esc50/ESC-50-master
```

Then re-run `python -m backend.audio_model.dataset_prep`.

## Port 8000 already in use

Either stop the conflicting process (`lsof -i :8000`) or override the port:

```bash
python -m uvicorn backend.api.main:app --port 8001
# and tell the frontend
VITE_BACKEND_URL=http://127.0.0.1:8001 npm run dev --prefix frontend
```

## Demo runner timing

`demo_runner.py` reports `end-to-end pipeline complete in <X>s`. On a cold start (model loading from disk) the first run is ~5 s; subsequent runs in the same Python process drop to ~1 s.
