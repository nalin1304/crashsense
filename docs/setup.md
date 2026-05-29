# Setup

CrashSense has been verified on macOS / Apple Silicon with **Python 3.14** and **Node.js 25**. It should work on any Python 3.10+ and Node 18+ install with internet access (for the one-time dataset download and pretrained ResNet-18 weights).

## Prerequisites

```bash
python3 --version    # 3.10 - 3.14
node --version       # 18+
npm --version        # 9+
```

If you are on macOS via Homebrew:

```bash
brew install python@3.14 node
```

## Automated setup

```bash
./scripts/setup.sh
```

That script does, in order:

1. Creates a virtual environment at `.venv/`.
2. Installs every backend dependency from `backend/requirements.txt` (~3 GB, mostly torch).
3. Installs frontend dependencies via `npm install`.
4. Downloads ESC-50 (~518 MB compressed, ~1.4 GB extracted) into `.cache/esc50/`.
5. Copies six categories into `data/raw_audio/{crash,noise}/` and applies six augmentations until each class has 560 samples.
6. Generates 1,120 mel spectrograms at `data/spectrograms/{crash,noise}/`.
7. Trains a ResNet-18 for 3 epochs (≥ 99 % validation accuracy on the first epoch in our runs).

You can re-run the script safely; it skips work that is already done.

## Manual setup

If you prefer to run each step individually:

```bash
# 1. Python venv + dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

# 2. Frontend dependencies
cd frontend && npm install && cd ..

# 3. Dataset preparation
python -m backend.audio_model.dataset_prep --target 600

# 4. Spectrogram generation
python -m backend.audio_model.spectrogram_gen

# 5. Training
python -m backend.audio_model.train --epochs 3
```

## Dataset notes

The original spec asked for Google AudioSet via the `audioset-download` package. In practice that pipeline is fragile: many videos referenced in the AudioSet ontology are deleted, the package depends on `youtube-dl` which is rate-limited, and partial downloads leave the dataset directory in an unusable state.

We therefore pivoted to ESC-50, which is a stable Creative Commons dataset hosted on GitHub. The category mapping was chosen for proximity to highway crash audio characteristics:

| Class | ESC-50 categories |
|---|---|
| `crash` | `car_horn`, `engine`, `breaking_glass` |
| `noise` | `wind`, `rain`, `crackling_fire` |

To reach the ≥ 500-per-class minimum, we apply six deterministic augmentations to each source clip:

- pitch shift +2 semitones
- pitch shift −2 semitones
- time stretch 0.9×
- time stretch 1.1×
- additive Gaussian noise at 18 dB SNR
- additive Gaussian noise at 12 dB SNR

This yields 560 clips per class — well above the 500 minimum required by Requirement 1.4 and balanced to within zero percent (Requirement 1.4 also caps the imbalance at 10 %).

## GPU acceleration (optional)

The default `torch==2.10.0` wheels on PyPI are CPU-only on macOS. On Linux with an NVIDIA GPU you can swap them for a CUDA build:

```bash
pip install torch==2.10.0+cu124 torchvision==0.25.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

Training drops from ~7 min/epoch to ~30 s/epoch on a modest GPU.

## Troubleshooting

See [docs/troubleshooting.md](troubleshooting.md).
