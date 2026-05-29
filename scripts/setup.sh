#!/usr/bin/env bash
# CrashSense one-time setup.
# Creates a Python venv, installs all deps, downloads ESC-50, generates
# spectrograms, trains the audio classifier, and installs frontend deps.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

step() { printf "\n\033[1;36m[setup]\033[0m %s\n" "$*"; }

step "Creating Python venv at .venv (skips if it already exists)"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
PY="$REPO_ROOT/.venv/bin/python"

step "Upgrading pip"
"$PY" -m pip install --upgrade pip wheel >/dev/null

step "Installing backend dependencies (this can take 5-10 minutes)"
"$PY" -m pip install -r backend/requirements.txt

step "Installing frontend dependencies"
( cd frontend && npm install )

step "Downloading dataset (ESC-50, ~518 MB; cached at .cache/esc50/)"
"$PY" -m backend.audio_model.dataset_prep --target 600

step "Generating mel spectrograms"
"$PY" -m backend.audio_model.spectrogram_gen

if [[ ! -f backend/audio_model/crash_detector.pth ]]; then
  step "Training the audio classifier (CPU: ~7 minutes per epoch)"
  step "  (Saved automatically every epoch; safe to interrupt with Ctrl-C)"
  "$PY" -m backend.audio_model.train --epochs 3 || true
else
  step "Reusing existing checkpoint at backend/audio_model/crash_detector.pth"
fi

step "Done. Run ./start.sh to launch the demo."
