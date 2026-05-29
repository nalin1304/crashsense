#!/usr/bin/env bash
# CrashSense launch script.
# Spins up the FastAPI backend on :8000, the Vite dev server on :5173,
# then fires the demo runner once the backend is ready.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Resolve Python interpreter — prefer the project venv if present.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

echo "[start] using python: $PY"
echo "[start] launching backend on :8000"
(
  cd "$REPO_ROOT"
  "$PY" -m uvicorn backend.api.main:app --reload --port 8000 --host 127.0.0.1
) &
BACKEND_PID=$!

cleanup() {
  echo "[start] stopping..."
  kill "$BACKEND_PID" 2>/dev/null || true
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[start] waiting for backend..."
for i in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[start] backend ready"
    break
  fi
  sleep 0.5
  if [[ $i -eq 40 ]]; then
    echo "[start] backend failed to come up; aborting" >&2
    exit 1
  fi
done

echo "[start] launching frontend dev server on :5173"
(
  cd "$REPO_ROOT/frontend"
  npm run dev
) &
FRONTEND_PID=$!

# Give the frontend a moment to bind, then fire the demo.
sleep 3
echo "[start] running demo_runner"
"$PY" -m backend.demo_runner --backend http://127.0.0.1:8000 || true

echo ""
echo "==============================================="
echo " CrashSense is running:"
echo "  Dashboard: http://127.0.0.1:5173"
echo "  API:       http://127.0.0.1:8000"
echo "  Press Ctrl-C to stop everything."
echo "==============================================="
echo ""

wait
