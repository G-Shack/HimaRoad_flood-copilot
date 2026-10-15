#!/bin/bash
# Start the web app with the copilot API (port 8000) and the analysis workbench
# for new live runs (port 8501) on this computer. Stop both with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export PYTHONPATH="$PWD"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/lib}"
export COPILOT_API_URL=http://127.0.0.1:8000
STREAMLIT=(--server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false)

.venv/bin/uvicorn web.server:app --host 127.0.0.1 --port 8000 &
.venv/bin/streamlit run app.py --server.port 8501 "${STREAMLIT[@]}" &
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM
echo "App: http://127.0.0.1:8000/?run=trishuli-2026-v3.0-r2"
echo "Workbench for new live analyses: http://127.0.0.1:8501"
wait
