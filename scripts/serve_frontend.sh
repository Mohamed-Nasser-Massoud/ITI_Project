#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8010}"

exec "$ROOT/.venv/bin/streamlit" run "$ROOT/frontend/app.py" \
  --server.headless true \
  --server.address "${UI_HOST:-127.0.0.1}" \
  --server.port "${UI_PORT:-8501}"
