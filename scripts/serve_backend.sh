#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/backend"
export HF_HOME="${HF_HOME:-$ROOT/.model_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$ROOT/.model_cache/hub}"

exec "$ROOT/.venv/bin/uvicorn" app.main:app --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8010}"
