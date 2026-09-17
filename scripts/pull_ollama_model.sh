#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_BIN="$ROOT/.tools/ollama/bin/ollama"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$ROOT/.ollama_models}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "Local Ollama binary is missing at $OLLAMA_BIN" >&2
  exit 1
fi

"$OLLAMA_BIN" pull "${OLLAMA_MODEL:-llama3.2:3b}"
