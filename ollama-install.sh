#!/usr/bin/env bash
set -euo pipefail

# Install Ollama entirely inside this project when it is not already present.
# This script deliberately does not invoke the system installer or sudo.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/.tools/ollama"
OLLAMA_BIN="$TARGET/bin/ollama"

if [[ -x "$OLLAMA_BIN" ]]; then
  "$OLLAMA_BIN" --version
  echo "Ollama is already installed at $TARGET"
  exit 0
fi

TMP="$ROOT/.tmp/ollama-install"
rm -rf "$TMP"
mkdir -p "$TMP/extracted" "$TARGET"
trap 'rm -rf "$TMP"' EXIT

ARCHIVE="$TMP/ollama-linux-amd64.tgz"
curl --fail --show-error --location \
  'https://ollama.com/download/ollama-linux-amd64.tgz' \
  --output "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$TMP/extracted"
cp -a "$TMP/extracted/bin" "$TARGET/"
cp -a "$TMP/extracted/lib" "$TARGET/"

"$OLLAMA_BIN" --version
echo "Ollama installed at $TARGET"
