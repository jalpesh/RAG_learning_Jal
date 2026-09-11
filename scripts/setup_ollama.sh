#!/usr/bin/env bash
# Install the Ollama models required by rag/config.py.
# Ollama must already be running; model keep-alive is sent per request by Python.
set -euo pipefail

OLLAMA_HOST_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
REQUIRED_MODELS=("qwen3:4b" "qwen3:8b")

command -v ollama >/dev/null 2>&1 || {
  echo "ERROR: Ollama is not installed. On macOS: brew install ollama" >&2
  exit 1
}

curl --fail --silent --show-error "${OLLAMA_HOST_URL}/api/tags" >/dev/null || {
  echo "ERROR: Ollama is not reachable at ${OLLAMA_HOST_URL}. Start it with: ollama serve" >&2
  exit 1
}

for model in "${REQUIRED_MODELS[@]}"; do
  if ollama show "${model}" >/dev/null 2>&1; then
    echo "present: ${model}"
  else
    echo "pulling: ${model}"
    ollama pull "${model}"
  fi
done

if [[ "${1:-}" == "--with-embed" ]]; then
  if ollama show nomic-embed-text >/dev/null 2>&1; then
    echo "present: nomic-embed-text"
  else
    echo "pulling: nomic-embed-text"
    ollama pull nomic-embed-text
  fi
fi

echo "Ollama setup complete. Pinned benchmark configs send keep_alive=-1s per request."
echo "Note: Ollama models are not GGUF files for llama-server; the full bootstrap installs those separately."
