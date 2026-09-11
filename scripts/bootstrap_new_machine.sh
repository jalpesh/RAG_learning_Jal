#!/usr/bin/env bash
# Idempotent macOS bootstrap for the complete local RAG stack.
# Re-run safely: completed installs/downloads and healthy services are skipped.
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_DIR="${ROOT_DIR}/.runtime"
LOG_DIR="${ROOT_DIR}/logs"
MODEL_DIR="${ROOT_DIR}/models"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

step=0
status() {
  step=$((step + 1))
  printf '\n[%d] %s\n' "${step}" "$1"
}

run() {
  if (( DRY_RUN )); then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

http_ready() {
  curl --fail --silent --max-time 2 "$1" >/dev/null 2>&1
}

wait_for_http() {
  local name=$1 url=$2 attempts=${3:-60}
  if (( DRY_RUN )); then
    echo "DRY RUN: would wait for ${name} at ${url}"
    return
  fi
  for ((i = 1; i <= attempts; i++)); do
    if http_ready "${url}"; then
      echo "ready: ${name}"
      return
    fi
    printf '\rwaiting for %s (%d/%d)' "${name}" "${i}" "${attempts}"
    sleep 2
  done
  printf '\n' >&2
  echo "ERROR: ${name} did not become ready; inspect ${LOG_DIR}" >&2
  exit 1
}

start_background() {
  local name=$1 pid_file=$2 log_file=$3
  shift 3
  if (( DRY_RUN )); then
    printf 'DRY RUN: start %s:' "${name}"
    printf ' %q' "$@"
    printf '\n'
    return
  fi
  nohup "$@" >>"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${pid_file}"
  echo "started: ${name} (pid ${pid}, log ${log_file})"
}

download_gguf() {
  local label=$1 url=$2 destination=$3 expected_bytes=$4
  local size=0
  if [[ -f "${destination}" ]]; then
    size=$(stat -f '%z' "${destination}" 2>/dev/null || stat -c '%s' "${destination}" 2>/dev/null || echo 0)
  fi
  if (( size == expected_bytes )); then
    echo "present: ${label} ($(du -h "${destination}" | awk '{print $1}'))"
    return
  fi
  echo "downloading/resuming: ${label}"
  if (( DRY_RUN )); then
    echo "DRY RUN: curl --location --continue-at - ${url}"
    return
  fi
  curl --fail --location --progress-bar --continue-at - \
    --output "${destination}.part" "${url}"
  size=$(stat -f '%z' "${destination}.part" 2>/dev/null || stat -c '%s' "${destination}.part" 2>/dev/null || echo 0)
  if (( size != expected_bytes )); then
    echo "ERROR: ${label} download is ${size} bytes; expected ${expected_bytes}." >&2
    echo "The partial file was kept so the next run can resume it." >&2
    exit 1
  fi
  mv "${destination}.part" "${destination}"
  echo "downloaded: ${label} ($(du -h "${destination}" | awk '{print $1}'))"
}

status "Checking platform and package manager"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this bootstrap currently supports macOS; use the README for Linux setup." >&2
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is required. Install it from https://brew.sh and rerun." >&2
  exit 1
fi

status "Installing missing command-line prerequisites"
for spec in "uv:uv" "ollama:ollama" "llama-server:llama.cpp"; do
  command_name=${spec%%:*}
  formula=${spec#*:}
  if command -v "${command_name}" >/dev/null 2>&1; then
    echo "present: ${command_name}"
  else
    echo "installing: ${formula}"
    run brew install "${formula}"
  fi
done

run mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}" "${MODEL_DIR}"

status "Ensuring Ollama is running"
if http_ready "http://127.0.0.1:11434/api/tags"; then
  echo "already running: Ollama"
else
  start_background "Ollama" "${RUNTIME_DIR}/ollama.pid" "${LOG_DIR}/ollama.log" ollama serve
  wait_for_http "Ollama" "http://127.0.0.1:11434/api/tags" 30
fi

status "Installing required Ollama models before starting the RAG services"
if (( DRY_RUN )); then
  echo "DRY RUN: ./scripts/setup_ollama.sh"
else
  "${ROOT_DIR}/scripts/setup_ollama.sh"
fi

status "Synchronizing the locked Python environment"
run uv sync --locked --project "${ROOT_DIR}"

status "Installing official llama.cpp GGUF models (downloads resume if interrupted)"
download_gguf "Qwen3 4B Q4_K_M" \
  "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf" \
  "${MODEL_DIR}/qwen3-4b-official.gguf" 2497280256
download_gguf "Qwen3 8B Q4_K_M" \
  "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf" \
  "${MODEL_DIR}/qwen3-8b-official.gguf" 5027783488

status "Ensuring the fast llama.cpp lane is running"
if http_ready "http://127.0.0.1:8082/health"; then
  echo "already running: llama.cpp fast lane on port 8082"
else
  start_background "llama.cpp fast lane" "${RUNTIME_DIR}/llama-fast.pid" \
    "${LOG_DIR}/llama-fast.log" llama-server \
    -m "${MODEL_DIR}/qwen3-4b-official.gguf" --host 127.0.0.1 \
    --port 8082 -ngl 99 -c 4096
  wait_for_http "llama.cpp fast lane" "http://127.0.0.1:8082/health" 60
fi

status "Ensuring the deep llama.cpp lane is running"
if http_ready "http://127.0.0.1:8081/health"; then
  echo "already running: llama.cpp deep lane on port 8081"
else
  start_background "llama.cpp deep lane" "${RUNTIME_DIR}/llama-deep.pid" \
    "${LOG_DIR}/llama-deep.log" llama-server \
    -m "${MODEL_DIR}/qwen3-8b-official.gguf" --host 127.0.0.1 \
    --port 8081 -ngl 99 -c 4096
  wait_for_http "llama.cpp deep lane" "http://127.0.0.1:8081/health" 60
fi

status "Checking the corpus"
document_count=$(find "${ROOT_DIR}/corpus" -maxdepth 1 -type f \( -iname '*.pdf' -o -iname '*.docx' \) | wc -l | tr -d ' ')
if (( document_count == 0 )); then
  echo "ACTION REQUIRED: no PDF/DOCX files are present in ${ROOT_DIR}/corpus"
  echo "Copy documents there and rerun this script; completed work will be skipped."
  exit 0
fi
echo "found: ${document_count} document(s)"

status "Ensuring the authless localhost RAG API is running and ingested"
if http_ready "http://127.0.0.1:8000/health"; then
  echo "already running: API; refreshing the corpus index"
  if (( DRY_RUN )); then
    echo "DRY RUN: POST http://127.0.0.1:8000/ingest"
  else
    curl --fail --silent --show-error --request POST \
      "http://127.0.0.1:8000/ingest" >/dev/null
    echo "complete: corpus re-ingested"
  fi
else
  start_background "RAG API" "${RUNTIME_DIR}/api.pid" "${LOG_DIR}/api.log" \
    uv run --directory "${ROOT_DIR}" uvicorn service.app:app \
    --host 127.0.0.1 --port 8000
  wait_for_http "RAG API (startup includes ingestion)" "http://127.0.0.1:8000/health" 180
fi

status "Final status"
echo "Ollama:    http://127.0.0.1:11434"
echo "Fast lane: http://127.0.0.1:8082"
echo "Deep lane: http://127.0.0.1:8081"
echo "RAG API:   http://127.0.0.1:8000"
echo "Logs:      ${LOG_DIR}"
echo "Bootstrap complete. Re-running this script is safe."
