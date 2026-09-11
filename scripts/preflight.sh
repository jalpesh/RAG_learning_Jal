#!/usr/bin/env bash
# Sanity-check the machine before trusting a single benchmark number.
set -euo pipefail

REQUIRED_MODELS=("qwen3:4b" "qwen3:8b")

echo "== hardware =="
sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m
if memory_bytes=$(sysctl -n hw.memsize 2>/dev/null); then
  echo "memory GB: $(( memory_bytes / 1073741824 ))"
else
  echo "memory GB: unavailable (non-macOS host)"
fi

echo -e "\n== ollama =="
command -v ollama >/dev/null 2>&1 || { echo "MISSING: install Ollama"; exit 1; }
ollama --version
curl --fail --silent --show-error http://127.0.0.1:11434/api/tags >/dev/null || {
  echo "server: DOWN (run: ollama serve)"
  exit 1
}
echo "server: up"
ollama list

missing=0
for model in "${REQUIRED_MODELS[@]}"; do
  if ollama show "${model}" >/dev/null 2>&1; then
    echo "model: ${model} ready"
  else
    echo "model: ${model} MISSING"
    missing=1
  fi
done
if (( missing )); then
  echo "Run ./scripts/setup_ollama.sh to install required models."
  exit 1
fi

echo -e "\n== metal / torch mps =="
python3 - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "mps available:", torch.backends.mps.is_available())
except Exception as e:
    print("torch not installed yet:", e)
PY

echo -e "\n== memory budget check =="
echo "8B q4_K_M ~5.0GB + KV@8k ~1.0GB + embedder ~0.1GB + python ~1.5GB = ~7.6GB"
echo "On 18GB that is comfortable. Do NOT also load a 14B - swap makes every number noise."

echo -e "\n== keep_alive =="
echo "Pinned configs send keep_alive=-1s in each /api/chat request; no shell variable is required."
