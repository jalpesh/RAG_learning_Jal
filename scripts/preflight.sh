#!/usr/bin/env bash
# Sanity-check the machine before trusting a single benchmark number.
set -u
echo "== hardware =="
sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m
echo "memory GB: $(( $(sysctl -n hw.memsize) / 1073741824 ))"

echo -e "\n== ollama =="
ollama --version || { echo "MISSING"; exit 1; }
curl -s http://127.0.0.1:11434/api/tags >/dev/null && echo "server: up" || echo "server: DOWN (run: ollama serve)"
ollama list

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
echo "For pinned configs: export OLLAMA_KEEP_ALIVE=-1 and restart ollama serve."
