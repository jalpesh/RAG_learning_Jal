from __future__ import annotations
import json
import time
import httpx
from rag.config import Config


def stream_answer(cfg: Config, messages: list[dict]) -> dict:
    """Stream and measure. Returns TTFT separately from total, because
    perceived latency is time-to-first-token, not wall time."""
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    n_tok = 0

    with httpx.Client(base_url=cfg.ollama_host, timeout=600) as client:
        with client.stream("POST", "/api/chat", json={
            "model": cfg.gen_model,
            "messages": messages,
            "stream": True,
            "keep_alive": cfg.gen_keep_alive,
            "think": cfg.gen_think,
            "options": {"num_predict": cfg.max_tokens},
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                ev = json.loads(line)
                piece = ev.get("message", {}).get("content", "")
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    parts.append(piece)
                    n_tok += 1
                if ev.get("done"):
                    break

    total = (time.perf_counter() - t0) * 1000
    gen_ms = max(total - (ttft or 0), 1e-6)
    return {
        "text": "".join(parts),
        "ttft_ms": round(ttft or total, 2),
        "total_ms": round(total, 2),
        "n_chunks_streamed": n_tok,
        "tok_per_s": round(n_tok / (gen_ms / 1000), 2),
    }
