from __future__ import annotations
import json
import time
import httpx
from rag.config import Config


def stream_answer(cfg: Config, messages: list[dict]) -> dict:
    """Stream from llama-server's OpenAI-compatible endpoint. Same output
    contract as rag.generate.ollama.stream_answer, so Pipeline doesn't care
    which backend it's talking to - only Config does.

    llama-server reports its own prompt/decode timings server-side (more
    precise than our wall-clock TTFT, which also includes HTTP/JSON
    overhead), so those ride along as extra fields when present.
    """
    t0 = time.perf_counter()
    ttft = None
    parts: list[str] = []
    n_tok = 0
    server_timings: dict = {}

    with httpx.Client(base_url=cfg.llama_host, timeout=600) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "model": cfg.gen_model,
            "messages": messages,
            "stream": True,
            "max_tokens": cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": cfg.gen_think},
            "stream_options": {"include_usage": True},
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                if ev.get("timings"):
                    server_timings = ev["timings"]
                choices = ev.get("choices") or []
                if not choices:
                    continue
                piece = choices[0].get("delta", {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    parts.append(piece)
                    n_tok += 1

    total = (time.perf_counter() - t0) * 1000
    gen_ms = max(total - (ttft or 0), 1e-6)
    out = {
        "text": "".join(parts),
        "ttft_ms": round(ttft or total, 2),
        "total_ms": round(total, 2),
        "n_chunks_streamed": n_tok,
        "tok_per_s": round(n_tok / (gen_ms / 1000), 2),
    }
    if server_timings:
        out["server_prompt_ms"] = server_timings.get("prompt_ms")
        out["server_prompt_tok_per_s"] = server_timings.get("prompt_per_second")
        out["server_predict_ms"] = server_timings.get("predicted_ms")
        out["server_predict_tok_per_s"] = server_timings.get("predicted_per_second")
    return out
