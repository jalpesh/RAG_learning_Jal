"""Measure how TTFT and throughput degrade under concurrent load.

    uv run python -m bench.concurrency --levels 1,2,4,8
    uv run python -m bench.concurrency --levels 1 --think

A burst of N simultaneous requests is the worst case, which is the number
worth planning capacity against - sustained arrival at the same rate is
always kinder than this.

Prompts are built once from real retrieved context (different question per
request, so only the system prompt prefix is shared - the realistic case
for N different users asking N different things).
"""
from __future__ import annotations
import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import httpx

from harness.spans import Run
from rag.config import Config
from rag.pipeline import Pipeline
from rag.generate.prompt import build_prompt

ROOT = Path(__file__).resolve().parents[1]


def build_prompts(cfg: Config, n: int) -> list[list[dict]]:
    """Real retrieved context, one distinct question per request."""
    questions = json.loads((ROOT / "harness" / "regression.json").read_text())["questions"]
    with Run(config="concurrency-prep", meta=cfg.as_dict()) as run:
        pipe = Pipeline(cfg, run)
        pipe.ingest(ROOT / "corpus")
        prompts = []
        for i in range(n):
            q = questions[i % len(questions)]["q"]
            qv = pipe.embedder.encode([q])[0]
            dense = pipe.index.dense(qv, cfg.candidates)
            hits = [pipe.index.chunks[j] for j, _ in dense[: cfg.top_k]]
            prompts.append(build_prompt(q, hits))
    return prompts


def fire(cfg: Config, messages: list[dict], think: bool, out: list) -> None:
    t0 = time.perf_counter()
    ttft = None
    n_content = 0
    n_reasoning = 0
    with httpx.Client(base_url=cfg.llama_host, timeout=600) as client:
        with client.stream("POST", "/v1/chat/completions", json={
            "model": cfg.gen_model,
            "messages": messages,
            "stream": True,
            "max_tokens": cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                ev = json.loads(payload)
                choices = ev.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if delta.get("reasoning_content"):
                    n_reasoning += 1
                if delta.get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    n_content += 1
    total = (time.perf_counter() - t0) * 1000
    out.append({"ttft_ms": ttft or total, "total_ms": total,
                "n_content": n_content, "n_reasoning": n_reasoning})


def run_level(cfg: Config, prompts: list[list[dict]], level: int, think: bool) -> dict:
    out: list[dict] = []
    threads = [threading.Thread(target=fire, args=(cfg, prompts[i], think, out))
               for i in range(level)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    ttfts = sorted(r["ttft_ms"] for r in out)
    total_tok = sum(r["n_content"] + r["n_reasoning"] for r in out)
    return {
        "level": level,
        "ttft_p50": statistics.median(ttfts),
        "ttft_max": max(ttfts),
        "wall_s": wall,
        "aggregate_tok_s": total_tok / wall,
        "reasoning_tok_median": statistics.median(r["n_reasoning"] for r in out),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--think", action="store_true", help="enable model reasoning")
    ap.add_argument("--host", default="http://127.0.0.1:8082")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()

    levels = [int(x) for x in a.levels.split(",")]
    cfg = Config(name="concurrency", embed_device="mps", embed_batch=128,
                 gen_backend="llamacpp", gen_model=a.model, llama_host=a.host,
                 max_tokens=a.max_tokens)

    prompts = build_prompts(cfg, max(levels))
    fire(cfg, prompts[0], a.think, [])   # warm the server; level 1 is otherwise a cold-start number
    print(f"\nmodel={a.model}  thinking={'ON' if a.think else 'off'}  "
          f"(burst of N simultaneous requests)\n")
    print(f"{'concurrent':>10} {'TTFT p50':>10} {'TTFT max':>10} "
          f"{'wall s':>8} {'agg tok/s':>10} {'think tok':>10}")
    for level in levels:
        r = run_level(cfg, prompts, level, a.think)
        print(f"{r['level']:>10} {r['ttft_p50']:>9.0f}ms {r['ttft_max']:>9.0f}ms "
              f"{r['wall_s']:>8.1f} {r['aggregate_tok_s']:>10.1f} "
              f"{r['reasoning_tok_median']:>10.0f}")


if __name__ == "__main__":
    main()
