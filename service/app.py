"""Resident FastAPI service: semantic cache -> router -> two generation
lanes -> confidence-gated auto-escalation, sharing one retrieval layer.

    uv run uvicorn service.app:app --port 8000

Prerequisites - two llama-server processes, already running:
    llama-server -m models/qwen3-4b-official.gguf --host 127.0.0.1 --port 8082 -ngl 99 -c 4096  # fast
    llama-server -m models/qwen3-8b-official.gguf --host 127.0.0.1 --port 8081 -ngl 99 -c 4096  # deep

Request flow for POST /query:
    1. Semantic cache check (embedding similarity, no LLM call at all on a hit)
    2. Router (rag.route.classify) - lexical + matched-source signals, no LLM call
    3. Fast lane answers synchronously, OR deep lane is enqueued and a job id
       returned immediately (never blocks the event loop - both lanes run via
       asyncio.to_thread so concurrent requests aren't serialized on one)
    4. Fast-lane answers are checked for low confidence (rag.confidence -
       ungrounded numbers or hedging language). A flagged answer is still
       returned immediately (never silently delayed), but a deep-lane
       verification job is kicked off in the background and its result
       replaces the cache entry once done.

Auth: set RAG_API_KEY before binding to anything but localhost. See
docs/02-findings.md's LAN pitfalls - the local-dev default (unset) has no
auth at all.
"""
from __future__ import annotations
import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from harness.spans import Run
from rag.cache import SemanticCache
from rag.confidence import is_low_confidence
from rag.config import Config
from rag.pipeline import Pipeline
from rag.route import classify

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
API_KEY = os.environ.get("RAG_API_KEY")

FAST_CFG = Config(name="service-fast", embed_device="mps", embed_batch=128,
                   gen_backend="llamacpp", gen_model="qwen3-4b",
                   llama_host="http://127.0.0.1:8082",
                   metadata_filter=True, gen_think=False, max_tokens=400)
DEEP_CFG = Config(name="service-deep", embed_device="mps", embed_batch=128,
                   gen_backend="llamacpp", gen_model="qwen3-8b",
                   llama_host="http://127.0.0.1:8081",
                   metadata_filter=True, gen_think=True, max_tokens=2000)

app = FastAPI(title="local-rag-lab")
pipelines: dict[str, Pipeline] = {}
jobs: dict[str, dict] = {}
cache = SemanticCache()


def _auth(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "missing or invalid API key")


def _strip_chunks(result: dict) -> dict:
    """chunks ride along internally for the confidence check; never hand
    raw chunk text back over the API - sources (filename + page) is the
    citation a caller needs, not a second copy of the retrieved text."""
    return {k: v for k, v in result.items() if k != "chunks"}


@app.on_event("startup")
def startup() -> None:
    fast_run = Run(config="service-fast")
    fast = Pipeline(FAST_CFG, fast_run)
    fast.ingest(CORPUS)

    # Deep lane reuses the fast lane's embedder + index - retrieval doesn't
    # depend on which generation model answers. Only cfg/run differ.
    deep_run = Run(config="service-deep")
    deep = Pipeline(DEEP_CFG, deep_run)
    deep.embedder = fast.embedder
    deep.index = fast.index

    pipelines["fast"] = fast
    pipelines["deep"] = deep


class QueryBody(BaseModel):
    question: str
    force_lane: Literal["fast", "deep"] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "lanes": list(pipelines.keys()), "cache_size": len(cache)}


@app.post("/query")
async def query(body: QueryBody, authorization: str | None = Header(None)) -> dict:
    _auth(authorization)
    fast = pipelines["fast"]

    qvec = await asyncio.to_thread(lambda: fast.embedder.encode([body.question])[0])
    if body.force_lane is None:
        hit = cache.get(qvec)
        if hit is not None:
            return {**_strip_chunks(hit), "lane": "cache"}

    lane = body.force_lane or classify(body.question, fast.index)

    if lane == "fast":
        t0 = time.perf_counter()
        res = await asyncio.to_thread(fast.query, body.question)
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)

        low_conf, reason = is_low_confidence(res["answer"], res["chunks"],
                                              question=body.question, idx=fast.index,
                                              sources_used=res["sources"])
        out = {"lane": "fast", "low_confidence": low_conf, **_strip_chunks(res)}
        if low_conf:
            out["low_confidence_reason"] = reason
            job_id = uuid.uuid4().hex[:12]
            jobs[job_id] = {"status": "running", "question": body.question,
                            "trigger": "auto-escalated: " + reason}
            out["verify_job"] = job_id
            asyncio.create_task(_run_deep(job_id, body.question, qvec, update_cache_only_if_better=True))
        else:
            cache.put(body.question, qvec, out)
        return out

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "question": body.question}
    asyncio.create_task(_run_deep(job_id, body.question, qvec))
    return {"lane": "deep", "job_id": job_id, "status": "running", "poll": f"/jobs/{job_id}"}


async def _run_deep(job_id: str, question: str, qvec, update_cache_only_if_better: bool = False) -> None:
    t0 = time.perf_counter()
    try:
        res = await asyncio.to_thread(pipelines["deep"].query, question)
        out = {"lane": "deep", **_strip_chunks(res)}
        jobs[job_id] = {"status": "done", "elapsed_s": round(time.perf_counter() - t0, 2), **out}
        # A verification job's whole point is to leave a trustworthy answer
        # in the cache for the next person who asks something similar -
        # the fast lane's shaky first answer never got cached in the first
        # place (see /query), so this is a plain write, not an overwrite race.
        cache.put(question, qvec, out)
    except Exception as exc:
        jobs[job_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(None)) -> dict:
    _auth(authorization)
    if job_id not in jobs:
        raise HTTPException(404, "no such job")
    return jobs[job_id]


@app.post("/ingest")
async def ingest(authorization: str | None = Header(None)) -> dict:
    """Re-embeds the whole corpus. Not incremental - see docs/02-findings.md
    Open Items. Clears the cache too: a stale cached answer citing an
    old version of a document is worse than a cache miss."""
    _auth(authorization)
    await asyncio.to_thread(pipelines["fast"].ingest, CORPUS)
    pipelines["deep"].embedder = pipelines["fast"].embedder
    pipelines["deep"].index = pipelines["fast"].index
    cache.clear()
    return {"status": "reingested"}
