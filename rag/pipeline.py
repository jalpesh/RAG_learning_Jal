from __future__ import annotations
from pathlib import Path
import numpy as np

from harness.spans import Run
from rag.config import Config
from rag.ingest.parse import parse_dir
from rag.ingest.chunk import chunk_pages
from rag.ingest.structured import parse_structured_dir, chunk_structured
from rag.ingest.embed import Embedder
from rag.retrieve.index import FlatIndex
from rag.retrieve.rerank import Reranker
from rag.confidence import conflicting_day_dates
from rag.generate import build_prompt, stream_answer


class Pipeline:
    """Resident RAG. Built once, queried many times.

    This object existing at all is the single biggest optimisation in the
    project: it holds the embedder in memory, so a query never pays a model
    load or a `import torch`. A CLI that constructs this per question can
    never be fast, no matter what else you tune.
    """

    def __init__(self, cfg: Config, run: Run):
        self.cfg, self.run = cfg, run
        self.embedder = Embedder(cfg)
        self.reranker = Reranker(cfg) if cfg.reranker else None
        self.index: FlatIndex | None = None

    # ---------- ingest ----------
    def ingest(self, corpus: Path) -> None:
        cfg, run = self.cfg, self.run

        with run.span("embed_model_load", backend=cfg.embed_backend,
                      device=cfg.embed_device, model=cfg.embed_model):
            self.embedder.load()

        if self.reranker is not None:
            with run.span("reranker_model_load", model=cfg.reranker_model,
                          device=cfg.reranker_device):
                self.reranker.load()

        structured = cfg.chunk_strategy == "structured"
        with run.span("parse", strategy=cfg.chunk_strategy) as s:
            if structured:
                units = parse_structured_dir(corpus)
                s["n_units"] = len(units)
            else:
                pages = parse_dir(corpus)
                s["n_pages"] = len(pages)
                s["n_suspect_scanned"] = sum(p["suspect_scanned"] for p in pages)

        with run.span("chunk", size=cfg.chunk_tokens, overlap=cfg.chunk_overlap,
                      strategy=cfg.chunk_strategy) as s:
            if structured:
                chunks = chunk_structured(units, cfg.chunk_tokens, cfg.chunk_overlap)
            else:
                chunks = chunk_pages(pages, cfg.chunk_tokens, cfg.chunk_overlap)
            s["n_chunks"] = len(chunks)

        with run.span("dedup") as s:
            seen, unique = set(), []
            for c in chunks:
                if c["id"] not in seen:
                    seen.add(c["id"])
                    unique.append(c)
            s["n_dropped"] = len(chunks) - len(unique)
            chunks = unique

        n_tok = sum(c["n_tokens"] for c in chunks)
        with run.span("embed_chunks", batch=cfg.embed_batch,
                      device=cfg.embed_device) as s:
            vecs = self.embedder.encode([c["text"] for c in chunks])
            s["n_chunks"] = len(chunks)
            s["n_tokens"] = n_tok
        # throughput is the number that actually explains the gap
        run.emit("embed_throughput", 0.0, tokens=n_tok, dim=int(vecs.shape[1]))

        with run.span("index_build", hybrid=cfg.hybrid) as s:
            self.index = FlatIndex(vecs, chunks)
            if cfg.hybrid:
                self.index.build_bm25()
            s["n_vectors"] = len(chunks)

    # ---------- query ----------
    def query(self, question: str) -> dict:
        cfg, run, idx = self.cfg, self.run, self.index
        assert idx is not None, "ingest() first"

        with run.span("embed_query"):
            qv = self.embedder.encode([question])[0]

        with run.span("retrieve", hybrid=cfg.hybrid, k=cfg.top_k) as s:
            allowed = None
            if cfg.metadata_filter:
                matched = idx.matched_sources(question)
                if matched:
                    allowed = idx.indices_for(matched)
                    s["metadata_filter_matched"] = sorted(matched)
            dense = idx.dense(qv, cfg.candidates, allowed=allowed)
            if cfg.hybrid:
                sparse = idx.sparse(question, cfg.candidates, allowed=allowed)
                candidates = FlatIndex.rrf(dense, sparse)[: cfg.candidates]
            else:
                candidates = [i for i, _ in dense]
            order = candidates[: cfg.top_k]
            s["n_candidates"] = len(candidates)

        if self.reranker is not None:
            with run.span("rerank", model=cfg.reranker_model) as s:
                pool = [idx.chunks[i] for i in candidates]
                picked = self.reranker.rerank(question, pool, cfg.top_k)
                order = [candidates[i] for i in picked]
                s["n_reranked"] = len(candidates)

        hits = [idx.chunks[i] for i in order]

        with run.span("prompt_build") as s:
            conflicts = conflicting_day_dates(hits)
            messages = build_prompt(question, hits, conflicts=conflicts or None)
            s["ctx_tokens"] = sum(h["n_tokens"] for h in hits)
            if conflicts:
                s["conflicts_detected"] = len(conflicts)

        with run.span("generate", model=cfg.gen_model,
                      keep_alive=cfg.gen_keep_alive) as s:
            out = stream_answer(cfg, messages)
            s.update({k: v for k, v in out.items() if k != "text"})

        return {"question": question, "answer": out["text"],
                "sources": [f"{h['source']} p.{h['page']}" for h in hits],
                "ttft_ms": out["ttft_ms"], "tok_per_s": out["tok_per_s"],
                "chunks": hits}
