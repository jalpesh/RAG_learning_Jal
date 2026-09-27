from __future__ import annotations
import numpy as np
from rank_bm25 import BM25Okapi
from rag.retrieve.filter import source_hints, matched_sources, loosely_referenced_sources
from rag.retrieve.category import classify_sources, category_matched_sources


class FlatIndex:
    """Exact cosine search over a float32 matrix, plus optional BM25.

    No HNSW, no vector DB. At corpus sizes below ~100k vectors an exact matmul
    is sub-millisecond and, crucially, has *zero build cost* - whereas HNSW
    charges you graph construction on every ingest to save query time you do
    not need. Introduce an ANN index when the harness proves flat has lost,
    not before.
    """

    def __init__(self, vectors: np.ndarray, chunks: list[dict]):
        self.v = np.ascontiguousarray(vectors, dtype=np.float32)
        self.chunks = chunks
        self._bm25 = None
        self.sources = sorted({c["source"] for c in chunks})
        self._hints = source_hints(self.sources)
        self._categories = classify_sources(chunks)

    def build_bm25(self) -> None:
        self._bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])

    def matched_sources(self, question: str) -> set[str]:
        """Which single source (if any) the question unambiguously names -
        see rag.retrieve.filter for the matching rule. Empty or multi-source
        means the caller should search everything, unfiltered."""
        return matched_sources(question, self.sources, self._hints)

    def loosely_referenced_sources(self, question: str) -> set[str]:
        """Every source the question's wording plausibly names, without
        matched_sources()'s single-owner-only restriction - for callers
        where a false positive is cheap and a false negative is not (the
        claims-grounding check, not retrieval filtering). See
        rag.retrieve.filter.loosely_referenced_sources."""
        return loosely_referenced_sources(question, self.sources)

    def category_matched_sources(self, question: str) -> set[str]:
        """Every source matching a document *category* the question names
        by plain-English word ("the resumes"), resolved from content, not
        filenames - see rag.retrieve.category. Fixes the specific case
        filename matching structurally cannot: two documents of the same
        real-world type with no shared distinguishing word in either name."""
        return category_matched_sources(question, self._categories)

    def indices_for(self, sources: set[str]) -> set[int]:
        return {i for i, c in enumerate(self.chunks) if c["source"] in sources}

    def dense(self, q: np.ndarray, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores = self.v @ q.ravel()          # vectors are pre-normalised
        pool = np.arange(len(scores)) if allowed is None else np.fromiter(allowed, dtype=np.int64)
        k = min(k, len(pool))
        top = pool[np.argsort(-scores[pool])[:k]]
        return [(int(i), float(scores[i])) for i in top]

    def sparse(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(query.lower().split())
        pool = np.arange(len(scores)) if allowed is None else np.fromiter(allowed, dtype=np.int64)
        k = min(k, len(pool))
        top = pool[np.argsort(-scores[pool])[:k]]
        return [(int(i), float(scores[i])) for i in top]

    @staticmethod
    def rrf(*rankings: list[tuple[int, float]], k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion. Score-free, so it needs no calibration
        between a cosine similarity and a BM25 score - which is exactly why it
        beats naive score addition."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (doc, _) in enumerate(ranking):
                fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
        return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
