from __future__ import annotations
import numpy as np
from rag.config import Config


class Reranker:
    """Cross-encoder second pass over the retrieved candidates.

    A cross-encoder scores (query, chunk) jointly, which is more accurate
    than comparing two separately-computed embeddings - but it costs one
    forward pass per candidate, not per corpus, which is exactly why it
    only ever runs on the ~20-candidate pool, never the whole index.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.cfg.reranker_model, device=self.cfg.reranker_device)

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[int]:
        """Score every chunk against the query, return the best `top_k`
        positions into `chunks`, best first."""
        pairs = [(query, c["text"]) for c in chunks]
        scores = self._model.predict(pairs)
        order = np.argsort(-scores)[:top_k]
        return [int(i) for i in order]
