from __future__ import annotations
import numpy as np


class SemanticCache:
    """In-memory, embedding-similarity answer cache. A repeat or
    near-duplicate question skips retrieval and generation entirely - the
    standard first move in production RAG cost/latency, because the same
    handful of questions get asked over and over in a shared corpus.

    Threshold calibrated against real embeddings, not guessed (see
    docs/02-findings.md): exact repeat = 1.00, a genuine paraphrase of the
    same question = 0.92, a *different* question on the same topic = 0.80.
    That 0.12 gap is the entire safety margin - returning a wrong cached
    answer is worse than a cache miss, so the default sits above the
    paraphrase score, not between it and the same-topic score. This means
    only close paraphrases and near-exact repeats hit; loosely-worded
    paraphrases safely miss and fall through to a real answer.

    v1 constraints, stated not hidden: unbounded (no LRU/TTL eviction),
    single-process (no shared cache across replicas), invalidated
    wholesale on re-ingest rather than per-document.
    """

    def __init__(self, threshold: float = 0.93):
        self.threshold = threshold
        self._questions: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._answers: list[dict] = []

    def get(self, qvec: np.ndarray) -> dict | None:
        if not self._vecs:
            return None
        sims = np.stack(self._vecs) @ qvec.ravel()
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return {**self._answers[best], "cache_sim": round(float(sims[best]), 4),
                    "cache_hit_for": self._questions[best]}
        return None

    def put(self, question: str, qvec: np.ndarray, answer: dict) -> None:
        self._questions.append(question)
        self._vecs.append(qvec)
        self._answers.append(answer)

    def clear(self) -> None:
        self._questions.clear()
        self._vecs.clear()
        self._answers.clear()

    def __len__(self) -> int:
        return len(self._vecs)
