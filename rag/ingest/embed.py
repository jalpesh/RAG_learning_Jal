from __future__ import annotations
import numpy as np
import httpx
from rag.config import Config


class Embedder:
    """Pluggable embedding backend.

    The whole point of this class is that swapping cpu -> mps, or batch 32 ->
    128, is a config change and nothing else. That is what makes the benchmark
    matrix honest: only one variable moves at a time.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        """Explicit, separately-timed load. Cold start is a first-class cost,
        not something to hide inside the first encode() call. No-ops on a
        second call - that's what makes Pipeline actually resident instead
        of re-paying the model load on every ingest()."""
        if self._model is not None or getattr(self, "_client", None) is not None:
            return
        if self.cfg.embed_backend == "st":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.cfg.embed_model, device=self.cfg.embed_device
            )
        else:
            self._client = httpx.Client(base_url=self.cfg.ollama_host, timeout=300)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.cfg.embed_backend == "st":
            vecs = self._model.encode(
                texts,
                batch_size=self.cfg.embed_batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vecs.astype(np.float32)

        # Ollama: send in explicit batches so batch size stays a real variable.
        out = []
        for i in range(0, len(texts), self.cfg.embed_batch):
            r = self._client.post("/api/embed", json={
                "model": self.cfg.embed_model,
                "input": texts[i:i + self.cfg.embed_batch],
            })
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        v = np.asarray(out, dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
