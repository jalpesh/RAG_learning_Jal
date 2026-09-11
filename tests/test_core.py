from __future__ import annotations

import unittest

import numpy as np

from rag.cache import SemanticCache
from rag.confidence import ungrounded_numbers
from rag.retrieve.index import FlatIndex


class SemanticCacheTests(unittest.TestCase):
    def test_exact_match_hits_and_clear_invalidates(self) -> None:
        cache = SemanticCache(threshold=0.93)
        vector = np.array([1.0, 0.0], dtype=np.float32)
        cache.put("question", vector, {"answer": "answer"})

        self.assertEqual(cache.get(vector)["answer"], "answer")
        cache.clear()
        self.assertIsNone(cache.get(vector))


class RetrievalTests(unittest.TestCase):
    def test_dense_search_respects_source_filter(self) -> None:
        chunks = [
            {"source": "a.pdf", "page": 1, "text": "alpha", "n_tokens": 1},
            {"source": "b.pdf", "page": 1, "text": "beta", "n_tokens": 1},
        ]
        index = FlatIndex(np.eye(2, dtype=np.float32), chunks)

        self.assertEqual(index.dense(np.array([1.0, 0.0]), 1)[0][0], 0)
        self.assertEqual(index.dense(np.array([1.0, 0.0]), 1, allowed={1})[0][0], 1)

    def test_rrf_rewards_documents_present_in_both_rankings(self) -> None:
        fused = FlatIndex.rrf([(0, 0.9), (1, 0.8)], [(1, 4.0), (2, 3.0)])
        self.assertEqual(fused[0], 1)


class ConfidenceTests(unittest.TestCase):
    def test_citation_page_is_not_treated_as_an_ungrounded_number(self) -> None:
        chunks = [{"text": "DOB: 20/05/1984"}]
        answer = "DOB is 20/05/1984 [Aadhaar.pdf p.1]."
        self.assertEqual(ungrounded_numbers(answer, chunks), set())


if __name__ == "__main__":
    unittest.main()
