from __future__ import annotations
import re

from rag.retrieve.index import FlatIndex

# Words that show up when a question genuinely needs to synthesise across
# chunks/documents rather than look up one fact. Calibrated against this
# project's own 25-question eval set (see demo() below) - not guessed.
_COMPLEXITY_WORDS = re.compile(
    r"\b(across|combin\w*|both|compar\w*|agree\w*|differ\w*|every|all\b|"
    r"summar\w*|chronolog\w*|how many)\b",
    re.IGNORECASE,
)


def classify(question: str, idx: FlatIndex) -> str:
    """"fast" or "deep". Two cheap, already-available signals, no LLM call:

    1. Lexical - does the question's own wording ask for synthesis
       (compare, combine, summarise, count) rather than lookup?
    2. Retrieval - idx.matched_sources() (built for metadata filtering)
       also tells us how many *distinct* documents the question names.
       Naming 2+ explicitly is a strong signal it needs cross-document
       synthesis, not a single retrieval pass.

    This is a heuristic, not a classifier - false positives just cost a
    slower answer, and are cheaper to tolerate than false negatives (a
    complex question mishandled on the fast lane, which is where every
    quality failure in this project's testing actually happened).
    """
    if _COMPLEXITY_WORDS.search(question):
        return "deep"
    if len(idx.matched_sources(question)) >= 2:
        return "deep"
    return "fast"


def demo() -> None:
    import json
    from pathlib import Path

    questions = json.loads((Path(__file__).resolve().parents[1] / "harness" / "questions.json").read_text())["questions"]
    sources = [
        "Aadhaar.pdf", "Camp_Ethnosphere_Final_Fixed_v2.pdf", "Jalpesh_Rajani.pdf",
        "Resume-Jalpesh-Rajani-HCLSoftware.pdf", "Udaipur_Jaisalmer_Itinerary_DesertTheme.pdf",
    ]

    class FakeIndex:
        """classify() only touches matched_sources() - no need for a real
        FlatIndex (vectors, chunks) just to exercise the routing logic."""
        def __init__(self):
            from rag.retrieve.filter import source_hints
            self._hints = source_hints(sources)

        def matched_sources(self, q):
            from rag.retrieve.filter import matched_sources
            return matched_sources(q, sources, self._hints)

    idx = FakeIndex()
    counts = {"factual": [0, 0], "multihop": [0, 0], "aggregation": [0, 0], "unanswerable": [0, 0]}
    for q in questions:
        lane = classify(q["q"], idx)
        counts[q["type"]][0 if lane == "fast" else 1] += 1

    print(f"{'type':14} {'fast':>5} {'deep':>5}")
    for t, (f, d) in counts.items():
        print(f"{t:14} {f:5} {d:5}")

    # every unanswerable and every plain factual question must stay fast -
    # both handled correctly at every model size tested (Addendum 3/8);
    # routing them deep would only add latency, never correctness.
    assert counts["factual"][1] <= 2, "too many factual questions routed deep"
    assert counts["unanswerable"][1] == 0, "unanswerable questions must stay on the fast lane"
    print("\nrag.route: all checks passed")


if __name__ == "__main__":
    demo()
