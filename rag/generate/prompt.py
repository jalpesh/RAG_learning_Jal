from __future__ import annotations

SYSTEM = (
    "You answer strictly from the provided context. If the context does not "
    "contain the answer, say so. Cite sources as [source p.N]. Be concise."
)


def build_prompt(question: str, chunks: list[dict],
                  conflicts: list[tuple[int, dict[str, str]]] | None = None) -> list[dict]:
    """System prompt first, retrieved context after.

    Order matters: llama.cpp / Ollama cache the KV for a stable prefix, so a
    fixed system block at position 0 is prefill you only pay once.

    `conflicts` (from rag.confidence.conflicting_day_dates, computed on
    these same chunks before this call) is what closes the Addendum 14 gap:
    detection alone didn't change the answer - the deep-lane escalation
    re-asked the identical question and picked the same side silently,
    because nothing told it a conflict existed. Telling it, here, is the
    fix - not a bigger model, not a retry."""
    ctx = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks
    )
    notice = ""
    if conflicts:
        lines = [
            f"Day {day_n}: " + "; ".join(f"{date} per {src}" for date, src in vals.items())
            for day_n, vals in conflicts
        ]
        notice = (
            "\n\nNote: your sources disagree with each other on the following - "
            "do not silently pick one. State that sources disagree and name what "
            "each one says:\n" + "\n".join(lines)
        )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}{notice}"},
    ]


def demo() -> None:
    chunks = [{"source": "a.pdf", "page": 1, "text": "Day 1 - 28 Nov"}]
    plain = build_prompt("When is Day 1?", chunks)
    assert "disagree" not in plain[1]["content"], "no conflicts given -> no notice"

    conflicts = [(1, {"28 Nov": "a.pdf", "28 Dec": "b.docx"})]
    flagged = build_prompt("When is Day 1?", chunks, conflicts=conflicts)
    assert "disagree" in flagged[1]["content"]
    assert "28 Nov per a.pdf" in flagged[1]["content"]
    assert "28 Dec per b.docx" in flagged[1]["content"]

    print("rag.generate.prompt: all checks passed")


if __name__ == "__main__":
    demo()
