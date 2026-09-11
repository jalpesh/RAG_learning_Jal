from __future__ import annotations

SYSTEM = (
    "You answer strictly from the provided context. If the context does not "
    "contain the answer, say so. Cite sources as [source p.N]. Be concise."
)


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """System prompt first, retrieved context after.

    Order matters: llama.cpp / Ollama cache the KV for a stable prefix, so a
    fixed system block at position 0 is prefill you only pay once.
    """
    ctx = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
    ]
