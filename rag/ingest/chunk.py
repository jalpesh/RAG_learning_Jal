from __future__ import annotations
import hashlib
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def chunk_pages(pages: list[dict], size: int, overlap: int) -> list[dict]:
    """Token-aware sliding window.

    Token-aware, not character-aware: the embedder's context limit is in
    tokens, and character windows silently truncate on dense text. Overlap is a
    tunable, not a constant - the harness decides its value, not folklore.
    """
    assert 0 <= overlap < size, "overlap must be smaller than the window"
    stride = size - overlap
    chunks: list[dict] = []
    for pg in pages:
        if not pg["text"]:
            continue
        ids = _ENC.encode(pg["text"])
        for start in range(0, max(len(ids), 1), stride):
            window = ids[start:start + size]
            if not window:
                break
            text = _ENC.decode(window)
            chunks.append({
                "id": hashlib.sha256(text.encode()).hexdigest()[:16],
                "text": text,
                "source": pg["source"],
                "page": pg["page"],
                "n_tokens": len(window),
            })
            if start + size >= len(ids):
                break
    return chunks
