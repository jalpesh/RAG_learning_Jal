from __future__ import annotations
import hashlib
from pathlib import Path

import fitz  # pymupdf
import docx  # python-docx

from rag.ingest.chunk import _ENC, chunk_pages

_HEADING_PREFIXES = ("heading", "title")


def parse_blocks_pdf(path: Path) -> list[dict]:
    """PDF paragraphs via pymupdf's own layout analysis, not blank-line
    guessing. get_text("blocks") returns pymupdf's already-segmented
    reading blocks (paragraphs/columns), each with a position - sorting by
    that position turns them into the same reading order a human would use,
    which is not guaranteed to be their raw extraction order."""
    doc = fitz.open(path)
    units = []
    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        for b in blocks:
            text = b[4].strip()
            if text:
                units.append({"source": path.name, "page": i + 1, "text": text, "kind": "block"})
    doc.close()
    return units


def parse_sections_docx(path: Path) -> list[dict]:
    """DOCX paragraphs already carry structure Word itself assigned -
    Heading 1/2/3 style names - that plain-text extraction (parse_docx in
    rag.ingest.parse) throws away by joining every paragraph into one blob.
    A new heading starts a new section; everything under it stays together
    until the next heading. Tables are their own units, never sliced
    mid-row - same convention as parse_docx's table handling."""
    d = docx.Document(str(path))
    units: list[dict] = []
    section: list[str] = []

    def flush() -> None:
        if section:
            units.append({"source": path.name, "page": 1, "text": "\n".join(section), "kind": "section"})
            section.clear()

    for p in d.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style_name = (p.style.name if p.style else "") or ""
        if style_name.lower().startswith(_HEADING_PREFIXES):
            flush()
        section.append(text)
    flush()

    for table in d.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            units.append({"source": path.name, "page": 1, "text": "\n".join(rows), "kind": "table"})
    return units


def parse_structured_dir(corpus: Path) -> list[dict]:
    """Same file-type dispatch as rag.ingest.parse.parse_dir, but returns
    structural units instead of one flat per-page text blob. .txt/.md have
    no exploitable structure - fall back to whole-file-as-one-unit, same
    convention parse_dir already uses for them."""
    out: list[dict] = []
    for p in sorted(corpus.rglob("*")):
        if p.suffix.lower() == ".pdf":
            out.extend(parse_blocks_pdf(p))
        elif p.suffix.lower() == ".docx":
            out.extend(parse_sections_docx(p))
        elif p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                out.append({"source": p.name, "page": 1, "text": text, "kind": "file"})
    return out


def chunk_structured(units: list[dict], size: int, overlap: int) -> list[dict]:
    """Greedily pack whole structural units into a chunk up to the token
    budget - never splitting a unit unless it alone exceeds `size`, in
    which case (and only then) it falls back to the ordinary sliding
    window, so an oversized paragraph/table still gets cut somewhere
    rather than producing one giant unembeddable chunk.

    No overlap between packed units - overlap exists to avoid losing
    continuity across an arbitrary token cut; packing whole units doesn't
    make one, so there's nothing for overlap to repair there. Overlap
    still applies within the sliding-window fallback for an oversized
    unit, where an arbitrary cut is unavoidable."""
    assert 0 <= overlap < size, "overlap must be smaller than the window"
    chunks: list[dict] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_source: str | None = None
    buf_page: int | None = None

    def flush() -> None:
        nonlocal buf, buf_tokens, buf_source, buf_page
        if buf:
            text = "\n\n".join(buf)
            chunks.append({
                "id": hashlib.sha256(text.encode()).hexdigest()[:16],
                "text": text, "source": buf_source, "page": buf_page,
                "n_tokens": buf_tokens,
            })
        buf, buf_tokens, buf_source, buf_page = [], 0, None, None

    for u in units:
        n = len(_ENC.encode(u["text"]))
        # A packed chunk is tagged with one page number for its citation -
        # letting it silently span pages would make that citation wrong for
        # part of its own content. Page changes flush, same as overflow.
        if buf_page is not None and (u["source"], u["page"]) != (buf_source, buf_page):
            flush()
        if n > size:
            flush()
            chunks.extend(chunk_pages([{"source": u["source"], "page": u["page"], "text": u["text"]}],
                                       size, overlap))
            continue
        if buf_tokens + n > size:
            flush()
        if buf_source is None:
            buf_source, buf_page = u["source"], u["page"]
        buf.append(u["text"])
        buf_tokens += n
    flush()
    return chunks


def demo() -> None:
    # A giant single unit must still get cut (via the sliding-window
    # fallback), not silently produce one unembeddable oversized chunk.
    giant = [{"source": "a.pdf", "page": 1, "text": "word " * 2000, "kind": "block"}]
    chunks = chunk_structured(giant, size=400, overlap=80)
    assert len(chunks) > 1, "an oversized unit must still be split"
    assert all(c["n_tokens"] <= 400 for c in chunks)

    # small units on the same "page" pack together instead of each
    # becoming its own tiny, mostly-wasted chunk.
    small = [
        {"source": "a.pdf", "page": 1, "text": "Section one, short.", "kind": "block"},
        {"source": "a.pdf", "page": 1, "text": "Section two, also short.", "kind": "block"},
    ]
    packed = chunk_structured(small, size=400, overlap=80)
    assert len(packed) == 1, "small units should pack into one chunk, not two"
    assert "Section one" in packed[0]["text"] and "Section two" in packed[0]["text"]

    # a heading never gets separated from the text under it - the whole
    # point of section-based DOCX chunking. parse_sections_docx keeps the
    # heading text inside the section it starts, so a real section unit
    # already reads "Executive Summary\n...".
    real_section = [{"source": "b.docx", "page": 1,
                      "text": "Executive Summary\nThe product does X and Y.", "kind": "section"}]
    out = chunk_structured(real_section, size=400, overlap=80)
    assert len(out) == 1 and out[0]["text"].startswith("Executive Summary")

    # regression: two small units on DIFFERENT pages must never pack into
    # one chunk under a single page tag - caught by testing against the
    # real Bullforce whitepaper, where page-1 content and page-2's table
    # of contents packed together under a "page: 1" citation that was
    # wrong for half the chunk's own text.
    cross_page = [
        {"source": "c.pdf", "page": 1, "text": "Page one content.", "kind": "block"},
        {"source": "c.pdf", "page": 2, "text": "Page two content.", "kind": "block"},
    ]
    split = chunk_structured(cross_page, size=400, overlap=80)
    assert len(split) == 2, "units on different pages must not share one chunk/page tag"
    assert split[0]["page"] == 1 and split[1]["page"] == 2

    print("rag.ingest.structured: all checks passed")


if __name__ == "__main__":
    demo()
