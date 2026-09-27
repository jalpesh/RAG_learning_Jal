from __future__ import annotations
import re

# Content-based, not filename-based - built because filename matching
# (matched_sources/loosely_referenced_sources in rag.retrieve.filter)
# structurally cannot fix this: "resume" is a distinctive filename token
# for only ONE of the two actual resumes in this corpus
# (Jalpesh_Rajani.pdf has no "resume" anywhere in its name). No amount of
# plural-matching on filenames could ever recognize both as the same
# category - the signal has to come from what the document actually says,
# not what it's called.
#
# Narrowly scoped to the one demonstrated failure (a generic "resumes"
# question losing to unrelated documents by raw embedding similarity),
# not a speculative general document-type taxonomy. Extend
# _CATEGORY_PATTERNS only when a second, real case shows up.

_CATEGORY_PATTERNS = {
    "resume": re.compile(r"\b\d+\s+years?\s+of\s+(progressive\s+)?experience\b", re.I),
}
_CATEGORY_WORDS = {
    "resume": re.compile(r"\bresumes?\b", re.I),  # matches the question's "resume" or "resumes"
}


def classify_sources(chunks: list[dict]) -> dict[str, set[str]]:
    """category -> set of source filenames whose earliest chunk matches
    that category's content signature. Uses each source's minimum-page
    chunk explicitly (not first-encountered-in-list), so this doesn't
    depend on chunks happening to arrive in file order."""
    first: dict[str, dict] = {}
    for c in chunks:
        cur = first.get(c["source"])
        if cur is None or c["page"] < cur["page"]:
            first[c["source"]] = c

    result: dict[str, set[str]] = {cat: set() for cat in _CATEGORY_PATTERNS}
    for source, c in first.items():
        for cat, pattern in _CATEGORY_PATTERNS.items():
            if pattern.search(c["text"]):
                result[cat].add(source)
    return result


def category_matched_sources(question: str, categories: dict[str, set[str]]) -> set[str]:
    """Which category the question names (by its plain-English word, not
    a filename) that resolves to 2+ actual documents. Only fires at 2+ -
    a single-document category match isn't this function's job (that's
    what metadata filtering already covers); this exists specifically for
    "the resumes"/"the reports" style references that name a document
    *type*, not a specific file."""
    hit: set[str] = set()
    for cat, sources in categories.items():
        word = _CATEGORY_WORDS.get(cat)
        if word and len(sources) >= 2 and word.search(question):
            hit |= sources
    return hit


def demo() -> None:
    chunks = [
        {"source": "Jalpesh_Rajani.pdf", "page": 1,
         "text": "Profile: 18 years of progressive experience in engineering leadership."},
        {"source": "Resume-Jalpesh-Rajani-HCLSoftware.pdf", "page": 1,
         "text": "Summary: Engineering Executive with 19 years of experience."},
        {"source": "Bullforce-tech_whitepaper_V1.0.pdf", "page": 1,
         "text": "This is the whitepaper of the broking house with technology capability."},
        {"source": "KoinX - Crypto Complete Tax Report - FY 2025.pdf", "page": 1,
         "text": "Complete Tax Report. Financial Year 2024-25."},
    ]
    categories = classify_sources(chunks)
    assert categories["resume"] == {"Jalpesh_Rajani.pdf", "Resume-Jalpesh-Rajani-HCLSoftware.pdf"}, categories["resume"]

    # the actual failing question, real corpus filenames
    got = category_matched_sources("Do the two resumes agree on the start and end dates of the Reliance Jio role?",
                                    categories)
    assert got == {"Jalpesh_Rajani.pdf", "Resume-Jalpesh-Rajani-HCLSoftware.pdf"}, got
    print("resumes-plural question -> both resumes matched by content, not filename")

    # a single resume mentioned generically must not trigger this (that's
    # metadata_filter's job, via a specific filename/keyword match) -
    # this function only fires when the *category* resolves to 2+ sources,
    # which is already true here regardless of question wording, so instead
    # check the negative: an unrelated question must not match.
    got = category_matched_sources("What is the boiling point of mercury?", categories)
    assert got == set(), got
    print("unrelated question -> no match")

    print("\nrag.retrieve.category: all checks passed")


if __name__ == "__main__":
    demo()
