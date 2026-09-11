from __future__ import annotations
import re
from pathlib import Path

# Filename-versioning noise, not real content words - would give false
# "distinctive" signal on words too generic to mean anything.
_STOPWORDS = {"final", "fixed", "copy", "new", "old", "draft"}


def _tokenize(filename: str) -> set[str]:
    # Path.stem strips whatever extension is actually present (.pdf, .docx,
    # ...) instead of hardcoding one - a filename-versioning "(1)" suffix
    # from a duplicate download is noise the same way "final"/"v6" are.
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    tokens = re.split(r"[_\-\s]+", stem.lower())
    return {t for t in tokens
            if len(t) >= 4 and t not in _STOPWORDS
            and not t.isdigit() and not re.fullmatch(r"v\d+", t)}


def source_hints(sources: list[str]) -> dict[str, set[str]]:
    """Per-source distinctive tokens from filenames. A token counts only if
    it belongs to exactly one source - so two resumes that both contain
    "jalpesh"/"rajani" never let either word falsely narrow the search to
    just one of them."""
    per_source = {s: _tokenize(s) for s in sources}
    owners: dict[str, int] = {}
    for toks in per_source.values():
        for t in toks:
            owners[t] = owners.get(t, 0) + 1
    return {s: {t for t in toks if owners[t] == 1} for s, toks in per_source.items()}


def matched_sources(question: str, sources: list[str], hints: dict[str, set[str]]) -> set[str]:
    """Which single source (if any) the question unambiguously names.
    Empty or multi-source means "don't filter, search everything" - the
    caller should fall back to normal full-corpus retrieval in that case,
    which is what keeps genuinely cross-document questions (the common
    failure mode this guards against, not this) working."""
    q = question.lower()
    exact = {s for s in sources if s.lower() in q}
    if len(exact) == 1:
        return exact
    return {s for s, toks in hints.items()
            if any(re.search(rf"\b{re.escape(t)}\b", q) for t in toks)}


def loosely_referenced_sources(question: str, sources: list[str], max_owners: int = 4) -> set[str]:
    """Like matched_sources, but without the single-owner-only restriction -
    and unlike a naive "any shared token counts" version, still excludes
    tokens that are basically the cluster's topic word, not a document
    identifier.

    matched_sources() is built for metadata filtering, where a false
    positive is costly (it narrows retrieval, and getting that wrong hides
    real content - Addendum 6), so it only trusts a token with exactly one
    owner. A claims-grounding check has the opposite cost profile: a false
    positive just means one extra, cheap escalation; a false negative
    hides a real hallucination - so this allows tokens shared by a *few*
    files (real near-duplicates: "DesertTheme" stops being unique the
    moment "DesertTheme_TIMINGS_BOOKINGS.pdf" also exists - Addendum 10 -
    but the question is still plausibly about one or both of them).

    The cap matters: without it, "itinerary" itself - present in ~12 of 31
    filenames in this corpus - would match nearly the whole cluster on any
    itinerary question, firing on essentially everything and becoming
    noise instead of a signal. `max_owners` keeps genuinely generic,
    cluster-wide words out while still catching small groups of
    near-duplicates."""
    q = question.lower()
    per_source = {s: _tokenize(s) for s in sources}
    owners: dict[str, int] = {}
    for toks in per_source.values():
        for t in toks:
            owners[t] = owners.get(t, 0) + 1

    hit = set()
    for s in sources:
        if s.lower() in q:
            hit.add(s)
            continue
        specific = {t for t in per_source[s] if owners[t] <= max_owners}
        if any(re.search(rf"\b{re.escape(t)}\b", q) for t in specific):
            hit.add(s)
    return hit


def demo() -> None:
    sources = [
        "Aadhaar.pdf",
        "Camp_Ethnosphere_Final_Fixed_v2.pdf",
        "Jalpesh_Rajani.pdf",
        "Resume-Jalpesh-Rajani-HCLSoftware.pdf",
        "Udaipur_Jaisalmer_Itinerary_DesertTheme.pdf",
    ]
    hints = source_hints(sources)

    # shared tokens never distinguish the two resumes from each other
    assert "jalpesh" not in hints["Jalpesh_Rajani.pdf"]
    assert "rajani" not in hints["Resume-Jalpesh-Rajani-HCLSoftware.pdf"]

    # exact filename mention resolves unambiguously
    got = matched_sources("How many years per Jalpesh_Rajani.pdf's profile?", sources, hints)
    assert got == {"Jalpesh_Rajani.pdf"}, got

    # natural-language reference to the other resume, via a distinctive token
    got = matched_sources("What does the HCL Software resume's summary claim?", sources, hints)
    assert got == {"Resume-Jalpesh-Rajani-HCLSoftware.pdf"}, got

    # plural "resumes" must NOT match singular "resume" - both docs needed
    got = matched_sources("Do the two resumes agree on the dates?", sources, hints)
    assert got == set(), got

    # two distinct sources named -> ambiguous on purpose, no filter
    got = matched_sources("Combining the Aadhaar card and Camp Ethnosphere form...", sources, hints)
    assert got == {"Aadhaar.pdf", "Camp_Ethnosphere_Final_Fixed_v2.pdf"}, got

    # nothing named -> no filter
    got = matched_sources("What is Jalpesh's father's name?", sources, hints)
    assert got == set(), got

    # loosely_referenced_sources: unlike matched_sources, still catches a
    # token even when a second file shares it and kills its "distinctive"
    # status - the real, live case that broke the claims-grounding check
    # (Addendum 12) until this function was added.
    sources2 = sources + ["Udaipur_Jaisalmer_Itinerary_DesertTheme_TIMINGS_BOOKINGS.pdf"]
    assert matched_sources("the DesertTheme itinerary", sources2, source_hints(sources2)) == set(), \
        "matched_sources should now see 'desertheme' as non-distinctive (2 owners)"
    loose = loosely_referenced_sources("the DesertTheme itinerary", sources2)
    assert "Udaipur_Jaisalmer_Itinerary_DesertTheme.pdf" in loose
    assert "Udaipur_Jaisalmer_Itinerary_DesertTheme_TIMINGS_BOOKINGS.pdf" in loose

    # but the cap must keep a genuinely cluster-wide word ("itinerary",
    # shared by many files) from matching nearly everything - a question
    # that only says "the itinerary" isn't naming multiple specific drafts.
    big_cluster = sources2 + [f"Udaipur_Jaisalmer_Itinerary_Draft_{i}.pdf" for i in range(6)]
    broad = loosely_referenced_sources("what happens on the itinerary?", big_cluster)
    assert len(broad) == 0, f"'itinerary' alone (shared by {len(big_cluster)} files) should not match anything, got {broad}"

    print("rag.retrieve.filter: all checks passed")


if __name__ == "__main__":
    demo()
