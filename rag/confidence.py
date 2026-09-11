from __future__ import annotations
import re

# Tried retrieval top-score as a confidence signal first - calibrated
# against real questions and rejected it. A plausible-but-unanswerable
# question ("chess tournament prize money", when the corpus never states
# one) scored *higher* (0.78) than a genuinely answerable one (Aadhaar
# DOB, 0.64) - retrieval score measures topical similarity to some chunk,
# not whether the specific fact is actually present. See docs/02-findings.md.
#
# What actually failed in this project's own testing (Addendum 5): the
# model computing a number instead of quoting one that was already stated.
# That failure is mechanically checkable - a genuinely quoted number
# appears verbatim in the context it was quoted from; a computed one
# usually doesn't.

_HEDGE = re.compile(
    r"\b(might be|may be|possibly|not (entirely )?certain|unclear|"
    r"i'm not sure|difficult to (determine|say)|unable to (determine|confirm)|"
    r"appears to|seems to|approximately|roughly)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\b\d{1,4}\b")
_CITATION = re.compile(r"\[[^\]]*\]")  # "[source p.1]" / "[Foo.pdf p.3]" - metadata, not a claimed fact


def ungrounded_numbers(answer: str, context_chunks: list[dict]) -> set[str]:
    """Numbers the answer states that don't appear anywhere in the chunks
    it was supposedly answering from. A non-empty result is a specific,
    checkable sign the model may have computed rather than quoted.

    Citation brackets are stripped first - "[source p.1]"'s page number is
    structural, not a fact drawn from the chunk text, and page numbers are
    never going to appear inside the chunk text itself. Without this, every
    single-digit-page citation false-positives - caught by testing this
    against a real, correct answer before trusting it (see docs/02-findings.md)."""
    context_text = " ".join(c["text"] for c in context_chunks)
    context_numbers = set(_NUMBER.findall(context_text))
    answer_numbers = set(_NUMBER.findall(_CITATION.sub("", answer)))
    return answer_numbers - context_numbers


def missing_named_sources(question: str, idx, sources_used: list[str]) -> set[str]:
    """Documents the QUESTION itself names that never actually made it into
    the retrieved context. A non-empty result is the exact mechanism behind
    a real, confirmed hallucination this project found (Addendum 11): asked
    whether two named itinerary drafts agreed on a schedule, retrieval
    pulled 5/5 chunks from just one of them, and the model confidently
    answered "yes, both..." anyway - it never saw the second document.

    Uses idx.loosely_referenced_sources(), not matched_sources() - the
    latter is built for metadata filtering, where a false positive is
    costly (it silently narrows retrieval), so it only trusts a token that
    belongs to exactly one source. That rule cost this check the actual
    live case it exists to catch (Addendum 12): "DesertTheme" stopped being
    "distinctive" the moment "DesertTheme_TIMINGS_BOOKINGS.pdf" also
    entered the corpus, even though the question was unambiguously about
    it. A false positive here just costs one cheap extra escalation; a
    false negative hides a real hallucination - the looser matcher is the
    correct tradeoff for this direction."""
    named = idx.loosely_referenced_sources(question)
    if len(named) < 2:
        return set()  # nothing to cross-check a single/no reference against
    retrieved = {s.rsplit(" p.", 1)[0] for s in sources_used}
    return named - retrieved


def is_low_confidence(answer: str, context_chunks: list[dict],
                       question: str | None = None, idx=None,
                       sources_used: list[str] | None = None) -> tuple[bool, str]:
    """(escalate?, reason). Cheap, no-extra-LLM-call checks - string
    matching, a regex, a set difference - not a model call, so this costs
    microseconds next to the multi-second thing it might trigger.
    question/idx/sources_used are optional so the numeric+hedge checks
    still work standalone (as in the original bug they were built for)."""
    ungrounded = ungrounded_numbers(answer, context_chunks)
    if ungrounded:
        return True, f"ungrounded numbers: {sorted(ungrounded)}"
    if _HEDGE.search(answer):
        return True, "hedging language"
    if question is not None and idx is not None and sources_used is not None:
        missing = missing_named_sources(question, idx, sources_used)
        if missing:
            return True, f"question named {sorted(missing)} but it was never retrieved"
    return False, ""


def demo() -> None:
    # The actual wrong/right answer pair from this session's own HCL-years
    # bug (Addendum 5) - this is what the check exists to catch.
    chunks = [{"text": "Innovative resume with 19 years of experience driving technical excellence"}]
    wrong = "The HCL Software resume's summary claims 20 years of experience."
    right = "The HCL Software resume's summary claims 19 years of experience."

    flag, reason = is_low_confidence(wrong, chunks)
    assert flag, "should have flagged the computed, ungrounded '20'"
    print(f"wrong answer -> escalate=True ({reason})")

    flag, reason = is_low_confidence(right, chunks)
    assert not flag, f"should NOT have flagged a correctly quoted number, got: {reason}"
    print("right answer -> escalate=False")

    hedged = "The document might mention around 19 years, but it's unclear."
    flag, reason = is_low_confidence(hedged, chunks)
    assert flag
    print(f"hedged answer -> escalate=True ({reason})")

    # regression case: a citation's own page number ("[source p.1]") is
    # metadata, not a claimed fact - must not trip the ungrounded check.
    cited = "The date of birth is 20/05/1984 [source p.1]."
    dob_chunks = [{"text": "DOB: 20/05/1984, address in Thane"}]
    flag, reason = is_low_confidence(cited, dob_chunks)
    assert not flag, f"citation page number should not count as ungrounded, got: {reason}"
    print("cited answer (page number in brackets) -> escalate=False")

    # regression case: the actual confirmed hallucination from Addendum 11.
    # Question names two drafts; retrieval only pulled chunks from one;
    # the model answered "yes, both..." anyway. Uses the REAL corpus's
    # full source list, not a hand-picked 2-file subset - the first version
    # of this test used just the 2 relevant files and passed even though
    # the live check failed (Addendum 12), because "DesertTheme" only loses
    # its distinctive-token status once the full 31-file corpus's
    # "DesertTheme_TIMINGS_BOOKINGS.pdf" is also present to collide with it.
    from pathlib import Path
    from rag.retrieve.filter import loosely_referenced_sources as _loose

    corpus_dir = Path(__file__).resolve().parents[1] / "corpus"
    sources = sorted(p.name for p in corpus_dir.iterdir() if p.suffix.lower() in (".pdf", ".docx")) \
        if corpus_dir.exists() else [
            "Udaipur_Jaisalmer_Itinerary_Optimized.pdf",
            "Udaipur_Jaisalmer_Itinerary_DesertTheme.pdf",
            "Udaipur_Jaisalmer_Itinerary_DesertTheme_TIMINGS_BOOKINGS.pdf",
        ]

    class FakeIndex:
        def loosely_referenced_sources(self, q):
            return _loose(q, sources)

    q = "Do the 'Optimized' and 'DesertTheme' itinerary drafts describe the same Day 1 schedule?"
    retrieved_sources_only = [f"Udaipur_Jaisalmer_Itinerary_Optimized.pdf p.{i}" for i in range(1, 6)]
    # realistic retrieved context, containing "Day 1" and "City Palace" -
    # every number the answer states IS grounded, isolating this test to
    # the claims-grounding path rather than tripping the numeric one.
    real_context = [{"text": "Day 1 - arrival in Udaipur, check-in, City Palace visit, Bagore Ki Haveli walk"}]
    hallucinated = ("Yes, both drafts describe the same Day 1 schedule: arrival, City Palace, "
                     "and Bagore Ki Haveli. [Udaipur_Jaisalmer_Itinerary_Optimized.pdf p.1]")
    flag, reason = is_low_confidence(hallucinated, real_context, question=q, idx=FakeIndex(),
                                      sources_used=retrieved_sources_only)
    assert flag and "DesertTheme" in reason, \
        f"should have caught DesertTheme never being retrieved despite being named, got: {reason}"
    print(f"confirmed-hallucination case -> escalate=True ({reason})")

    print("\nrag.confidence: all checks passed")


if __name__ == "__main__":
    demo()
