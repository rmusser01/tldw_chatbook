"""Citation verification for the deep-search pipeline (task-16331).

Port of tldw_server dev's verbatim-first checking design (Claims_Extraction/
alignment.py + RAG guardrails quote citations) onto the chatbook deep-search
evidence shape (``FinalAnswerDict.evidence`` entries with 1..N ids):

- ``[n]`` markers the synthesis LLM emits must resolve to a real evidence id;
  unknown ids are flagged inline as ``[n?]`` and counted, never deleted.
- Quoted spans in the answer (>= 4 chars, per the server guardrails threshold)
  are matched against the scraped ``original_content`` (falling back to the
  LLM ``content`` summary) with a three-rung ladder: exact substring, then
  casefold/whitespace-normalized containment, then a bounded token-level
  fuzzy window (difflib over token lists, never quadratic over raw chars).
- Sentences carrying no citation marker are counted as informational
  "uncited" -- reported, not rewritten.

Everything here is pure string work: no network calls, no LLM calls, so it
composes freely with the deep-search deadline/budget machinery.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "CITATION_MARKER_RE",
    "QUOTE_SPAN_RE",
    "match_quote_in_sources",
    "summarize_for_footer",
    "verify_citations",
]

# Numeric-only so markdown links "[text](url)" and LaTeX "\[ \sin(x) \]"
# (both appear in the answer_synthesis prompt's formatting rules) can never
# be mistaken for citation markers.
CITATION_MARKER_RE = re.compile(r"\[(\d{1,4})\]")
# Double or single quoted spans; curly doubles included since the synthesis
# model sometimes emits them. >= 4 inner chars matches the server threshold.
QUOTE_SPAN_RE = re.compile(r"[\"“'‘]([^\"”'’]{4,})[\"”'’]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\w+")

# Server guardrails.py only quotes-checks spans >= 4 chars (QUOTE_MIN_CHARS);
# the regex enforces it structurally, this constant documents the pairing.
QUOTE_MIN_CHARS = 4

# Fuzzy rung bounds (server alignment.py accepts span matches at 0.6; chatbook
# stays stricter at 0.75 since deep-search quotes are short and a false
# "verified" is worse than a flagged one). Window: len(q)..len(q)+8, tighter
# than the server's 8x cap.
_FUZZY_RATIO_THRESHOLD = 0.75
_FUZZY_MAX_EXTRA_TOKENS = 8
# Defensive work guard: original_content is scrape-capped upstream, but a
# pathological entry must not turn quote checking into a long scan.
_MAX_SOURCE_CHARS_FOR_FUZZY = 100_000


def _normalize_for_match(text: str) -> str:
    """Casefold + collapse all whitespace runs (server `_normalized_for_match`
    plus case-insensitivity, as in claims_engine's quote check)."""
    return " ".join(text.split()).casefold()


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.casefold())


def _fuzzy_token_match(quote_tokens: Sequence[str], source_text: str) -> bool:
    """Bounded token-level fuzzy fallback: slide windows sized len(q)..len(q)+8
    over the source tokens and accept a SequenceMatcher ratio >= 0.85."""
    if not quote_tokens or len(source_text) > _MAX_SOURCE_CHARS_FOR_FUZZY:
        return False
    source_tokens = _tokens(source_text)
    n = len(quote_tokens)
    if not source_tokens or n > len(source_tokens):
        return False
    max_window = min(2 * n, n + _FUZZY_MAX_EXTRA_TOKENS)
    matcher = SequenceMatcher(None, list(quote_tokens), source_tokens[:max_window])
    for start in range(0, len(source_tokens) - n + 1):
        for window in range(n, max_window + 1):
            end = start + window
            if end > len(source_tokens):
                break
            matcher.set_seq2(source_tokens[start:end])
            if matcher.ratio() >= _FUZZY_RATIO_THRESHOLD:
                return True
    return False


def match_quote_in_sources(quote: str, source_texts: Sequence[str]) -> Dict[str, Any]:
    """Match one quoted span against source texts with the verbatim-first
    ladder: exact substring, then normalized containment, then bounded fuzzy.

    Args:
        quote: The quoted span from the answer text.
        source_texts: Candidate source texts (raw scrapes first, then LLM
            summaries), searched in order.

    Returns:
        Dict with ``matched`` (bool), ``level`` (``"exact"``,
        ``"normalized"``, ``"fuzzy"``, or ``None`` when unmatched), and
        ``source_index`` (0-based index into ``source_texts``, or ``None``).
    """
    quote = (quote or "").strip()
    if not quote:
        return {"matched": False, "level": None, "source_index": None}

    # Rung 1: exact verbatim substring.
    for idx, source in enumerate(source_texts):
        if source and quote in source:
            return {"matched": True, "level": "exact", "source_index": idx}

    # Rung 2: casefold + whitespace-normalized containment.
    normalized_quote = _normalize_for_match(quote)
    for idx, source in enumerate(source_texts):
        if source and normalized_quote in _normalize_for_match(source):
            return {"matched": True, "level": "normalized", "source_index": idx}

    # Rung 3: bounded token-level fuzzy window.
    quote_tokens = _tokens(quote)
    if len(quote_tokens) >= 2:
        for idx, source in enumerate(source_texts):
            if source and _fuzzy_token_match(quote_tokens, source):
                return {"matched": True, "level": "fuzzy", "source_index": idx}

    return {"matched": False, "level": None, "source_index": None}


def _extract_source_texts(evidence: Sequence[Mapping[str, Any]]) -> List[str]:
    """Flatten evidence entries into matchable source texts: the raw scrape
    (``original_content``) first, then the LLM summary (``content``)."""
    texts: List[str] = []
    for entry in evidence:
        if not isinstance(entry, Mapping):
            continue
        for key in ("original_content", "content"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def verify_citations(
    answer_text: str, evidence: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Verify the ``[n]`` markers and quoted spans of a synthesized answer
    against the run's evidence (task-16331).

    Args:
        answer_text: The synthesis LLM's answer text.
        evidence: ``FinalAnswerDict.evidence`` entries
            (``{id, content, original_content, ...}``).

    Returns:
        Dict with ``markers_total``/``markers_resolved``/
        ``unknown_marker_ids``, ``quotes_checked``/``quotes_verified``/
        ``quotes_misquoted``, ``uncited_sentences``, and ``annotated_text``
        (the answer with unknown ids flagged inline as ``[n?]``; flagged
        statements stay visible -- nothing is ever deleted).
    """
    answer_text = answer_text or ""
    known_ids = {
        entry.get("id")
        for entry in evidence
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), int)
    }

    unknown_ids: List[int] = []
    resolved = 0
    annotated = answer_text

    def _annotate(match: "re.Match[str]") -> str:
        nonlocal resolved
        marker_id = int(match.group(1))
        if marker_id in known_ids:
            resolved += 1
            return match.group(0)
        if marker_id not in unknown_ids:
            unknown_ids.append(marker_id)
        return f"[{marker_id}?]"

    annotated = CITATION_MARKER_RE.sub(_annotate, answer_text)
    markers_total = len(CITATION_MARKER_RE.findall(answer_text))

    source_texts = _extract_source_texts(evidence)
    quotes_checked = 0
    quotes_verified = 0
    for span in QUOTE_SPAN_RE.findall(answer_text):
        quotes_checked += 1
        if match_quote_in_sources(span, source_texts)["matched"]:
            quotes_verified += 1

    # Uncited counting runs on the ORIGINAL answer (task-16814): the
    # annotated form rewrites unknown markers to "[n?]", which the numeric
    # marker regex cannot see -- those sentences ATTEMPTED citations and
    # must not be miscounted as uncited.
    original_sentences = [
        s for s in _SENTENCE_SPLIT_RE.split(answer_text) if s.strip()
    ]
    uncited_sentences = sum(
        1 for s in original_sentences if not CITATION_MARKER_RE.search(s)
    )

    # Per-claim detail (task-16325): every sentence carrying at least one
    # citation marker becomes a claim record -- resolved source ids, unknown
    # ids, per-sentence quote verdicts, and a supported/unverified status.
    # Uncited sentences are counted above but are not claims (no citation to
    # verify). Status is "supported" only when every marker resolves AND
    # every quote in the sentence verified. Sentences come from the ORIGINAL
    # answer: the annotated form's "[n?]" flags do not match the marker
    # regex, and the claim record should quote what was actually written.
    claims: List[Dict[str, Any]] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer_text):
        if not sentence.strip():
            continue
        sentence_markers = CITATION_MARKER_RE.findall(sentence)
        if not sentence_markers:
            continue
        sentence_source_ids: List[int] = []
        sentence_unknown: List[int] = []
        for marker in sentence_markers:
            marker_id = int(marker)
            if marker_id in known_ids:
                if marker_id not in sentence_source_ids:
                    sentence_source_ids.append(marker_id)
            elif marker_id not in sentence_unknown:
                sentence_unknown.append(marker_id)
        sentence_quotes_checked = 0
        sentence_quotes_verified = 0
        for span in QUOTE_SPAN_RE.findall(sentence):
            sentence_quotes_checked += 1
            if match_quote_in_sources(span, source_texts)["matched"]:
                sentence_quotes_verified += 1
        supported = (
            not sentence_unknown
            and sentence_quotes_verified == sentence_quotes_checked
        )
        claims.append(
            {
                "claim_id": f"claim-{len(claims) + 1}",
                "text": sentence,
                "source_ids": sentence_source_ids,
                "unknown_marker_ids": sentence_unknown,
                "quotes_checked": sentence_quotes_checked,
                "quotes_verified": sentence_quotes_verified,
                "status": "supported" if supported else "unverified",
            }
        )

    return {
        "markers_total": markers_total,
        "markers_resolved": resolved,
        "unknown_marker_ids": unknown_ids,
        "quotes_checked": quotes_checked,
        "quotes_verified": quotes_verified,
        "quotes_misquoted": quotes_checked - quotes_verified,
        "uncited_sentences": uncited_sentences,
        "annotated_text": annotated,
        "claims": claims,
    }


def summarize_for_footer(verification: Optional[Mapping[str, Any]]) -> str:
    """Render a verification summary as a compact footer segment; sections
    with nothing to report are omitted so a fully-clean run stays quiet.
    ``None`` (branch without a verdict) renders as the empty string."""
    if not verification:
        return ""
    markers_total = int(verification.get("markers_total") or 0)
    if markers_total <= 0:
        return ""
    parts = [
        f"Citations: {int(verification.get('markers_resolved') or 0)}/{markers_total} resolved"
    ]
    unknown = verification.get("unknown_marker_ids") or []
    if unknown:
        parts[0] += f" ({len(unknown)} unknown)"
    quotes_checked = int(verification.get("quotes_checked") or 0)
    if quotes_checked > 0:
        quote_bit = f"quotes {int(verification.get('quotes_verified') or 0)}/{quotes_checked} verified"
        misquoted = int(verification.get("quotes_misquoted") or 0)
        if misquoted:
            quote_bit += f" ({misquoted} misquoted)"
        parts.append(quote_bit)
    uncited = int(verification.get("uncited_sentences") or 0)
    if uncited:
        parts.append(f"{uncited} uncited sentence(s)")
    return ", ".join(parts)
