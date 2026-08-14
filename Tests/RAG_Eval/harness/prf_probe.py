# Tests/RAG_Eval/harness/prf_probe.py
"""PRF probe machinery: which terms the feedback set contributes, and how.

TASK-15965 Phase A. Pseudo-relevance feedback is the first P2c FEATURE
candidate, and it is probed before it is built -- the discipline that
retired the expansion, acronym and compositional premises. This module is
the probe's whole mechanism, kept pure and separate from the run so the
run's numbers can be evidence about PRF rather than about the probe:

1. `derive_expansion_terms` -- top-N terms by TF over the top-M
   pseudo-relevant documents, function words and the query's own
   vocabulary excluded (the classic RM3-shaped derivation, at its
   simplest honest form).
2. `compose_prf_expression` -- the second pass: the query's content terms
   OR-extended with those expansion terms.
3. `compose_feedback_expression` -- the ONE licensed variant's first
   pass (spec Step 0): an OR of the query's content terms, run FOR
   FEEDBACK SELECTION ONLY when the shipped AND-strict first pass fires
   on too few of the target queries to feed PRF anything at all.
4. `ProbeQueryResult` -- the per-query table row the probe report prints
   verbatim.

**Everything token-shaped here is the engine's own.** `RAGService`'s
static helpers do the tokenizing, the stopword test and the quoting, and
`_FTS5_STOPWORDS` is imported rather than copied. That is the TASK-15400
probe-fidelity lesson applied to construction as well as to execution:
a probe that builds a *similar* expression measures a query the engine
would never run. `Tests/RAG_Eval/test_prf_probe.py` pins
`compose_feedback_expression` byte-identical to what the engine's own
`or` construction emits, so "the same helpers" cannot drift into "the
same idea".

**Why per-token quoting is not optional in a probe.** The expansion terms
come from DOCUMENT text. Query-box input at least passes through a
product path; a corpus does not. A bare join would let a document
containing `Obsidian-3` raise `OperationalError('no such column: 3')` --
or, quieter and worse, let a document's own `OR` re-read the expression
as boolean logic and change what the probe measured without failing.
`_quote_fts5_token` is the single place that property is implemented and
this module inherits it.

**Nothing here executes a query.** The passes, the content fetches and
the verdict belong to the probe RUN, which is gated
(`harness_gate()`); this module is imported by always-on pure tests and
must stay importable with no env var set.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Collection, Sequence

from tldw_chatbook.RAG_Search.simplified.rag_service import (
    _FTS5_STOPWORDS,
    RAGService,
)

__all__ = [
    "DEFAULT_N_TERMS",
    "DEFAULT_TOP_M",
    "FTS5_STOPWORDS",
    "ProbeQueryResult",
    "compose_feedback_expression",
    "compose_prf_expression",
    "derive_expansion_terms",
]

#: The pre-registered BASE grid point (spec: "N=8, M=5 to start"). The
#: full grid {N: 4/8/16} x {M: 3/5/10} runs only if the base point shows
#: signal, and every point run is recorded.
DEFAULT_N_TERMS = 8
DEFAULT_TOP_M = 5

#: The engine's function-word list, re-exported so a probe run names one
#: source for it. 67 words, pinned in `Tests/RAG_Search/
#: test_fts5_match_construction.py`; NEVER copied here -- a probe that
#: trims a different set of words is expanding a different query.
FTS5_STOPWORDS = _FTS5_STOPWORDS

#: What FTS5's default tokenizer indexes: runs of alphanumerics, with `_`
#: excluded from the run the way `RAGService._fts5_term_key` reads it. A
#: derived term that is not an index term can never match a document, so
#: derivation and indexing must agree on where a word ends.
_TERM_RUN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _index_terms(text: str) -> list[str]:
    """The lowercased index terms of a piece of text, in order.

    Args:
        text: Document text, or a raw query token.

    Returns:
        Every alphanumeric run, case-folded; empty for text with none.
    """
    return [run.lower() for run in _TERM_RUN_RE.findall(text)]


def derive_expansion_terms(
    docs: Sequence[str],
    *,
    query_terms: Collection[str],
    n_terms: int,
    stopwords: Collection[str],
) -> tuple[str, ...]:
    """The top-N characteristic terms of a pseudo-relevant document set.

    Term frequency over the WHOLE fed set (total occurrences, not
    document frequency): with M as small as 3-10, document frequency
    collapses to a handful of distinct values and ties on nearly
    everything, which is precisely the regime a tie-break has to survive.

    Two exclusions, both load-bearing:

    * **Function words.** Raw TF over any English text puts "the" and
      "of" on top, and an OR-shaped second pass led by those matches
      essentially the corpus.
    * **The query's own terms.** They are already in the second pass via
      its query side; re-deriving them would spend the N budget on terms
      that expand nothing. Query terms are compared as INDEX terms, so a
      raw token (`spindle-bearing`) excludes both of the words FTS5
      indexes it as.

    Ordering is total: count descending, then alphabetically. Without the
    second key the order falls to dictionary insertion -- i.e. to
    whichever document the probe happened to read first -- and a grid
    point stops being reproducible in exactly the small-count regime
    where ties are the rule.

    Args:
        docs: The pseudo-relevant documents' text, already fetched (the
            probe pays a content fetch per fed row; four-seam media and
            conversation rows carry label snippets, not text).
        query_terms: The query's raw tokens, excluded from the result.
        n_terms: The N cut; ``<= 0`` yields no terms.
        stopwords: The function words to exclude -- `FTS5_STOPWORDS` on
            every probe path.

    Returns:
        At most ``n_terms`` lowercased index terms, most frequent first,
        ties broken alphabetically. Empty when the documents contribute
        nothing -- the structural case where a zero-row first pass feeds
        PRF nothing at all.
    """
    if n_terms <= 0:
        return ()

    excluded = {term.lower() for term in stopwords}
    for token in query_terms:
        excluded.update(_index_terms(token))

    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(
            term for term in _index_terms(doc) if term not in excluded
        )

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(term for term, _count in ranked[:n_terms])


def _quoted_content_tokens(query: str) -> list[str]:
    """The query's content tokens, individually quoted by the engine.

    The exact three-step the engine's own content-token forms run
    (`RAGService._fts5_match_expressions`): tokenize once, drop the
    function words, quote each survivor.

    Args:
        query: Raw query text.

    Returns:
        One quoted FTS5 term per surviving token, in query order.
    """
    return [
        RAGService._quote_fts5_token(token)
        for token in RAGService._fts5_query_tokens(query)
        if not RAGService._is_fts5_stopword(token)
    ]


def compose_prf_expression(
    query: str, expansion_terms: Sequence[str]
) -> str:
    """The PRF second pass: the query's content terms, OR-extended.

    Query side first in query order, then the expansion terms in their
    derived order, so a reported expression reads against the term list
    that produced it. The join is an OR because the expansion's value is
    REACH -- an AND with terms the target may not carry would narrow the
    pass that is supposed to widen it.

    An expansion term with no alphanumeric run is dropped rather than
    quoted: `""` inside a MATCH expression is a syntax error, not an
    inert token.

    Args:
        query: The raw query.
        expansion_terms: Terms from `derive_expansion_terms` (or any
            sequence -- this function assumes nothing about their
            provenance, which is why it quotes them).

    Returns:
        The MATCH expression, or ``""`` when nothing survives. ``""``
        means "no rows, skip the query" -- the engine's existing skip
        contract; an empty string handed to MATCH raises.
    """
    quoted = _quoted_content_tokens(query)
    quoted += [
        RAGService._quote_fts5_token(term)
        for term in expansion_terms
        if _TERM_RUN_RE.search(term)
    ]
    return " OR ".join(quoted)


def compose_feedback_expression(query: str) -> str:
    """The ONE licensed variant's first pass (spec Step 0).

    The shipped four-seam first pass ANDs across the query's terms, so a
    paraphrase query that shares no content word with any document
    returns zero rows -- and a zero-row first pass feeds PRF nothing,
    structurally. The spec pre-registers exactly one fallback for that
    regime: run the first pass in an OR-of-content-terms form FOR
    FEEDBACK SELECTION ONLY. Users would still see the shipped results;
    this wider pass exists only to choose the pseudo-relevant set, which
    is classic PRF over a candidate set.

    Byte-identical to what the engine's `or` construction emits (pinned
    in `test_prf_probe.py`), which is what makes a result measured
    through it a result about a form the engine can actually run.

    Args:
        query: The raw query.

    Returns:
        The MATCH expression, or ``""`` when trimming leaves nothing --
        the skip contract again.
    """
    return " OR ".join(_quoted_content_tokens(query))


@dataclass(frozen=True, slots=True)
class ProbeQueryResult:
    """One query's row in the probe report, printed verbatim.

    Every column answers a question the spec's admission bar or its
    honesty notes ask, which is why none of them has a default -- a
    forgotten column is a report that quietly cannot answer one of them.

    Attributes:
        query_id: The golden query's id. Gains and losses are reported BY
            ID (the TASK-15700 lost-column discipline); an aggregate
            cannot show which query paid for which rescue.
        category: The golden category -- the probe's populations
            (paraphrase/vocabulary_mismatch as target, negation as
            regression guard, negatives as junk guard) behave differently
            enough that a mixed average means nothing.
        fireable: Whether the first pass returned anything to feed from.
            Step 0's census answer for this query, and the arc's central
            structural question.
        first_pass_rows: Rows the first pass returned.
        fed_docs: How many of those rows' documents actually fed term
            derivation (the top-M cut).
        content_fetches: Reads spent fetching text for the fed rows. The
            PRICE, reported rather than hidden: four-seam media and
            conversation rows carry label snippets ("Matched media ·
            {type}"), not text, so without the fetch the feed silently
            skews toward notes and prompts.
        target_rank_before: The target's 1-based rank in the first pass,
            or ``None`` for a miss. Never 0 -- a miss is an absence.
        target_rank_after: The same after the expanded second pass. The
            column that lets the report state losses as well as gains.
        rows_after: Rows the second pass returned -- the negation and
            negatives guards are read off this.
    """

    query_id: str
    category: str
    fireable: bool
    first_pass_rows: int
    fed_docs: int
    content_fetches: int
    target_rank_before: int | None
    target_rank_after: int | None
    rows_after: int
