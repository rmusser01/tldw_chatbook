# Tests/RAG_Eval/test_prf_probe_run.py
r"""THE PRF PROBE RUN: the fireability census, the grid, the mechanical verdict.

TASK-15965 Phase A, Task 2. `harness/prf_probe.py` is the mechanism (pure,
always-on-tested); this module is the one place it meets a real corpus, a
real index and the product's own four-seam keyword path. It prints three
things and asserts a fourth:

1. **STEP 0 — the fireability census**, run FIRST because it can decide the
   whole arc in one pass: for each of the 22 plain-failing queries
   (paraphrase 13 + vocabulary_mismatch 9, derived from the golden set at
   run time rather than hardcoded), does the SHIPPED first pass return any
   rows at all? A zero-row first pass feeds PRF nothing, structurally. At
   `< 5/22` the spec's ONE licensed variant activates
   (`compose_feedback_expression`, feedback selection only) and every table
   below says so.
2. **The grid** — the base point N=8/M=5 first, and the full
   {4,8,16}x{3,5,10} only if the base point rescues at least one query.
   Every point that runs is recorded, including the ones that move nothing.
3. **The guards** — the currently-hitting plain population derived from a
   fresh baseline pass (gains AND losses by query id: the TASK-15700
   lost-column discipline), the 3 negation queries (rows + the documents
   the second pass added), and the 7 negatives. Every miss is diagnosed by
   MECHANISM at `DEEP_K`, because "lost" without a cause is a number a
   reader completes from their own prior.
4. The assertions pin **the instrument, never the outcome**: that the census
   covered 22 queries, that every pass routed through `local-fts`, that the
   grid's base point is the pre-registered one, that the seam was left
   exactly as it was found. A test that asserted "PRF rescues 5" would turn
   the arc's pre-registered NULL into a red test, which is the opposite of
   fail-first.

**How the probe hands the engine its own expression.** The four-seam path
builds a MATCH string with `build_fts_match_query` at four call sites
(`library_local_rag_search_service.py:494/528/548/586`) and hands it to the
`db.search_*`-family functions. For a non-shipped pass this module
substitutes THAT ONE FUNCTION for the duration of one call and leaves every
other line of the path alone -- the scope handling, the seam fan-out and
ordering, the row builders, the empty/blocked outcomes are all the product's.
Replicating the call sites by hand was the alternative and is strictly
worse: it would re-implement the scoped-seam exclusions and the row shapes,
and the TASK-15400 probe-fidelity lesson is that a probe measuring a
*similar* path measures a query the engine would never run. The SHIPPED
pass is run with no substitution at all, so the before-columns are the
product's own behaviour by construction rather than by argument.

**Four properties of this corpus/harness that the numbers cannot be read
without.**

* *The plain path here is three seams, not four.* `build_eval_runtime` wires
  `prompt_scope_service=None` (documented there as a real gap: the Library's
  plain fan-out reaches prompts, this harness's does not), so the 5 prompt
  queries are unreachable in every plain pass and never enter the hitter
  population. The engine-side prompts sub-leg is a hybrid-mode thing and
  hybrid is not this probe's population.
* *"Top-M" is ROTATION order, not relevance order.* The four-seam path
  interleaves its seams rank-fairly -- TASK-16071 replaced the fixed-order
  concatenation with `interleave_rankings`, so the merged list runs each
  seam's rank-1 row, then each seam's rank-2 row, and so on -- and the rows
  still carry no cross-seam score at all (`_note_row` and friends set
  `"score": None` deliberately), which is why rank POSITION within a seam is
  the only comparable signal there is. Classic PRF feeds from the top-M by
  relevance; the honest statement of what this probe feeds from is "the
  first M documents the product would show", which is what a plain-profile
  user's PRF would have to feed from too. That is not a cosmetic
  distinction: when the merge changed, the fed set's label-only share (media
  and conversation rows, which carry no text in the row itself) went 39/211
  to 113/211 at the base point. WHAT a top-M consumer sees moved, not just
  the order it sees it in.
* *The merge order used to interact with EXPRESSION BREADTH; now the
  per-seam BUDGET does*, which is why `_run_oracle_control` runs under
  several term SELECTORS rather than one. `top_k` is a PER-SEAM limit, and
  under a rank-fair rotation over this harness's three seams each seam holds
  only about ceil(K/3) of the K merged slots -- so a target ranked deeper
  than that WITHIN ITS OWN SEAM is unreachable however well it matched, and
  15 of the 22 targets are media or conversation documents. How hard that
  bites is **not a fixed property of the path**: it depends on how broad the
  expansion is, which is a property of the SELECTOR. Measured with oracle
  feeds (the target document itself), the pre-registered TF-8 selector
  leaves 14/22 observable and a rarest-8-by-corpus-DF selector of the same
  shape leaves 19/22. Before TASK-16071 those same two rows read 8/22 and
  15/22, under a harsher constraint of a different kind: the merge
  CONCATENATED in seam order, so a full notes seam pushed out every media
  and conversation target whatever its own rank. That component is gone --
  the entire 8->14 and 15->19 gain is the conversation column (0/6 -> 6/6,
  2/6 -> 6/6), whose targets were all rank 1 in their own seam -- and what
  survives is per-seam VOLUME. An earlier version of this module printed the
  8/22 as "could not have been rescued by any real feedback set"; a review
  refuted that by running it, and the refutation is now part of the
  instrument rather than a note about it. Measure the ceiling before reading
  the floor, and measure it under more than one selector before calling it a
  ceiling at all.
* *The second pass drops the plural/singular widening the first pass has.*
  `build_fts_match_query` widens each term into an OR-group of naive
  variants; `compose_prf_expression` is the engine's own content-token form
  and does not, so a target reached only through a variant would be
  genuinely unreachable in the second pass. That was a live hypothesis for
  the losses and the DEEP_K diagnosis REFUTED it: on this corpus every lost
  hitter is still matched at k=200 (0 unmatched, measured), so the losses
  are dilution and nothing else. The hypothesis is kept written down because
  a corpus edit could revive it, and because a loss column that only ever
  said "lost" would have let the wrong cause ride.

**PRF fires only when it has feedback.** When the feed produces no expansion
terms the second pass is NOT run and the after-state IS the shipped
first pass -- the feature semantics ("nothing to expand with, so show the
user what they already get"), and the reason the spec calls the negatives
guard STRUCTURAL under the shipped regime: a zero-row first pass cannot
produce a second pass to add rows with. Under the licensed variant the feed
fires on negatives too, terms are derived, the second pass runs, and the
same guard becomes LIVE.

Skipped unless `RAG_EVAL=1` plus the embeddings extras plus a warm model
cache -- the same gate every harness module uses, never a new one:

    RAG_EVAL=1 .venv/bin/pytest Tests/RAG_Eval/test_prf_probe_run.py -s
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from Tests.RAG_Eval.harness.environment import harness_gate

pytestmark = harness_gate()

#: Result cap and the depth every rank here is stated at -- the harness's own
#: k, so a rank this probe reports is the rank `run_eval` would score.
K = 10

#: Depth of the LOSS DIAGNOSIS pass, and the reason it exists. `top_k` is a
#: PER-SEAM cap in the four-seam path, so a target missing from a pass run at
#: k=10 has two completely different possible causes, and the merged list
#: cannot tell them apart: it was still matched but ranked past its own
#: seam's 10 rows (DILUTION -- a ranking problem), or the composed second
#: pass does not match the document at all (CONSTRUCTION -- the four-seam
#: builder's plural/singular widening is not in `compose_prf_expression`, so
#: a document reached only through a variant is genuinely unreachable). Both
#: read as "absent" at k=10. Re-running the SAME expression deep separates
#: them, and the answer changes what a Phase B would have to fix.
DEEP_K = 200

#: The pre-registered base grid point (spec: "N=8, M=5 to start"). Pinned
#: against `prf_probe`'s own constants inside the run.
BASE_POINT: tuple[int, int] = (8, 5)

#: The pre-registered full grid, run ONLY if the base point shows signal.
GRID_N: tuple[int, ...] = (4, 8, 16)
GRID_M: tuple[int, ...] = (3, 5, 10)

#: The probe's target population: the plain-failing cells PRF claims to fix.
TARGET_CATEGORIES: tuple[str, ...] = ("paraphrase", "vocabulary_mismatch")

#: The regression-guard populations, named separately because they answer
#: different clauses of the bar.
NEGATION_CATEGORY = "negation"
NEGATIVE_CATEGORY = "negative"

#: Step 0's threshold: at or above it the base grid runs on the shipped
#: first pass; below it the ONE licensed variant activates.
FIREABILITY_FLOOR = 5

#: The bar's first clause: rescued queries needed for ADMIT.
RESCUE_FLOOR = 5

#: What a plain pass must report as its backend. Asserting it is what proves
#: the substitution did not accidentally re-route the search.
PLAIN_BACKEND = "local-fts"


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CensusRow:
    """One query's Step 0 row: what each candidate first pass returned.

    Both passes are recorded for every query even when only one regime is
    active, because "the shipped pass fires on 2 and the variant on 21" and
    "neither fires" are different findings that would otherwise produce the
    same activated-variant headline.
    """

    query_id: str
    category: str
    shipped_rows: int
    shipped_docs: int
    variant_rows: int
    variant_docs: int
    variant_expression: str


@dataclass(frozen=True, slots=True)
class _OracleRow:
    """One target query under a PERFECT feed — the rescue-channel control.

    A null rescue count only means "PRF did not rescue" if a rescue was
    OBSERVABLE at all. The four-seam path interleaves its seams rank-fairly
    (TASK-16071) and its rows carry no cross-seam score, so under this
    harness's three-seam fan-out each seam holds only about ceil(k/3) of the
    k merged slots: a target ranked deeper than that WITHIN ITS OWN SEAM is
    pushed past rank 10 no matter how good the expansion was. 15 of the 22
    targets are media or conversation documents (measured), so that is not a
    hypothetical. Before TASK-16071 the constraint was harsher and of a
    different kind -- the merge CONCATENATED in seam order, so once an
    expanded pass filled the notes seam's `top_k` rows every media and
    conversation target was pushed out whatever its own rank. That component
    is gone; per-seam volume is what remains.

    This control removes the FEED from the question by feeding PRF the
    TARGET DOCUMENT ITSELF — the best feed any feedback set could supply.
    The resulting expression, and therefore the rank, is a JOINT property
    of that feed and the SELECTOR that built the expression from it (this
    module measures 14/22 vs 19/22 vs 22/22 over the identical feed under
    three selectors — see the printed selector table): an oracle row's
    rank is a ceiling only UNDER THE SELECTOR THAT BUILT IT, never an
    absolute. It is a CONTROL and never an input to the verdict — an
    oracle feed is not something a retrieval system has.
    """

    query_id: str
    category: str
    target_slug: str
    target_source_type: str
    terms: tuple[str, ...]
    rank_at_k: int | None
    position: int | None
    docs_returned: int
    #: Where the SAME expression puts the target at k=DEEP_K. An oracle feed
    #: is derived from the document itself, so the expression must match it;
    #: this column is what turns that from an argument into a measurement,
    #: and it is what proves an oracle miss is seam displacement rather than
    #: a broken control.
    deep_position: int | None


@dataclass(frozen=True, slots=True)
class _QueryPoint:
    """One query at one grid point: the probe row plus its working."""

    result: Any  # ProbeQueryResult
    terms: tuple[str, ...]
    expression: str
    fed_slugs: tuple[str, ...]
    fed_docs_with_row_text: int
    docs_before: tuple[str, ...]
    docs_after: tuple[str, ...]
    slugs_lost: tuple[str, ...]
    slugs_gained: tuple[str, ...]
    docs_added: int
    #: The target's UNCAPPED 1-based position in the second pass's document
    #: list, or None when it is absent entirely. The column that separates
    #: the two ways a hitter can be "lost": DILUTION (still returned, pushed
    #: past rank 10 by expansion-term matches) and ABSENCE (the composed
    #: second pass does not match the document at all — the real cost of
    #: dropping the four-seam builder's plural/singular widening). Reporting
    #: one number for both would attribute every loss to whichever
    #: mechanism the reader already believed in.
    target_position_after: int | None
    #: Whether the DEEP_K diagnosis pass ran for this query, and where the
    #: target landed in it. `deep_ran=False` means the question did not
    #: arise (the target was already at k, or PRF never fired);
    #: `deep_ran=True, deep_position=None` is the only reading that means
    #: "the composed second pass does not match this document at all".
    deep_ran: bool
    deep_position: int | None


@dataclass(frozen=True, slots=True)
class _PointReport:
    """Everything one (N, M) grid point produced."""

    n_terms: int
    top_m: int
    wall_s: float
    points: tuple[_QueryPoint, ...]
    content_fetches: int
    #: Which term selector produced this point. Only the pre-registered one
    #: reaches the verdict; the label is carried so a table can never show a
    #: control row and a verdict row without saying which is which.
    selector: str

    def by_category(self, categories: Sequence[str]) -> tuple[_QueryPoint, ...]:
        return tuple(
            point
            for point in self.points
            if point.result.category in categories
        )

    @property
    def rescued(self) -> tuple[_QueryPoint, ...]:
        """Target-population queries that missed before and hit after."""
        return tuple(
            point
            for point in self.by_category(TARGET_CATEGORIES)
            if point.result.target_rank_before is None
            and point.result.target_rank_after is not None
        )

    @property
    def lost(self) -> tuple[_QueryPoint, ...]:
        """Any query that held a target document before and lost it after."""
        return tuple(point for point in self.points if point.slugs_lost)


# ---------------------------------------------------------------------------
# The passes
# ---------------------------------------------------------------------------


def _run_pass(
    runtime: Any,
    seam: Any,
    service_module: Any,
    lookup: Mapping[tuple[str, str], str],
    query_text: str,
    scope: Any,
    *,
    expression: str | None = None,
    top_k: int = K,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    """Run ONE four-seam keyword pass and canonicalize its rows.

    Args:
        runtime: The live `EvalRuntime` (owns the only loop the service's
            pools are bound to).
        seam: A `LibraryLocalRagSearchService` over that runtime's app.
        service_module: The seam's own module -- the substitution target.
        lookup: `slug_lookup_from(runtime.slug_to_source)`.
        query_text: The raw golden query. Always passed, even when an
            expression is substituted: the seam validates it and the notes
            and media seams take it as their `query=` argument exactly as
            production does.
        scope: The query's `EffectiveScope`, or None.
        expression: ``None`` runs the SHIPPED pass with no substitution at
            all. A string is handed at the four `build_fts_match_query` call
            sites instead. ``""`` is the engine's skip contract -- an empty
            MATCH string either raises or, worse, falls back to the raw
            quoted query at the notes seam's `safe_search_term` line -- so it
            returns no rows without touching the DB.

    Returns:
        ``(rows, canonicalized doc ids)``.

    Raises:
        AssertionError: The seam erred, or reported a backend other than the
            four-seam keyword path.
    """
    from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids
    from Tests.RAG_Eval.harness.runner import SOURCE_TYPES, _extract_rows

    if expression is not None and not expression:
        return (), ()

    if expression is None:
        result = runtime.run(
            seam.search(query_text, SOURCE_TYPES, "rag", top_k=top_k, scope=scope)
        )
    else:
        original = service_module.build_fts_match_query
        service_module.build_fts_match_query = (
            lambda _query, _expression=expression: _expression
        )
        try:
            result = runtime.run(
                seam.search(query_text, SOURCE_TYPES, "rag", top_k=top_k, scope=scope)
            )
        finally:
            service_module.build_fts_match_query = original

    rows, backend, error = _extract_rows(result)
    if error is not None:
        raise AssertionError(
            f"the four-seam pass erred for {query_text!r} "
            f"(expression={expression!r}): {error}"
        )
    if backend != PLAIN_BACKEND:
        raise AssertionError(
            f"the pass for {query_text!r} routed to {backend!r}, not "
            f"{PLAIN_BACKEND!r} -- it did not measure the four-seam path"
        )
    return tuple(rows), tuple(rows_to_doc_ids(rows, lookup))


def _fetch_document_text(
    runtime: Any, slug: str, media_db: Any, chachanotes_db: Any
) -> str:
    """One real content read for one fed document -- the PRICE, paid.

    The spec's row-content fact, verified again here: only `_note_row`
    carries the document's text; `_media_row`'s snippet is
    ``"Matched media · {type}"`` and `_conversation_row`'s is
    ``"Matched conversation · N messages"``. Deriving terms from the rows
    alone would silently feed PRF notes and nothing else. Nothing in
    `prf_probe` enforces this fetch -- the run does, and counts it.

    Args:
        runtime: The live runtime (its `slug_to_source` resolves the slug).
        slug: A canonicalized fixture slug from a first-pass row.
        media_db: The scratch `MediaDatabase`.
        chachanotes_db: The scratch `CharactersRAGDB`.

    Returns:
        The document's full text.

    Raises:
        AssertionError: The slug is unknown, or its source type has no read
            path here (a prompt row cannot reach a plain pass in this
            harness; if one ever does, the feed must not silently skip it).
    """
    entry = runtime.slug_to_source.get(slug)
    if entry is None:
        raise AssertionError(
            f"fed row {slug!r} is not a corpus document; the feed cannot be "
            "priced or derived from"
        )
    source_type, source_id = entry
    if source_type == "media":
        row = media_db.get_media_by_id(int(source_id))
        return str((row or {}).get("content") or "")
    if source_type == "note":
        row = chachanotes_db.get_note_by_id(source_id)
        return str((row or {}).get("content") or "")
    if source_type == "conversation":
        messages = chachanotes_db.get_messages_for_conversation(
            source_id, limit=500, include_image_data=False
        )
        return "\n".join(str(message.get("content") or "") for message in messages)
    raise AssertionError(
        f"fed row {slug!r} has source_type {source_type!r}, which this probe "
        "has no content read for"
    )


@dataclass(frozen=True, slots=True)
class _Selector:
    """One way of choosing expansion terms, plus how it composes them.

    The probe's verdict is computed from the PRE-REGISTERED selector alone
    (`derive_expansion_terms`, RM3 tf/|D|). The others exist because a
    review refuted a claim this module used to print: that the oracle
    ceiling (8/22 as it then read; 14/22 since TASK-16071) was a property of
    the four-seam path. It is not -- it is a property of how BROAD the
    chosen terms are, and the only way to say so honestly is to vary the
    selector and hold everything else fixed.

    Attributes:
        name: Short label used in every table.
        disclosure: What this selector changes relative to the
            pre-registered one, stated in the report so a reader never has
            to infer which variables moved.
        pre_registered: True only for the spec's own derivation. Anything
            False is a CONTROL and can never reach the verdict.
        select: ``(docs, query_terms=, n_terms=, stopwords=) -> terms``.
        compose: ``(query, terms) -> MATCH expression``.
    """

    name: str
    disclosure: str
    pre_registered: bool
    select: Callable[..., tuple[str, ...]]
    compose: Callable[[str, Sequence[str]], str]


def _corpus_document_frequency(corpus: Sequence[Any]) -> dict[str, int]:
    """Term -> how many corpus documents contain it.

    Over document CONTENT, which is exactly what `_fetch_document_text`
    feeds derivation, so a "rare" term is rare in the text PRF actually
    reads rather than in some other view of the corpus.
    """
    from Tests.RAG_Eval.harness.prf_probe import _index_terms

    frequency: Counter[str] = Counter()
    for doc in corpus:
        frequency.update(set(_index_terms(doc.content)))
    return dict(frequency)


def _rarest_terms(
    docs: Sequence[str],
    *,
    query_terms: Sequence[str],
    n_terms: int,
    stopwords: Sequence[str],
    document_frequency: Mapping[str, int],
) -> tuple[str, ...]:
    """The N rarest terms of the fed set, by corpus document frequency.

    The single-variable counterpart to `derive_expansion_terms`: same fed
    documents, same exclusions (function words, the query's own index
    terms), same total ordering discipline -- only the RANKING KEY changes,
    from "heaviest by tf/|D|" to "rarest by corpus DF". TF picks terms that
    are frequent in the fed set, and on this corpus that means broad ones
    ("rather", "once", "each"); DF picks terms few documents share.

    `_index_terms` is imported rather than re-derived for the same reason
    `prf_probe` states in its own docstring: a probe that tokenizes
    differently from the engine measures a query the engine would not run.

    Args:
        docs: The fed documents' text.
        query_terms: The query's raw tokens, excluded as index terms.
        n_terms: The N cut; ``<= 0`` yields no terms.
        stopwords: Function words to exclude (`FTS5_STOPWORDS` everywhere).
        document_frequency: `_corpus_document_frequency` output.

    Returns:
        At most ``n_terms`` index terms, rarest first, ties alphabetical.
    """
    from Tests.RAG_Eval.harness.prf_probe import _index_terms

    if n_terms <= 0:
        return ()
    excluded = {term.lower() for term in stopwords}
    for token in query_terms:
        excluded.update(_index_terms(token))
    candidates = {
        term
        for doc in docs
        for term in _index_terms(doc)
        if term not in excluded
    }
    ranked = sorted(
        candidates, key=lambda term: (document_frequency.get(term, 0), term)
    )
    return tuple(ranked[:n_terms])


def _compose_terms_only(query: str, terms: Sequence[str]) -> str:
    """An OR of the expansion terms with the QUERY SIDE dropped entirely.

    The narrow endpoint of the breadth axis, and the one composition here
    that is not `compose_prf_expression`. It exists to show that the
    observability ceiling moves all the way to 22/22 when the expression
    stops matching everything the query's own content words match -- i.e.
    that breadth, not the seam order alone, was what bound. Since TASK-16071
    removed seam order from the merge entirely, breadth against the per-seam
    budget is the whole of what still binds here. Because it
    changes TWO things at once (selector AND composition) it is labelled an
    ILLUSTRATION rather than a controlled comparison wherever it is printed.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    quoted = [
        RAGService._quote_fts5_token(term)
        for term in terms
        if any(character.isalnum() for character in term)
    ]
    return " OR ".join(quoted)


#: Source types whose four-seam ROW carries the document's own text. Only
#: notes: `_note_row`'s snippet is `item["content"]`, while `_media_row` and
#: `_conversation_row` emit the labels "Matched media · {type}" and "Matched
#: conversation · N messages" (`library_local_rag_search_service.py:1081-1120`).
#: This is the whole reason `_fetch_document_text` exists, and counting it
#: turns the spec's row-content fact into a measured column.
_SOURCE_TYPES_WITH_ROW_TEXT: frozenset[str] = frozenset({"note"})


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _slug_rank(doc_ids: Sequence[str], slugs: Sequence[str]) -> int | None:
    """Best 1-based rank any of ``slugs`` reached within the first K ids."""
    from Tests.RAG_Eval.harness.fixture_probe import rank_of

    ranks = [rank_of(doc_ids, slug, K) for slug in slugs]
    found = [rank for rank in ranks if rank is not None]
    return min(found) if found else None


def _slug_position(doc_ids: Sequence[str], slugs: Sequence[str]) -> int | None:
    """Best UNCAPPED 1-based position any of ``slugs`` reached, or None.

    `_slug_rank` at the measured depth answers the bar; this answers *why* a
    query missed it. Same function, no k.
    """
    from Tests.RAG_Eval.harness.fixture_probe import rank_of

    ranks = [rank_of(doc_ids, slug, len(doc_ids)) for slug in slugs]
    found = [rank for rank in ranks if rank is not None]
    return min(found) if found else None


def _run_oracle_control(
    *,
    targets: Sequence[Any],
    n_terms: int,
    scopes: Mapping[str, Any],
    runtime: Any,
    seam: Any,
    service_module: Any,
    lookup: Mapping[tuple[str, str], str],
    media_db: Any,
    chachanotes_db: Any,
    selector: _Selector,
) -> tuple[_OracleRow, ...]:
    """Feed PRF the target document itself and see whether it can win.

    The rescue-channel measurement (see `_OracleRow`). Not part of the grid
    and not part of the verdict: a real feedback set never contains the
    answer by construction, so an oracle rescue proves only that the CHANNEL
    is open under this selector, and an oracle miss proves it was shut
    before PRF was asked -- **under this selector**. That qualifier is the
    whole correction: run this with one selector and the number reads as a
    property of the retrieval path, which a review measured to be false.
    """
    from Tests.RAG_Eval.harness.prf_probe import FTS5_STOPWORDS

    rows: list[_OracleRow] = []
    for query in targets:
        slug = query.relevant_slugs[0]
        text = _fetch_document_text(runtime, slug, media_db, chachanotes_db)
        terms = selector.select(
            [text],
            query_terms=query.query.split(),
            n_terms=n_terms,
            stopwords=FTS5_STOPWORDS,
        )
        expression = selector.compose(query.query, terms) if terms else ""
        _rows, docs = _run_pass(
            runtime,
            seam,
            service_module,
            lookup,
            query.query,
            scopes[query.id],
            expression=expression,
        )
        rank_at_k = _slug_rank(docs, (slug,))
        deep_position: int | None = None
        if rank_at_k is None and expression:
            _deep_rows, deep_docs = _run_pass(
                runtime,
                seam,
                service_module,
                lookup,
                query.query,
                scopes[query.id],
                expression=expression,
                top_k=DEEP_K,
            )
            deep_position = _slug_position(deep_docs, (slug,))
        rows.append(
            _OracleRow(
                query_id=query.id,
                category=query.category,
                target_slug=slug,
                target_source_type=runtime.slug_to_source[slug][0],
                terms=terms,
                rank_at_k=rank_at_k,
                position=_slug_position(docs, (slug,)),
                docs_returned=len(docs),
                deep_position=deep_position,
            )
        )
    return tuple(rows)


def _run_point(
    *,
    n_terms: int,
    top_m: int,
    golden: Sequence[Any],
    shipped: Mapping[str, tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]],
    feed_docs: Mapping[str, tuple[str, ...]],
    fireable: Mapping[str, bool],
    scopes: Mapping[str, Any],
    runtime: Any,
    seam: Any,
    service_module: Any,
    lookup: Mapping[tuple[str, str], str],
    media_db: Any,
    chachanotes_db: Any,
    selector: _Selector,
) -> _PointReport:
    """Run every golden query at one (N, M) grid point.

    The shipped first pass and the feed pass are computed once per run and
    passed in: neither depends on N, M or the selector, and re-running them
    per point would price the probe rather than the feature.

    ``selector`` is the pre-registered TF derivation for every point the
    VERDICT reads. Points run under any other selector are axis controls and
    are kept in a separate list by the caller, never handed to
    `_format_verdict` -- a control that can reach the verdict is a grid
    search wearing a different hat.
    """
    from Tests.RAG_Eval.harness.prf_probe import FTS5_STOPWORDS, ProbeQueryResult

    started = time.perf_counter()
    points: list[_QueryPoint] = []
    fetches = 0
    for query in golden:
        shipped_rows, shipped_docs = shipped[query.id]
        fed_slugs = tuple(feed_docs[query.id][:top_m])

        texts: list[str] = []
        for slug in fed_slugs:
            texts.append(_fetch_document_text(runtime, slug, media_db, chachanotes_db))
            fetches += 1

        terms = selector.select(
            texts,
            query_terms=query.query.split(),
            n_terms=n_terms,
            stopwords=FTS5_STOPWORDS,
        )
        expression = selector.compose(query.query, terms) if terms else ""
        if terms and expression:
            _after_rows, docs_after = _run_pass(
                runtime,
                seam,
                service_module,
                lookup,
                query.query,
                scopes[query.id],
                expression=expression,
            )
            rows_after = len(_after_rows)
        else:
            # PRF did not fire: no feedback, no second pass, and the user
            # keeps exactly the shipped result. See the module docstring.
            docs_after = shipped_docs
            rows_after = len(shipped_rows)

        before_set = set(shipped_docs[:K])
        after_set = set(docs_after[:K])
        relevant = tuple(query.relevant_slugs)

        # The loss diagnosis (see DEEP_K): run only when the target is
        # missing at k, which is exactly when "dilution or construction?"
        # is an open question. Skipped when PRF did not fire, because then
        # the deep pass would re-measure the SHIPPED expression and answer a
        # question nobody asked.
        deep_position: int | None = None
        deep_ran = False
        if relevant and _slug_rank(docs_after, relevant) is None and expression:
            deep_ran = True
            _deep_rows, deep_docs = _run_pass(
                runtime,
                seam,
                service_module,
                lookup,
                query.query,
                scopes[query.id],
                expression=expression,
                top_k=DEEP_K,
            )
            deep_position = _slug_position(deep_docs, relevant)

        points.append(
            _QueryPoint(
                result=ProbeQueryResult(
                    query_id=query.id,
                    category=query.category,
                    fireable=fireable[query.id],
                    first_pass_rows=len(shipped_rows),
                    fed_docs=len(fed_slugs),
                    content_fetches=len(fed_slugs),
                    target_rank_before=_slug_rank(shipped_docs, relevant),
                    target_rank_after=_slug_rank(docs_after, relevant),
                    rows_after=rows_after,
                ),
                terms=terms,
                expression=expression,
                fed_slugs=fed_slugs,
                fed_docs_with_row_text=sum(
                    1
                    for slug in fed_slugs
                    if runtime.slug_to_source[slug][0] in _SOURCE_TYPES_WITH_ROW_TEXT
                ),
                docs_before=shipped_docs,
                docs_after=docs_after,
                slugs_lost=tuple(
                    slug
                    for slug in relevant
                    if slug in before_set and slug not in after_set
                ),
                slugs_gained=tuple(
                    slug
                    for slug in relevant
                    if slug not in before_set and slug in after_set
                ),
                docs_added=len(after_set - before_set),
                target_position_after=_slug_position(docs_after, relevant),
                deep_ran=deep_ran,
                deep_position=deep_position,
            )
        )
    return _PointReport(
        n_terms=n_terms,
        top_m=top_m,
        wall_s=time.perf_counter() - started,
        points=tuple(points),
        content_fetches=fetches,
        selector=selector.name,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _rank(value: int | None) -> str:
    return "miss" if value is None else str(value)


def _miss_mechanism(point: _QueryPoint) -> str:
    """Why the target is not at k after the second pass, in three readings.

    `top_k` is a PER-SEAM cap, so "absent from the merged list" and "not
    matched" are different facts that look identical at k=10. Only the
    DEEP_K pass can separate them (see `DEEP_K`), and only its two outcomes
    are allowed to claim the strong reading.
    """
    if point.target_position_after is not None:
        return (
            f"MERGE-DISPLACED: still returned, at merged position "
            f"{point.target_position_after}, behind earlier seams"
        )
    if not point.deep_ran:
        return "not returned (no second pass ran, so nothing displaced it)"
    if point.deep_position is not None:
        return (
            f"SEAM-DISPLACED: the same expression at k={DEEP_K} returns it at "
            f"position {point.deep_position}, so it matched and lost its "
            f"own seam's {K} slots to expansion-term rows"
        )
    return (
        f"UNMATCHED: the composed second pass does not return it even at "
        f"k={DEEP_K} — the expansion never reached the document"
    )


def _format_census(
    rows: Sequence[_CensusRow], *, fireable_shipped: int, fireable_variant: int, regime: str
) -> str:
    lines = [
        f"STEP 0 — fireability census over the {len(rows)} plain-failing queries "
        f"({'+'.join(TARGET_CATEGORIES)}), @k={K}",
        f"  shipped four-seam AND-strict first pass: {fireable_shipped}/{len(rows)} "
        f"return any rows (floor for the base regime: {FIREABILITY_FLOOR})",
        f"  licensed OR-feedback variant (feedback selection only): "
        f"{fireable_variant}/{len(rows)}",
        f"  ACTIVE REGIME: {regime}",
        "",
    ]
    header = (
        f"{'query_id':<26}{'category':<21}{'shipped_rows':>13}{'shipped_docs':>13}"
        f"{'fires':>7}{'variant_rows':>13}{'variant_docs':>13}{'fires':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row.query_id:<26}{row.category:<21}{row.shipped_rows:>13}"
            f"{row.shipped_docs:>13}{'yes' if row.shipped_rows else 'no':>7}"
            f"{row.variant_rows:>13}{row.variant_docs:>13}"
            f"{'yes' if row.variant_rows else 'no':>7}"
        )
    return "\n".join(lines)


def _format_oracle_comparison(
    runs: Sequence[tuple[_Selector, int, tuple[_OracleRow, ...]]],
) -> str:
    """The headline the single-selector version got wrong.

    One selector produces one number that reads as a property of the
    retrieval path. Several selectors, same oracle feed, same composition
    (except where disclosed), show what the number is actually a property
    of. This table IS the correction, rendered where the claim used to be.
    """
    types = ("note", "media", "conversation")
    lines = [
        "CONTROL — the rescue channel under DIFFERENT TERM SELECTORS "
        "(oracle feed: the TARGET DOCUMENT itself)",
        "",
        "  What is fixed across the rows: the four-seam path, the oracle "
        "feed, k, the corpus. What varies: which",
        "  terms are chosen (and, for the disclosed illustration row, "
        "whether the query's own tokens stay in the",
        "  expression). Read this BEFORE the floor: it says how many cells "
        "a rescue could have been seen in at all.",
        "",
    ]
    header = (
        f"{'selector':<34}{'N':>3}{'top-' + str(K):>8}"
        + "".join(f"{name:>14}" for name in types)
        + "  what it changes"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for selector, n_terms, rows in runs:
        reachable = sum(1 for row in rows if row.rank_at_k is not None)
        cells = []
        for source_type in types:
            group = [row for row in rows if row.target_source_type == source_type]
            hits = sum(1 for row in group if row.rank_at_k is not None)
            cells.append(f"{hits}/{len(group)}")
        lines.append(
            f"{selector.name:<34}{n_terms:>3}{f'{reachable}/{len(rows)}':>8}"
            + "".join(f"{cell:>14}" for cell in cells)
            + f"  {selector.disclosure}"
        )
    lines.append("")
    lines.append(
        "  THE READING. The plain path now interleaves its seams RANK-FAIRLY "
        "(TASK-16071), but top_k is still a"
    )
    lines.append(
        "  PER-SEAM limit, so across this harness's three seams one seam "
        f"holds only about ceil({K}/3) of the {K} merged"
    )
    lines.append(
        "  slots: a target ranked deeper than that WITHIN ITS OWN SEAM "
        "cannot be seen however well it matched."
    )
    lines.append(
        "  Rank position, not seam order, is what decides now. Before "
        "TASK-16071 this merge CONCATENATED in seam"
    )
    lines.append(
        "  order and buried every media/conversation target behind a full "
        "notes seam regardless of match quality;"
    )
    lines.append(
        "  that component is what the conversation column above lost (0/6 "
        "and 2/6 then, 6/6 in both rows now)."
    )
    lines.append(
        "  How hard the SURVIVING constraint bites depends on EXPANSION "
        "BREADTH, which is the selector's property,"
    )
    lines.append(
        "  not the path's — the rows above are the same path at different "
        "breadths. So 'this target could not have"
    )
    lines.append(
        "  been rescued by any feedback set' is NOT a conclusion this "
        "control supports, and an earlier version of"
    )
    lines.append("  this module printed it anyway.")
    return "\n".join(lines)


def _format_oracle(
    rows: Sequence[_OracleRow], *, n_terms: int, selector: _Selector
) -> str:
    """One selector's rescue-channel detail table (see `_OracleRow`)."""
    reachable = sum(1 for row in rows if row.rank_at_k is not None)
    returned = sum(1 for row in rows if row.position is not None)
    by_type: dict[str, list[_OracleRow]] = {}
    for row in rows:
        by_type.setdefault(row.target_source_type, []).append(row)

    lines = [
        f"  oracle detail — selector: {selector.name} (N={n_terms}); "
        f"{selector.disclosure}",
        f"  {reachable}/{len(rows)} targets reach the top-{K}; "
        f"{returned}/{len(rows)} are returned at any depth.",
    ]
    for source_type in sorted(by_type):
        group = by_type[source_type]
        hits = sum(1 for row in group if row.rank_at_k is not None)
        lines.append(
            f"  by target source type — {source_type}: {hits}/{len(group)} "
            f"reach the top-{K}"
        )
    matched_deep = sum(
        1
        for row in rows
        if row.rank_at_k is not None or row.deep_position is not None
    )
    lines.append(
        f"  {matched_deep}/{len(rows)} oracle expressions DO match their "
        f"target (at k={DEEP_K}) — so every miss here is displacement under "
        "THIS selector, not a control that failed to reach its document."
    )
    lines.append("")
    header = (
        f"{'query_id':<26}{'target':<34}{'type':<14}{'terms':>7}"
        f"{'rank@k':>8}{'position':>10}{'deep':>7}{'docs':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row.query_id:<26}{row.target_slug:<34}{row.target_source_type:<14}"
            f"{len(row.terms):>7}{_rank(row.rank_at_k):>8}"
            f"{_rank(row.position):>10}"
            # "n/a" rather than "miss" when the deep pass never ran: a
            # rank-1 row printing "miss" in the deep column reads as a
            # contradiction, and the control's whole job is to be readable.
            f"{('n/a' if row.rank_at_k is not None else _rank(row.deep_position)):>7}"
            f"{row.docs_returned:>7}"
        )
    return "\n".join(lines)


def _format_point(report: _PointReport, *, regime: str, with_terms: bool) -> str:
    targets = report.by_category(TARGET_CATEGORIES)
    with_row_text = sum(point.fed_docs_with_row_text for point in report.points)
    fed_total = sum(point.result.fed_docs for point in report.points)
    lines = [
        f"GRID POINT N={report.n_terms} M={report.top_m} "
        f"[selector: {report.selector}] — regime: {regime} — "
        f"{len(report.rescued)}/{len(targets)} rescued, "
        f"{len(report.lost)} hitter(s) lost, "
        f"{report.content_fetches} content fetches, "
        f"{report.wall_s:.1f}s wall",
        f"  price: {fed_total} fed documents over {len(report.points)} queries; "
        f"{with_row_text} of them ({(100.0 * with_row_text / fed_total) if fed_total else 0.0:.0f}%) "
        "carried text in the seam row itself — the rest are label-only rows "
        "(media/conversation), so every fed row is fetched",
        "",
    ]
    header = (
        f"{'query_id':<26}{'category':<21}{'fire':>6}{'rows1':>7}{'fed':>5}"
        f"{'fetch':>7}{'terms':>7}{'before':>8}{'after':>7}{'pos2':>7}"
        f"{'rows2':>7}  outcome"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for point in targets:
        result = point.result
        if result.target_rank_before is None and result.target_rank_after is not None:
            outcome = "RESCUED"
        elif result.target_rank_before is not None and result.target_rank_after is None:
            outcome = "LOST"
        elif not point.terms:
            outcome = "prf did not fire"
        else:
            outcome = f"no change ({_miss_mechanism(point)})"
        lines.append(
            f"{result.query_id:<26}{result.category:<21}"
            f"{'yes' if result.fireable else 'no':>6}{result.first_pass_rows:>7}"
            f"{result.fed_docs:>5}{result.content_fetches:>7}{len(point.terms):>7}"
            f"{_rank(result.target_rank_before):>8}"
            f"{_rank(result.target_rank_after):>7}"
            f"{_rank(point.target_position_after):>7}{result.rows_after:>7}  {outcome}"
        )

    if with_terms:
        lines.append("")
        lines.append("expansion terms (this point only):")
        for point in targets:
            lines.append(
                f"  {point.result.query_id:<26} fed={list(point.fed_slugs)}"
            )
            lines.append(f"  {'':<26} terms={list(point.terms)}")
    return "\n".join(lines)


def _format_guards(report: _PointReport, hitters: Mapping[str, tuple[str, ...]]) -> str:
    lines = [
        f"GUARDS at N={report.n_terms} M={report.top_m}",
        "",
        f"  hitter population (fresh baseline pass, all categories): "
        f"{len(hitters)} queries hold a target in the shipped top-{K}",
    ]
    lost = report.lost
    if lost:
        merge_displaced = sum(
            1 for point in lost if point.target_position_after is not None
        )
        seam_displaced = sum(
            1
            for point in lost
            if point.target_position_after is None and point.deep_position is not None
        )
        unmatched = len(lost) - merge_displaced - seam_displaced
        lines.append(
            f"    {len(lost)} LOST — {merge_displaced} merge-displaced, "
            f"{seam_displaced} seam-displaced, {unmatched} unmatched "
            f"(diagnosed at k={DEEP_K}; see `_miss_mechanism`)"
        )
        for point in lost:
            lines.append(
                f"    LOST {point.result.query_id} ({point.result.category}): "
                f"{list(point.slugs_lost)} — rank "
                f"{_rank(point.result.target_rank_before)} -> "
                f"{_rank(point.result.target_rank_after)}; "
                f"{_miss_mechanism(point)}"
            )
    else:
        lines.append("    zero losses")

    lines.append("")
    lines.append("  negation guard (rows + documents the second pass added):")
    for point in report.by_category((NEGATION_CATEGORY,)):
        lines.append(
            f"    {point.result.query_id:<26} rows {point.result.first_pass_rows} -> "
            f"{point.result.rows_after}; new docs {point.docs_added}; "
            f"terms {len(point.terms)}; "
            f"{'PRF fired' if point.terms else 'PRF did not fire'}"
        )

    lines.append("")
    negatives = report.by_category((NEGATIVE_CATEGORY,))
    live = [point for point in negatives if point.terms]
    lines.append(
        f"  negatives guard ({len(negatives)} queries): "
        + (
            f"LIVE for {len(live)}/{len(negatives)} (the feed fired, so a "
            "second pass ran and could add rows)"
            if live
            else "STRUCTURAL for all — the first pass returned nothing to "
            "feed from, so PRF never fired and the guard cannot bind"
        )
    )
    for point in negatives:
        lines.append(
            f"    {point.result.query_id:<26} rows {point.result.first_pass_rows} -> "
            f"{point.result.rows_after}; new docs {point.docs_added}; "
            f"terms {len(point.terms)}"
        )
    return "\n".join(lines)


def _format_verdict(
    points: Sequence[_PointReport], hitters: Mapping[str, tuple[str, ...]], regime: str
) -> tuple[str, bool]:
    """Apply the spec's pre-registered bar to every point that ran.

    Returns:
        ``(rendered verdict, admitted)``. ADMIT requires ONE point to meet
        clauses 1-3 together; clause 4 (negation) is a reported measurement
        the spec pre-registers as expected-to-bind, never a gate.

    **One deliberate difference from the spec's wording, disclosed rather
    than silently reconciled.** The spec states clause 3 as "zero new ROWS
    on negatives"; this implements it as zero new DOCUMENTS in the top-K
    (`docs_added`), the document-level unit every other number in this
    harness is stated in (`canonicalize.rows_to_doc_ids`). On this run the
    difference is immaterial -- the negatives return zero rows AND zero
    documents before and after, so both readings give 0 -- but a future run
    where a second pass returned several chunks of one already-present
    document would score 0 here and non-zero on the spec's literal wording.
    The row counts are printed beside the document counts in the guard
    table so a reader can apply either reading.
    """
    lines = ["THE VERDICT — the spec's pre-registered bar, clause by clause", ""]
    admitted_points: list[_PointReport] = []
    for report in points:
        targets = report.by_category(TARGET_CATEGORIES)
        rescued = report.rescued
        lost = report.lost
        negative_added = sum(
            point.docs_added for point in report.by_category((NEGATIVE_CATEGORY,))
        )
        clause1 = len(rescued) >= RESCUE_FLOOR
        clause2 = not lost
        clause3 = negative_added == 0
        if clause1 and clause2 and clause3:
            admitted_points.append(report)
        lines.append(
            f"  N={report.n_terms:<3} M={report.top_m:<3} "
            f"[1] rescued {len(rescued)}/{len(targets)} >= {RESCUE_FLOOR}: "
            f"{'PASS' if clause1 else 'FAIL'}   "
            f"[2] hitters lost {len(lost)} == 0: {'PASS' if clause2 else 'FAIL'}   "
            f"[3] new negative docs {negative_added} == 0: "
            f"{'PASS' if clause3 else 'FAIL'}"
        )
    lines.append("")
    lines.append(
        f"  [4] negation guard: reported per point above (pre-registered as "
        f"expected to bind; never a gate)."
    )
    lines.append("")
    lines.append(f"  regime: {regime}")
    lines.append(f"  hitter population: {len(hitters)} queries")
    lines.append("")
    if admitted_points:
        best = admitted_points[0]
        lines.append(
            f"  VERDICT: ADMIT — clauses 1-3 met at N={best.n_terms} "
            f"M={best.top_m} ({len(admitted_points)} of {len(points)} points "
            "met the bar). Tasks 3-4 proceed."
        )
    else:
        lines.append(
            f"  VERDICT: NULL — no grid point of the {len(points)} run met "
            "the bar. Task 3 records the null; Task 4 is skipped."
        )
    return "\n".join(lines), bool(admitted_points)


# ---------------------------------------------------------------------------
# The one gated test
# ---------------------------------------------------------------------------


def test_the_prf_probe_over_the_real_fixtures(tmp_path, capsys):
    """Census, grid, guards, verdict — one runtime, one pass through Phase A.

    One test function rather than four, against this directory's usual
    "one concern per test" rule, for a reason the rule does not cover: the
    census DECIDES which regime the grid runs in, and the runtime that
    answers it costs minutes to build (172 documents through the real
    writers and a real embedding model). Splitting would either rebuild it
    per concern or smuggle it into a module-scoped fixture, which
    `test_harness_run.py` documents as broken here (a module-scoped fixture
    is set up before `conftest.py`'s function-scoped model-cache fixture, so
    it runs against the sandboxed HOME and misses the cache).
    """
    import tldw_chatbook.Library.library_local_rag_search_service as service_module
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    import tldw_chatbook

    from Tests.RAG_Eval.harness.canonicalize import slug_lookup_from
    from Tests.RAG_Eval.harness.goldenset import load_fixtures
    from Tests.RAG_Eval.harness.ingest import build_eval_runtime
    from Tests.RAG_Eval.harness.prf_probe import (
        DEFAULT_N_TERMS,
        DEFAULT_TOP_M,
        compose_feedback_expression,
        compose_prf_expression,
        derive_expansion_terms,
    )
    from Tests.RAG_Eval.harness.runner import build_query_scope

    # Import provenance, printed as evidence line 1 of the report: a
    # measurement against a foreign checkout's package is a measurement of
    # somebody else's branch (the input-latency audit's editable-install
    # trap).
    provenance = tldw_chatbook.__file__

    assert (DEFAULT_N_TERMS, DEFAULT_TOP_M) == BASE_POINT, (
        f"the probe's base point {BASE_POINT} is not the machinery's "
        f"({DEFAULT_N_TERMS}, {DEFAULT_TOP_M}) — the pre-registered point "
        "moved under the run"
    )

    corpus, golden = load_fixtures()
    targets = [query for query in golden if query.category in TARGET_CATEGORIES]
    assert len(targets) == 22, (
        f"the target population is {len(targets)} queries, not the 22 the "
        "spec pre-registered; the census would answer a different question"
    )

    runtime = build_eval_runtime(corpus, tmp_path)
    search_config = runtime.service.config.search
    original_mode = search_config.default_search_mode
    original_builder = service_module.build_fts_match_query

    census: list[_CensusRow] = []
    close_error: Exception | None = None
    try:
        search_config.default_search_mode = "plain"
        seam = LibraryLocalRagSearchService(runtime.app)
        lookup = slug_lookup_from(runtime.slug_to_source)
        media_db = runtime.app.media_reading_scope_service.local_service.media_db
        chachanotes_db = runtime.app.chachanotes_db
        scopes = {
            query.id: build_query_scope(runtime.slug_to_source, query)
            for query in golden
        }

        # --- the shipped first pass, once, for every query -----------------
        # Every query, not just the 22: the hitter population the bar's loss
        # clause is defined against is DERIVED from this pass rather than
        # hardcoded (plain keyword recall is not 1.000, and the scoped
        # category's plain hits belong in the guard too).
        shipped: dict[str, tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]] = {}
        variant: dict[str, tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]] = {}
        variant_expressions: dict[str, str] = {}
        for query in golden:
            shipped[query.id] = _run_pass(
                runtime, seam, service_module, lookup, query.query, scopes[query.id]
            )
            expression = compose_feedback_expression(query.query)
            variant_expressions[query.id] = expression
            variant[query.id] = _run_pass(
                runtime,
                seam,
                service_module,
                lookup,
                query.query,
                scopes[query.id],
                expression=expression,
            )

        for query in targets:
            shipped_rows, shipped_docs = shipped[query.id]
            variant_rows, variant_docs = variant[query.id]
            census.append(
                _CensusRow(
                    query_id=query.id,
                    category=query.category,
                    shipped_rows=len(shipped_rows),
                    shipped_docs=len(shipped_docs),
                    variant_rows=len(variant_rows),
                    variant_docs=len(variant_docs),
                    variant_expression=variant_expressions[query.id],
                )
            )
        fireable_shipped = sum(1 for row in census if row.shipped_rows)
        fireable_variant = sum(1 for row in census if row.variant_rows)
        variant_active = fireable_shipped < FIREABILITY_FLOOR
        regime = (
            "LICENSED OR-FEEDBACK VARIANT (feedback selection only; the "
            "before-columns remain the SHIPPED pass)"
            if variant_active
            else "SHIPPED four-seam AND-strict first pass"
        )

        feed = variant if variant_active else shipped
        feed_docs = {qid: docs for qid, (_rows, docs) in feed.items()}
        fireable = {
            query.id: bool(feed_docs[query.id]) for query in golden
        }

        hitters = {
            query.id: tuple(
                slug
                for slug in query.relevant_slugs
                if slug in set(shipped[query.id][1][:K])
            )
            for query in golden
        }
        hitters = {qid: slugs for qid, slugs in hitters.items() if slugs}

        # --- the selectors --------------------------------------------------
        # TF is the spec's own derivation and the ONLY one the verdict reads.
        # The other two exist because a review measured that this module's
        # single-selector oracle number was a property of TF's breadth, not
        # of the retrieval path, and printed the wrong conclusion from it.
        document_frequency = _corpus_document_frequency(corpus)
        tf_selector = _Selector(
            name="TF tf/|D| (PRE-REGISTERED)",
            disclosure="the spec's own derivation — the verdict reads this row only",
            pre_registered=True,
            select=derive_expansion_terms,
            compose=compose_prf_expression,
        )
        rare_selector = _Selector(
            name="rarest-by-corpus-DF",
            disclosure="ranking key only: tf/|D| -> corpus DF ascending",
            pre_registered=False,
            select=lambda docs, **kwargs: _rarest_terms(
                docs, document_frequency=document_frequency, **kwargs
            ),
            compose=compose_prf_expression,
        )
        rare_narrow_selector = _Selector(
            name="rarest-1, query side dropped",
            disclosure="ILLUSTRATION: changes TWO things (N=1 rarest AND no query tokens)",
            pre_registered=False,
            select=lambda docs, **kwargs: _rarest_terms(
                docs, document_frequency=document_frequency, **kwargs
            ),
            compose=_compose_terms_only,
        )

        def oracle_for(selector: _Selector, n_terms: int) -> tuple[_OracleRow, ...]:
            return _run_oracle_control(
                targets=targets,
                n_terms=n_terms,
                scopes=scopes,
                runtime=runtime,
                seam=seam,
                service_module=service_module,
                lookup=lookup,
                media_db=media_db,
                chachanotes_db=chachanotes_db,
                selector=selector,
            )

        oracle_runs = [
            (tf_selector, BASE_POINT[0], oracle_for(tf_selector, BASE_POINT[0])),
            (rare_selector, BASE_POINT[0], oracle_for(rare_selector, BASE_POINT[0])),
            (rare_narrow_selector, 1, oracle_for(rare_narrow_selector, 1)),
        ]
        oracle = oracle_runs[0][2]

        # --- the grid ------------------------------------------------------
        def point(
            n_terms: int, top_m: int, selector: _Selector = tf_selector
        ) -> _PointReport:
            return _run_point(
                n_terms=n_terms,
                top_m=top_m,
                golden=golden,
                shipped=shipped,
                feed_docs=feed_docs,
                fireable=fireable,
                scopes=scopes,
                runtime=runtime,
                seam=seam,
                service_module=service_module,
                lookup=lookup,
                media_db=media_db,
                chachanotes_db=chachanotes_db,
                selector=selector,
            )

        base = point(*BASE_POINT)
        reports = [base]
        # The full grid runs ONLY on signal at the base point. A null there
        # is the null: the spec forbids searching the grid until something
        # moves, and the ledger's small-M note is the reason the sweep is
        # worth running when there IS signal (it says whether the movement
        # was the feed or the mechanism).
        if base.rescued:
            for n_terms in GRID_N:
                for top_m in GRID_M:
                    if (n_terms, top_m) == BASE_POINT:
                        continue
                    reports.append(point(n_terms, top_m))

        # --- the TF-vs-DF axis, on the REAL feed ----------------------------
        # Not a grid point and not a verdict input: a different DERIVATION,
        # outside the pre-registration entirely. It exists because the oracle
        # comparison above raises the obvious question -- if breadth is what
        # binds, does a narrow selector rescue anything when it is fed a real
        # feedback set rather than the answer? Answering it with a
        # measurement is cheaper than leaving it as a threat to the null.
        axis_reports = [
            point(8, BASE_POINT[1], rare_selector),
            point(4, BASE_POINT[1], rare_selector),
        ]
    finally:
        search_config.default_search_mode = original_mode
        service_module.build_fts_match_query = original_builder
        try:
            runtime.close()
        except Exception as exc:  # pragma: no cover - reported, not raised
            close_error = exc

    verdict_text, admitted = _format_verdict(reports, hitters, regime)
    with capsys.disabled():
        print("\n" + "=" * 78)
        print(f"PRF PROBE (TASK-15965 Phase A) — import provenance: {provenance}")
        print(
            f"corpus {len(corpus)} docs · golden {len(golden)} queries · "
            f"targets {len(targets)} · k={K}"
        )
        print("=" * 78)
        print(
            _format_census(
                census,
                fireable_shipped=fireable_shipped,
                fireable_variant=fireable_variant,
                regime=regime,
            )
        )
        print("")
        print(_format_oracle_comparison(oracle_runs))
        for selector, n_terms, rows in oracle_runs:
            print("")
            print(_format_oracle(rows, n_terms=n_terms, selector=selector))
        for report in reports:
            print("")
            print(
                _format_point(
                    report,
                    regime=regime,
                    with_terms=(report.n_terms, report.top_m) == BASE_POINT,
                )
            )
            print("")
            print(_format_guards(report, hitters))
        print("")
        print(
            "AXIS CONTROL — the same REAL variant feed, derived with the "
            "rarest-by-corpus-DF selector instead of TF."
        )
        print(
            "  Outside the pre-registration, so it can never ADMIT; it is "
            "here to answer 'does a narrower selector"
        )
        print("  rescue anything on a real feed?' with a measurement.")
        for report in axis_reports:
            print("")
            print(_format_point(report, regime=regime, with_terms=False))
            print("")
            print(_format_guards(report, hitters))
        print("")
        print(verdict_text)
        print("=" * 78)
    if close_error is not None:
        print(f"NOTE: runtime.close() failed after the probe: {close_error!r}")

    # Instrument pins ONLY. The verdict above is data: an arc whose
    # pre-registered NULL reddens a test cannot report a null honestly.
    assert len(census) == len(targets), (
        f"the census covered {len(census)} of {len(targets)} target queries"
    )
    for selector, _n_terms, rows in oracle_runs:
        assert len(rows) == len(targets), (
            f"the rescue-channel control under {selector.name!r} covered "
            f"{len(rows)} of {len(targets)} target queries"
        )
    assert sum(
        1 for selector, _n, _rows in oracle_runs if selector.pre_registered
    ) == 1, (
        "exactly one oracle row may be the pre-registered selector"
    )
    assert len(oracle_runs) > 1, (
        "the oracle must run under more than one selector — a single-selector "
        "ceiling reads as a property of the retrieval path, which a review "
        "measured to be false"
    )
    # The verdict must never see a control point. Honestly stated: this is a
    # TEST-LEVEL assertion over the list the verdict already consumed —
    # `_format_verdict` itself never inspects `.selector`, so a bug that fed
    # it a control point would print a wrong verdict first and red HERE one
    # step later (the test still fails; the printed text is what a -s reader
    # would have seen in between). The list separation in the body is the
    # primary defense; this pins it against drift.
    assert all(report.selector.endswith("(PRE-REGISTERED)") for report in reports), (
        f"a non-pre-registered selector reached the verdict: "
        f"{[report.selector for report in reports]}"
    )
    assert axis_reports and all(
        not report.selector.endswith("(PRE-REGISTERED)") for report in axis_reports
    ), "the axis control must run, and must not be the pre-registered selector"
    assert reports, "no grid point ran at all"
    assert (reports[0].n_terms, reports[0].top_m) == BASE_POINT, (
        "the first grid point must be the pre-registered base point"
    )
    if len(reports) > 1:
        assert len(reports) == len(GRID_N) * len(GRID_M), (
            f"the full grid ran {len(reports)} points, not "
            f"{len(GRID_N) * len(GRID_M)}"
        )
    for report in [*reports, *axis_reports]:
        assert len(report.points) == len(golden), (
            f"N={report.n_terms} M={report.top_m} [{report.selector}] probed "
            f"{len(report.points)} of {len(golden)} queries"
        )
        assert report.content_fetches > 0, (
            f"N={report.n_terms} M={report.top_m} [{report.selector}] paid no "
            "content fetches — the feed was derived from label snippets, not "
            "documents"
        )
    assert service_module.build_fts_match_query is original_builder, (
        "the probe left the seam's MATCH builder substituted"
    )
    assert search_config.default_search_mode == original_mode
    assert isinstance(admitted, bool)
