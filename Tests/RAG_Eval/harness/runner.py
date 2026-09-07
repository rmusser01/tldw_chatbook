# Tests/RAG_Eval/harness/runner.py
"""Three-mode retrieval eval: run the golden set through the Library seam.

One `run_eval` call answers the question P1 exists to answer — *what does
each retrieval mode actually buy?* — by running every golden query three
times, once per `default_search_mode`, through
`LibraryLocalRagSearchService` (the production seam, not the engine), and
scoring the results per capability category.

Four decisions worth knowing before reading a number this produces:

**Modes are flipped, not rebuilt.** Each pass sets
``runtime.service.config.search.default_search_mode``; the seam's
`_search_rag` re-reads it per query and routes accordingly — ``plain`` to
the Library's four-seam keyword path, ``semantic`` to the vector path,
``hybrid`` to the engine's RRF fusion. The runtime, the index and the
embedded corpus are therefore identical across modes, which is what makes
the cross-mode deltas comparable at all. The report records the
``runtime_backend`` each pass reported so a pass that silently failed to
re-route is visible rather than merely wrong.

**Metrics are document-level.** Rows are canonicalized to fixture slugs by
`canonicalize.rows_to_doc_ids` before scoring; see that module for why a
chunk is not a document and why unmapped rows are kept.

**Negatives are measured, never averaged.** A query with no relevant
document has no meaningful precision or recall (recall over an empty
relevant set is 0.0 by convention, which would drag every average toward
zero for reasons that say nothing about retrieval). They are reported
separately: how much a mode returned anyway, and how confident it was.

**Scoped queries are measured, and averaged only into their own cell.** A
query in the ``scoped`` category runs under a real `EffectiveScope` built
from the runtime's own ids (`build_query_scope`) and passed to the seam's
`scope=` parameter — the same object production passes. It is excluded from
the cross-mode overall row, though no longer for the reason it originally
was: until TASK-15020/B1 a scope forced the seam to divert a hybrid profile
to the semantic path (the engine's allowlist pushdown was semantic-only), so
a scoped row's "hybrid" and "semantic" columns were one measurement wearing
two names. That is over — the allowlists reach both engine legs, and a
scoped query now routes exactly as its mode says, which makes the modes
genuinely comparable to each other on a scoped query. The exclusion stands
on what was always the deeper reason: **a scoped query is asked over a
different universe.** Its haystack is the hundred documents of its scope;
every other query's is the whole corpus. Recall@10 over 100 documents and
recall@10 over 172 are two different questions, and one average over both
answers neither. The overall row is also what the baseline gate compares
across re-stamps, so folding scoped in would let an edit to a *scope* — a
fixture-side decision the size and composition pins exist to make
deliberate — move the headline retrieval number of the whole harness.

Which route each scoped query actually took is still recorded per query
(``runtime_backend`` + ``route_notes``), so a change in routing shows up as
a change in the report rather than only in the score. That record is what
made B1's flip readable, and it is what would make a regression back to the
divert readable too.

**An erroring query is reported, not scored.** The run continues, the error
is recorded against that query and surfaced in the mode's ``errors`` list,
and the query is left out of the averages — a failed query silently scored
as 0.0 would look like a quality regression instead of a broken seam. The
shortfall stays visible because ``overall["num_queries"]`` is the count that
was actually averaged.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tldw_chatbook.Chat.rag_scope import EffectiveScope
from tldw_chatbook.RAG_Search.eval.metrics import evaluate_retrieval_batch
from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids, slug_lookup_from
from Tests.RAG_Eval.harness.goldenset import (
    NEGATIVE_CATEGORY,
    SCOPEABLE_SOURCE_TYPES,
    SCOPED_CATEGORY,
    GoldenQuery,
)

__all__ = [
    "EvalReport",
    "MODES",
    "ModeReport",
    "NegativeProbe",
    "QueryOutcome",
    "SOURCE_TYPES",
    "UNAVERAGED_CATEGORIES",
    "build_query_scope",
    "count_scored",
    "run_eval",
]

#: The three retrieval modes, in report order. These are the values
#: `SearchConfig.default_search_mode` takes and the seam's `_search_rag`
#: routes on.
MODES: tuple[str, ...] = ("semantic", "plain", "hybrid")

#: The Library scope identifiers passed to the seam (plural — the scope
#: vocabulary, not the ingestion one). All FOUR corpus source types: the
#: seam degrades a hybrid profile to semantic when NONE of the selected
#: types is one the engine's FTS leg can serve, and these four are exactly
#: that set since TASK-15020/B2 (`_FTS_SERVABLE_SOURCE_TYPES`), so dropping
#: one would silently make the hybrid pass a second semantic pass.
#:
#: `prompts` is load-bearing twice over and was added by B2: without it the
#: seam never asks the engine for the prompts sub-leg (`keyword_source_
#: types` is built from this tuple), AND the source-type post-filter drops
#: every prompt row that came back anyway. Either alone is enough to make
#: the prompt category read 0.000 while the sub-leg works perfectly.
SOURCE_TYPES: tuple[str, ...] = ("media", "notes", "conversations", "prompts")

#: Metric keys `evaluate_retrieval_batch` returns (short names — see the
#: port note in `RAG_Search/eval/metrics.py`).
_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "mrr", "ndcg", "f1")

#: Categories measured, but never folded into the cross-mode overall row:
#: negatives (no relevant document, so no meaningful precision/recall) and
#: scoped (a different haystack, so a different question — see the module
#: docstring; the ROUTING reason this line used to give died with B1).
#: One definition, because every "how many queries did that average cover"
#: count in this package must be the same count: `count_scored` below, the
#: `averaged` set in `_build_mode_report`, and the fusion sweep's own header
#: all read this.
UNAVERAGED_CATEGORIES: tuple[str, ...] = (NEGATIVE_CATEGORY, SCOPED_CATEGORY)


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One golden query, run once, in one mode."""

    query_id: str
    query: str
    category: str
    retrieved_doc_ids: tuple[str, ...]
    relevant_slugs: tuple[str, ...]
    rows_returned: int
    latency_s: float
    runtime_backend: str
    top_score: float | None
    top_vector_score: float | None
    error: str | None
    #: The seam's own routing disclosures for this query (its
    #: ``diagnostics[LIBRARY_RAG_ROUTE_NOTES_KEY]``), e.g. "a hybrid profile
    #: ran semantic because a scope is active". Empty for a query that ran
    #: exactly as its profile configures. Recorded because the backend label
    #: alone says which path ran but not *why* it was chosen.
    route_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot of this query's outcome.

        Returns:
            The dataclass fields as a plain dict, with tuples rendered as
            lists and ``latency_s`` converted to ``latency_ms``.
        """
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "relevant_slugs": list(self.relevant_slugs),
            "rows_returned": self.rows_returned,
            "latency_ms": self.latency_s * 1000.0,
            "runtime_backend": self.runtime_backend,
            "route_notes": list(self.route_notes),
            "top_score": self.top_score,
            "top_vector_score": self.top_vector_score,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class NegativeProbe:
    """What a mode did with a query that has no relevant document.

    Attributes:
        docs_at_k: Distinct documents returned within the first ``k``
            canonicalized results — the keyword-mode "results returned"
            measure, and the one number a negative query can be wrong about.
        top_score: The best score any row carried, in the mode's own score
            kind: a cosine similarity under ``semantic``, a fused RRF score
            under ``hybrid``, and ``None`` under ``plain`` (the four-seam
            keyword path deliberately emits no scores — see `_conversation_row`
            in the seam).
        top_vector_score: The best *vector* similarity available. Identical
            to ``top_score`` under ``semantic``; read out of hybrid's
            ``provenance["hybrid_fusion"]["vector_score"]`` under ``hybrid``,
            where it is ``None`` for a row only the FTS leg returned.
    """

    query_id: str
    query: str
    rows_returned: int
    docs_at_k: int
    top_score: float | None
    top_vector_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "rows_returned": self.rows_returned,
            "docs_at_k": self.docs_at_k,
            "top_score": self.top_score,
            "top_vector_score": self.top_vector_score,
        }


@dataclass(frozen=True, slots=True)
class ModeReport:
    """Everything one retrieval mode produced over the whole golden set."""

    mode: str
    k: int
    queries: tuple[QueryOutcome, ...]
    overall: dict[str, float]
    per_category: dict[str, dict[str, float]]
    negatives: tuple[NegativeProbe, ...]
    latency: dict[str, float]
    runtime_backends: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]
    #: Mean distinct documents returned per scored query. Reported next to
    #: precision because the ported `precision_at_k` divides by
    #: ``min(k, len(retrieved))`` — literally ``len(retrieved_ids[:k])`` —
    #: and not by `k` (Task 2 pinned that convention against the server
    #: module): a mode that returns one correct document scores P@10 = 1.0,
    #: and a mode that returns ten with one correct scores 0.1. Without this
    #: column the precision row reads as a quality ranking when it is partly
    #: a verbosity ranking.
    mean_docs_at_k: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "k": self.k,
            "overall": dict(self.overall),
            "mean_docs_at_k": self.mean_docs_at_k,
            "per_category": {
                category: dict(metrics)
                for category, metrics in self.per_category.items()
            },
            "negatives": [probe.to_dict() for probe in self.negatives],
            "latency": dict(self.latency),
            "runtime_backends": list(self.runtime_backends),
            "errors": [list(error) for error in self.errors],
            "queries": [outcome.to_dict() for outcome in self.queries],
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The three-mode result of one `run_eval`."""

    k: int
    modes: dict[str, ModeReport]
    source_types: tuple[str, ...] = SOURCE_TYPES
    num_queries: int = 0
    #: Queries the overall row averages: everything that is neither negative
    #: nor scoped. Both exclusions are visible as their own counts, so a
    #: shrinking ``num_scored`` can always be accounted for.
    num_scored: int = 0
    num_negative: int = 0
    num_scoped: int = 0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot (Task 7's baselines are written from this)."""
        return {
            "k": self.k,
            "source_types": list(self.source_types),
            "num_queries": self.num_queries,
            "num_scored": self.num_scored,
            "num_negative": self.num_negative,
            "num_scoped": self.num_scoped,
            "modes": {
                mode: report.to_dict() for mode, report in self.modes.items()
            },
        }

    def format_summary(self) -> str:
        """Render the human-readable table, including the plain-vs-hybrid delta."""
        return _format_summary(self)


def run_eval(
    runtime: Any,
    golden: Sequence[GoldenQuery],
    k: int = 10,
    *,
    modes: Sequence[str] = MODES,
    source_types: Sequence[str] = SOURCE_TYPES,
) -> EvalReport:
    """Run every golden query through the seam once per retrieval mode.

    Args:
        runtime: A live `EvalRuntime` (Task 5). Every async call is driven
            through ``runtime.run`` — the runtime owns the only event loop
            its service's pools are bound to.
        golden: The validated golden query set.
        k: Result cap per query and the ``@k`` of every metric.
        modes: Retrieval modes to run, in report order.
        source_types: Library scope identifiers to search.

    Returns:
        An `EvalReport` with per-mode, per-category metrics, negative
        probes, latencies and per-query detail.

    Raises:
        ValueError: ``golden`` is empty, ``k < 1``, ``modes`` is empty, or a
            scoped query's scope cannot be built against this runtime (see
            `build_query_scope`). An empty run would report a full set of 0.0
            metrics, which reads as a total retrieval failure rather than as
            no measurement.
    """
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not golden:
        raise ValueError("refusing to run an eval over an empty golden set")
    if not modes:
        raise ValueError("refusing to run an eval over no modes")

    seam = LibraryLocalRagSearchService(runtime.app)
    lookup = slug_lookup_from(runtime.slug_to_source)
    scope = tuple(source_types)

    # Built once, before any mode runs: a scope that cannot be built is a
    # fixture/runtime mismatch, and discovering it three modes deep would
    # waste the whole pass. The same object is reused across modes, which is
    # also what makes the cross-mode comparison of a scoped query honest.
    queries_with_scope = tuple(
        (query, build_query_scope(runtime.slug_to_source, query)) for query in golden
    )

    search_config = runtime.service.config.search
    original_mode = getattr(search_config, "default_search_mode", None)
    mode_reports: dict[str, ModeReport] = {}
    try:
        for mode in modes:
            search_config.default_search_mode = mode
            outcomes = tuple(
                _run_query(seam, runtime, query, k, scope, lookup, query_scope)
                for query, query_scope in queries_with_scope
            )
            mode_reports[mode] = _build_mode_report(mode, k, outcomes)
    finally:
        # The runtime is usually discarded after a run, but leaving a
        # caller's service on whichever mode happened to be last would make
        # any later use of it depend on this function's loop order.
        if original_mode is not None:
            search_config.default_search_mode = original_mode

    negatives = sum(1 for query in golden if query.category == NEGATIVE_CATEGORY)
    scoped = sum(1 for query in golden if query.category == SCOPED_CATEGORY)
    return EvalReport(
        k=k,
        modes=mode_reports,
        source_types=scope,
        num_queries=len(golden),
        num_scored=count_scored(golden),
        num_negative=negatives,
        num_scoped=scoped,
    )


def count_scored(golden: Sequence[GoldenQuery]) -> int:
    """How many of ``golden`` the overall row actually averages.

    The one definition of that count. A report header that says "38 scored"
    while the row beneath it averaged 32 is not a rounding difference — it is
    a claim about coverage, and the two numbers drifting apart is exactly the
    "plausible numbers that mean something else" failure this harness exists
    to prevent. Every caller that labels an averaged number reads it here.

    Args:
        golden: The queries a run covers.

    Returns:
        The count excluding every `UNAVERAGED_CATEGORIES` member.
    """
    return sum(
        1 for query in golden if query.category not in UNAVERAGED_CATEGORIES
    )


def build_query_scope(
    slug_to_source: Mapping[str, tuple[str, str]], query: GoldenQuery
) -> EffectiveScope | None:
    """Translate a scoped query's fixture slugs into the seam's scope object.

    The harness must pass the *production* scope object, not a slug list or a
    source-type filter: `EffectiveScope` is what `LibraryLocalRagSearchService.
    search(scope=...)` accepts, and its allowlist carries the runtime ids the
    real writers assigned — which is why the translation can only happen here,
    against a live runtime's `slug_to_source` map.

    Every failure raises rather than degrading. Dropping an unknown slug would
    silently narrow the scope (a smaller haystack scores better for a reason
    no report would show), and running a scoped query unscoped would report an
    unscoped measurement in a scoped cell.

    Args:
        slug_to_source: The runtime's fixture slug -> (source_type, source_id)
            map (`EvalRuntime.slug_to_source`).
        query: The golden query to build a scope for.

    Returns:
        ``None`` for every category except ``scoped`` — an unscoped search,
        which is what the seam's ``scope=None`` means. For a scoped query, an
        `EffectiveScope` in state ``"scoped"`` whose allowlist holds one
        non-empty frozenset of runtime ids per source type.

    Raises:
        ValueError: The query is scoped but carries no ``scope_slugs``, names
            a slug this runtime never ingested, or names a document whose
            source type is outside the scope vocabulary.
    """
    if query.category != SCOPED_CATEGORY:
        return None
    if not query.scope_slugs:
        raise ValueError(
            f"golden query {query.id!r} is category {SCOPED_CATEGORY!r} but "
            "carries no scope_slugs; running it would measure unscoped "
            "retrieval in a scoped cell"
        )

    allowlist: dict[str, set[str]] = {}
    for slug in query.scope_slugs:
        entry = slug_to_source.get(slug)
        if entry is None:
            raise ValueError(
                f"golden query {query.id!r} scopes slug {slug!r}, which this "
                "runtime never ingested; the scope would be narrower than the "
                "fixture asks for"
            )
        source_type, source_id = entry
        if source_type not in SCOPEABLE_SOURCE_TYPES:
            raise ValueError(
                f"golden query {query.id!r} scopes slug {slug!r}, whose source "
                f"type {source_type!r} is outside the scope vocabulary "
                f"(scopeable: {', '.join(SCOPEABLE_SOURCE_TYPES)})"
            )
        allowlist.setdefault(source_type, set()).add(str(source_id))

    return EffectiveScope(
        state="scoped",
        allowlist={
            source_type: frozenset(ids) for source_type, ids in allowlist.items()
        },
        cause=None,
    )


def _run_query(
    seam: Any,
    runtime: Any,
    query: GoldenQuery,
    k: int,
    source_types: tuple[str, ...],
    lookup: Mapping[tuple[str, str], str],
    scope: EffectiveScope | None = None,
) -> QueryOutcome:
    """Run one query and canonicalize its rows, recording rather than raising.

    ``scope`` is passed straight through to the seam's own ``scope=``
    parameter — `None` for every unscoped query, which is byte-for-byte the
    call the harness has always made.
    """
    start = time.perf_counter()
    error: str | None = None
    result: Any = None
    try:
        result = runtime.run(
            seam.search(query.query, source_types, "rag", top_k=k, scope=scope)
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_s = time.perf_counter() - start

    rows: list[Mapping[str, Any]] = []
    backend = ""
    route_notes: tuple[str, ...] = ()
    if error is None:
        rows, backend, error = _extract_rows(result)
        route_notes = _route_notes(result)

    # Canonicalization failures are NOT caught: they mean the seam's row
    # shape changed under the harness, which must fail the run loudly rather
    # than degrade one query's score.
    doc_ids = tuple(rows_to_doc_ids(rows, lookup))
    top_score, top_vector_score = _top_scores(rows)
    return QueryOutcome(
        query_id=query.id,
        query=query.query,
        category=query.category,
        retrieved_doc_ids=doc_ids,
        relevant_slugs=tuple(query.relevant_slugs),
        rows_returned=len(rows),
        latency_s=latency_s,
        runtime_backend=backend,
        top_score=top_score,
        top_vector_score=top_vector_score,
        error=error,
        route_notes=route_notes,
    )


def _route_notes(result: Any) -> tuple[str, ...]:
    """The seam's routing disclosures for one result, if it carried any.

    Both seam shapes keep them in the same place
    (``diagnostics[LIBRARY_RAG_ROUTE_NOTES_KEY]``), and a result that ran
    exactly as its profile configures carries no ``diagnostics`` key at all —
    which reads as "no disclosure", not as an error.
    """
    # Imported here, not at module scope: this module is imported by the
    # directory's ALWAYS-ON tests, and the Library state module pulls the
    # app's config chain in behind it.
    from tldw_chatbook.Library.library_rag_state import LIBRARY_RAG_ROUTE_NOTES_KEY

    diagnostics = (
        result.get("diagnostics")
        if isinstance(result, Mapping)
        else getattr(result, "diagnostics", None)
    )
    if not isinstance(diagnostics, Mapping):
        return ()
    notes = diagnostics.get(LIBRARY_RAG_ROUTE_NOTES_KEY)
    if not isinstance(notes, (list, tuple)):
        return ()
    return tuple(str(note) for note in notes)


def _extract_rows(result: Any) -> tuple[list[Mapping[str, Any]], str, str | None]:
    """Normalize the seam's two return shapes into (rows, backend, error).

    The seam returns a mapping for a search that ran, and a
    `LibraryRagSearchOutcome` for blocked/empty states. ``blocked`` is an
    error here — it means retrieval never happened (no runtime, no seams) —
    while ``empty`` is a legitimate zero-result answer.
    """
    if isinstance(result, Mapping):
        raw_rows = result.get("results") or []
        backend = str(result.get("runtime_backend") or "")
        return list(raw_rows), backend, None

    status = getattr(result, "status", None)
    if isinstance(status, str):
        backend = str(getattr(result, "runtime_backend", "") or "")
        if status == "blocked":
            recovery = getattr(result, "recovery_state", None)
            detail = getattr(recovery, "title", None) or type(recovery).__name__
            return [], backend, f"seam returned status=blocked ({detail})"
        return list(getattr(result, "results", ()) or ()), backend, None

    return [], "", f"unrecognized seam result shape: {type(result).__name__}"


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def _top_scores(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    """Best score and best vector similarity across a query's rows.

    Mode-agnostic by construction: when any row carries hybrid's fusion
    block the vector similarity is read from there (it is the only place it
    survives fusion); otherwise the row score *is* the vector similarity
    under ``semantic``, and is absent entirely under ``plain``.
    """
    scores = [
        score
        for score in (_coerce_float(row.get("score")) for row in rows)
        if score is not None
    ]
    top_score = max(scores) if scores else None

    fused_vector_scores: list[float] = []
    saw_fusion = False
    for row in rows:
        provenance = row.get("provenance")
        fusion = (
            provenance.get("hybrid_fusion") if isinstance(provenance, Mapping) else None
        )
        if not isinstance(fusion, Mapping):
            continue
        saw_fusion = True
        vector_score = _coerce_float(fusion.get("vector_score"))
        if vector_score is not None:
            fused_vector_scores.append(vector_score)
    if saw_fusion:
        return top_score, (max(fused_vector_scores) if fused_vector_scores else None)
    return top_score, top_score


def _percentile_ms(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile, in milliseconds.

    Nearest-rank rather than interpolated: with 44 samples an interpolated
    p95 invents a latency no query actually had.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1] * 1000.0


def _build_mode_report(
    mode: str, k: int, outcomes: Sequence[QueryOutcome]
) -> ModeReport:
    scored = [
        outcome
        for outcome in outcomes
        if outcome.category != NEGATIVE_CATEGORY and outcome.error is None
    ]
    # Scoped queries ARE scored — into their own cell — but never into the
    # cross-mode overall row: they are asked over their scope, not over the
    # corpus, so averaging them in would mix two different haystacks into one
    # number. Same mechanism as the negative exclusion above, one category
    # further out.
    averaged = [
        outcome
        for outcome in scored
        if outcome.category not in UNAVERAGED_CATEGORIES
    ]
    overall = _metrics_for(averaged, k)
    per_category: dict[str, dict[str, float]] = {}
    for category in sorted({outcome.category for outcome in scored}):
        per_category[category] = _metrics_for(
            [outcome for outcome in scored if outcome.category == category], k
        )

    negatives = tuple(
        NegativeProbe(
            query_id=outcome.query_id,
            query=outcome.query,
            rows_returned=outcome.rows_returned,
            docs_at_k=len(outcome.retrieved_doc_ids[:k]),
            top_score=outcome.top_score,
            top_vector_score=outcome.top_vector_score,
        )
        for outcome in outcomes
        if outcome.category == NEGATIVE_CATEGORY and outcome.error is None
    )

    latencies = [outcome.latency_s for outcome in outcomes]
    latency = {
        "count": float(len(latencies)),
        "mean_ms": (sum(latencies) / len(latencies) * 1000.0) if latencies else 0.0,
        "p95_ms": _percentile_ms(latencies, 0.95),
        "max_ms": (max(latencies) * 1000.0) if latencies else 0.0,
        "total_s": sum(latencies),
    }

    return ModeReport(
        mode=mode,
        k=k,
        queries=tuple(outcomes),
        overall=overall,
        per_category=per_category,
        negatives=negatives,
        latency=latency,
        runtime_backends=tuple(
            sorted({outcome.runtime_backend for outcome in outcomes if outcome.runtime_backend})
        ),
        errors=tuple(
            (outcome.query_id, outcome.error)
            for outcome in outcomes
            if outcome.error is not None
        ),
        mean_docs_at_k=(
            # Over the SAME set the overall row averages: it is read as a
            # companion to that row's precision.
            sum(len(outcome.retrieved_doc_ids[:k]) for outcome in averaged)
            / len(averaged)
            if averaged
            else 0.0
        ),
    )


def _metrics_for(outcomes: Iterable[QueryOutcome], k: int) -> dict[str, float]:
    pairs = [
        (list(outcome.retrieved_doc_ids), list(outcome.relevant_slugs))
        for outcome in outcomes
    ]
    return evaluate_retrieval_batch(pairs, k=k)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_summary(report: EvalReport) -> str:
    k = report.k
    lines: list[str] = []
    lines.append(
        f"Retrieval eval @k={k} — {report.num_queries} golden queries "
        f"({report.num_scored} scored, {report.num_negative} negative, "
        f"{report.num_scoped} scoped) over {'/'.join(report.source_types)}"
    )
    lines.append("")

    header = (
        f"{'mode':<10}{'P@k':>8}{'R@k':>8}{'MRR':>8}{'NDCG':>8}{'F1':>8}"
        f"{'docs':>7}{'n':>5}{'mean ms':>10}{'p95 ms':>9}{'errors':>8}  backend"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for mode, mode_report in report.modes.items():
        overall = mode_report.overall
        lines.append(
            f"{mode:<10}"
            f"{overall['precision']:>8.3f}{overall['recall']:>8.3f}"
            f"{overall['mrr']:>8.3f}{overall['ndcg']:>8.3f}{overall['f1']:>8.3f}"
            f"{mode_report.mean_docs_at_k:>7.1f}"
            f"{int(overall['num_queries']):>5}"
            f"{mode_report.latency['mean_ms']:>10.1f}"
            f"{mode_report.latency['p95_ms']:>9.1f}"
            f"{len(mode_report.errors):>8}"
            f"  {','.join(mode_report.runtime_backends) or '-'}"
        )
    lines.append(
        "'docs' = mean distinct documents returned per scored query. P@k "
        "divides by min(k, len(retrieved)), NOT by k, so a mode that returns "
        "little scores high precision for returning little."
    )
    lines.append(
        "plain-mode MRR/NDCG do not measure ranking: its rows carry no score, "
        "and the seams are merged by rank-fair rotation (TASK-16071) rather "
        "than by any cross-seam relevance signal, so for that mode those two "
        "columns track recall."
    )

    categories = sorted(
        {
            category
            for mode_report in report.modes.values()
            for category in mode_report.per_category
        }
    )
    if categories:
        lines.append("")
        lines.append(f"per category — recall@{k} (precision@{k})")
        column = f"{'category':<22}" + "".join(
            f"{mode:>20}" for mode in report.modes
        )
        lines.append(column)
        lines.append("-" * len(column))
        for category in categories:
            cells = []
            for mode_report in report.modes.values():
                metrics = mode_report.per_category.get(category)
                cells.append(
                    "                   -"
                    if metrics is None
                    else f"{metrics['recall']:>13.3f} ({metrics['precision']:.3f})"
                )
            lines.append(f"{category:<22}" + "".join(cells))

    lines.append("")
    lines.append(
        f"negatives (excluded from every average above) — {report.num_negative} "
        "queries with no relevant document"
    )
    for mode, mode_report in report.modes.items():
        probes = mode_report.negatives
        if not probes:
            lines.append(f"  {mode:<10} no probes recorded")
            continue
        mean_docs = sum(probe.docs_at_k for probe in probes) / len(probes)
        returned_any = sum(1 for probe in probes if probe.docs_at_k)
        top_scores = [
            probe.top_score for probe in probes if probe.top_score is not None
        ]
        vector_scores = [
            probe.top_vector_score
            for probe in probes
            if probe.top_vector_score is not None
        ]
        lines.append(
            f"  {mode:<10} returned something for {returned_any}/{len(probes)}; "
            f"mean {mean_docs:.1f} docs@{k}; "
            f"max top score {_optional(max(top_scores) if top_scores else None)}; "
            f"max top vector score "
            f"{_optional(max(vector_scores) if vector_scores else None)}"
        )

    lines.extend(_scoped_lines(report))

    plain = report.modes.get("plain")
    hybrid = report.modes.get("hybrid")
    if plain is not None and hybrid is not None:
        lines.append("")
        lines.append(_delta_line(plain, hybrid, k))
    return "\n".join(lines)


def _scoped_lines(report: EvalReport) -> list[str]:
    """The scoped section: which route each scoped query actually took.

    Omitted entirely for a set with no scoped queries (the header's scoped
    count is always shown, so "0 scoped" is never silent). The route, not
    just the score, is the point. It was written when a scope diverted a
    hybrid profile to semantic, so that the hybrid column of a scoped row
    announced itself as a second semantic column on the face of the report
    rather than in a docstring; TASK-15020/B1 ended the divert, and this
    section is now what shows — in the same place, in the same shape — that
    every scoped query ran its profile's own route with nothing to disclose.
    A silent divert is the failure it exists to prevent, in either direction.
    """
    if not report.num_scoped:
        return []
    lines = ["", (
        f"scoped (excluded from every average above) — {report.num_scoped} "
        "queries run under a real retrieval scope; reported in their own "
        "category cell"
    )]
    for mode, mode_report in report.modes.items():
        outcomes = [
            outcome
            for outcome in mode_report.queries
            if outcome.category == SCOPED_CATEGORY
        ]
        if not outcomes:
            lines.append(f"  {mode:<10} no scoped queries ran")
            continue
        for outcome in outcomes:
            notes = "; ".join(outcome.route_notes) or "no routing disclosure"
            lines.append(
                f"  {mode:<10} {outcome.query_id:<28} -> "
                f"{outcome.runtime_backend or '-'}  ({notes})"
            )
    return lines


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _delta_line(plain: ModeReport, hybrid: ModeReport, k: int) -> str:
    """The four-seam-keyword vs hybrid comparison, overall and on keyword queries.

    The keyword-category row is the one that answers "does fusion cost us
    the literal matches the four-seam path already found", which is the
    specific risk of routing a hybrid profile away from the Library's own
    keyword seams.
    """
    parts = []
    for label, source_plain, source_hybrid in (
        ("overall", plain.overall, hybrid.overall),
        (
            "keyword-category",
            plain.per_category.get("keyword"),
            hybrid.per_category.get("keyword"),
        ),
    ):
        if not source_plain or not source_hybrid:
            continue
        cells = []
        for key, name in (("precision", "P"), ("recall", "R"), ("ndcg", "NDCG")):
            before = source_plain[key]
            after = source_hybrid[key]
            cells.append(f"{name} {before:.3f}->{after:.3f} ({after - before:+.3f})")
        parts.append(f"{label}: " + " · ".join(cells))
    return f"four-seam keyword (plain) vs hybrid @k={k} — " + " | ".join(parts)
