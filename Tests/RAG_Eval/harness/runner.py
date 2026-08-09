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

from tldw_chatbook.RAG_Search.eval.metrics import evaluate_retrieval_batch

from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids, slug_lookup_from
from Tests.RAG_Eval.harness.goldenset import NEGATIVE_CATEGORY, GoldenQuery

__all__ = [
    "EvalReport",
    "MODES",
    "ModeReport",
    "NegativeProbe",
    "QueryOutcome",
    "SOURCE_TYPES",
    "run_eval",
]

#: The three retrieval modes, in report order. These are the values
#: `SearchConfig.default_search_mode` takes and the seam's `_search_rag`
#: routes on.
MODES: tuple[str, ...] = ("semantic", "plain", "hybrid")

#: The Library scope identifiers passed to the seam (plural — the scope
#: vocabulary, not the ingestion one). All three corpus source types, and
#: `media` in particular: the seam degrades a hybrid profile to semantic
#: when media is deselected, so dropping it would silently make the hybrid
#: pass a second semantic pass.
SOURCE_TYPES: tuple[str, ...] = ("media", "notes", "conversations")

#: Metric keys `evaluate_retrieval_batch` returns (short names — see the
#: port note in `RAG_Search/eval/metrics.py`).
_METRIC_KEYS: tuple[str, ...] = ("precision", "recall", "mrr", "ndcg", "f1")


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "relevant_slugs": list(self.relevant_slugs),
            "rows_returned": self.rows_returned,
            "latency_ms": self.latency_s * 1000.0,
            "runtime_backend": self.runtime_backend,
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
    #: precision because the ported `precision_at_k` divides by the number
    #: of results actually returned, not by `k` (Task 2 pinned that
    #: convention against the server module): a mode that returns one
    #: correct document scores P@10 = 1.0, and a mode that returns ten with
    #: one correct scores 0.1. Without this column the precision row reads
    #: as a quality ranking when it is partly a verbosity ranking.
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
    num_scored: int = 0
    num_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot (Task 7's baselines are written from this)."""
        return {
            "k": self.k,
            "source_types": list(self.source_types),
            "num_queries": self.num_queries,
            "num_scored": self.num_scored,
            "num_negative": self.num_negative,
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
        ValueError: ``golden`` is empty, ``k < 1``, or ``modes`` is empty.
            An empty run would report a full set of 0.0 metrics, which reads
            as a total retrieval failure rather than as no measurement.
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

    search_config = runtime.service.config.search
    original_mode = getattr(search_config, "default_search_mode", None)
    mode_reports: dict[str, ModeReport] = {}
    try:
        for mode in modes:
            search_config.default_search_mode = mode
            outcomes = tuple(
                _run_query(seam, runtime, query, k, scope, lookup) for query in golden
            )
            mode_reports[mode] = _build_mode_report(mode, k, outcomes)
    finally:
        # The runtime is usually discarded after a run, but leaving a
        # caller's service on whichever mode happened to be last would make
        # any later use of it depend on this function's loop order.
        if original_mode is not None:
            search_config.default_search_mode = original_mode

    negatives = sum(1 for query in golden if query.category == NEGATIVE_CATEGORY)
    return EvalReport(
        k=k,
        modes=mode_reports,
        source_types=scope,
        num_queries=len(golden),
        num_scored=len(golden) - negatives,
        num_negative=negatives,
    )


def _run_query(
    seam: Any,
    runtime: Any,
    query: GoldenQuery,
    k: int,
    source_types: tuple[str, ...],
    lookup: Mapping[tuple[str, str], str],
) -> QueryOutcome:
    """Run one query and canonicalize its rows, recording rather than raising."""
    start = time.perf_counter()
    error: str | None = None
    result: Any = None
    try:
        result = runtime.run(
            seam.search(query.query, source_types, "rag", top_k=k)
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_s = time.perf_counter() - start

    rows: list[Mapping[str, Any]] = []
    backend = ""
    if error is None:
        rows, backend, error = _extract_rows(result)

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
    )


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
    overall = _metrics_for(scored, k)
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
            sum(len(outcome.retrieved_doc_ids[:k]) for outcome in scored) / len(scored)
            if scored
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
        f"({report.num_scored} scored, {report.num_negative} negative) over "
        f"{'/'.join(report.source_types)}"
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
        "'docs' = mean distinct documents returned per scored query; P@k "
        "divides by that, not by k (see ModeReport.mean_docs_at_k)."
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

    plain = report.modes.get("plain")
    hybrid = report.modes.get("hybrid")
    if plain is not None and hybrid is not None:
        lines.append("")
        lines.append(_delta_line(plain, hybrid, k))
    return "\n".join(lines)


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
