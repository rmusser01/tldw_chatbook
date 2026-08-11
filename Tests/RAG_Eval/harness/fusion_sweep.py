# Tests/RAG_Eval/harness/fusion_sweep.py
"""Fusion strategy sweep: run the golden set through hybrid under N settings.

TASK-4110's decision is made by measurement, not argument. This module is the
measurement tool: it takes a matrix of `Strategy` values (rrf_k, hybrid pool
multiplier, hybrid alpha), runs the whole golden set through **hybrid mode
only** once per strategy on P1's runner, records the rescue verdict for the
vector-blind fixture, and applies the spec's decision rule mechanically.

Four things about it are load-bearing.

**It reuses P1's runner rather than re-implementing scoring.** Each pass is a
`run_eval(..., modes=("hybrid",))` call, so the per-category metrics, the
negative probes, the latency percentiles and the canonicalization are the
same code the P1 gate scores with. The only thing this module adds is a
per-strategy config flip, the rescue probe, and the table.

**The config is flipped and restored in `finally`.** Exactly P1's pattern for
`default_search_mode`, extended to the three fusion fields Task 3 threaded
plus the mode itself. A sweep that crashed mid-matrix must not hand the
caller's service back on whatever knobs the last attempt happened to set.
The search cache key covers all three resolved fusion values (Task 3), so a
stale entry can no longer report every strategy as "no effect"; the sweep
still clears the cache between passes as belt-and-braces and to keep memory
flat over a long matrix.

**The two rescue senses are distinct, and the structural one is arithmetic.**
`fts_only_beats_vector_rank(alpha, rrf_k)` answers AC#4's question — what is
the best vector-only rank an FTS-only rank-1 row can beat — from the fusion
formula itself, so it does not depend on which documents happened to be in
the fixture corpus. It reproduces the spec's diagnosis exactly: at the
shipped 0.7/60 an FTS-only row only wins past vector rank 82, while the
hybrid legs fetch 20 candidates, which is why the measured baseline has the
fixture sorting 21st behind 20 vector rows. AC#3's sense — did the fixture
actually surface in the top-10 — is measured, and the mechanism (merged via a
widened pool, or fts-only via the weighting) is read out of the row's own
`hybrid_fusion` metadata rather than inferred.

**Pool widening can never be the sole winner.** A wider pool widens the
candidate window, which makes the structural threshold easier to fall inside
— so "threshold inside the window" alone would let pool widening satisfy a
guarantee it structurally cannot provide (the fixture is vector-POOR, not
vector-ABSENT; a genuinely vector-absent document is still unreachable at any
pool size). `qualify` therefore requires a *weighting* lever to have moved as
its own clause, and `Strategy.changed_levers` deliberately does not count the
pool multiplier as one.

Nothing here is imported by the application.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence

from loguru import logger

from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids, slug_lookup_from
from Tests.RAG_Eval.harness.goldenset import GoldenQuery
from Tests.RAG_Eval.harness.runner import (
    SOURCE_TYPES,
    ModeReport,
    # Private on purpose in runner.py, imported here rather than duplicated:
    # normalizing the seam's two return shapes is the runner's own contract
    # with the seam, and a second copy of it in this module would drift the
    # first time the seam grows a third shape.
    _extract_rows,
    count_scored,
    run_eval,
)

__all__ = [
    "ALPHA_COMBO_STRATEGIES",
    "BASE_STRATEGIES",
    "CONTROL",
    "CONTROL_NAME",
    "DEFAULT_K",
    "HYBRID_MODE",
    "LEVER_PRECEDENCE",
    "REGRESSION_METRICS",
    "RESCUE_QUERY_ID",
    "RESCUE_TARGET_SLUG",
    "Qualification",
    "RescueVerdict",
    "Strategy",
    "StrategyReport",
    "SweepReport",
    "WARN_BAND",
    "combined_strategy",
    "format_matrix",
    "fts_only_beats_vector_rank",
    "lever_rank",
    "qualify",
    "run_full_matrix",
    "run_fusion_sweep",
    "select_winner",
    "worst_regression",
]

#: The only mode a fusion sweep measures. Semantic and plain do not read any
#: of these knobs, so running them would triple the cost for three identical
#: copies of the same numbers.
HYBRID_MODE = "hybrid"

#: Result cap and metric @k, matching the P1 gate's.
DEFAULT_K = 10

#: The strategy every other strategy is compared against, by name.
CONTROL_NAME = "control"

#: The vector-blind fixture TASK-4110 was raised on: plain finds it at rank
#: 1, semantic does not return it at all (it sits ~rank 22 in the index), and
#: hybrid drops it entirely at the shipped defaults.
RESCUE_QUERY_ID = "kw-plant-maintenance-record"
RESCUE_TARGET_SLUG = "note-saltmarsh-hide"

#: The gate's warn band (`baseline_io`'s). A per-category cell that falls by
#: more than this disqualifies a strategy.
WARN_BAND = 0.02

#: Metrics clause (b) protects. Precision is deliberately absent: it moves
#: mechanically with row counts (P@k divides by ``min(k, len(retrieved))``,
#: not by k), so a strategy that returns more rows scores lower precision for
#: reasons that say nothing about quality. It is reported, not gated.
REGRESSION_METRICS: tuple[str, ...] = ("recall", "mrr", "ndcg")

#: Tie-break order among weighting levers (spec's clause c). ``quota`` has no
#: field on `Strategy` — the quota mechanism is only built if no parameter
#: strategy qualifies (the spec's YAGNI ordering) — but its precedence is
#: fixed here so a later task cannot quietly reorder it to suit a result.
LEVER_PRECEDENCE: tuple[str, ...] = ("rrf_k", "quota", "hybrid_alpha")

#: Config fields the sweep writes, and therefore must put back.
_RESTORED_FIELDS: tuple[str, ...] = (
    "rrf_k",
    "hybrid_pool_multiplier",
    "hybrid_alpha",
    "default_search_mode",
)


@dataclass(frozen=True, slots=True)
class Strategy:
    """One point in the fusion parameter space.

    Attributes:
        name: Short label; also the matrix column header, so keep it under
            ~10 characters.
        rrf_k: ``config.search.rrf_k`` — the RRF denominator constant.
        hybrid_pool_multiplier: ``config.search.hybrid_pool_multiplier`` —
            each hybrid leg over-fetches ``top_k * this`` candidates. Note
            the semantic leg's raw vector-store fetch compounds this with the
            module-level ``SEARCH_RESULT_MULTIPLIER``; the number that
            matters for fusion is the leg's returned candidate budget, which
            is ``top_k * this``.
        hybrid_alpha: ``config.search.hybrid_alpha`` — the vector leg's blend
            weight.
    """

    name: str
    rrf_k: int
    hybrid_pool_multiplier: int
    hybrid_alpha: float

    def apply(self, search_config: Any) -> None:
        """Write this strategy's three knobs onto a live `SearchConfig`."""
        search_config.rrf_k = self.rrf_k
        search_config.hybrid_pool_multiplier = self.hybrid_pool_multiplier
        search_config.hybrid_alpha = self.hybrid_alpha

    def changed_fields(self, baseline: "Strategy") -> tuple[str, ...]:
        """Config fields this strategy moves relative to ``baseline``."""
        return tuple(
            field
            for field in ("rrf_k", "hybrid_pool_multiplier", "hybrid_alpha")
            if getattr(self, field) != getattr(baseline, field)
        )

    def changed_levers(self, baseline: "Strategy") -> tuple[str, ...]:
        """The **weighting** levers this strategy moves.

        `hybrid_pool_multiplier` is excluded by construction: widening the
        candidate pool changes which documents fusion sees, never how fusion
        weighs them, so it cannot satisfy AC#4's structural guarantee. Its
        exclusion here is the single place that rule is enforced.
        """
        return tuple(
            field for field in self.changed_fields(baseline) if field != "hybrid_pool_multiplier"
        )

    def deviation(self, baseline: "Strategy") -> float:
        """Relative L1 distance from ``baseline`` — the 'smallest deviation' metric.

        Relative rather than absolute so a 40-unit move in `rrf_k` and a
        0.15 move in `hybrid_alpha` are on one scale at all.
        """
        total = 0.0
        for field in ("rrf_k", "hybrid_pool_multiplier", "hybrid_alpha"):
            base = float(getattr(baseline, field))
            mine = float(getattr(self, field))
            total += abs(mine - base) / abs(base) if base else abs(mine - base)
        return total

    def vector_window(self, k: int) -> int:
        """Candidate rows the vector leg is asked for at this top_k.

        The window AC#4's structural question is asked inside: an FTS-only
        row can only outrank a vector-only row that fusion actually saw.
        """
        return k * self.hybrid_pool_multiplier

    def describe(self) -> str:
        return (
            f"rrf_k={self.rrf_k}, pool=x{self.hybrid_pool_multiplier}, "
            f"alpha={self.hybrid_alpha:.2f}"
        )


#: The control: ADR-005's server-parity constants, which were the SHIPPED
#: defaults when this sweep was run. Task 5 then shipped the winner, so
#: `rrf_k=60` is now the pre-decision baseline rather than the live default
#: (`config.DEFAULT_HYBRID_RRF_K` = 5). It stays 60 deliberately — a control
#: re-pointed at the value the experiment chose would compare the winner to
#: itself and every delta in the table would read +0.000.
CONTROL = Strategy(CONTROL_NAME, rrf_k=60, hybrid_pool_multiplier=2, hybrid_alpha=0.7)

#: The spec's matrix, phase 1: control, the rrf_k sweep at the shipped pool
#: and alpha, and the pool sweep at the shipped weighting. The combination is
#: derived from these results (`combined_strategy`), not fixed here.
BASE_STRATEGIES: tuple[Strategy, ...] = (
    CONTROL,
    Strategy("k20", rrf_k=20, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
    Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
    Strategy("k5", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
    Strategy("pool3", rrf_k=60, hybrid_pool_multiplier=3, hybrid_alpha=0.7),
    Strategy("pool5", rrf_k=60, hybrid_pool_multiplier=5, hybrid_alpha=0.7),
)

#: Phase 3, run ONLY when nothing above qualifies (the spec's last resort:
#: alpha is a global penalty on vector quality, so it is never a first move).
ALPHA_COMBO_STRATEGIES: tuple[Strategy, ...] = (
    Strategy("a.60+k20", rrf_k=20, hybrid_pool_multiplier=2, hybrid_alpha=0.60),
    Strategy("a.55", rrf_k=60, hybrid_pool_multiplier=2, hybrid_alpha=0.55),
)


def fts_only_beats_vector_rank(alpha: float, rrf_k: int) -> Optional[int]:
    """Best vector-only rank an FTS-only rank-1 row outranks, or None.

    The fused score of a row only the keyword leg returned is
    ``(1 - alpha) / (rrf_k + 1)``; of a row only the vector leg returned at
    rank ``r``, ``alpha / (rrf_k + r)``. The first *outranks* the second when

        r >= alpha * (rrf_k + 1) / (1 - alpha) - rrf_k

    ``>=`` rather than ``>`` because an exact score tie is decided, not
    arbitrary: `reciprocal_rank_fusion` sorts by ``(-score, fts_rank,
    vector_rank)`` with absent ranks at infinity, so on a tie the FTS-only
    row (fts_rank 1) sorts ahead of the vector-only row (fts_rank absent).
    At alpha 0.7 / rrf_k 20 the boundary is exactly 29 — 0.3/21 == 0.7/49 —
    and rank 29 does go to the keyword row.

    Args:
        alpha: Vector-leg blend weight in [0, 1].
        rrf_k: RRF denominator constant.

    Returns:
        The best (smallest) vector-only rank the FTS-only row outranks, or
        ``None`` when it never can (``alpha == 1``: the FTS leg carries no
        weight at all).

    Note:
        The epsilon is not cosmetic: the boundary is exactly integral at
        several of the values being swept, and in binary floating point
        ``0.7 * 21 / (1 - 0.7) - 20`` evaluates to 28.999999999999993 while
        the same expression with other constants can land just above the
        integer. Nudging before `ceil` makes the answer the mathematical one
        in both directions rather than an artefact of the rounding.
    """
    if alpha >= 1.0:
        return None
    threshold = alpha * (rrf_k + 1) / (1.0 - alpha) - rrf_k
    return max(1, math.ceil(threshold - 1e-9))


def lever_rank(levers: Collection[str]) -> int:
    """Precedence of a strategy's most-preferred weighting lever.

    Lower is better. A strategy that moves no weighting lever ranks last (it
    cannot satisfy AC#4 at all, and `qualify` rejects it outright — this only
    keeps the ordering total).
    """
    ranks = [
        LEVER_PRECEDENCE.index(lever)
        for lever in levers
        if lever in LEVER_PRECEDENCE
    ]
    return min(ranks) if ranks else len(LEVER_PRECEDENCE)


@dataclass(frozen=True, slots=True)
class RescueVerdict:
    """What one strategy did with the vector-blind fixture.

    Attributes:
        present: Did the target document appear within the top-k the user
            would see? This is AC#3's cell.
        rank: 1-based document rank of the target, or None when absent. Only
            observable inside the top-k the seam returned — a fixture that
            sorts 16th under a top-10 search is simply "absent" here, because
            that is exactly what the product would show.
        mechanism: ``merged`` (both legs found it — the widened-pool rescue),
            ``fts-only`` (the keyword leg alone, carried by the weighting),
            ``vector-only``, ``absent``, or ``unknown`` when the row carried
            no `hybrid_fusion` block.
        run_rank: The same rank as computed by the sweep's `run_eval` pass.
            Recorded to cross-check the probe against the scored run; a
            disagreement means the two calls did not see the same retrieval.
    """

    query_id: str
    target_slug: str
    present: bool
    rank: Optional[int]
    mechanism: str
    fts_rank: Optional[int]
    vector_rank: Optional[int]
    docs_returned: int
    run_rank: Optional[int]

    @property
    def consistent_with_run(self) -> bool:
        return self.rank == self.run_rank


@dataclass(frozen=True, slots=True)
class StrategyReport:
    """One strategy's whole result: hybrid metrics + the rescue verdict."""

    strategy: Strategy
    hybrid: ModeReport
    rescue: RescueVerdict


@dataclass(frozen=True, slots=True)
class Qualification:
    """Why one strategy does or does not satisfy the spec's decision rule."""

    strategy_name: str
    weighting_changed: bool
    structural_ok: bool
    structural_threshold: Optional[int]
    vector_window: int
    rescued: bool
    rescue_mechanism: str
    regression_ok: bool
    worst_regression: Optional[tuple[str, str, float]]

    @property
    def qualifies(self) -> bool:
        return (
            self.weighting_changed
            and self.structural_ok
            and self.rescued
            and self.regression_ok
        )

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why it does not qualify — empty when it does."""
        reasons: list[str] = []
        if not self.weighting_changed:
            reasons.append(
                "no weighting lever moved (pool widening alone can never "
                "satisfy AC#4)"
            )
        if not self.structural_ok:
            threshold = (
                "never" if self.structural_threshold is None
                else f">= {self.structural_threshold}"
            )
            reasons.append(
                f"an FTS-only rank-1 row only outranks vector rank {threshold}, "
                f"but the leg fetches {self.vector_window}"
            )
        if not self.rescued:
            reasons.append("the fixture did not reach the top-k")
        if not self.regression_ok and self.worst_regression is not None:
            category, metric, delta = self.worst_regression
            reasons.append(f"{category}/{metric} {delta:+.3f} exceeds the warn band")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Every strategy's result for one matrix run."""

    k: int
    entries: tuple[StrategyReport, ...]
    rescue_query_id: str
    target_slug: str
    source_types: tuple[str, ...]
    num_queries: int
    num_scored: int
    control_name: str = CONTROL_NAME

    def control(self) -> StrategyReport:
        """The baseline entry.

        Raises:
            ValueError: No entry is named `control_name`. Every clause of the
                decision rule is stated relative to the shipped defaults, so
                a matrix without them cannot be scored at all — better a loud
                refusal than a table silently graded against its own first row.
        """
        for entry in self.entries:
            if entry.strategy.name == self.control_name:
                return entry
        raise ValueError(
            f"the sweep has no {self.control_name!r} entry; the decision rule "
            "is defined relative to the shipped defaults"
        )

    def with_entries(self, more: Iterable[StrategyReport]) -> "SweepReport":
        """A copy with extra strategy results appended (phases 2 and 3)."""
        return replace(self, entries=self.entries + tuple(more))

    def qualifications(self) -> tuple[Qualification, ...]:
        control = self.control()
        return tuple(qualify(entry, control, k=self.k) for entry in self.entries)

    def winner(self) -> Optional[StrategyReport]:
        return select_winner(self)

    def format_matrix(self) -> str:
        return format_matrix(self)


def worst_regression(
    candidate: ModeReport, control: ModeReport
) -> Optional[tuple[str, str, float]]:
    """The most negative per-category recall/MRR/NDCG delta vs the control.

    Args:
        candidate: The strategy's hybrid mode report.
        control: The control strategy's hybrid mode report.

    Returns:
        ``(category, metric, delta)`` for the worst cell — the delta is
        ``candidate - control``, so a regression is negative — or ``None``
        when the control scored no categories at all. A category the control
        scored and the candidate did not counts as a full loss of that cell,
        never as "no change": a vanished category is the loudest possible
        regression and must not read as a silent zero.
    """
    worst: Optional[tuple[str, str, float]] = None
    for category, control_cells in sorted(control.per_category.items()):
        candidate_cells = candidate.per_category.get(category)
        for metric in REGRESSION_METRICS:
            before = float(control_cells.get(metric, 0.0))
            if candidate_cells is None:
                delta = -before
            else:
                delta = float(candidate_cells.get(metric, 0.0)) - before
            if worst is None or delta < worst[2]:
                worst = (category, metric, delta)
    return worst


def qualify(
    entry: StrategyReport, control: StrategyReport, *, k: int = DEFAULT_K
) -> Qualification:
    """Apply the spec's decision rule to one strategy.

    Four clauses, all required:

    1. a **weighting** lever moved (`rrf_k`, `hybrid_alpha`, or — when it
       exists — a quota); pool widening alone never counts;
    2. AC#4's structural sense: under those weights an FTS-only rank-1 row
       outranks at least one vector-only row *inside the candidate window
       fusion actually sees*;
    3. AC#3's fixture sense: the vector-blind fixture reached the top-k;
    4. no per-category recall/MRR/NDCG cell fell by more than `WARN_BAND`.

    Args:
        entry: The strategy to judge.
        control: The shipped-defaults entry every clause is relative to.
        k: The run's top-k, which sets the candidate window.

    Returns:
        A `Qualification` carrying each clause's answer and its evidence.
    """
    strategy = entry.strategy
    baseline = control.strategy
    threshold = fts_only_beats_vector_rank(strategy.hybrid_alpha, strategy.rrf_k)
    window = strategy.vector_window(k)
    worst = worst_regression(entry.hybrid, control.hybrid)
    # `> WARN_BAND` — the band itself is inside the gate, matching the P1
    # gate's own wording. The epsilon absorbs binary-float noise on an
    # exactly-at-the-band delta (0.65 - 0.02 does not round-trip exactly).
    regression_ok = worst is None or worst[2] >= -WARN_BAND - 1e-9
    return Qualification(
        strategy_name=strategy.name,
        weighting_changed=bool(strategy.changed_levers(baseline)),
        structural_ok=threshold is not None and threshold <= window,
        structural_threshold=threshold,
        vector_window=window,
        rescued=entry.rescue.present,
        rescue_mechanism=entry.rescue.mechanism,
        regression_ok=regression_ok,
        worst_regression=worst,
    )


def _preference_key(entry: StrategyReport, baseline: Strategy) -> tuple:
    """Clause (c): the smallest deviation among qualifiers.

    Ordered by the lever's precedence first, then by how many knobs moved at
    all (so a widened pool rides along only when its weighting twin did not
    qualify on its own), then by relative deviation, then by name for
    determinism.
    """
    return (
        lever_rank(entry.strategy.changed_levers(baseline)),
        len(entry.strategy.changed_fields(baseline)),
        entry.strategy.deviation(baseline),
        entry.strategy.name,
    )


def select_winner(report: SweepReport) -> Optional[StrategyReport]:
    """The winning strategy, or None when the matrix is BLOCKED.

    Args:
        report: A sweep containing the control.

    Returns:
        The qualifying strategy with the smallest deviation, or ``None`` when
        nothing qualifies — which the spec calls a finding, not a failure: the
        owner then chooses the trade-off from the matrix.

    Raises:
        ValueError: The report has no control entry.
    """
    control = report.control()
    qualified = [
        entry
        for entry in report.entries
        if qualify(entry, control, k=report.k).qualifies
    ]
    if not qualified:
        return None
    return min(qualified, key=lambda e: _preference_key(e, control.strategy))


def _family_key(entry: StrategyReport, baseline: Strategy, k: int) -> tuple:
    """Rank one strategy within its own single-knob family.

    Structurally viable first, then rescuing the fixture, then the better
    fused rank, then the better overall NDCG, then the smaller deviation.
    """
    threshold = fts_only_beats_vector_rank(
        entry.strategy.hybrid_alpha, entry.strategy.rrf_k
    )
    structural_ok = threshold is not None and threshold <= entry.strategy.vector_window(k)
    return (
        not structural_ok,
        not entry.rescue.present,
        entry.rescue.rank if entry.rescue.rank is not None else math.inf,
        -float(entry.hybrid.overall.get("ndcg", 0.0)),
        entry.strategy.deviation(baseline),
        entry.strategy.name,
    )


def combined_strategy(report: SweepReport) -> Optional[Strategy]:
    """Phase 2: the best rrf_k variant paired with the best pool variant.

    'The two best combined' means the best of each *family*, not the two best
    rows overall — combining two rrf_k values is not a thing you can do. Alpha
    is left at the control's value; it is the last-resort lever.

    Args:
        report: The phase-1 sweep (must contain the control).

    Returns:
        The combined strategy, or ``None`` when either family is missing from
        the matrix.
    """
    control = report.control()
    baseline = control.strategy
    k_family = [
        entry
        for entry in report.entries
        if entry.strategy.changed_fields(baseline) == ("rrf_k",)
    ]
    pool_family = [
        entry
        for entry in report.entries
        if entry.strategy.changed_fields(baseline) == ("hybrid_pool_multiplier",)
    ]
    if not k_family or not pool_family:
        return None
    best_k = min(k_family, key=lambda e: _family_key(e, baseline, report.k))
    best_pool = min(pool_family, key=lambda e: _family_key(e, baseline, report.k))
    return Strategy(
        name=f"k{best_k.strategy.rrf_k}+pool{best_pool.strategy.hybrid_pool_multiplier}",
        rrf_k=best_k.strategy.rrf_k,
        hybrid_pool_multiplier=best_pool.strategy.hybrid_pool_multiplier,
        hybrid_alpha=baseline.hybrid_alpha,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _build_seam(runtime: Any) -> Any:
    from tldw_chatbook.Library.library_local_rag_search_service import (
        LibraryLocalRagSearchService,
    )

    return LibraryLocalRagSearchService(runtime.app)


def _fusion_block(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    fusion = provenance.get("hybrid_fusion")
    return fusion if isinstance(fusion, Mapping) else None


def _mechanism(fusion: Optional[Mapping[str, Any]]) -> str:
    if fusion is None:
        return "unknown"
    fts_rank = fusion.get("fts_rank")
    vector_rank = fusion.get("vector_rank")
    if fts_rank is not None and vector_rank is not None:
        return "merged"
    if fts_rank is not None:
        return "fts-only"
    if vector_rank is not None:
        return "vector-only"
    return "unknown"


def _rescue_probe(
    seam: Any,
    runtime: Any,
    lookup: Mapping[tuple[str, str], str],
    query: GoldenQuery,
    k: int,
    source_types: tuple[str, ...],
    target_slug: str,
    hybrid: ModeReport,
) -> RescueVerdict:
    """Re-run the rescue query and read the target row's fusion provenance.

    `run_eval` scores documents, so its `QueryOutcome` knows the target's
    rank but not *how* it got there. One extra call answers that; it is
    served from the search cache the scored pass just populated (same query,
    same top_k, same fusion key), so it costs a dictionary lookup rather than
    a second retrieval.
    """
    result = runtime.run(seam.search(query.query, source_types, "rag", top_k=k))
    rows, _backend, _error = _extract_rows(result)
    doc_ids = rows_to_doc_ids(rows, lookup)

    rank: Optional[int] = None
    if target_slug in doc_ids[:k]:
        rank = doc_ids.index(target_slug) + 1

    fusion: Optional[Mapping[str, Any]] = None
    for row in rows:
        if rows_to_doc_ids([row], lookup) == [target_slug]:
            fusion = _fusion_block(row)
            break

    run_rank: Optional[int] = None
    for outcome in hybrid.queries:
        if outcome.query_id == query.id:
            ids = list(outcome.retrieved_doc_ids[:k])
            if target_slug in ids:
                run_rank = ids.index(target_slug) + 1
            break

    return RescueVerdict(
        query_id=query.id,
        target_slug=target_slug,
        present=rank is not None,
        rank=rank,
        mechanism=_mechanism(fusion) if rank is not None else "absent",
        fts_rank=fusion.get("fts_rank") if fusion else None,
        vector_rank=fusion.get("vector_rank") if fusion else None,
        docs_returned=len(doc_ids),
        run_rank=run_rank,
    )


def run_fusion_sweep(
    runtime: Any,
    golden: Sequence[GoldenQuery],
    strategies: Sequence[Strategy] = BASE_STRATEGIES,
    k: int = DEFAULT_K,
    *,
    source_types: Sequence[str] = SOURCE_TYPES,
    rescue_query_id: str = RESCUE_QUERY_ID,
    target_slug: str = RESCUE_TARGET_SLUG,
    seam: Any = None,
) -> SweepReport:
    """Run the golden set through hybrid once per strategy.

    Args:
        runtime: A live `EvalRuntime`. One per process — `RAGService.close`
            clears process-global pools — so every strategy shares it and
            every async call goes through ``runtime.run``.
        golden: The validated golden query set.
        strategies: The matrix to measure, in report order.
        k: Result cap and metric @k.
        source_types: Library scope identifiers to search.
        rescue_query_id: The golden query the rescue verdict is read from.
        target_slug: The fixture document that query must surface.
        seam: Optional pre-built `LibraryLocalRagSearchService`; built from
            ``runtime.app`` when omitted.

    Returns:
        A `SweepReport`. It has no control entry unless `strategies` contains
        one — phases 2 and 3 append to the phase-1 report rather than being
        scored on their own.

    Raises:
        ValueError: Empty matrix, ``k < 1``, or no golden query with
            `rescue_query_id`.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not strategies:
        raise ValueError("refusing to run a sweep over an empty strategy matrix")
    rescue_query = next((q for q in golden if q.id == rescue_query_id), None)
    if rescue_query is None:
        raise ValueError(
            f"the golden set has no query {rescue_query_id!r}; the rescue "
            "verdict is the point of this sweep"
        )

    if seam is None:
        seam = _build_seam(runtime)
    lookup = slug_lookup_from(runtime.slug_to_source)
    scope = tuple(source_types)
    search_config = runtime.service.config.search
    saved = {field: getattr(search_config, field) for field in _RESTORED_FIELDS}

    entries: list[StrategyReport] = []
    try:
        for strategy in strategies:
            strategy.apply(search_config)
            # Belt-and-braces: Task 3 put the three resolved fusion values in
            # the hybrid cache key, so passes can no longer share an entry;
            # clearing anyway keeps a long matrix's memory flat and means a
            # future cache-key regression cannot silently flatten the sweep.
            runtime.service.clear_cache()
            logger.info(f"fusion sweep: {strategy.name} ({strategy.describe()})")
            report = run_eval(
                runtime, golden, k=k, modes=(HYBRID_MODE,), source_types=scope
            )
            hybrid = report.modes[HYBRID_MODE]
            # `run_eval` restores whatever mode it found; the probe needs
            # hybrid, and the finally below puts the caller's value back.
            search_config.default_search_mode = HYBRID_MODE
            verdict = _rescue_probe(
                seam, runtime, lookup, rescue_query, k, scope, target_slug, hybrid
            )
            entries.append(StrategyReport(strategy, hybrid, verdict))
    finally:
        for field, value in saved.items():
            setattr(search_config, field, value)

    # Literally the function `run_eval` counts with, not a second predicate
    # that happens to agree: the sweep's header labels the same averaged row
    # `run_eval` produced, and a local copy of the rule silently over-counted
    # the moment a second unaveraged category (scoped) existed. Pinned by
    # `test_fusion_decision_rule.test_the_sweep_counts_scored_queries_the_way_run_eval_does`.
    return SweepReport(
        k=k,
        entries=tuple(entries),
        rescue_query_id=rescue_query_id,
        target_slug=target_slug,
        source_types=scope,
        num_queries=len(golden),
        num_scored=count_scored(golden),
    )


def run_full_matrix(
    runtime: Any,
    golden: Sequence[GoldenQuery],
    k: int = DEFAULT_K,
    **kwargs: Any,
) -> SweepReport:
    """The spec's whole matrix: base sweep, derived combination, last resort.

    Phase 2 (the combination) is derived from phase 1's numbers, and phase 3
    (alpha) runs **only** when nothing else qualifies — the spec's YAGNI
    ordering, so the expensive last-resort lever is never measured for the
    sake of a fuller table.
    """
    report = run_fusion_sweep(runtime, golden, BASE_STRATEGIES, k=k, **kwargs)

    combination = combined_strategy(report)
    if combination is not None:
        report = report.with_entries(
            run_fusion_sweep(runtime, golden, (combination,), k=k, **kwargs).entries
        )

    if select_winner(report) is None:
        logger.warning(
            "fusion sweep: no parameter strategy qualified; measuring the "
            "alpha combinations (last resort)"
        )
        report = report.with_entries(
            run_fusion_sweep(
                runtime, golden, ALPHA_COMBO_STRATEGIES, k=k, **kwargs
            ).entries
        )
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(value: Optional[float], width: int = 8) -> str:
    return f"{'-':>{width}}" if value is None else f"{value:>{width}.3f}"


def format_matrix(report: SweepReport) -> str:
    """Render the decision table: one row per strategy, then the verdict."""
    control = report.control()
    baseline = control.strategy
    lines: list[str] = []
    lines.append(
        f"Fusion strategy sweep @k={report.k} — hybrid mode only, "
        f"{report.num_queries} golden queries ({report.num_scored} scored) over "
        f"{'/'.join(report.source_types)}"
    )
    lines.append(
        f"control = {baseline.name} ({baseline.describe()}); "
        f"rescue fixture = {report.rescue_query_id} -> {report.target_slug}"
    )
    lines.append("")

    header = (
        f"{'strategy':<11}{'rrf_k':>6}{'pool':>5}{'alpha':>7}"
        f"{'P@k':>8}{'R@k':>8}{'MRR':>8}{'NDCG':>8}{'docs':>6}"
        f"{'fts>vec':>8}{'win':>5}{'rescue':>8}{'rank':>6}{'mech':>11}"
        f"{'worst':>8}{'qual':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for entry in report.entries:
        strategy = entry.strategy
        verdict = qualify(entry, control, k=report.k)
        overall = entry.hybrid.overall
        threshold = verdict.structural_threshold
        worst = verdict.worst_regression
        lines.append(
            f"{strategy.name:<11}{strategy.rrf_k:>6}"
            f"{('x' + str(strategy.hybrid_pool_multiplier)):>5}"
            f"{strategy.hybrid_alpha:>7.2f}"
            f"{_cell(overall.get('precision'))}{_cell(overall.get('recall'))}"
            f"{_cell(overall.get('mrr'))}{_cell(overall.get('ndcg'))}"
            f"{entry.hybrid.mean_docs_at_k:>6.1f}"
            f"{('never' if threshold is None else threshold):>8}"
            f"{verdict.vector_window:>5}"
            f"{('yes' if verdict.rescued else 'no'):>8}"
            f"{(entry.rescue.rank if entry.rescue.rank is not None else '-'):>6}"
            f"{entry.rescue.mechanism:>11}"
            f"{(worst[2] if worst else 0.0):>+8.3f}"
            f"{('YES' if verdict.qualifies else 'no'):>6}"
        )
    lines.append(
        "'fts>vec' = best vector-only rank an FTS-only rank-1 row outranks "
        "under these weights; 'win' = candidate rows the vector leg is asked "
        "for. AC#4's structural sense holds when fts>vec <= win."
    )
    lines.append(
        "'worst' = most negative per-category recall/MRR/NDCG delta vs the "
        f"control; a strategy is disqualified below -{WARN_BAND:.3f}. Precision "
        "moves mechanically with 'docs' (P@k divides by min(k, len(retrieved)), "
        "not by k) and is reported, not gated."
    )

    categories = sorted(
        {
            category
            for entry in report.entries
            for category in entry.hybrid.per_category
        }
    )
    if categories:
        lines.append("")
        lines.append(f"per-category cells (hybrid) @k={report.k}")
        column = f"{'category/metric':<30}" + "".join(
            f"{entry.strategy.name:>11}" for entry in report.entries
        )
        lines.append(column)
        lines.append("-" * len(column))
        for category in categories:
            for metric in ("recall", "mrr", "ndcg", "precision"):
                cells = "".join(
                    _cell(
                        entry.hybrid.per_category.get(category, {}).get(metric),
                        width=11,
                    )
                    for entry in report.entries
                )
                lines.append(f"{category + '/' + metric:<30}{cells}")

    lines.append("")
    lines.append(
        "qualification (decision rule: a weighting lever moved AND its "
        "structural threshold falls inside the candidate window AND the "
        "fixture reached the top-k AND no gated cell fell by more than "
        f"{WARN_BAND:.3f})"
    )
    for entry in report.entries:
        verdict = qualify(entry, control, k=report.k)
        if verdict.qualifies:
            lines.append(
                f"  {entry.strategy.name:<11} QUALIFIES — fixture rescued at "
                f"rank {entry.rescue.rank} ({entry.rescue.mechanism}); "
                f"worst gated cell "
                f"{(verdict.worst_regression[2] if verdict.worst_regression else 0.0):+.3f}"
            )
        else:
            lines.append(
                f"  {entry.strategy.name:<11} no — " + "; ".join(verdict.reasons)
            )

    lines.append("")
    winner = select_winner(report)
    if winner is None:
        lines.append(
            "DECISION: BLOCKED — no strategy satisfies the rule. The spec says "
            "this is a finding, not a failure: report the matrix and let the "
            "owner choose the trade-off (or build the keyword slot quota)."
        )
    else:
        verdict = qualify(winner, control, k=report.k)
        lines.append(
            f"DECISION: WINNER = {winner.strategy.name} ({winner.strategy.describe()})"
        )
        lines.append(
            f"  AC#4 structural: an FTS-only rank-1 row outranks vector ranks "
            f">= {verdict.structural_threshold} and the leg fetches "
            f"{verdict.vector_window} — satisfied under the shipped weighting."
        )
        lines.append(
            f"  AC#3 fixture: {report.target_slug} at rank {winner.rescue.rank} "
            f"via {winner.rescue.mechanism} "
            f"(fts_rank={winner.rescue.fts_rank}, "
            f"vector_rank={winner.rescue.vector_rank})."
        )
        lines.append(
            "  clause (b): worst gated cell "
            f"{(verdict.worst_regression[2] if verdict.worst_regression else 0.0):+.3f}"
            f" ({verdict.worst_regression[0] if verdict.worst_regression else 'n/a'}"
            f"/{verdict.worst_regression[1] if verdict.worst_regression else 'n/a'})."
        )
        lines.append(
            f"  clause (c): deviation {winner.strategy.deviation(baseline):.3f}, "
            f"levers {list(winner.strategy.changed_levers(baseline))}, "
            f"fields {list(winner.strategy.changed_fields(baseline))}."
        )
    return "\n".join(lines)
