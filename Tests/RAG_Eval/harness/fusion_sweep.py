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

**TASK-15400 added a second axis to the same machinery**: the keyword leg's
FTS5 MATCH construction (`SearchConfig.fts_match_construction`). It rides
`Strategy` like the fusion knobs — one field, written by `apply`, restored in
`finally` — but it is measured by a different rule, and three things about
that are load-bearing.

*The census is leg-level.* The fusion arc's question was about fusion, so
its numbers came out of `run_eval`. This arc's question is about what the
KEYWORD LEG finds at all, and the semantic leg masks that for every source
type except prompts. `keyword_leg_census` therefore calls the engine's
`_keyword_search` directly, once per golden query, and counts the queries
whose target lands in the leg's own top-k. That is the number the spec's
decision rule maximizes: the control (pre-arc `and`) scores 20 over the 53
non-negative queries, TASK-15400's winner `and_stopword_trim` scores 21, and
the construction shipped since 2026-08-13 (`and_then_prefix`, by owner
ruling over the rule's tie-break — see `SearchConfig`) scores 23.

*The control row self-checks before anything else runs — and it is not a
cache alarm.* Be precise about what that check can and cannot see, because
the failure it is named after is the one this arc most fears. The census
calls `_keyword_search`, which NEVER touches `self.cache` (only `search()`
does — `rag_service.py:1240/:1334`), so a stale cache cannot move the census
in either direction. What the control row actually catches is **census-method
drift**: the counting method, the corpus or the golden set having moved out
from under the control's 20, and — via `_validate_constructions` beside it —
a construction VOCABULARY drift that would silently degrade every non-control
row to the control and report all six censuses equal.

What protects the SCORED passes from a shared cache is different and lives
elsewhere: Task 1 put the construction in the hybrid cache key, and this
loop clears the cache before every pass (count- and order-pinned). And the
discriminator for a reader who suspects staleness anyway is printed on the
matrix's face: prompts are FTS-only by construction, so a run where the
census moves under `or`/`and_or` while the prompt HYBRID cells do not is
staleness, not the semantic leg masking the change.

*The 4110 decision rule does not apply to this axis.* `qualify` /
`select_winner` / `format_matrix` implement the FUSION rule (a weighting
lever moved, the structural threshold, the vector-blind rescue). The
construction sweep renders through `format_construction_matrix`, and its
winner — biggest census subject to the spec's hard constraints — is applied
in writing against that table. `changed_levers` deliberately does not count
the construction, for the same reason it does not count pool widening: it
changes which documents fusion SEES, never how fusion weighs them.

Nothing here is imported by the application.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Optional, Sequence

from loguru import logger

from Tests.RAG_Eval.harness.canonicalize import rows_to_doc_ids, slug_lookup_from
from Tests.RAG_Eval.harness.goldenset import NEGATIVE_CATEGORY, GoldenQuery
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
    "CONSTRUCTION_CONTROL_NAME",
    "CONSTRUCTION_STRATEGIES",
    "CONTROL",
    "CONTROL_NAME",
    "DEFAULT_K",
    "FTS_MATCH_OR_FORM",
    "FTS_MATCH_PREFIX_FORM",
    "HYBRID_MODE",
    "LEVER_PRECEDENCE",
    "NEAR_PROBE_DISTANCE",
    "REGRESSION_METRICS",
    "RESCUE_QUERY_ID",
    "RESCUE_TARGET_SLUG",
    "SHIPPED_CONTROL_CENSUS",
    "LegCensus",
    "NegativeComposition",
    "ProbeCensus",
    "Qualification",
    "RescueVerdict",
    "Strategy",
    "StrategyReport",
    "SweepReport",
    "WARN_BAND",
    "WIDENING_FORMS",
    "combined_strategy",
    "format_construction_matrix",
    "format_matrix",
    "format_probe_table",
    "fts_only_beats_vector_rank",
    "keyword_leg_census",
    "lever_rank",
    "lost_census_queries",
    "near_probe_expression",
    "negative_composition",
    "prefix_probe_expression",
    "qualify",
    "rescued_zero_row_queries",
    "run_construction_sweep",
    "run_full_matrix",
    "run_fusion_sweep",
    "run_near_prefix_probes",
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

#: Config fields the sweep writes, and therefore must put back. The
#: construction joined them with TASK-15400: it is a live `SearchConfig`
#: field, so a sweep that did not restore it would re-point every later
#: search in the process at whichever candidate ran last.
_RESTORED_FIELDS: tuple[str, ...] = (
    "rrf_k",
    "hybrid_pool_multiplier",
    "hybrid_alpha",
    "default_search_mode",
    "fts_match_construction",
)

#: The knobs `Strategy` moves, in report order. `deviation` reads only the
#: three NUMERIC ones — the construction is categorical, so "how far is
#: `or` from `and`" has no answer to put on a scale with rrf_k.
_STRATEGY_FIELDS: tuple[str, ...] = (
    "rrf_k",
    "hybrid_pool_multiplier",
    "hybrid_alpha",
    "fts_match_construction",
)
_NUMERIC_FIELDS: tuple[str, ...] = (
    "rrf_k",
    "hybrid_pool_multiplier",
    "hybrid_alpha",
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
        fts_match_construction: ``config.search.fts_match_construction`` —
            the keyword leg's FTS5 MATCH construction (TASK-15400), one of
            ``and`` / ``and_stopword_trim`` / ``or`` / ``and_then_or``. It
            defaults to the pre-15400 ``and``, which is what keeps every
            pre-15400 strategy tuple meaning exactly what it meant — and,
            deliberately, what `BASE_STRATEGIES` and
            `ALPHA_COMBO_STRATEGIES` still run under: the fusion/rrf_k
            matrix's construction axis is pinned to the PRE-ARC construction
            for comparability with the P2ab weighting arc. Those rows
            therefore no longer measure the shipped retrieval path, and a
            future rrf_k re-tune read off them would be optimizing a
            construction that does not ship.
    """

    name: str
    rrf_k: int
    hybrid_pool_multiplier: int
    hybrid_alpha: float
    fts_match_construction: str = "and"

    def apply(self, search_config: Any) -> None:
        """Write this strategy's knobs onto a live `SearchConfig`."""
        search_config.rrf_k = self.rrf_k
        search_config.hybrid_pool_multiplier = self.hybrid_pool_multiplier
        search_config.hybrid_alpha = self.hybrid_alpha
        search_config.fts_match_construction = self.fts_match_construction

    def changed_fields(self, baseline: "Strategy") -> tuple[str, ...]:
        """Config fields this strategy moves relative to ``baseline``."""
        return tuple(
            field
            for field in _STRATEGY_FIELDS
            if getattr(self, field) != getattr(baseline, field)
        )

    def changed_levers(self, baseline: "Strategy") -> tuple[str, ...]:
        """The **weighting** levers this strategy moves.

        `hybrid_pool_multiplier` is excluded by construction: widening the
        candidate pool changes which documents fusion sees, never how fusion
        weighs them, so it cannot satisfy AC#4's structural guarantee. Its
        exclusion here is the single place that rule is enforced.

        `fts_match_construction` (TASK-15400) is excluded for exactly the
        same reason and must stay excluded: it changes which documents the
        keyword leg FINDS. Counting it as a weighting lever would let a
        construction row satisfy AC#4's first clause — a guarantee about
        fusion arithmetic — by changing something fusion never reads.
        """
        return tuple(
            field
            for field in self.changed_fields(baseline)
            if field not in ("hybrid_pool_multiplier", "fts_match_construction")
        )

    def deviation(self, baseline: "Strategy") -> float:
        """Relative L1 distance from ``baseline`` — the 'smallest deviation' metric.

        Relative rather than absolute so a 40-unit move in `rrf_k` and a
        0.15 move in `hybrid_alpha` are on one scale at all.
        """
        total = 0.0
        for field in _NUMERIC_FIELDS:
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


# ---------------------------------------------------------------------------
# TASK-15400: the MATCH-construction axis
# ---------------------------------------------------------------------------

#: The construction sweep's control row — the arc's BEFORE state, named
#: after the construction rather than "control" so the table reads as what
#: it is. This was the SHIPPED construction when the sweep ran; the default
#: has since moved twice (TASK-15400 to `and_stopword_trim`, 2026-08-12;
#: TASK-15700 to `and_then_prefix`, 2026-08-13), and this row is the pre-arc
#: baseline every other row is read against under BOTH matrices — which is
#: exactly why it is pinned to a construction rather than to "whatever
#: ships". It must NOT be re-pointed at the current default: doing so
#: would delete the baseline the matrix is compared to and silently collapse
#: rows 1 and 2 into one measurement.
CONSTRUCTION_CONTROL_NAME = "and"

#: What the CONTROL construction (`and`, the pre-arc default) scores on the
#: leg-level census: the golden queries whose target enters the KEYWORD
#: LEG's own top-10. Measured in TASK-15020/B2's authoring pass and recorded
#: in the spec (`keyword` 13/16 + `scoped` 7/7 = 20; every other category 0,
#: and the 7 negatives have no target to find), then reproduced exactly by
#: the control row when the sweep ran. **The value stays 20 even though the
#: shipped default moved** — it describes the control row, not the shipped
#: construction (which measures 21: the same 20 plus `pm-vendor-chaser`).
#: The control row must reproduce it or the sweep is measuring something
#: other than the construction — see `_check_control_census`.
SHIPPED_CONTROL_CENSUS = 20

#: `metadata["fts_match"]`'s OR-form value, mirrored from the engine's
#: `rag_service.FTS_MATCH_OR`. Mirrored rather than imported so the harness's
#: ALWAYS-ON tests do not drag the engine module in at import time; the two
#: are pinned equal by `test_the_or_form_stamp_is_the_engines_own_constant`.
#:
#: It names the FORM that matched a row, never its position: under the `or`
#: construction every keyword row carries it as a PRIMARY, and only under
#: `and_then_or` does it mean "the fallback fired". Fallback-ness is
#: therefore a function of (construction, form), which is why the negative
#: composition is recorded per row and read beside the construction column.
FTS_MATCH_OR_FORM = "or"

#: The same, for the PREFIX form TASK-15700's two new rows can run
#: (`rag_service.FTS_MATCH_PREFIX`; pinned equal by
#: `test_the_prefix_form_stamp_is_the_engines_own_constant`).
FTS_MATCH_PREFIX_FORM = "prefix"

#: The forms that WIDEN a query beyond the implicit AND over its own tokens
#: — what the negative composition counts (TASK-15700 extends it from the OR
#: form alone to any of these, which is what "count any non-primary form"
#: means with the vocabulary the engine actually stamps).
#:
#: Read this list beside the CONSTRUCTION column, never alone. The AND form
#: is the primary of every construction that can run it, so it is never here;
#: but a widening form is a FALLBACK's under `and_then_or`/`and_then_prefix`
#: and a PRIMARY's under `or`/`prefix` — same stamp, opposite meaning for the
#: tie-break. Counting the form (and saying so) is what keeps the number
#: honest under both; deriving "non-primary" per row would need the engine's
#: table here and would silently read 0 for the two widening-primary rows,
#: exactly where the noise cost is highest.
WIDENING_FORMS = (FTS_MATCH_OR_FORM, FTS_MATCH_PREFIX_FORM)

#: The pre-registered candidates, at the SHIPPED fusion parameters
#: (`SearchConfig`'s own defaults: rrf_k 5, pool x2, alpha 0.7 — pinned
#: against that class rather than copied from the spec's prose). Holding
#: fusion fixed is what makes the census column attributable to the
#: construction alone; a row that also moved rrf_k would confound the two
#: arcs' levers in one number. Names stay <= 10 characters — the matrix
#: column width is the dataclass's own stated rule.
#:
#: FOUR rows for TASK-15400, plus TWO for TASK-15700's re-run under the
#: form-tiered merge. The first four keep their exact meaning — same names,
#: same constructions, same fusion parameters, control still first — because
#: the re-run is read against the same baseline (`SHIPPED_CONTROL_CENSUS`
#: stays 20). The new rows are the spec's promotion of the 15400 sweep's
#: report-only prefix probe (`prefix`, the best of the two probes at 3
#: rescues) and its composition with the AND primary (`and_pfx`), which is
#: the row that matters if `and_or` STILL loses the vector-blind fixture
#: under the fixed merge.
CONSTRUCTION_STRATEGIES: tuple[Strategy, ...] = (
    Strategy("and", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
             fts_match_construction="and"),
    Strategy("and_trim", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
             fts_match_construction="and_stopword_trim"),
    Strategy("or", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
             fts_match_construction="or"),
    Strategy("and_or", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
             fts_match_construction="and_then_or"),
    Strategy("prefix", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
             fts_match_construction="prefix"),
    Strategy("and_pfx", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
             fts_match_construction="and_then_prefix"),
)

#: The token distance the NEAR probe asks for. FTS5's default is also 10;
#: it is written out because a probe's parameter belongs in the report.
NEAR_PROBE_DISTANCE = 10


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
class LegCensus:
    """What the KEYWORD LEG alone found, over the whole golden set.

    The number TASK-15400's decision rule maximizes. It is deliberately not
    a `run_eval` number: the semantic leg masks the keyword leg's misses for
    every source type except prompts, so a fused metric cannot answer "what
    does the keyword leg find at all".

    Attributes:
        k: The leg's top-k the count was taken inside.
        hits: Queries whose target appears in the leg's top-k.
        scoreable: Queries that HAVE a target (every non-negative query) —
            the census's denominator. A negative has nothing to find, so
            counting it would put a query that cannot be right into a rate
            about how often the leg is right.
        queries: Every query run, negatives included.
        hit_queries: The ids behind `hits`, so a candidate's gains and
            losses can be attributed query by query rather than by delta.
        zero_row_queries: Ids the leg returned NOTHING for, negatives
            included — the 40-of-60 set the arc was raised on, and exactly
            the set the NEAR/prefix probes run over.
        per_category: category -> (hits, scoreable in that category).
    """

    k: int
    hits: int
    scoreable: int
    queries: int
    hit_queries: tuple[str, ...]
    zero_row_queries: tuple[str, ...]
    per_category: Mapping[str, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class NegativeComposition:
    """What the keyword leg put into hybrid's top-k for absent-topic queries.

    The spec's NEGATIVE-COMPOSITION RECORD: the gated negative probes cannot
    move here (the vector leg already fills k and a fused FTS-only row
    cannot outscore the vector rank-1), but the composition can — junk rows
    can take the rescue slots. Recorded, not gated: fewer is better, and it
    feeds the tie-break as a named trade-off rather than a surprise.

    Attributes:
        queries: Negative queries measured.
        fallback_rows: FTS-only rows inside the top-k carrying a WIDENING
            form — the OR form or (TASK-15700) the prefix form,
            `WIDENING_FORMS`. Under `and_then_or`/`and_then_prefix` these ARE
            fallback rows; under `or`/`prefix` they are that construction's
            primaries — the construction column disambiguates, which is why
            this counts the FORM and says so. The name is kept from
            TASK-15400 because the report column and every reader's mental
            model are attached to it; what it counts widened, not what it
            means.
        fts_only_rows: All FTS-only rows inside the top-k, whatever form
            matched them. The denominator `fallback_rows` is read against.
    """

    queries: int
    fallback_rows: int
    fts_only_rows: int


@dataclass(frozen=True, slots=True)
class ProbeCensus:
    """One report-only MATCH variant, run over the zero-row queries.

    The spec's fifth axis: `NEAR` and prefix get one probe each, not a
    matrix row, and are promoted to a full row only if a probe beats the
    best swept candidate's census.

    Attributes:
        name: ``near`` or ``prefix``.
        expression_sample: One rendered expression, so the report shows what
            was actually asked rather than what the code was meant to ask.
        queries: Zero-row queries probed.
        queries_with_rows: How many returned anything at all.
        hits: How many returned their target inside the leg's top-k. A
            RESCUE count, comparable with the matrix's ``resc`` column and
            NEVER with a row's full census (which counts the ~20 queries the
            control already answers and no probe was ever run over).
        negative_queries_with_rows: How many of the probed NEGATIVES started
            returning rows; a variant's noise cost, in the same units the
            negative-composition record uses.
    """

    name: str
    expression_sample: str
    queries: int
    queries_with_rows: int
    hits: int
    negative_queries_with_rows: int


@dataclass(frozen=True, slots=True)
class StrategyReport:
    """One strategy's whole result: hybrid metrics + the rescue verdict.

    The last four fields are TASK-15400's instrumentation and are populated
    only by an instrumented (construction) pass; they stay ``None`` for the
    fusion matrix, which never measured them.
    """

    strategy: Strategy
    hybrid: ModeReport
    rescue: RescueVerdict
    census: Optional[LegCensus] = None
    negatives: Optional[NegativeComposition] = None
    elapsed_s: Optional[float] = None

    @property
    def census_hits(self) -> Optional[int]:
        """Golden queries whose target entered the keyword leg's top-k."""
        return None if self.census is None else self.census.hits

    @property
    def negative_fallback_rows(self) -> Optional[int]:
        """OR-form FTS-only rows inside hybrid top-k across the negatives."""
        return None if self.negatives is None else self.negatives.fallback_rows


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


# ---------------------------------------------------------------------------
# TASK-15400's instrumentation: the leg census, the negative composition,
# and the two report-only probes
# ---------------------------------------------------------------------------


def _leg_rows(results: Iterable[Any]) -> list[Mapping[str, Any]]:
    """Engine `SearchResult`s -> the row shape `rows_to_doc_ids` reads.

    The engine leg returns objects with a `metadata` dict; the seam returns
    dicts with a `provenance` block. Canonicalization is the harness's ONE
    definition of "which fixture is this row" (prefix-stripping and the
    source-type alias table included), so the leg is adapted into that shape
    rather than given a second, subtly different resolver.
    """
    rows: list[Mapping[str, Any]] = []
    for result in results:
        metadata = getattr(result, "metadata", None)
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        rows.append(
            {
                # `_resolve`'s prefix retry is what makes the bare-id and the
                # `media_7` form both land on the same fixture.
                "source_id": metadata.get("source_id") or getattr(result, "id", ""),
                "provenance": metadata,
            }
        )
    return rows


def keyword_leg_census(
    runtime: Any,
    golden: Sequence[GoldenQuery],
    k: int = DEFAULT_K,
    *,
    lookup: Optional[Mapping[tuple[str, str], str]] = None,
) -> LegCensus:
    """Count the golden queries whose target the KEYWORD LEG itself returns.

    One direct `RAGService._keyword_search` call per query — the private
    method on purpose, and for the same reason `_extract_rows` is imported
    from the runner: this is the harness's contract with the leg, and a
    second implementation of it here would drift. Going through the seam
    instead would measure fusion, which is precisely what this number must
    not do.

    The call is the ASYNC keyed path, driven through ``runtime.run``. The
    sync cache twins (`simple_cache.py`'s `get_sync`/`set_sync`) render the
    ``and`` key for every construction, so a census that ever reached them
    would report one number six times — the exact shape of the failure the
    control-row self-check exists to catch.

    Note for a future editor: do NOT turn this into a rank-ORDER assertion
    on a small corpus. bm25's IDF term goes to zero when a term appears in
    every document, so on a handful of documents every OR-form row scores
    -0.0 and their order is an artefact of the fixture, not of the
    construction (TASK-15400, Task 1 review).

    Args:
        runtime: A live `EvalRuntime`.
        golden: The validated golden query set.
        k: The leg's top-k, and the window a target must land inside.
        lookup: Optional pre-built slug lookup; built from the runtime when
            omitted.

    Returns:
        A `LegCensus` over every query, with the hit count taken only over
        the ones that have a target.
    """
    if lookup is None:
        lookup = slug_lookup_from(runtime.slug_to_source)

    hits: list[str] = []
    zero_rows: list[str] = []
    per_category: dict[str, list[int]] = {}
    scoreable = 0
    for query in golden:
        results = runtime.run(
            runtime.service._keyword_search(
                query.query, top_k=k, include_citations=False
            )
        )
        doc_ids = rows_to_doc_ids(_leg_rows(results), lookup)
        if not doc_ids:
            zero_rows.append(query.id)
        if query.category == NEGATIVE_CATEGORY:
            # A negative has no target (the golden-set validator enforces
            # that): it is in `queries` and can be in `zero_row_queries`, but
            # it can never be a hit and is not part of the denominator.
            # Classified by CATEGORY, the harness's own single definition of
            # what a negative is (`runner.NEGATIVE_CATEGORY`).
            continue
        scoreable += 1
        cell = per_category.setdefault(query.category, [0, 0])
        cell[1] += 1
        if any(slug in doc_ids[:k] for slug in query.relevant_slugs):
            hits.append(query.id)
            cell[0] += 1

    return LegCensus(
        k=k,
        hits=len(hits),
        scoreable=scoreable,
        queries=len(golden),
        hit_queries=tuple(hits),
        zero_row_queries=tuple(zero_rows),
        per_category={
            category: (cell[0], cell[1]) for category, cell in sorted(per_category.items())
        },
    )


def negative_composition(
    seam: Any,
    runtime: Any,
    golden: Sequence[GoldenQuery],
    k: int = DEFAULT_K,
    source_types: Sequence[str] = SOURCE_TYPES,
) -> NegativeComposition:
    """Count the keyword leg's WIDENED rows inside hybrid top-k for negatives.

    Re-runs each negative through the seam the scored pass just ran, so the
    search cache serves it (same query, same top_k, same key) and the rows
    are the ones that pass actually produced. `run_eval`'s `NegativeProbe`
    records how MANY documents came back; this records where they came
    from, which is the thing the gate is blind to.

    Args:
        seam: The `LibraryLocalRagSearchService` the pass used.
        runtime: The live `EvalRuntime`.
        golden: The golden set (negatives are selected out of it here).
        k: Result cap.
        source_types: Library scope identifiers to search.

    Returns:
        A `NegativeComposition` over the negative queries.
    """
    scope = tuple(source_types)
    negatives = [
        query for query in golden if query.category == NEGATIVE_CATEGORY
    ]
    fallback_rows = 0
    fts_only_rows = 0
    for query in negatives:
        result = runtime.run(seam.search(query.query, scope, "rag", top_k=k))
        rows, _backend, _error = _extract_rows(result)
        for row in rows[:k]:
            fusion = _fusion_block(row)
            if fusion is None:
                continue
            if fusion.get("fts_rank") is None or fusion.get("vector_rank") is not None:
                continue
            fts_only_rows += 1
            provenance = row.get("provenance")
            form = (
                provenance.get("fts_match") if isinstance(provenance, Mapping) else None
            )
            if form in WIDENING_FORMS:
                fallback_rows += 1
    return NegativeComposition(
        queries=len(negatives),
        fallback_rows=fallback_rows,
        fts_only_rows=fts_only_rows,
    )


def _content_tokens(service: Any, query: str) -> list[str]:
    """The query's content tokens, quoted by the ENGINE's own quoter.

    Both probes are one-variable-at-a-time moves off the `and_trim` row
    (the content-token AND), so they share its token set: what the probe
    changes is the JOIN (proximity) or the term (prefix), never the
    quoting — the injection safety of TASK-3995 is the engine's, borrowed
    here rather than reimplemented.
    """
    return [
        service._quote_fts5_token(token)
        for token in service._fts5_query_tokens(query)
        if not service._is_fts5_stopword(token)
    ]


def near_probe_expression(service: Any, query: str) -> str:
    """FTS5 proximity over the content tokens, in the FUNCTION form.

    FTS5 has no infix `NEAR` (FTS3/4 did): ``"a" NEAR "b"`` parses as an
    implicit AND over three terms, one of them the literal word "near". It
    does not raise — it silently matches nothing — so a probe written that
    way would report "NEAR rescues nothing" for a reason with no connection
    to proximity. Pinned against real SQLite in
    `test_fts5_reads_an_infix_near_as_a_bare_token_not_as_proximity`.

    Returns:
        ``NEAR("a" "b", N)``, or ``""`` when the query has no content tokens
        (``NEAR(, 10)`` is a syntax error; ``""`` is the leg's existing
        "no rows, no database lookup" contract).
    """
    quoted = _content_tokens(service, query)
    if not quoted:
        return ""
    return f"NEAR({' '.join(quoted)}, {NEAR_PROBE_DISTANCE})"


def prefix_probe_expression(service: Any, query: str) -> str:
    """Implicit AND over PREFIX terms — the star goes outside the quotes.

    FTS5 reads ``"tok"*`` as "a phrase whose last token is a prefix"; the
    star inside the quotes would be part of the literal string and match
    nothing. Pinned against real SQLite in
    `test_fts5_prefix_syntax_is_the_star_outside_the_quotes`.
    """
    quoted = _content_tokens(service, query)
    if not quoted:
        return ""
    return " ".join(f"{token}*" for token in quoted)


#: The probes, in report order.
_PROBE_BUILDERS: tuple[tuple[str, Callable[[Any, str], str]], ...] = (
    ("near", near_probe_expression),
    ("prefix", prefix_probe_expression),
)


@contextmanager
def _probe_expression_seam(
    service: Any, builder: Callable[[Any, str], str]
) -> Iterator[None]:
    """Drive every sub-leg with a probe expression, then put the seam back.

    All four sub-legs build their MATCH through `_fts5_match_expressions`,
    so patching that one bound attribute reaches the whole leg — including
    the early-exit check — without a second copy of the sub-leg fan-out.
    The restore is unconditional: a probe that left the seam patched would
    silently change every later row of the sweep.
    """
    attribute = "_fts5_match_expressions"
    had_own = attribute in vars(service)
    original = vars(service).get(attribute)
    setattr(service, attribute, lambda query: (builder(service, query), None))
    try:
        yield
    finally:
        if had_own:
            setattr(service, attribute, original)
        else:
            delattr(service, attribute)


def run_near_prefix_probes(
    runtime: Any,
    golden: Sequence[GoldenQuery],
    query_ids: Collection[str],
    k: int = DEFAULT_K,
    *,
    lookup: Optional[Mapping[tuple[str, str], str]] = None,
) -> tuple[ProbeCensus, ...]:
    """Run the NEAR and prefix variants over the zero-row queries only.

    Report-only by the spec's own scoping: these are one probe each, not
    matrix rows, and are promoted to a full row only if a probe's census
    beats the best swept candidate's. Running them over the queries the
    control already answers would measure nothing — those queries are not
    what either variant is for.

    Args:
        runtime: A live `EvalRuntime`.
        golden: The golden set.
        query_ids: The zero-row query ids (a control census's
            `zero_row_queries`).
        k: The leg's top-k.
        lookup: Optional pre-built slug lookup.

    Returns:
        One `ProbeCensus` per variant, in report order.
    """
    if lookup is None:
        lookup = slug_lookup_from(runtime.slug_to_source)
    wanted = set(query_ids)
    probed = [query for query in golden if query.id in wanted]
    service = runtime.service

    censuses: list[ProbeCensus] = []
    for name, builder in _PROBE_BUILDERS:
        sample = ""
        with_rows = 0
        hits = 0
        negatives_with_rows = 0
        with _probe_expression_seam(service, builder):
            for query in probed:
                sample = sample or builder(service, query.query)
                results = runtime.run(
                    service._keyword_search(query.query, top_k=k, include_citations=False)
                )
                doc_ids = rows_to_doc_ids(_leg_rows(results), lookup)
                if doc_ids:
                    with_rows += 1
                    if query.category == NEGATIVE_CATEGORY:
                        negatives_with_rows += 1
                if any(slug in doc_ids[:k] for slug in query.relevant_slugs):
                    hits += 1
        censuses.append(
            ProbeCensus(
                name=name,
                expression_sample=sample,
                queries=len(probed),
                queries_with_rows=with_rows,
                hits=hits,
                negative_queries_with_rows=negatives_with_rows,
            )
        )
    return tuple(censuses)


def _validate_constructions(strategies: Sequence[Strategy]) -> None:
    """Refuse a matrix naming a construction the ENGINE does not know.

    The engine resolves an unrecognized `fts_match_construction` to the
    shipped ``and`` with ONE warning per service instance
    (`_resolved_fts_match_construction`) — a deliberate fail-safe for
    production that is a silent flattener for a sweep: every non-control row
    would measure the control, all six censuses would read 20, the control's
    own self-check would PASS, and the table would report "the construction makes
    no difference". That is the 4110 failure through a different door, so the
    vocabulary is checked against the engine's own tuple rather than trusted
    as six string literals.

    Args:
        strategies: The matrix about to run.

    Raises:
        ValueError: A strategy names a construction outside
            `rag_service.FTS_MATCH_CONSTRUCTIONS`.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        FTS_MATCH_CONSTRUCTIONS,
    )

    unknown = [
        (strategy.name, strategy.fts_match_construction)
        for strategy in strategies
        if strategy.fts_match_construction not in FTS_MATCH_CONSTRUCTIONS
    ]
    if unknown:
        raise ValueError(
            f"these rows name constructions the engine does not know: "
            f"{unknown}. The engine would degrade each of them to "
            f"{FTS_MATCH_CONSTRUCTIONS[0]!r} with one warning and the sweep "
            f"would report the control's numbers under another row's name. "
            f"Valid: {', '.join(FTS_MATCH_CONSTRUCTIONS)}."
        )


def rescued_zero_row_queries(
    entry: StrategyReport, control: LegCensus
) -> tuple[str, ...]:
    """Which of the control's zero-row queries THIS row's leg now answers.

    The only number a probe's `hits` can honestly be compared against. A
    probe runs over the ~40 zero-row queries alone, so its census is a
    RESCUE count; a swept row's census is over all 53 scoreable queries and
    carries the control's own 20 inside it. Comparing those two directly
    sets the promotion bar ~20 too high and would have Task 3 writing
    "neither probe beats the winner" off a mis-scaled comparison.

    Computed from the ids rather than as a delta, so a row that gains three
    and loses one is not reported as having rescued two.

    Args:
        entry: A row of an instrumented sweep.
        control: The control row's census.

    Returns:
        The rescued query ids, in the control's zero-row order; empty when
        the row carries no census.
    """
    if entry.census is None:
        return ()
    found = set(entry.census.hit_queries)
    return tuple(
        query_id for query_id in control.zero_row_queries if query_id in found
    )


def lost_census_queries(
    entry: StrategyReport, control: LegCensus
) -> tuple[str, ...]:
    """Which of the CONTROL's census hits this row's leg no longer answers.

    Control-relative, NOT shipped-relative: the baseline is the pre-arc
    ``and`` control row's census, so this is a row's cost against the
    15400-era baseline the whole table is calibrated to. For a row's cost
    against a specific candidate default (e.g. the shipped
    ``and_stopword_trim``), pass THAT row's census as ``control`` — the
    function is baseline-agnostic; only the matrix's printed column fixes
    the baseline to the control row.

    THE COLUMN THE 15400 MATRIX DID NOT HAVE, and the one a widening row
    has to be read against (TASK-15700 review). Every other number on the
    table is blind to a self-inflicted loss:

    * `census` is NET — three gains and three losses render as "no change".
    * `resc` is gains-ONLY, and says so in its own docstring.
    * `zero` counts legs that returned NOTHING, so a leg that returned ten
      rows without the target is invisible to it.

    The loss is not hypothetical. A widening form is a superset at the MATCH
    level but NOT at the RETURNED-ROW level: each sub-leg's SQL is
    bm25-ordered and LIMITED, so the widened rows compete for that sub-leg's
    own slots before the merge is ever consulted. Measured during review —
    12 prefix-competitor documents plus one exact-match document, query
    "wombat log" at top_k=5 — `and_stopword_trim` finds the exact document
    and `prefix` returns five rows with it ABSENT: self-displacement inside
    ONE sub-leg, which the tiered merge cannot protect against because there
    is only one tier involved.

    Computed from the ids, exactly like `rescued_zero_row_queries` and for
    the same reason: a row that gains three and loses one must not report as
    "+2 and nothing lost".

    Args:
        entry: A row of an instrumented sweep.
        control: The control row's census.

    Returns:
        The lost query ids, in the control's hit order; empty when the row
        carries no census.
    """
    if entry.census is None:
        return ()
    found = set(entry.census.hit_queries)
    return tuple(
        query_id for query_id in control.hit_queries if query_id not in found
    )


def _check_control_census(
    strategy: Strategy, census: LegCensus, expected: int
) -> None:
    """Raise unless the control row reproduces its expected control census.

    A METHOD check, not a cache alarm: `_keyword_search` never reads
    `self.cache`, so nothing about cache state can move this number (see the
    module docstring). It catches the census method, the corpus or the golden
    set drifting away from the figure the whole decision rule is calibrated
    against — before the rest of the matrix spends a gated run's worth of
    time producing numbers measured against a moved baseline.

    Raised from inside the pass loop on the FIRST row, before its scored pass
    runs, which is why the control row is required to be first.
    """
    if census.hits == expected:
        return
    raise ValueError(
        f"the control row {strategy.name!r} "
        f"({strategy.fts_match_construction}) scored a keyword-leg census of "
        f"{census.hits}/{census.scoreable}, not the CONTROL's expected "
        f"{expected} (this is the control row's own number, not the shipped "
        "construction's — the shipped construction scores differently). The "
        "counting method, the corpus or the golden set has moved away from "
        "the number this arc's decision rule is calibrated against — "
        "reconcile the method against the control census before trusting "
        "any row, and do NOT edit the expected number to match. Per category: "
        f"{ {name: f'{hit}/{total}' for name, (hit, total) in census.per_category.items()} }; "
        f"{len(census.zero_row_queries)} of {census.queries} queries returned "
        "no rows at all."
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
    control_name: str = CONTROL_NAME,
    instrument: bool = False,
    expected_control_census: Optional[int] = None,
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
        control_name: Which row the report's `control()` resolves to. The
            construction matrix names its control after the PRE-ARC
            construction (`and`, the arc's BEFORE state) rather than
            "control" — it is deliberately NOT whichever construction
            currently ships (`and_then_prefix` since 2026-08-13).
        instrument: Record TASK-15400's per-row instrumentation (the
            keyword-leg census and the negative composition). Off for the
            fusion matrix, which never measured either.
        expected_control_census: The census the control row must reproduce.
            ``None`` skips the self-check entirely (and is what the fusion
            matrix passes, since it has no census at all).

    Returns:
        A `SweepReport`. It has no control entry unless `strategies` contains
        one — phases 2 and 3 append to the phase-1 report rather than being
        scored on their own.

    Raises:
        ValueError: Empty matrix, ``k < 1``, no golden query with
            `rescue_query_id`, a self-check whose control row is not first,
            or a control row whose census does not reproduce
            `expected_control_census`.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not strategies:
        raise ValueError("refusing to run a sweep over an empty strategy matrix")
    if expected_control_census is not None and strategies[0].name != control_name:
        # The self-check's whole value is that it fires BEFORE the rest of
        # the matrix runs; a control row buried at position three would turn
        # it into a post-mortem.
        raise ValueError(
            f"the control row {control_name!r} must be the FIRST strategy of a "
            f"self-checked sweep, but the matrix starts with "
            f"{strategies[0].name!r}"
        )
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
            started = time.perf_counter()
            strategy.apply(search_config)
            # Belt-and-braces: Task 3 put the three resolved fusion values in
            # the hybrid cache key, so passes can no longer share an entry;
            # clearing anyway keeps a long matrix's memory flat and means a
            # future cache-key regression cannot silently flatten the sweep.
            # TASK-15400 made this load-bearing rather than merely prudent:
            # the construction axis is measured across passes on ONE runtime,
            # and the control row's census is what proves the clearing works
            # (`test_a_warm_cache_cannot_blind_the_control_census`).
            runtime.service.clear_cache()
            logger.info(f"fusion sweep: {strategy.name} ({strategy.describe()})")
            census: Optional[LegCensus] = None
            if instrument:
                census = keyword_leg_census(runtime, golden, k=k, lookup=lookup)
                logger.info(
                    f"  keyword-leg census: {census.hits}/{census.scoreable} "
                    f"({len(census.zero_row_queries)} zero-row queries)"
                )
                if (
                    strategy.name == control_name
                    and expected_control_census is not None
                ):
                    _check_control_census(strategy, census, expected_control_census)
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
            composition = (
                negative_composition(seam, runtime, golden, k, scope)
                if instrument
                else None
            )
            entries.append(
                StrategyReport(
                    strategy,
                    hybrid,
                    verdict,
                    census=census,
                    negatives=composition,
                    elapsed_s=time.perf_counter() - started,
                )
            )
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
        control_name=control_name,
    )


def run_construction_sweep(
    runtime: Any,
    golden: Sequence[GoldenQuery],
    strategies: Sequence[Strategy] = CONSTRUCTION_STRATEGIES,
    k: int = DEFAULT_K,
    *,
    source_types: Sequence[str] = SOURCE_TYPES,
    rescue_query_id: str = RESCUE_QUERY_ID,
    target_slug: str = RESCUE_TARGET_SLUG,
    seam: Any = None,
    expected_control_census: Optional[int] = SHIPPED_CONTROL_CENSUS,
) -> SweepReport:
    """TASK-15400's matrix: the same pass loop, instrumented and self-checked.

    A thin wrapper on purpose. The config restore, the per-pass cache clear
    and the rescue probe are disciplines this arc inherits rather than
    reimplements — a second loop would be a second place for them to rot.

    Args:
        runtime: A live `EvalRuntime`.
        golden: The validated golden query set.
        strategies: The construction matrix; the control row must be first.
        k: Result cap and metric @k.
        source_types: Library scope identifiers to search.
        rescue_query_id: The vector-blind fixture's query (hard constraint
            (a) is read off its verdict).
        target_slug: That fixture's document.
        seam: Optional pre-built seam.
        expected_control_census: The census the control row must reproduce
            (`SHIPPED_CONTROL_CENSUS` — the control's own number, not the
            shipped construction's); ``None`` disables the self-check.

    Returns:
        A `SweepReport` whose entries carry the census and the negative
        composition. Render it with `format_construction_matrix` — the
        fusion rule (`format_matrix`) answers a different arc's question.
    """
    _validate_constructions(strategies)
    return run_fusion_sweep(
        runtime,
        golden,
        strategies,
        k=k,
        source_types=source_types,
        rescue_query_id=rescue_query_id,
        target_slug=target_slug,
        seam=seam,
        control_name=CONSTRUCTION_CONTROL_NAME,
        instrument=True,
        expected_control_census=expected_control_census,
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


def format_construction_matrix(report: SweepReport) -> str:
    """Render TASK-15400's decision table: one row per MATCH construction.

    Deliberately NOT `format_matrix`. That table applies the fusion arc's
    decision rule (`qualify`), whose clauses are about weighting levers and
    RRF arithmetic — every construction row would fail its first clause for
    a reason that says nothing about this arc. The winner here is the
    biggest census subject to the spec's hard constraints, and it is applied
    in writing against these numbers.

    Args:
        report: An instrumented sweep (`run_construction_sweep`).

    Returns:
        The table: census, gated cells, the vector-blind fixture's verdict
        and the negative composition, per construction.
    """
    control = report.control()
    lines: list[str] = []
    lines.append(
        f"MATCH construction sweep @k={report.k} — hybrid mode only, "
        f"{report.num_queries} golden queries ({report.num_scored} scored) over "
        f"{'/'.join(report.source_types)}"
    )
    lines.append(
        f"control = {control.strategy.name} "
        f"({control.strategy.fts_match_construction}; "
        f"{control.strategy.describe()}); rescue fixture = "
        f"{report.rescue_query_id} -> {report.target_slug}"
    )
    denominator = (
        "" if control.census is None
        else f" out of {control.census.scoreable} non-negative queries"
    )
    lines.append(
        "'census' = golden queries whose target enters the keyword leg's own "
        f"top-{report.k}{denominator}, measured leg-level (a direct "
        "`_keyword_search` pass, no fusion). NOT 'reaches fusion': hybrid "
        f"over-fetches top_k x pool = {control.strategy.vector_window(report.k)} "
        "candidates per leg, so a query can be a census miss and still have "
        "its target inside the fused pool. 'resc' = how many of the CONTROL's "
        "zero-row queries this row's leg now answers (the number a probe's "
        "count is comparable with); 'lost' = how many of the CONTROL's census "
        "hits this row's leg NO LONGER answers; 'zero' = queries the leg "
        "returned nothing for, negatives included."
    )
    lines.append(
        "'lost' (like 'resc') is measured against the PRE-ARC CONTROL row, "
        "not against whichever construction ships: for the cost of row X "
        "versus a candidate default Y, diff the two rows' own hit_queries "
        "(call lost_census_queries with Y's census as the baseline) — "
        "reading X's 'lost' cell as its cost versus the shipped row "
        "misstates it whenever the shipped row itself moved."
    )
    lines.append(
        "'lost' is not derivable from the other columns and is the one to "
        "read before crediting a widening row: 'census' is NET (three gains "
        "and three losses render as no change), 'resc' is gains-only, and "
        "'zero' only catches legs that returned NOTHING. A widening form is "
        "a superset at the MATCH level but NOT at the returned-row level — "
        "each sub-leg's query is bm25-ordered and LIMITED, so widened rows "
        "compete for that sub-leg's own slots BEFORE the merge is consulted "
        "(measured: 12 prefix-competitor docs + 1 exact-match doc, 'wombat "
        "log' at top_k=5 — `and_stopword_trim` finds the exact doc, `prefix` "
        "returns 5 rows without it). Hard constraint (a) is therefore NOT "
        "structurally safe for a widening-PRIMARY row (`or`, `prefix`): it "
        "can lose the vector-blind fixture's own keyword row inside one "
        "sub-leg, where tiering has nothing to tier. `and_then_prefix` and "
        "`and_then_or` are safe on that axis by construction — a non-empty "
        "primary is never widened."
    )
    lines.append("")

    header = (
        f"{'row':<10}{'construction':>19}{'census':>8}{'resc':>6}{'lost':>6}"
        f"{'zero':>6}"
        f"{'P@k':>8}{'R@k':>8}{'MRR':>8}{'NDCG':>8}{'docs':>6}"
        f"{'rescue':>8}{'rank':>6}{'mech':>11}"
        f"{'neg-wide':>10}{'neg-fts':>8}{'secs':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for entry in report.entries:
        strategy = entry.strategy
        overall = entry.hybrid.overall
        census = entry.census
        negatives = entry.negatives
        rescues = (
            "-"
            if control.census is None or census is None
            else len(rescued_zero_row_queries(entry, control.census))
        )
        lost = (
            "-"
            if control.census is None or census is None
            else len(lost_census_queries(entry, control.census))
        )
        lines.append(
            f"{strategy.name:<10}{strategy.fts_match_construction:>19}"
            f"{('-' if census is None else census.hits):>8}"
            f"{rescues:>6}"
            f"{lost:>6}"
            f"{('-' if census is None else len(census.zero_row_queries)):>6}"
            f"{_cell(overall.get('precision'))}{_cell(overall.get('recall'))}"
            f"{_cell(overall.get('mrr'))}{_cell(overall.get('ndcg'))}"
            f"{entry.hybrid.mean_docs_at_k:>6.1f}"
            f"{('yes' if entry.rescue.present else 'NO'):>8}"
            f"{(entry.rescue.rank if entry.rescue.rank is not None else '-'):>6}"
            f"{entry.rescue.mechanism:>11}"
            f"{('-' if negatives is None else negatives.fallback_rows):>10}"
            f"{('-' if negatives is None else negatives.fts_only_rows):>8}"
            f"{(0.0 if entry.elapsed_s is None else entry.elapsed_s):>7.1f}"
        )
    lines.append(
        f"'neg-wide' = FTS-only rows carrying a WIDENING form "
        f"({'/'.join(WIDENING_FORMS)}) inside hybrid top-k across the "
        "negatives ('neg-fts' = all FTS-only rows there). RENAMED from "
        "TASK-15400's 'neg-or', and its vocabulary GREW: that column counted "
        "the OR form alone, this one counts any widening form."
    )
    lines.append(
        "READ neg-wide AGAINST THE CONSTRUCTION COLUMN, and do not tie-break "
        "across the two kinds of row. Under `and_then_or`/`and_then_prefix` "
        "these rows are fallbacks — the leg widened only where it found "
        "nothing. Under `or`/`prefix` the widening form IS the primary, so "
        "EVERY keyword row carries the widening stamp and neg-wide == "
        "neg-fts by construction, not by measurement: the column then says "
        "'this leg returned rows' rather than 'this leg added noise', and "
        "the fewer-is-better tie-break cannot rank such a row against a "
        "fallback row at all."
    )
    lines.append(
        "The rename does NOT move the 15400 four rows' numbers: no "
        "construction outside the two prefix-bearing ones can stamp the "
        "prefix form, so those four rows count exactly what they counted "
        "before. Recorded for the tie-break (fewer is better), never gated."
    )
    lines.append(
        "'rescue' is hard constraint (a): the vector-blind fixture must keep "
        "its hybrid rescue. A 'NO' disqualifies the row whatever its census."
    )
    lines.append(
        "STALENESS CHECK (read before attributing any flat row): prompts are "
        "FTS-only by construction — no indexer writes them to the vector "
        "store — so a run where the census moves under `or`/`and_then_or` "
        "while the prompt/* HYBRID cells below do NOT move is a stale-cache "
        "result, not the semantic leg masking the change. The census itself "
        "cannot see cache state (`_keyword_search` never reads the cache); "
        "this pair of columns is what can."
    )

    census_entries = [entry for entry in report.entries if entry.census is not None]
    if census_entries:
        categories = sorted(
            {
                category
                for entry in census_entries
                for category in entry.census.per_category
            }
        )
        lines.append("")
        lines.append(f"keyword-leg census by category (hits/scoreable) @k={report.k}")
        column = f"{'category':<24}" + "".join(
            f"{entry.strategy.name:>11}" for entry in census_entries
        )
        lines.append(column)
        lines.append("-" * len(column))
        for category in categories:
            cells = ""
            for entry in census_entries:
                cell = entry.census.per_category.get(category)
                cells += f"{('-' if cell is None else f'{cell[0]}/{cell[1]}'):>11}"
            lines.append(f"{category:<24}{cells}")

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
        "DECISION: not computed here. TASK-15400's rule is 'the biggest "
        "census subject to the hard constraints (vector-blind rescue kept; "
        f"no gated cell down more than {WARN_BAND:.3f}; per-token quoting "
        "intact), ties broken by fewest extra FTS queries then smallest code "
        "delta' — applied in writing against this table, with the negative "
        "composition named as the trade-off it is."
    )
    return "\n".join(lines)


def format_probe_table(
    probes: Sequence[ProbeCensus], *, rescues_to_beat: Optional[int] = None
) -> str:
    """Render the NEAR/prefix probe results.

    Args:
        probes: The probe censuses, in report order.
        rescues_to_beat: The best swept candidate's RESCUE count — how many
            of the control's zero-row queries it answers
            (`rescued_zero_row_queries`), never its full census. A probe only
            ever runs over the zero-row queries, so its count is a rescue
            count; printing a full census as the bar would set it ~20 too
            high and make every probe look hopeless by arithmetic.

    Returns:
        The probe table, labelled report-only.
    """
    lines = [
        "NEAR / prefix probes (report-only) — run over the zero-row queries "
        "ONLY, as one-variable moves off the content-token AND: 'near' adds "
        f"proximity (NEAR(..., {NEAR_PROBE_DISTANCE})), 'prefix' adds prefix "
        "matching. 'rescues' counts probed queries whose target reached the "
        "leg's top-k — the same units as the matrix's 'resc' column, NOT a "
        "row's full census.",
        "NEAR's ceiling is a theorem, not a measurement: proximity only "
        "NARROWS, so NEAR over the content tokens matches a subset of the "
        "content-token AND (`and_trim`) — it can never rescue more than "
        "`and_trim` does, which the P2ab attribution measured at 1 of the 40. "
        "A nonzero NEAR row is therefore a subset finding, and a zero row is "
        "the expected one; neither is evidence about proximity's value in "
        "general.",
    ]
    header = (
        f"{'probe':<8}{'queries':>9}{'with rows':>11}{'rescues':>9}"
        f"{'negatives w/ rows':>20}  example expression"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for probe in probes:
        lines.append(
            f"{probe.name:<8}{probe.queries:>9}{probe.queries_with_rows:>11}"
            f"{probe.hits:>9}{probe.negative_queries_with_rows:>20}  "
            f"{probe.expression_sample or '(no content tokens)'}"
        )
    if rescues_to_beat is not None:
        lines.append(
            f"the number to beat is {rescues_to_beat} — the best swept "
            "candidate's RESCUE count over these same zero-row queries (its "
            "full census also carries the control's own hits, which no probe "
            "was ever run over). A probe at or below it stays a probe."
        )
    return "\n".join(lines)
