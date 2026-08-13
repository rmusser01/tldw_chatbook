# Tests/RAG_Eval/test_fusion_decision_rule.py
"""Always-on tests for the fusion sweep's PURE parts.

Everything here runs with no env var, no model and no corpus: the strategy
dataclass, the RRF arithmetic that decides AC#4's structural sense, the
qualification check that implements the spec's decision rule, the winner
tie-break, the derived combination strategy, and the matrix renderer. The
measurement itself lives in `test_fusion_sweep.py` behind the harness gate.

The point of splitting them is that the decision rule must be *provable*
without a five-minute run: a rule nobody can red on demand is a rule that
quietly drifts to fit whatever the numbers turned out to be. Every clause of
the spec's rule gets a hand-built `SweepReport` that isolates it — including
the clause that pool widening alone can never be the winner, which is the
one an over-eager reading of "it rescued the fixture!" would break first.
"""
from __future__ import annotations

import asyncio
import inspect
import sqlite3
from types import SimpleNamespace

import pytest

from Tests.RAG_Eval.harness import fusion_sweep
from Tests.RAG_Eval.harness.fusion_sweep import (
    ALPHA_COMBO_STRATEGIES,
    BASE_STRATEGIES,
    CONSTRUCTION_CONTROL_NAME,
    CONSTRUCTION_STRATEGIES,
    CONTROL,
    CONTROL_NAME,
    FTS_MATCH_OR_FORM,
    FTS_MATCH_PREFIX_FORM,
    LEVER_PRECEDENCE,
    NEAR_PROBE_DISTANCE,
    SHIPPED_CONTROL_CENSUS,
    WARN_BAND,
    WIDENING_FORMS,
    LegCensus,
    NegativeComposition,
    Qualification,
    RescueVerdict,
    Strategy,
    StrategyReport,
    SweepReport,
    combined_strategy,
    format_construction_matrix,
    format_probe_table,
    fts_only_beats_vector_rank,
    keyword_leg_census,
    lever_rank,
    near_probe_expression,
    negative_composition,
    prefix_probe_expression,
    qualify,
    run_construction_sweep,
    run_fusion_sweep,
    run_near_prefix_probes,
    select_winner,
    worst_regression,
)
from Tests.RAG_Eval.harness.goldenset import GoldenQuery
from Tests.RAG_Eval.harness.runner import EvalReport, ModeReport

K = 10

_METRIC_DEFAULTS = {
    "precision": 0.50,
    "recall": 0.60,
    "mrr": 0.70,
    "ndcg": 0.65,
    "f1": 0.55,
    "num_queries": 11.0,
}

CATEGORIES = ("keyword", "paraphrase", "vocabulary_mismatch")


def metrics(**overrides: float) -> dict[str, float]:
    """One metric cell dict, with only the values a test cares about moved."""
    return {**_METRIC_DEFAULTS, **overrides}


def mode_report(
    *,
    per_category: dict[str, dict[str, float]] | None = None,
    overall: dict[str, float] | None = None,
    mean_docs_at_k: float = 6.0,
) -> ModeReport:
    """A hybrid `ModeReport` carrying only what the decision rule reads."""
    return ModeReport(
        mode="hybrid",
        k=K,
        queries=(),
        overall=overall or metrics(num_queries=33.0),
        per_category=per_category
        or {category: metrics() for category in CATEGORIES},
        negatives=(),
        latency={"count": 33.0, "mean_ms": 12.0, "p95_ms": 20.0, "max_ms": 30.0,
                 "total_s": 0.4},
        runtime_backends=("rag-hybrid",),
        errors=(),
        mean_docs_at_k=mean_docs_at_k,
    )


def rescue(
    *,
    present: bool = True,
    rank: int | None = 1,
    mechanism: str = "merged",
    fts_rank: int | None = 1,
    vector_rank: int | None = 22,
) -> RescueVerdict:
    return RescueVerdict(
        query_id="kw-plant-maintenance-record",
        target_slug="note-saltmarsh-hide",
        present=present,
        rank=rank if present else None,
        mechanism=mechanism,
        fts_rank=fts_rank,
        vector_rank=vector_rank,
        docs_returned=10,
        run_rank=rank if present else None,
    )


MISSED = RescueVerdict(
    query_id="kw-plant-maintenance-record",
    target_slug="note-saltmarsh-hide",
    present=False,
    rank=None,
    mechanism="absent",
    fts_rank=None,
    vector_rank=None,
    docs_returned=10,
    run_rank=None,
)


def entry(
    strategy: Strategy,
    *,
    verdict: RescueVerdict | None = None,
    per_category: dict[str, dict[str, float]] | None = None,
    overall: dict[str, float] | None = None,
) -> StrategyReport:
    return StrategyReport(
        strategy=strategy,
        hybrid=mode_report(per_category=per_category, overall=overall),
        rescue=verdict if verdict is not None else rescue(),
    )


def sweep(*entries: StrategyReport, k: int = K) -> SweepReport:
    return SweepReport(
        k=k,
        entries=tuple(entries),
        rescue_query_id="kw-plant-maintenance-record",
        target_slug="note-saltmarsh-hide",
        source_types=("media", "notes", "conversations"),
        num_queries=44,
        num_scored=33,
    )


# ---------------------------------------------------------------------------
# The strategy dataclass
# ---------------------------------------------------------------------------


def test_strategy_carries_the_three_config_knobs_and_applies_them():
    class FakeSearchConfig:
        rrf_k = 60
        hybrid_pool_multiplier = 2
        hybrid_alpha = 0.7

    strategy = Strategy("k10+pool3", rrf_k=10, hybrid_pool_multiplier=3, hybrid_alpha=0.7)
    config = FakeSearchConfig()
    strategy.apply(config)

    assert (config.rrf_k, config.hybrid_pool_multiplier, config.hybrid_alpha) == (
        10,
        3,
        0.7,
    )


def test_changed_fields_and_levers_separate_weighting_from_pool_widening():
    pool_only = Strategy("pool5", rrf_k=60, hybrid_pool_multiplier=5, hybrid_alpha=0.7)
    k_only = Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7)
    both = Strategy("k10+pool3", rrf_k=10, hybrid_pool_multiplier=3, hybrid_alpha=0.7)

    assert pool_only.changed_fields(CONTROL) == ("hybrid_pool_multiplier",)
    assert pool_only.changed_levers(CONTROL) == (), (
        "widening the candidate pool is not a weighting lever — treating it as "
        "one is exactly how pool widening would sneak in as a sole winner"
    )
    assert k_only.changed_levers(CONTROL) == ("rrf_k",)
    assert both.changed_fields(CONTROL) == ("rrf_k", "hybrid_pool_multiplier")
    assert both.changed_levers(CONTROL) == ("rrf_k",)


def test_deviation_is_relative_so_the_smallest_move_wins():
    k20 = Strategy("k20", rrf_k=20, hybrid_pool_multiplier=2, hybrid_alpha=0.7)
    k10 = Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7)
    k5 = Strategy("k5", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7)

    assert CONTROL.deviation(CONTROL) == 0.0
    assert k20.deviation(CONTROL) < k10.deviation(CONTROL) < k5.deviation(CONTROL)


def test_the_vector_window_is_the_legs_requested_candidate_budget():
    assert CONTROL.vector_window(10) == 20
    assert Strategy("p5", 60, 5, 0.7).vector_window(10) == 50


def test_the_shipped_base_matrix_is_the_one_the_spec_names():
    names = tuple(s.name for s in BASE_STRATEGIES)
    assert names[0] == CONTROL_NAME
    assert CONTROL in BASE_STRATEGIES
    assert {s.rrf_k for s in BASE_STRATEGIES if s.changed_levers(CONTROL)} == {5, 10, 20}
    assert {
        s.hybrid_pool_multiplier
        for s in BASE_STRATEGIES
        if s.changed_fields(CONTROL) == ("hybrid_pool_multiplier",)
    } == {3, 5}
    assert all(s.hybrid_alpha == CONTROL.hybrid_alpha for s in BASE_STRATEGIES), (
        "alpha is the last-resort lever; no base strategy may move it"
    )
    assert all(
        s.hybrid_alpha != CONTROL.hybrid_alpha for s in ALPHA_COMBO_STRATEGIES
    )


# ---------------------------------------------------------------------------
# AC#4's structural sense — the RRF arithmetic, pinned to the spec's numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,rrf_k,expected",
    [
        # The spec's diagnosis, verbatim: at the shipped defaults an FTS-only
        # rank-1 row only outranks vector rows past ~82.
        (0.7, 60, 83),
        # Exactly-integral boundaries: 0.3/21 == 0.7/49 and 0.3/6 == 0.7/14,
        # and a tie goes to the keyword row (fusion sorts fts_rank ahead of an
        # absent one). These two are also where naive float rounding is wrong.
        (0.7, 20, 29),
        (0.7, 5, 9),
        (0.7, 10, 16),
        # "Retuning alpha alone needs alpha < 0.567 before FTS rank 1 beats
        # vector rank 20" — so the threshold crosses 20 between these two.
        (0.56, 60, 18),
        (0.58, 60, 25),
    ],
)
def test_fts_only_beats_vector_rank_matches_the_specs_arithmetic(alpha, rrf_k, expected):
    assert fts_only_beats_vector_rank(alpha, rrf_k) == expected


def test_the_alpha_threshold_the_spec_named_is_where_vector_rank_20_flips():
    """0.567 is the spec's stated crossover; pin it from both sides."""
    assert fts_only_beats_vector_rank(0.567, 60) == 20
    assert fts_only_beats_vector_rank(0.57, 60) > 20


def test_an_all_vector_blend_never_lets_an_fts_only_row_win():
    assert fts_only_beats_vector_rank(1.0, 60) is None


def test_an_all_keyword_blend_wins_from_the_first_rank():
    assert fts_only_beats_vector_rank(0.0, 60) == 1


# ---------------------------------------------------------------------------
# Regression detection (clause b)
# ---------------------------------------------------------------------------


def test_worst_regression_finds_the_most_negative_recall_mrr_or_ndcg_cell():
    control = mode_report()
    candidate = mode_report(
        per_category={
            "keyword": metrics(recall=0.55),  # -0.05
            "paraphrase": metrics(ndcg=0.60),  # -0.05
            "vocabulary_mismatch": metrics(mrr=0.61),  # -0.09  <- worst
        }
    )
    category, metric, delta = worst_regression(candidate, control)
    assert (category, metric) == ("vocabulary_mismatch", "mrr")
    assert delta == pytest.approx(-0.09, abs=1e-9)


def test_worst_regression_ignores_precision_moves():
    control = mode_report()
    candidate = mode_report(
        per_category={c: metrics(precision=0.10) for c in CATEGORIES}
    )
    category, metric, delta = worst_regression(candidate, control)
    assert metric != "precision"
    assert delta == pytest.approx(0.0)


def test_a_category_that_vanished_counts_as_a_full_loss():
    control = mode_report()
    candidate = mode_report(
        per_category={c: metrics() for c in CATEGORIES if c != "keyword"}
    )
    category, metric, delta = worst_regression(candidate, control)
    assert category == "keyword"
    assert delta < -WARN_BAND


# ---------------------------------------------------------------------------
# The decision rule itself
# ---------------------------------------------------------------------------


def test_the_control_never_qualifies_it_is_the_thing_being_beaten():
    control = entry(CONTROL, verdict=MISSED)
    result = qualify(control, control, k=K)
    assert isinstance(result, Qualification)
    assert not result.qualifies
    assert not result.weighting_changed
    assert not result.structural_ok


def test_pool_widening_alone_never_qualifies_even_when_it_rescues_the_fixture():
    """The clause the spec added on self-review.

    The pool-only strategy below is built to be maximally tempting: it
    rescues the fixture at rank 1 by merging it, it regresses nothing, and
    its window is wide enough that the raw RRF arithmetic (threshold 83)
    falls inside it. It still must not qualify, because AC#4's structural
    guarantee has to hold under the SHIPPED weighting, and widening a pool
    changes no weight.
    """
    control = entry(CONTROL, verdict=MISSED)
    pool_only = entry(
        Strategy("pool10", rrf_k=60, hybrid_pool_multiplier=10, hybrid_alpha=0.7),
        verdict=rescue(rank=1, mechanism="merged"),
    )
    result = qualify(pool_only, control, k=K)

    assert pool_only.strategy.vector_window(K) >= fts_only_beats_vector_rank(0.7, 60)
    assert result.rescued
    assert result.regression_ok
    assert not result.weighting_changed
    assert not result.qualifies
    assert any("weighting" in reason for reason in result.reasons)


def test_a_weighting_change_whose_threshold_misses_the_window_does_not_qualify():
    control = entry(CONTROL, verdict=MISSED)
    k20 = entry(
        Strategy("k20", rrf_k=20, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
        verdict=rescue(rank=4),
    )
    result = qualify(k20, control, k=K)

    assert result.weighting_changed
    assert result.structural_threshold == 29
    assert result.vector_window == 20
    assert not result.structural_ok
    assert not result.qualifies


def test_a_weighting_change_that_rescues_and_regresses_nothing_qualifies():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(
        Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
        verdict=rescue(rank=3, mechanism="fts-only", vector_rank=None),
    )
    result = qualify(k10, control, k=K)

    assert result.qualifies
    assert result.structural_ok and result.rescued and result.regression_ok
    assert result.rescue_mechanism == "fts-only"
    assert result.reasons == ()


def test_a_strategy_that_never_rescues_the_fixture_does_not_qualify():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(
        Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
        verdict=MISSED,
    )
    result = qualify(k10, control, k=K)

    assert result.structural_ok
    assert not result.rescued
    assert not result.qualifies


def test_a_cell_down_more_than_the_warn_band_disqualifies():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(
        Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
        per_category={
            "keyword": metrics(),
            "paraphrase": metrics(ndcg=0.65 - 0.03),
            "vocabulary_mismatch": metrics(),
        },
    )
    result = qualify(k10, control, k=K)

    assert not result.regression_ok
    assert not result.qualifies
    assert result.worst_regression[0] == "paraphrase"


def test_a_cell_down_by_exactly_the_warn_band_still_qualifies():
    """'regresses by more than 0.02' — the band itself is inside the gate."""
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(
        Strategy("k10", rrf_k=10, hybrid_pool_multiplier=2, hybrid_alpha=0.7),
        per_category={
            "keyword": metrics(),
            "paraphrase": metrics(ndcg=0.65 - WARN_BAND),
            "vocabulary_mismatch": metrics(),
        },
    )
    result = qualify(k10, control, k=K)

    assert result.regression_ok
    assert result.qualifies


# ---------------------------------------------------------------------------
# Tie-breaks (clause c)
# ---------------------------------------------------------------------------


def test_the_lever_tie_break_order_is_rrf_k_then_quota_then_alpha():
    assert LEVER_PRECEDENCE == ("rrf_k", "quota", "hybrid_alpha")
    assert (
        lever_rank(("rrf_k",))
        < lever_rank(("quota",))
        < lever_rank(("hybrid_alpha",))
    )
    assert lever_rank(("hybrid_alpha", "rrf_k")) == lever_rank(("rrf_k",)), (
        "a combination is ranked by its most-preferred lever"
    )
    assert lever_rank(()) > lever_rank(("hybrid_alpha",))


def test_the_winner_prefers_rrf_k_over_an_alpha_combination():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=rescue(rank=3))
    alpha = entry(Strategy("a0.55", 60, 2, 0.55), verdict=rescue(rank=2))
    report = sweep(control, alpha, k10)

    winner = select_winner(report)
    assert winner is not None and winner.strategy.name == "k10"


def test_the_winner_drops_pool_widening_that_did_not_earn_its_keep():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=rescue(rank=6))
    k10_pool3 = entry(Strategy("k10+pool3", 10, 3, 0.7), verdict=rescue(rank=1))
    report = sweep(control, k10, k10_pool3)

    winner = select_winner(report)
    assert winner is not None and winner.strategy.name == "k10", (
        "when the weighting change alone already qualifies, the widened pool "
        "is a second knob moved for nothing"
    )


def test_pool_widening_wins_a_slot_when_it_is_what_rescues_the_fixture():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=MISSED)
    k10_pool3 = entry(Strategy("k10+pool3", 10, 3, 0.7), verdict=rescue(rank=1))
    report = sweep(control, k10, k10_pool3)

    winner = select_winner(report)
    assert winner is not None and winner.strategy.name == "k10+pool3"


def test_among_equal_qualifiers_the_smallest_deviation_wins():
    control = entry(CONTROL, verdict=MISSED)
    k5 = entry(Strategy("k5", 5, 2, 0.7), verdict=rescue(rank=1))
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=rescue(rank=5))
    report = sweep(control, k5, k10)

    winner = select_winner(report)
    assert winner is not None and winner.strategy.name == "k10", (
        "a better fixture rank is not a licence to deviate further; the rule "
        "prefers the smallest deviation among QUALIFIERS"
    )


def test_no_qualifier_means_no_winner_rather_than_the_least_bad_one():
    control = entry(CONTROL, verdict=MISSED)
    k20 = entry(Strategy("k20", 20, 2, 0.7), verdict=rescue(rank=1))
    pool5 = entry(Strategy("pool5", 60, 5, 0.7), verdict=rescue(rank=1))
    report = sweep(control, k20, pool5)

    assert select_winner(report) is None


# ---------------------------------------------------------------------------
# The derived combination (phase 2 of the matrix)
# ---------------------------------------------------------------------------


def test_the_combination_pairs_the_best_k_variant_with_the_best_pool_variant():
    control = entry(CONTROL, verdict=MISSED)
    k5 = entry(Strategy("k5", 5, 2, 0.7), verdict=MISSED)
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=MISSED)
    k20 = entry(Strategy("k20", 20, 2, 0.7), verdict=MISSED)
    pool3 = entry(Strategy("pool3", 60, 3, 0.7), verdict=rescue(rank=1))
    pool5 = entry(Strategy("pool5", 60, 5, 0.7), verdict=rescue(rank=2))
    report = sweep(control, k5, k10, k20, pool3, pool5)

    combination = combined_strategy(report)
    assert combination is not None
    # k20's threshold (30) falls outside its window (20), so it is not
    # structurally viable; between k5 and k10 the smaller deviation wins.
    assert (combination.rrf_k, combination.hybrid_pool_multiplier) == (10, 3)
    assert combination.hybrid_alpha == CONTROL.hybrid_alpha
    assert combination.name not in {e.strategy.name for e in report.entries}


def test_no_combination_when_a_family_is_missing():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=MISSED)
    assert combined_strategy(sweep(control, k10)) is None


# ---------------------------------------------------------------------------
# The matrix renderer
# ---------------------------------------------------------------------------


def test_format_matrix_prints_every_strategy_the_control_and_the_verdict():
    control = entry(CONTROL, verdict=MISSED)
    k10 = entry(Strategy("k10", 10, 2, 0.7), verdict=rescue(rank=4, mechanism="fts-only"))
    pool3 = entry(Strategy("pool3", 60, 3, 0.7), verdict=rescue(rank=1))
    report = sweep(control, k10, pool3)

    rendered = report.format_matrix()

    for name in ("control", "k10", "pool3"):
        assert name in rendered
    assert "kw-plant-maintenance-record" in rendered
    assert "note-saltmarsh-hide" in rendered
    assert "fts-only" in rendered
    # Every category the run scored has to appear, or a regression could hide
    # in a cell the table never printed.
    for category in CATEGORIES:
        assert category in rendered
    assert "WINNER" in rendered


def test_format_matrix_says_blocked_when_nothing_qualifies():
    control = entry(CONTROL, verdict=MISSED)
    pool5 = entry(Strategy("pool5", 60, 5, 0.7), verdict=rescue(rank=1))
    rendered = sweep(control, pool5).format_matrix()

    assert "BLOCKED" in rendered
    assert "WINNER" not in rendered


def test_a_report_without_a_control_refuses_to_be_scored():
    k10 = entry(Strategy("k10", 10, 2, 0.7))
    with pytest.raises(ValueError, match="control"):
        select_winner(sweep(k10))


# ---------------------------------------------------------------------------
# The runner's config discipline — driven with fakes, no model, no corpus
# ---------------------------------------------------------------------------


class FakeSearchConfig:
    def __init__(self) -> None:
        self.rrf_k = 60
        self.hybrid_pool_multiplier = 2
        self.hybrid_alpha = 0.7
        self.default_search_mode = "semantic"
        # Present because the real `SearchConfig` has carried it since
        # TASK-15400's seam landed, and the sweep now saves/restores it.
        self.fts_match_construction = "and"

    def snapshot(self) -> tuple:
        return (
            self.rrf_k,
            self.hybrid_pool_multiplier,
            self.hybrid_alpha,
            self.default_search_mode,
        )


class FakeService:
    def __init__(self) -> None:
        self.config = SimpleNamespace(search=FakeSearchConfig())
        self.cache_clears = 0

    def clear_cache(self) -> None:
        self.cache_clears += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.service = FakeService()
        self.slug_to_source = {"note-saltmarsh-hide": ("note", "1")}
        self.app = object()

    def run(self, awaitable):
        # The fake seam returns a plain mapping, not a coroutine.
        return awaitable


class FakeSeam:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    def search(self, query, source_types, profile, top_k=10):
        self.calls.append((query, tuple(source_types), profile, top_k))
        return {"results": list(self.rows), "runtime_backend": "rag-hybrid"}


TARGET_ROW = {
    "source_id": "1",
    "provenance": {
        "source_type": "note",
        "hybrid_fusion": {"fts_rank": 1, "vector_rank": 22},
    },
}

GOLDEN = (
    GoldenQuery(
        id="kw-plant-maintenance-record",
        query="plant maintenance record",
        category="keyword",
        relevant_slugs=("note-saltmarsh-hide",),
    ),
)


def _fake_run_eval(seen: list[tuple]):
    def run_eval(runtime, golden, k=10, *, modes=(), source_types=()):
        seen.append(runtime.service.config.search.snapshot())
        return EvalReport(
            k=k,
            modes={"hybrid": mode_report()},
            source_types=tuple(source_types),
            num_queries=len(golden),
            num_scored=len(golden),
            num_negative=0,
        )

    return run_eval


def test_the_sweep_counts_scored_queries_the_way_run_eval_does(monkeypatch):
    """The sweep's header must label the row `run_eval` actually averaged.

    These were two implementations of one rule, agreeing only by coincidence:
    the sweep subtracted negatives, `run_eval` subtracts negatives AND scoped.
    The moment a scoped fixture existed the header would have over-counted —
    silently, because nothing compared the two numbers. They now share one
    function, and this is the test that reds if a local copy comes back.
    """
    from Tests.RAG_Eval.harness.runner import count_scored

    golden = (
        *GOLDEN,  # one keyword query — the only averaged one here
        GoldenQuery(
            id="neg-nothing", query="nothing", category="negative", relevant_slugs=()
        ),
        GoldenQuery(
            id="sc-scoped",
            query="scoped",
            category="scoped",
            relevant_slugs=("note-saltmarsh-hide",),
            scope_slugs=("note-saltmarsh-hide",),
        ),
    )
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))

    report = run_fusion_sweep(FakeRuntime(), golden, BASE_STRATEGIES, k=K, seam=FakeSeam([TARGET_ROW]))

    # Stated independently of both implementations: three queries, one of
    # which is averaged (negative and scoped are each excluded, for their own
    # reasons).
    assert report.num_queries == 3
    assert report.num_scored == 1
    assert count_scored(golden) == 1


def test_the_sweep_applies_every_strategy_clears_the_cache_and_restores_config(
    monkeypatch,
):
    seen: list[tuple] = []
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval(seen))
    runtime = FakeRuntime()
    seam = FakeSeam([TARGET_ROW])

    report = run_fusion_sweep(runtime, GOLDEN, BASE_STRATEGIES, k=K, seam=seam)

    assert [s.strategy for s in report.entries] == list(BASE_STRATEGIES)
    assert [(k, pool, alpha) for k, pool, alpha, _ in seen] == [
        (s.rrf_k, s.hybrid_pool_multiplier, s.hybrid_alpha) for s in BASE_STRATEGIES
    ], "each pass must run under its own strategy's knobs"
    assert runtime.service.cache_clears == len(BASE_STRATEGIES), (
        "belt-and-braces: the hybrid cache key covers the knobs, but a stale "
        "entry must not be able to blind the sweep"
    )
    assert len(seam.calls) == len(BASE_STRATEGIES)
    assert runtime.service.config.search.snapshot() == (60, 2, 0.7, "semantic"), (
        "the sweep must hand the caller's service back exactly as it found it"
    )


def test_the_rescue_probe_reads_the_fusion_metadata_for_the_mechanism(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    report = run_fusion_sweep(
        FakeRuntime(), GOLDEN, (CONTROL,), k=K, seam=FakeSeam([TARGET_ROW])
    )

    verdict = report.entries[0].rescue
    assert verdict.present and verdict.rank == 1
    assert verdict.mechanism == "merged"
    assert (verdict.fts_rank, verdict.vector_rank) == (1, 22)


def test_a_row_only_the_keyword_leg_found_is_reported_as_fts_only(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    row = {
        "source_id": "1",
        "provenance": {
            "source_type": "note",
            "hybrid_fusion": {"fts_rank": 1, "vector_rank": None},
        },
    }
    report = run_fusion_sweep(
        FakeRuntime(), GOLDEN, (CONTROL,), k=K, seam=FakeSeam([row])
    )
    assert report.entries[0].rescue.mechanism == "fts-only"


def test_a_missing_target_is_reported_absent_not_guessed_at(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    report = run_fusion_sweep(
        FakeRuntime(), GOLDEN, (CONTROL,), k=K, seam=FakeSeam([])
    )
    verdict = report.entries[0].rescue
    assert not verdict.present
    assert verdict.rank is None
    assert verdict.mechanism == "absent"


def test_config_is_restored_even_when_a_pass_explodes(monkeypatch):
    def exploding_run_eval(runtime, golden, k=10, *, modes=(), source_types=()):
        raise RuntimeError("the vector store fell over")

    monkeypatch.setattr(fusion_sweep, "run_eval", exploding_run_eval)
    runtime = FakeRuntime()

    # Deliberately a strategy that moves ALL THREE knobs off the control:
    # crashing on a strategy that happens to equal the shipped defaults would
    # leave the config correct by accident and pass with no restore at all.
    off_default = (Strategy("k5+pool3+a.55", rrf_k=5, hybrid_pool_multiplier=3, hybrid_alpha=0.55),)
    with pytest.raises(RuntimeError, match="fell over"):
        run_fusion_sweep(runtime, GOLDEN, off_default, k=K, seam=FakeSeam([]))

    assert runtime.service.config.search.snapshot() == (60, 2, 0.7, "semantic"), (
        "a crashed sweep must not leave the caller's service on the last "
        "strategy it tried"
    )


def test_the_sweep_refuses_a_rescue_query_the_golden_set_does_not_have(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    with pytest.raises(ValueError, match="no-such-query"):
        run_fusion_sweep(
            FakeRuntime(),
            GOLDEN,
            (CONTROL,),
            k=K,
            seam=FakeSeam([]),
            rescue_query_id="no-such-query",
        )


def test_the_sweep_refuses_an_empty_strategy_matrix(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    with pytest.raises(ValueError):
        run_fusion_sweep(FakeRuntime(), GOLDEN, (), k=K, seam=FakeSeam([]))


# ---------------------------------------------------------------------------
# TASK-15400: the MATCH-construction axis, its instrumentation, and the
# self-check that stops a cache-blinded sweep from reporting "no difference".
#
# Everything here is synthetic — fake service, fake leg, fake seam — except
# the two FTS5 syntax pins, which run real SQLite because the thing they
# check is FTS5's own parse.
# ---------------------------------------------------------------------------


def leg_row(source_type: str, source_id: str, **metadata) -> SimpleNamespace:
    """One engine-leg `SearchResult`-shaped row (what `_keyword_search` returns)."""
    return SimpleNamespace(
        id=f"{source_type}_{source_id}",
        score=1.0,
        document="...",
        metadata={"source_type": source_type, "source_id": source_id, **metadata},
    )


NOTE_ROW = leg_row("note", "1")
OTHER_ROW = leg_row("media", "99")

CENSUS_GOLDEN = (
    GoldenQuery(
        id="kw-hit",
        query="wombat kiln",
        category="keyword",
        relevant_slugs=("note-saltmarsh-hide",),
    ),
    GoldenQuery(
        id="pm-miss",
        query="how do I write a shift log",
        category="paraphrase",
        relevant_slugs=("media-shift",),
    ),
    GoldenQuery(
        id="neg-absent",
        query="quantum wombat futures",
        category="negative",
        relevant_slugs=(),
    ),
)

CENSUS_SLUGS = {"note-saltmarsh-hide": ("note", "1"), "media-shift": ("media", "7")}

#: The leg's answers for `CENSUS_GOLDEN`: one query finds its target, one
#: returns nothing at all, and the negative returns a row that is nobody's
#: target (so it is neither a hit nor a zero-row query).
CENSUS_ROWS = {
    "wombat kiln": [NOTE_ROW],
    "how do I write a shift log": [],
    "quantum wombat futures": [OTHER_ROW],
}


class FakeLegService(FakeService):
    """A service whose keyword leg is a dictionary, plus a cache that lies.

    ``stale_rows`` models a runtime handed to the sweep WARM: until
    `clear_cache` runs, every leg call returns the same stale answer no
    matter which construction was just applied. That is the exact shape of
    the failure the control-row self-check exists to catch (TASK-4110's
    "k doesn't matter" report, one arc earlier).
    """

    def __init__(self, rows_by_query=None, *, stale_rows=None) -> None:
        super().__init__()
        self.config.search.fts_match_construction = "and"
        self.rows_by_query = dict(rows_by_query or {})
        self.stale_rows = stale_rows
        self.async_leg_calls: list[tuple[str, int]] = []
        self.sync_leg_calls = 0
        self.constructions_seen: list[str] = []
        #: Cache clears and leg calls, interleaved in the order they happened
        #: — the order pin's evidence.
        self.events: list[str] = []

    async def _keyword_search(self, query, top_k, include_citations=True, **kwargs):
        self.async_leg_calls.append((query, top_k))
        self.events.append(f"leg:{query}")
        self.constructions_seen.append(self.config.search.fts_match_construction)
        rows = (
            self.stale_rows
            if self.stale_rows is not None
            else self.rows_by_query.get(query, [])
        )
        return list(rows)[:top_k]

    def keyword_search_sync(self, *args, **kwargs):
        """The SYNC twin (`simple_cache.py:483`/`:711` render the "and" key for
        every construction). Nothing in the sweep may reach it."""
        self.sync_leg_calls += 1
        return []

    def clear_cache(self) -> None:
        super().clear_cache()
        self.events.append("clear")
        self.stale_rows = None


class FakeLegRuntime:
    """A `FakeRuntime` whose `run` actually drives coroutines."""

    def __init__(self, service, slug_to_source=None) -> None:
        self.service = service
        self.slug_to_source = dict(slug_to_source or CENSUS_SLUGS)
        self.app = object()

    def run(self, awaitable):
        if inspect.isawaitable(awaitable):
            return asyncio.run(awaitable)
        return awaitable


# ---------------------------------------------------------------------------
# The axis itself
# ---------------------------------------------------------------------------


def test_the_construction_round_trips_through_apply():
    config = FakeSearchConfig()
    config.fts_match_construction = "and"

    Strategy(
        "and_or", rrf_k=5, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
        fts_match_construction="and_then_or",
    ).apply(config)

    assert config.fts_match_construction == "and_then_or"
    assert (config.rrf_k, config.hybrid_pool_multiplier, config.hybrid_alpha) == (
        5, 2, 0.7,
    )


def test_every_pre_15400_strategy_still_means_exactly_what_it_meant():
    """The axis defaults to the pre-15400 construction (`and`), not to
    whichever construction currently ships (`and_stopword_trim` 2026-08-11 →
    2026-08-13, `and_then_prefix` since), so the fusion matrix that already
    ran keeps measuring what it measured. Written against the ROLE rather
    than the name on purpose: this docstring has now been stale twice."""
    for strategy in (*BASE_STRATEGIES, *ALPHA_COMBO_STRATEGIES):
        assert strategy.fts_match_construction == "and", strategy.name
    # ...and one tuple, spelled out: the control is still the 4110 control.
    assert CONTROL == Strategy(
        "control", rrf_k=60, hybrid_pool_multiplier=2, hybrid_alpha=0.7,
        fts_match_construction="and",
    )
    assert Strategy("pool5", 60, 5, 0.7).changed_fields(CONTROL) == (
        "hybrid_pool_multiplier",
    )


def test_the_construction_is_a_changed_field_but_never_a_weighting_lever():
    """It changes which documents the keyword leg FINDS, not how fusion weighs
    them — the same argument that keeps pool widening out of `changed_levers`."""
    construction = Strategy("and_or", 60, 2, 0.7, "and_then_or")

    assert construction.changed_fields(CONTROL) == ("fts_match_construction",)
    assert construction.changed_levers(CONTROL) == ()


def test_the_sweep_restores_the_construction_it_found(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    runtime = FakeRuntime()
    runtime.service.config.search.fts_match_construction = "and"

    run_fusion_sweep(
        runtime,
        GOLDEN,
        (Strategy("or", 5, 2, 0.7, "or"),),
        k=K,
        seam=FakeSeam([TARGET_ROW]),
    )

    assert runtime.service.config.search.fts_match_construction == "and", (
        "a sweep that leaves the caller's service on the last construction it "
        "tried would silently re-point every later search in the process"
    )


def test_the_construction_matrix_is_the_six_the_specs_pre_registered():
    """DISCLOSED (2026-08-13, TASK-15700 Task 2): four rows became SIX.

    The 15400 spec pre-registered four; the 15700 spec adds two for the
    re-run under the form-tiered merge (`prefix`, the promotion of the
    report-only probe that rescued 3, and `and_pfx`, its composition with
    the AND primary). The four original rows are asserted UNCHANGED —
    position, name and construction — because the re-run is read against
    the same control baseline; a row that quietly changed meaning would
    make the two matrices incomparable.
    """
    names = tuple(s.name for s in CONSTRUCTION_STRATEGIES)
    assert names == ("and", "and_trim", "or", "and_or", "prefix", "and_pfx")
    assert names[0] == CONSTRUCTION_CONTROL_NAME, "the control row must be first"
    assert all(len(name) <= 10 for name in names), "the matrix column is 10 wide"
    assert tuple(s.fts_match_construction for s in CONSTRUCTION_STRATEGIES) == (
        "and", "and_stopword_trim", "or", "and_then_or",
        "prefix", "and_then_prefix",
    )
    # The 15400 rows, still meaning exactly what they meant: same order, same
    # constructions, appended to rather than rewritten.
    assert names[:4] == ("and", "and_trim", "or", "and_or")
    assert tuple(
        s.fts_match_construction for s in CONSTRUCTION_STRATEGIES[:4]
    ) == ("and", "and_stopword_trim", "or", "and_then_or")
    # The SHIPPED fusion parameters, held fixed: this arc measures the
    # construction, and a row that also moved rrf_k would confound the two.
    for strategy in CONSTRUCTION_STRATEGIES:
        assert (strategy.rrf_k, strategy.hybrid_pool_multiplier, strategy.hybrid_alpha) == (
            5, 2, 0.7,
        ), strategy.name


def test_the_construction_rows_ride_the_engines_own_shipped_defaults():
    """Read off `SearchConfig`, not copied from the spec's prose.

    DISCLOSED ORACLE FLIP (2026-08-11, TASK-15400 Task 4, sweep row
    `and_trim`): the construction was part of this equality — the control
    row's `"and"` WAS `SearchConfig`'s default. Task 4 shipped the sweep's
    winner, so the engine defaulted to `"and_stopword_trim"` while the
    control row deliberately stays `"and"`: the control column is the arc's
    BEFORE-state, the one `SHIPPED_CONTROL_CENSUS == 20` was measured
    against, and re-pointing it at the new default would delete the
    baseline the matrix is read against (and silently turn rows 1 and 2
    into the same measurement). The fusion parameters still ride the
    engine's defaults — that half is what stops the matrix drifting off the
    configuration it claims to measure.

    **DISCLOSED ORACLE FLIP #2 (2026-08-13, TASK-15700 Task 4): the engine's
    default moved again, `"and_stopword_trim"` → `"and_then_prefix"`.** The
    control row is UNCHANGED at `"and"` for exactly the reason above — it is
    still the pre-arc BEFORE-state both matrices are read against. What
    changed is only the right-hand side of the last assertion, and the
    wording of why it holds: `and_then_prefix` is **not** the re-run's
    computed winner. The pre-registered rule, applied verbatim, tied
    `prefix` and `and_then_prefix` at census 23 (measurement-identical on
    every captured axis) and its tie-break — fewest extra FTS statements,
    240 vs 460 — selected `prefix`. The OWNER RULED `and_then_prefix` ships
    instead, applying the standing stability-over-quick-wins ruling to a
    dimension the tie-break predates (structural immunity to intra-sub-leg
    self-displacement, at 220 extra statements and zero measured retrieval
    difference). This assertion therefore pins an OWNER DECISION, not a
    measured optimum — do not "correct" it back to the rule's output.
    """
    from tldw_chatbook.RAG_Search.simplified.config import SearchConfig

    shipped = SearchConfig()
    control = CONSTRUCTION_STRATEGIES[0]
    assert (
        control.rrf_k,
        control.hybrid_pool_multiplier,
        control.hybrid_alpha,
    ) == (
        shipped.rrf_k,
        shipped.hybrid_pool_multiplier,
        shipped.hybrid_alpha,
    )
    assert control.fts_match_construction == "and"
    assert shipped.fts_match_construction == "and_then_prefix", (
        "the engine's default is no longer the construction TASK-15700's "
        "owner ruling shipped; if it moved again, the construction matrix "
        "needs re-running before this control row means anything"
    )
    # ...and the winner the arc chose IS one of the swept rows, so the
    # shipped default is a construction this matrix actually measured.
    assert shipped.fts_match_construction in {
        strategy.fts_match_construction for strategy in CONSTRUCTION_STRATEGIES
    }


def test_every_construction_row_names_a_construction_the_engine_KNOWS():
    """Vocabulary drift is a SILENT flattener, so it is checked, not trusted.

    The engine resolves an unknown `fts_match_construction` to the shipped
    `and` with one warning per service instance. A renamed value would leave
    every non-control row measuring the control: six censuses of 20, a
    control row whose self-check passes, and a table saying "the
    construction makes no difference" — TASK-4110's failure through a
    different door.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        FTS_MATCH_CONSTRUCTIONS,
    )

    values = tuple(s.fts_match_construction for s in CONSTRUCTION_STRATEGIES)
    assert set(values) <= set(FTS_MATCH_CONSTRUCTIONS)
    assert set(values) == set(FTS_MATCH_CONSTRUCTIONS), (
        "the matrix must sweep every construction the engine can be put in, "
        "or a candidate ships unmeasured"
    )
    # ...and the gate the sweep itself runs accepts the shipped matrix. The
    # membership assertions above are this function's premise; calling it is
    # what proves the two new rows (TASK-15700) get past the guard rather
    # than stopping the sweep on their first invocation.
    fusion_sweep._validate_constructions(CONSTRUCTION_STRATEGIES)


def test_a_construction_the_engine_does_not_know_stops_the_sweep(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    rows = (
        Strategy("and", 5, 2, 0.7, "and"),
        Strategy("typo", 5, 2, 0.7, "and_then_OR"),  # a plausible rename
    )

    with pytest.raises(ValueError, match="does not know"):
        run_construction_sweep(
            FakeLegRuntime(FakeLegService(CENSUS_ROWS)),
            CENSUS_GOLDEN,
            rows,
            k=K,
            seam=FakeSeam([]),
            rescue_query_id="kw-hit",
            target_slug="note-saltmarsh-hide",
            expected_control_census=1,
        )


def test_the_or_form_stamp_is_the_engines_own_constant():
    """The negative-composition counter reads `metadata["fts_match"]`; a rename
    on the engine side must not leave this module counting a dead string."""
    from tldw_chatbook.RAG_Search.simplified.rag_service import FTS_MATCH_OR

    assert FTS_MATCH_OR_FORM == FTS_MATCH_OR


def test_the_prefix_form_stamp_is_the_engines_own_constant():
    """Same guarantee for TASK-15700's second widening form."""
    from tldw_chatbook.RAG_Search.simplified.rag_service import FTS_MATCH_PREFIX

    assert FTS_MATCH_PREFIX_FORM == FTS_MATCH_PREFIX


def test_the_widening_forms_are_every_non_and_form_the_engine_can_stamp():
    """The counter's vocabulary, derived from the ENGINE's own table.

    `WIDENING_FORMS` is what the negative composition counts. A construction
    that adds a THIRD widening form (or renames one) and does not appear
    here would make the noise column read 0 for the row that produces the
    most noise — the counter would be blind exactly where it is load-
    bearing. `FTS_MATCH_FORMS_BY_CONSTRUCTION` is the engine's own statement
    of every form the leg can stamp, so it is the oracle rather than a
    second hand-maintained list.

    The oracle is EVERY non-AND form in that table, primary or fallback —
    not the fallback column alone. A widening form is a fallback's under
    `and_then_or`/`and_then_prefix` and a PRIMARY's under `or`/`prefix`, and
    it is noise in a negative's top-k either way; scoping the oracle to
    fallbacks would red the day someone ships a widening construction that
    has no fallback at all, which is precisely the shape this arc already
    ships twice.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        FTS_MATCH_AND,
        FTS_MATCH_FORMS_BY_CONSTRUCTION,
    )

    stampable = {
        form
        for forms in FTS_MATCH_FORMS_BY_CONSTRUCTION.values()
        for form in forms
        if form is not None
    }
    assert stampable - {FTS_MATCH_AND} == set(WIDENING_FORMS), (
        f"engine non-AND forms {sorted(stampable - {FTS_MATCH_AND})} vs "
        f"counted {sorted(WIDENING_FORMS)}"
    )
    # ...and the AND form is deliberately NOT one of them: it is some
    # construction's primary everywhere it appears, so counting it would
    # report the shipped default as pure noise.
    assert FTS_MATCH_AND not in WIDENING_FORMS
    # Every fallback form is still covered — the property the counter's
    # original scoping was reaching for, now a consequence rather than the
    # definition.
    fallback_forms = {
        fallback
        for _primary, fallback in FTS_MATCH_FORMS_BY_CONSTRUCTION.values()
        if fallback is not None
    }
    assert fallback_forms <= set(WIDENING_FORMS)


# ---------------------------------------------------------------------------
# The census — leg-level, the number the whole decision rule maximizes
# ---------------------------------------------------------------------------


def test_the_census_counts_targets_inside_the_keyword_legs_top_k():
    service = FakeLegService(CENSUS_ROWS)
    census = keyword_leg_census(FakeLegRuntime(service), CENSUS_GOLDEN, k=K)

    assert isinstance(census, LegCensus)
    assert census.hits == 1
    assert census.hit_queries == ("kw-hit",)
    # The denominator is every NON-NEGATIVE query: a negative has no target,
    # so counting it would put a query that cannot be right into a rate that
    # says how often the leg is right.
    assert census.scoreable == 2
    assert census.queries == 3
    assert census.per_category["keyword"] == (1, 1)
    assert census.per_category["paraphrase"] == (0, 1)


def test_the_census_records_zero_row_queries_across_the_whole_set():
    """The 40-of-60 number the arc was raised on counts negatives too — they
    are the queries the probes are then run over."""
    service = FakeLegService({**CENSUS_ROWS, "quantum wombat futures": []})
    census = keyword_leg_census(FakeLegRuntime(service), CENSUS_GOLDEN, k=K)

    assert census.zero_row_queries == ("pm-miss", "neg-absent")
    assert census.hits == 1


def test_scoped_queries_are_part_of_the_census_population():
    """The population decision, pinned — it was not, and a mutation excluding
    scoped queries left every other test green.

    Scoped queries count because they HIT today (7/7 of the control's 20 are
    scoped), so dropping them would silently move the number the whole
    decision rule is calibrated against. The census asks them UNSCOPED — a
    leg-level question about the whole corpus — which is also how the
    control's 20 was measured.
    """
    scoped = GoldenQuery(
        id="sc-scoped",
        query="saltmarsh hide",
        category="scoped",
        relevant_slugs=("note-saltmarsh-hide",),
        scope_slugs=("note-saltmarsh-hide",),
    )
    golden = (CENSUS_GOLDEN[0], scoped)
    service = FakeLegService({**CENSUS_ROWS, "saltmarsh hide": [NOTE_ROW]})

    census = keyword_leg_census(FakeLegRuntime(service), golden, k=K)

    assert census.scoreable == 2
    assert census.hits == 2 and "sc-scoped" in census.hit_queries
    assert census.per_category["scoped"] == (1, 1)


def test_a_row_no_fixture_claims_is_not_a_census_hit():
    service = FakeLegService({**CENSUS_ROWS, "how do I write a shift log": [OTHER_ROW]})
    census = keyword_leg_census(FakeLegRuntime(service), CENSUS_GOLDEN, k=K)

    assert census.hits == 1, "an unclaimed row occupies a slot; it is not a hit"
    assert census.zero_row_queries == ()


def test_the_census_drives_the_async_keyed_leg_and_never_a_sync_twin():
    """The handover from Task 1: `simple_cache.py`'s SYNC twins render the
    "and" key for EVERY construction, so a census that ever reached them
    would report the same number six times."""
    service = FakeLegService(CENSUS_ROWS)
    keyword_leg_census(FakeLegRuntime(service), CENSUS_GOLDEN, k=K)

    assert [query for query, _ in service.async_leg_calls] == [
        query.query for query in CENSUS_GOLDEN
    ]
    assert all(top_k == K for _, top_k in service.async_leg_calls)
    assert service.sync_leg_calls == 0


# ---------------------------------------------------------------------------
# The negative-composition record
# ---------------------------------------------------------------------------


def hybrid_row(source_id: str, *, fts_rank, vector_rank, fts_match="and") -> dict:
    provenance = {
        "source_type": "note",
        "hybrid_fusion": {"fts_rank": fts_rank, "vector_rank": vector_rank},
    }
    if fts_match is not None:
        provenance["fts_match"] = fts_match
    return {"source_id": source_id, "provenance": provenance}


NEGATIVE_GOLDEN = (
    GoldenQuery("neg-one", "quantum wombat futures", "negative", ()),
)


def test_only_fts_only_rows_in_a_widening_form_count_as_fallback_rows():
    """DISCLOSED (2026-08-13, TASK-15700 Task 2): the counted vocabulary grew.

    It counted the OR form alone; it now counts any WIDENING form, which the
    prefix row makes a live distinction rather than a hypothetical one. The
    other two clauses are unchanged: an AND-form row is not noise, and a row
    the vector leg also returned is not FTS-only.
    """
    rows = [
        hybrid_row("1", fts_rank=1, vector_rank=None, fts_match="or"),      # counted
        hybrid_row("2", fts_rank=2, vector_rank=None, fts_match="and"),     # AND form
        hybrid_row("3", fts_rank=3, vector_rank=4, fts_match="or"),         # merged
        hybrid_row("4", fts_rank=None, vector_rank=1, fts_match=None),      # vector
        hybrid_row("5", fts_rank=4, vector_rank=None, fts_match="prefix"),  # counted
    ]
    composition = negative_composition(
        FakeSeam(rows), FakeRuntime(), NEGATIVE_GOLDEN, K, ("media",)
    )

    assert isinstance(composition, NegativeComposition)
    assert composition.fallback_rows == 2
    assert composition.fts_only_rows == 3, (
        "the denominator the fallback count is read against — how many rows "
        "the keyword leg put into these results at all"
    )
    assert composition.queries == 1


def test_a_prefix_form_row_is_counted_under_both_prefix_constructions():
    """The counter reads the FORM, and the construction column disambiguates.

    Under `and_then_prefix` a prefix row IS a fallback; under `prefix` it is
    that construction's primary. Both are noise in a negative's top-k and
    both are counted — the number would be uninterpretable if the widest
    candidate reported zero, which is exactly what a "count only fallbacks"
    rule would produce for the row with the most to answer for.
    """
    rows = [hybrid_row("1", fts_rank=1, vector_rank=None, fts_match="prefix")]

    composition = negative_composition(
        FakeSeam(rows), FakeRuntime(), NEGATIVE_GOLDEN, K, ("media",)
    )

    assert composition.fallback_rows == 1
    assert composition.fts_only_rows == 1


def test_the_negative_composition_only_looks_at_negative_queries():
    composition = negative_composition(
        FakeSeam([hybrid_row("1", fts_rank=1, vector_rank=None, fts_match="or")]),
        FakeRuntime(),
        CENSUS_GOLDEN,
        K,
        ("media",),
    )
    assert composition.queries == 1, "only `neg-absent` is a negative"
    assert composition.fallback_rows == 1


# ---------------------------------------------------------------------------
# The control-row self-check — the cache-blindness alarm
# ---------------------------------------------------------------------------


def _construction_rows(*constructions: str) -> tuple[Strategy, ...]:
    return tuple(
        Strategy(name, 5, 2, 0.7, name if name != "and_or" else "and_then_or")
        for name in constructions
    )


def _run_instrumented(monkeypatch, service, *, expected, strategies=None, seen=None):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval(seen if seen is not None else []))
    return run_construction_sweep(
        FakeLegRuntime(service),
        CENSUS_GOLDEN,
        strategies if strategies is not None else _construction_rows("and", "or"),
        k=K,
        seam=FakeSeam([hybrid_row("1", fts_rank=1, vector_rank=None, fts_match="or")]),
        rescue_query_id="kw-hit",
        target_slug="note-saltmarsh-hide",
        expected_control_census=expected,
    )


def test_a_control_census_that_matches_lets_the_sweep_proceed(monkeypatch):
    service = FakeLegService(CENSUS_ROWS)
    seen: list[tuple] = []

    report = _run_instrumented(monkeypatch, service, expected=1, seen=seen)

    assert [e.strategy.name for e in report.entries] == ["and", "or"]
    assert report.control().strategy.name == CONSTRUCTION_CONTROL_NAME
    assert report.entries[0].census_hits == 1
    assert report.entries[0].negative_fallback_rows == 1
    assert len(seen) == 2, "both rows ran their scored pass"


def test_a_control_census_mismatch_raises_before_any_other_row_runs(monkeypatch):
    service = FakeLegService(CENSUS_ROWS)
    seen: list[tuple] = []

    with pytest.raises(ValueError, match="census"):
        _run_instrumented(monkeypatch, service, expected=20, seen=seen)

    assert seen == [], (
        "the alarm must fire BEFORE the matrix spends five minutes producing "
        "numbers nobody can trust"
    )
    assert service.constructions_seen == ["and"] * len(CENSUS_GOLDEN), (
        "no other construction may have reached the leg"
    )


def test_every_pass_clears_the_cache_before_it_measures_anything(monkeypatch):
    """THE MUTATION TEST for `clear_cache` in the pass loop — an ORDER pin.

    Be exact about what this does and does not prove. Today's census cannot
    be blinded by a stale cache at all: `_keyword_search` never reads
    `self.cache` (only `search()` does, `rag_service.py:1240/:1334`). What
    protects the SCORED passes is the construction being in the cache key
    (Task 1) plus this clear, and what this test pins is that the clear
    happens FIRST — before the pass measures anything, census or scored
    pass. Drop it from the loop and the recorded order loses its "clear",
    which is the regression an editor would otherwise make silently.

    The stale-serving fake below models a leg whose answers ARE cache-served
    — which today's is not — so the ordering pin keeps its teeth if that
    ever changes.
    """
    service = FakeLegService(CENSUS_ROWS, stale_rows=[OTHER_ROW])

    report = _run_instrumented(monkeypatch, service, expected=1)

    # Two passes, each opening with a clear and then measuring.
    assert service.events[:2] == ["clear", f"leg:{CENSUS_GOLDEN[0].query}"]
    assert service.events.count("clear") == 2
    for position, event in enumerate(service.events):
        if event.startswith("leg:"):
            assert "clear" in service.events[:position], (
                "a pass measured before it cleared: "
                f"{service.events[:position + 1]}"
            )
    assert report.entries[0].census_hits == 1


def test_a_stale_leg_would_be_caught_by_the_control_row(monkeypatch):
    """The counterfactual behind the pin above, stated rather than implied.

    Served stale answers, the control's census is NOT the shipped number —
    so the self-check WOULD catch a leg whose results were cache-served.
    That is a property of this fake, not of today's engine, and the
    docstring on `_check_control_census` says so.
    """
    blinded = keyword_leg_census(
        FakeLegRuntime(FakeLegService(CENSUS_ROWS, stale_rows=[OTHER_ROW])),
        CENSUS_GOLDEN,
        k=K,
    )
    assert blinded.hits == 0

    with pytest.raises(ValueError, match="census"):
        fusion_sweep._check_control_census(CONSTRUCTION_STRATEGIES[0], blinded, 1)


def test_the_self_check_says_what_it_checks_and_what_it_cannot(monkeypatch):
    """The message must not sell a cache alarm it cannot be.

    The overclaim this replaces would have had Task 3 reading a passing
    control row as proof the passes were not cache-flattened. They are
    protected — by the keyed cache and the per-pass clear — but not by this.
    """
    census = LegCensus(
        k=K, hits=3, scoreable=53, queries=60, hit_queries=("a", "b", "c"),
        zero_row_queries=("d",), per_category={"keyword": (3, 16)},
    )
    with pytest.raises(ValueError) as excinfo:
        fusion_sweep._check_control_census(CONSTRUCTION_STRATEGIES[0], census, 20)

    message = str(excinfo.value)
    assert "cache" not in message.lower(), (
        "the census cannot see cache state; naming it here would send the "
        "next reader hunting the wrong failure"
    )
    for expected in ("counting method", "corpus", "golden set", "3/53", "keyword"):
        assert expected in message


def test_the_self_check_needs_the_control_row_first(monkeypatch):
    monkeypatch.setattr(fusion_sweep, "run_eval", _fake_run_eval([]))
    with pytest.raises(ValueError, match="control"):
        run_construction_sweep(
            FakeLegRuntime(FakeLegService(CENSUS_ROWS)),
            CENSUS_GOLDEN,
            _construction_rows("or", "and"),
            k=K,
            seam=FakeSeam([]),
            rescue_query_id="kw-hit",
            target_slug="note-saltmarsh-hide",
            expected_control_census=1,
        )


def test_the_shipped_control_census_is_the_number_task_7_measured():
    assert SHIPPED_CONTROL_CENSUS == 20


# ---------------------------------------------------------------------------
# The NEAR / prefix probes — report-only, and FTS5's real syntax
# ---------------------------------------------------------------------------


def test_the_near_probe_uses_fts5s_function_form():
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    expression = near_probe_expression(RAGService, "the shift log summary")

    assert expression == f'NEAR("shift" "log" "summary", {NEAR_PROBE_DISTANCE})'


def test_the_prefix_probe_puts_the_star_after_the_closing_quote():
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    assert prefix_probe_expression(RAGService, "the shift log") == '"shift"* "log"*'


def test_an_all_stopword_query_probes_to_nothing_rather_than_a_syntax_error():
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    # `NEAR(, 10)` is an FTS5 syntax error; "" is the leg's existing
    # "return no rows without a database lookup" contract.
    assert near_probe_expression(RAGService, "the and of") == ""
    assert prefix_probe_expression(RAGService, "the and of") == ""


def test_fts5_reads_an_infix_near_as_a_bare_token_not_as_proximity():
    """WHY the probe builds `NEAR(a b, N)` and not `"a" NEAR "b"`.

    FTS5 (unlike FTS3/4) has no infix NEAR: `"a" NEAR "b"` parses as an
    implicit AND over three terms, one of which is the literal word "near".
    It does not raise — it silently returns nothing — so a probe written that
    way would have reported "NEAR rescues 0 of 40" for a reason that has
    nothing to do with proximity.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        # Row 1 is what the probe is FOR: the two terms, adjacent, no "near".
        db.execute("INSERT INTO t(body) VALUES ('the shift log summary for supervisors')")
        # Row 2 contains the WORD "near" and the two terms scattered — the
        # only row the infix spelling matches. Without it the claim below
        # would be 0 == 0 and prove nothing.
        db.execute(
            "INSERT INTO t(body) VALUES "
            "('the log is filed near the end of every shift handover')"
        )

        def rows(expression: str) -> list[int]:
            return [
                row[0]
                for row in db.execute(
                    "SELECT rowid FROM t WHERE t MATCH ?", (expression,)
                ).fetchall()
            ]

        # The infix spelling matches row 2 ONLY — the row containing the
        # literal word "near", where the terms are 9 tokens apart — and
        # misses row 1, where they are adjacent. It is an AND over three
        # terms, demonstrably, not proximity in any direction.
        assert rows('"shift" NEAR "log"') == [2]
        assert rows('"shift" "near" "log"') == [2], "...it is this AND, exactly"
        # The function form is proximity: both rows are within 10 tokens.
        assert sorted(rows(near_probe_expression(RAGService, "shift log"))) == [1, 2]
        # ...and it discriminates on distance, which the infix form cannot:
        assert rows('NEAR("shift" "log", 2)') == [1]
    finally:
        db.close()


def test_fts5_prefix_syntax_is_the_star_outside_the_quotes():
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    db = sqlite3.connect(":memory:")
    try:
        db.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        db.execute("INSERT INTO t(body) VALUES ('templates of building rough turns')")

        def hits(expression: str) -> int:
            return len(
                db.execute("SELECT rowid FROM t WHERE t MATCH ?", (expression,)).fetchall()
            )

        # The shipped AND misses the plural; the prefix form is what widens it.
        assert hits('"template" "building"') == 0
        assert hits(prefix_probe_expression(RAGService, "template building")) == 1

        # Injection safety survives the star, and the star is what makes the
        # difference visible: appended to a BARE token FTS5 parses the token
        # as column syntax and raises, which is the failure mode a probe
        # that built its own expressions would have shipped.
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            hits("templ-3*")
        assert hits(prefix_probe_expression(RAGService, "templ-3")) == 0
        assert prefix_probe_expression(RAGService, "templ-3") == '"templ-3"*'
        # An embedded quote is doubled, not escaped out of: the user's `OR`
        # stays a word (and is trimmed as the stopword it is), never an
        # operator that would have made this hostile query match.
        hostile = prefix_probe_expression(RAGService, 'templates" OR "wombat')
        assert hostile == '"templates"""* """wombat"*'
        assert hits(hostile) == 0
    finally:
        db.close()


def test_the_prefix_construction_builds_the_probes_own_expression():
    """THE PROVENANCE PIN (TASK-15700): the row inherits the probe's lead.

    The `prefix` matrix row exists because THIS probe rescued 3 of the 40
    zero-row golden queries in the 15400 sweep — the only variant that
    rescued anything. That lead transfers to the shipped construction only
    if the construction sends the same expression: a different join (an OR
    of prefixes rather than the probe's implicit AND), a different token set
    (stopwords kept), or the star inside the quotes would each be a
    DIFFERENT query, and the 3 rescues would be evidence for something the
    engine does not do.

    Checked over shapes that separate every one of those: multi-token
    (join), a stopword-bearing query (token set), an all-stopword query
    (the empty contract), a hostile token (quoting), and a single token
    (where an OR join and an AND join coincide and would hide a difference).
    """
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    from tldw_chatbook.RAG_Search.simplified.rag_service import (
        FTS_MATCH_CONSTRUCTION_PREFIX,
        RAGService,
    )

    config = RAGConfig()
    config.embedding.model = "mock"
    config.embedding.device = "cpu"
    config.vector_store.type = "memory"
    config.vector_store.persist_directory = None
    config.search.enable_cache = False
    config.search.fts_match_construction = FTS_MATCH_CONSTRUCTION_PREFIX
    service = RAGService(config)

    for query in (
        "the shift log",
        "template building",
        "notes about the vendor",
        "what about the",
        "templ-3",
        'templates" OR "wombat',
        "wombat",
    ):
        primary, fallback = service._fts5_match_expressions(query)
        assert primary == prefix_probe_expression(RAGService, query), query
        assert fallback is None, query


def with_engine_tokenizer(service: FakeLegService) -> FakeLegService:
    """Lend the fake leg the ENGINE's real tokenizer, quoter and stopword test.

    They are `@staticmethod`s, so they can be borrowed without standing a
    service up — and borrowing them rather than faking them is the point: a
    fake quoter would let a probe builder pass this test while producing
    an expression the real engine would never send.
    """
    from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

    service._fts5_query_tokens = RAGService._fts5_query_tokens
    service._quote_fts5_token = RAGService._quote_fts5_token
    service._is_fts5_stopword = RAGService._is_fts5_stopword
    return service


def test_the_probes_run_only_the_named_queries_and_restore_the_builder():
    service = with_engine_tokenizer(FakeLegService(CENSUS_ROWS))
    service._fts5_match_expressions = lambda query: ("original", None)
    original = service._fts5_match_expressions

    probes = run_near_prefix_probes(
        FakeLegRuntime(service), CENSUS_GOLDEN, ("pm-miss",), k=K
    )

    assert [probe.name for probe in probes] == ["near", "prefix"]
    assert all(probe.queries == 1 for probe in probes)
    assert [query for query, _ in service.async_leg_calls] == [
        "how do I write a shift log"
    ] * 2, "only the zero-row queries the caller named may be probed"
    assert service._fts5_match_expressions is original, (
        "a probe that leaves the injection seam patched would silently change "
        "every later row of the sweep"
    )


def test_a_rows_rescues_are_counted_over_the_controls_zero_row_queries():
    """The only number a probe's count is comparable with.

    A swept row's census counts all 53 scoreable queries — including the ~20
    the control already answers, which no probe was ever run over. Comparing
    a probe's ~40-query count against that sets the promotion bar ~20 too
    high, and the write-up would then say "neither probe beats the winner"
    off pure arithmetic.

    Counted from the ids rather than as a census delta, so a row that gains
    three and loses one reports three rescues, not two.
    """
    control_census = LegCensus(
        k=K, hits=2, scoreable=5, queries=6, hit_queries=("a", "b"),
        zero_row_queries=("c", "d", "e", "neg"), per_category={},
    )
    candidate = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[3],
        hybrid=mode_report(),
        rescue=rescue(),
        census=LegCensus(
            k=K, hits=4, scoreable=5, queries=6,
            hit_queries=("a", "c", "d", "e"),  # gained three, LOST "b"
            zero_row_queries=(), per_category={},
        ),
    )

    rescued = fusion_sweep.rescued_zero_row_queries(candidate, control_census)

    assert rescued == ("c", "d", "e")
    assert len(rescued) == 3 != candidate.census_hits - control_census.hits, (
        "a net delta would have said 2; the rescue count is what the probes "
        "are measured against"
    )


def test_a_rows_losses_are_counted_over_the_controls_census_hits():
    """THE COLUMN THE 15400 MATRIX DID NOT HAVE (TASK-15700 review).

    A widening form is a superset at the MATCH level and NOT at the
    RETURNED-ROW level: each sub-leg's SQL is bm25-ordered and LIMITED, so
    the widened rows compete for that sub-leg's own slots before the merge
    is consulted. Measured in review — 12 prefix-competitor documents plus
    one exact-match document, "wombat log" at top_k=5 — `and_stopword_trim`
    finds the exact document and `prefix` returns five rows without it.

    Every other column is blind to that: `census` is NET, `resc` is
    gains-only by its own docstring, and `zero` only counts legs that
    returned nothing at all. This fixture is exactly that blind spot — three
    gains and one loss, a net +3 census — and the loss must still be
    reported as one.
    """
    control_census = LegCensus(
        k=K, hits=2, scoreable=5, queries=6, hit_queries=("a", "b"),
        zero_row_queries=("c", "d", "e", "neg"), per_category={},
    )
    candidate = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[4],
        hybrid=mode_report(),
        rescue=rescue(),
        census=LegCensus(
            k=K, hits=4, scoreable=5, queries=6,
            hit_queries=("a", "c", "d", "e"),  # gained three, LOST "b"
            zero_row_queries=(), per_category={},
        ),
    )

    lost = fusion_sweep.lost_census_queries(candidate, control_census)

    assert lost == ("b",)
    # ...and none of the columns that DO exist would have said so.
    assert candidate.census_hits > control_census.hits, "census is net-positive"
    assert fusion_sweep.rescued_zero_row_queries(candidate, control_census) == (
        "c", "d", "e",
    ), "resc is gains-only"
    assert candidate.census.zero_row_queries == (), "zero sees an empty leg only"


def test_a_row_that_loses_nothing_reports_no_losses():
    """The column must not fire on a row that only gained.

    Otherwise `lost` would read as noise on every row and Task 3 would learn
    to ignore it — the failure mode of a column that is always non-zero.
    """
    control_census = LegCensus(
        k=K, hits=2, scoreable=5, queries=6, hit_queries=("a", "b"),
        zero_row_queries=("c",), per_category={},
    )
    candidate = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[1],
        hybrid=mode_report(),
        rescue=rescue(),
        census=LegCensus(
            k=K, hits=3, scoreable=5, queries=6,
            hit_queries=("a", "b", "c"), zero_row_queries=(), per_category={},
        ),
    )

    assert fusion_sweep.lost_census_queries(candidate, control_census) == ()
    # A row with no census at all cannot claim a clean sheet either way.
    censusless = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[1], hybrid=mode_report(), rescue=rescue(),
    )
    assert fusion_sweep.lost_census_queries(censusless, control_census) == ()


def test_the_matrix_prints_what_a_widening_row_loses():
    """The `lost` cell reaches the TABLE FACE, which is what Task 3 copies.

    Constraint (a) is not structurally safe for a widening-PRIMARY row, so
    the reading has to be column-supported rather than derived from
    `hit_queries` by hand at write-up time.
    """
    control_census = LegCensus(
        k=K, hits=2, scoreable=3, queries=4, hit_queries=("kw-hit", "kw-exact"),
        zero_row_queries=("pm-miss",), per_category={"keyword": (2, 2)},
    )
    control = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[0], hybrid=mode_report(),
        rescue=rescue(), census=control_census,
        negatives=NegativeComposition(queries=1, fallback_rows=0, fts_only_rows=1),
    )
    widening = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[4],  # the `prefix` row
        hybrid=mode_report(), rescue=rescue(),
        census=LegCensus(
            k=K, hits=2, scoreable=3, queries=4,
            # Gained the zero-row query, LOST its own exact-match hit: the
            # self-displacement the review measured, at fixture scale.
            hit_queries=("kw-hit", "pm-miss"), zero_row_queries=(),
            per_category={"keyword": (2, 2)},
        ),
        negatives=NegativeComposition(queries=1, fallback_rows=3, fts_only_rows=3),
    )
    report = SweepReport(
        k=K, entries=(control, widening),
        rescue_query_id="kw-hit", target_slug="note-saltmarsh-hide",
        source_types=("media",), num_queries=4, num_scored=3, control_name="and",
    )

    rendered = format_construction_matrix(report)

    assert "lost" in rendered
    row = next(line for line in rendered.splitlines() if line.startswith("prefix"))
    # census, resc, lost — the net census is UNCHANGED at 2 while one hit was
    # traded for another, which is precisely why the third number is needed.
    assert row.split()[2:5] == ["2", "1", "1"], row
    assert "'census' is NET" in rendered
    assert "bm25-ordered and LIMITED" in rendered
    assert "not structurally safe" in rendered.lower()


def test_the_neg_wide_legend_states_both_comparability_facts():
    """Task 3 copies the table FACE, so the caveats must live on it.

    Two facts a reader cannot recover from the numbers: (1) for a
    widening-PRIMARY row every keyword row carries the widening stamp, so
    `neg-wide == neg-fts` by construction and the fewer-is-better tie-break
    cannot rank such a row against a fallback row; (2) the column was
    renamed from TASK-15400's `neg-or` and its vocabulary grew, while the
    original four rows' numbers are unchanged by that (no other
    construction can stamp `prefix`).
    """
    control_census = LegCensus(
        k=K, hits=1, scoreable=2, queries=3, hit_queries=("kw-hit",),
        zero_row_queries=("pm-miss",), per_category={"keyword": (1, 1)},
    )
    control = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[0], hybrid=mode_report(),
        rescue=rescue(), census=control_census,
        negatives=NegativeComposition(queries=1, fallback_rows=0, fts_only_rows=1),
    )
    report = SweepReport(
        k=K, entries=(control,), rescue_query_id="kw-hit",
        target_slug="note-saltmarsh-hide", source_types=("media",),
        num_queries=3, num_scored=2, control_name="and",
    )

    rendered = format_construction_matrix(report)

    assert "neg-wide" in rendered
    assert "RENAMED from" in rendered and "neg-or" in rendered
    assert "neg-wide == neg-fts by construction" in rendered
    assert "cannot rank such a row against a fallback row" in rendered
    # The mitigation, stated rather than left to be rediscovered.
    assert "does NOT move the 15400 four rows' numbers" in rendered
    # ...and the legend names the widening forms from the constant, so a
    # third form cannot appear in the counter and be absent from the prose.
    for form in WIDENING_FORMS:
        assert form in rendered


def test_the_matrix_prints_rescues_beside_the_census():
    control_census = LegCensus(
        k=K, hits=1, scoreable=2, queries=3, hit_queries=("kw-hit",),
        zero_row_queries=("pm-miss",), per_category={"keyword": (1, 1)},
    )
    control = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[0], hybrid=mode_report(),
        rescue=rescue(), census=control_census,
        negatives=NegativeComposition(queries=1, fallback_rows=0, fts_only_rows=1),
    )
    candidate = StrategyReport(
        strategy=CONSTRUCTION_STRATEGIES[3], hybrid=mode_report(), rescue=rescue(),
        census=LegCensus(
            k=K, hits=2, scoreable=2, queries=3,
            hit_queries=("kw-hit", "pm-miss"), zero_row_queries=(),
            per_category={"keyword": (1, 1), "paraphrase": (1, 1)},
        ),
        negatives=NegativeComposition(queries=1, fallback_rows=2, fts_only_rows=3),
    )
    report = SweepReport(
        k=K, entries=(control, candidate),
        rescue_query_id="kw-hit", target_slug="note-saltmarsh-hide",
        source_types=("media",), num_queries=3, num_scored=2, control_name="and",
    )

    rendered = format_construction_matrix(report)

    assert "resc" in rendered
    # The candidate's row carries census 2 AND rescues 1 — the two numbers a
    # reader must not conflate, printed side by side.
    row = next(line for line in rendered.splitlines() if line.startswith("and_or"))
    assert row.split()[2:4] == ["2", "1"]
    assert "NOT 'reaches fusion'" in rendered, (
        "a census hit is 'in the leg's own top-10', not 'reaches fusion' — "
        "hybrid over-fetches top_k x pool"
    )
    assert "STALENESS CHECK" in rendered


def test_the_probe_report_is_readable_on_its_own():
    service = with_engine_tokenizer(
        FakeLegService({"how do I write a shift log": [leg_row("media", "7")]})
    )

    probes = run_near_prefix_probes(
        FakeLegRuntime(service), CENSUS_GOLDEN, ("pm-miss",), k=K
    )
    rendered = format_probe_table(probes, rescues_to_beat=1)

    assert "near" in rendered and "prefix" in rendered
    assert "report-only" in rendered
    assert probes[0].hits == 1
    # The bar is a RESCUE count, and the table says so — the mis-scaled
    # comparison (a probe's ~40-query count against a row's full census) is
    # the one arithmetic error that would decide the promotion question
    # wrongly without anyone noticing.
    assert "rescues" in rendered and "RESCUE count" in rendered
    assert "full census" in rendered
    # NEAR's ceiling is a theorem, and the text Task 3 copies must carry it:
    # proximity only narrows, so NEAR over content tokens matches a subset of
    # `and_trim`, which rescues 1 of the 40.
    assert "NARROWS" in rendered and "and_trim" in rendered


# ---------------------------------------------------------------------------
# The construction table
# ---------------------------------------------------------------------------


def test_the_construction_matrix_prints_the_census_and_the_composition(monkeypatch):
    service = FakeLegService(CENSUS_ROWS)
    report = _run_instrumented(monkeypatch, service, expected=1)

    rendered = format_construction_matrix(report)

    assert "and_or" not in rendered, "only the rows that ran are in the table"
    for name in ("and", "or"):
        assert name in rendered
    assert "census" in rendered
    assert "keyword leg" in rendered
    # The 4110 rule (`qualify`) does NOT decide this arc: its winner is the
    # biggest census subject to the spec's hard constraints, applied in
    # writing. A table that printed "QUALIFIES" here would answer a question
    # nobody asked.
    assert "QUALIFIES" not in rendered and "WINNER" not in rendered
