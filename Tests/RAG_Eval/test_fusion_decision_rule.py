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

from types import SimpleNamespace

import pytest

from Tests.RAG_Eval.harness import fusion_sweep
from Tests.RAG_Eval.harness.fusion_sweep import (
    ALPHA_COMBO_STRATEGIES,
    BASE_STRATEGIES,
    CONTROL,
    CONTROL_NAME,
    LEVER_PRECEDENCE,
    WARN_BAND,
    Qualification,
    RescueVerdict,
    Strategy,
    StrategyReport,
    SweepReport,
    combined_strategy,
    fts_only_beats_vector_rank,
    lever_rank,
    qualify,
    run_fusion_sweep,
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
