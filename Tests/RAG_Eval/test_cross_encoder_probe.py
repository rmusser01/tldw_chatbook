# Tests/RAG_Eval/test_cross_encoder_probe.py
"""Always-on tests for the cross-encoder probe's MECHANISM.

TASK-16965 Task 2. The gated run (`test_cross_encoder_probe_run.py`) needs a
real corpus, a real index and a real model; these do not, and they exist for
one reason: **the pre-registered decision rule must be pinned by a test
before it is applied to any number.** A rule that lives only inside a print
statement can be edited after the fact and nobody notices; a rule with
tests has to be edited in a reviewable diff.

Same split as the PRF precedent (`test_prf_probe.py` next to
`test_prf_probe_run.py`): mechanism here, meeting-reality there.
"""
from __future__ import annotations

import pytest

from Tests.RAG_Eval.harness.baseline_io import FAIL_BAND
from Tests.RAG_Eval.harness.cross_encoder_probe import (
    ARM_A,
    ARM_B,
    TOLERANCE,
    VERDICT_METRICS,
    ModeArm,
    Verdict,
    arm_verdict,
    compose_arc_verdict,
    metric_moves,
    reorder_rows,
    rows_to_search_results,
)

FLAT = {"precision": 0.5, "recall": 0.5, "mrr": 0.5, "ndcg": 0.5, "f1": 0.5}


def _arm(
    mode: str = "semantic",
    *,
    before: dict[str, float] | None = None,
    after: dict[str, float] | None = None,
    before_cats: dict[str, dict[str, float]] | None = None,
    after_cats: dict[str, dict[str, float]] | None = None,
) -> ModeArm:
    return ModeArm(
        arm=ARM_A,
        mode=mode,
        depth=10,
        before=before or dict(FLAT),
        after=after or dict(FLAT),
        before_per_category=before_cats or {},
        after_per_category=after_cats or {},
        rows_scored=600,
        rows_failed=0,
        empty_document_rows=0,
        row_order_changes=100,
        queries_reordered=60,
        queries_doc_order_changed=40,
        predict_seconds=1.0,
    )


def test_the_tolerance_is_the_gates_own_band_not_a_copy():
    """The plan says "the gate's tolerance (0.05)"; drift would be silent."""
    assert TOLERANCE == FAIL_BAND == 0.05


def test_nothing_moving_is_a_null():
    verdict, reasons = arm_verdict([_arm("semantic"), _arm("hybrid")])
    assert verdict is Verdict.NULL
    assert reasons == ()


def test_movement_inside_the_tolerance_is_still_a_null():
    """A tie is never a gain -- the burden is on the strategy."""
    after = {**FLAT, "mrr": 0.5 + TOLERANCE}
    verdict, _ = arm_verdict([_arm("semantic", after=after), _arm("hybrid")])
    assert verdict is Verdict.NULL


def test_one_metric_on_one_mode_beyond_tolerance_is_a_help():
    after = {**FLAT, "ndcg": 0.5 + TOLERANCE + 0.001}
    verdict, reasons = arm_verdict([_arm("semantic"), _arm("hybrid", after=after)])
    assert verdict is Verdict.HELPED
    assert any("hybrid/overall ndcg" in reason for reason in reasons)


def test_an_overall_regression_beyond_tolerance_is_harm():
    after = {**FLAT, "mrr": 0.5 - TOLERANCE - 0.001}
    verdict, reasons = arm_verdict([_arm("semantic", after=after), _arm("hybrid")])
    assert verdict is Verdict.HARMED
    assert any("semantic/overall mrr" in reason for reason in reasons)


def test_a_gain_that_costs_a_category_is_harm_not_help():
    """HELPED's own text is a conjunction: gain AND no category regressing."""
    arm = _arm(
        "semantic",
        after={**FLAT, "mrr": 0.9},
        before_cats={"paraphrase": dict(FLAT)},
        after_cats={"paraphrase": {**FLAT, "ndcg": 0.5 - TOLERANCE - 0.001}},
    )
    verdict, reasons = arm_verdict([arm, _arm("hybrid")])
    assert verdict is Verdict.HARMED
    assert any("semantic/paraphrase ndcg" in reason for reason in reasons)


def test_a_category_present_on_one_side_only_is_skipped_not_invented():
    arm = _arm(
        "semantic",
        before_cats={"paraphrase": dict(FLAT)},
        after_cats={},
    )
    verdict, _ = arm_verdict([arm, _arm("hybrid")])
    assert verdict is Verdict.NULL


def test_a_verdict_over_no_arms_raises_rather_than_reading_as_null():
    with pytest.raises(ValueError, match="no measured arms"):
        arm_verdict([])


def test_a_non_verdict_mode_raises():
    """`plain` is the identity by census; scoring it would be a category error."""
    with pytest.raises(ValueError, match="non-verdict mode"):
        arm_verdict([_arm("plain")])


def test_a_missing_metric_raises_rather_than_defaulting_to_zero():
    with pytest.raises(KeyError):
        metric_moves(FLAT, {"precision": 0.5})


def test_metric_moves_cover_exactly_the_pre_registered_metrics():
    moves = metric_moves(FLAT, FLAT)
    assert tuple(move.metric for move in moves) == VERDICT_METRICS


def test_harm_in_one_arm_is_never_covered_by_help_in_the_other():
    verdict, reason = compose_arc_verdict(
        {ARM_A: Verdict.HARMED, ARM_B: Verdict.HELPED}
    )
    assert verdict is Verdict.HARMED
    assert ARM_A in reason


def test_help_in_either_arm_composes_to_help():
    verdict, _ = compose_arc_verdict({ARM_A: Verdict.NULL, ARM_B: Verdict.HELPED})
    assert verdict is Verdict.HELPED


def test_two_nulls_compose_to_null():
    verdict, reason = compose_arc_verdict({ARM_A: Verdict.NULL, ARM_B: Verdict.NULL})
    assert verdict is Verdict.NULL
    assert "0.050" in reason


def test_composing_no_arms_raises():
    with pytest.raises(ValueError, match="no arms"):
        compose_arc_verdict({})


def test_search_results_carry_unique_ids_so_the_rerank_cache_cannot_collide():
    """Positional ids would give a query's semantic and hybrid windows one key."""
    rows = [{"snippet": "a", "score": 0.9}, {"snippet": "b", "score": 0.1}]
    semantic = rows_to_search_results(rows, window_id="A|semantic|q1")
    hybrid = rows_to_search_results(rows, window_id="A|hybrid|q1")
    assert {r.id for r in semantic}.isdisjoint({r.id for r in hybrid})


def test_search_results_take_text_from_the_snippet_and_tolerate_a_none_score():
    rows = [{"snippet": "the text", "score": None}, {"score": 0.4}]
    results = rows_to_search_results(rows, window_id="A|plain|q1")
    assert [r.document for r in results] == ["the text", ""]
    assert [r.score for r in results] == [0.0, 0.4]
    assert [r.metadata["probe_row_index"] for r in results] == [0, 1]


def test_reorder_rows_applies_the_permutation_to_the_original_rows():
    rows = [{"source_id": "1"}, {"source_id": "2"}, {"source_id": "3"}]
    reranked = rows_to_search_results(rows, window_id="A|semantic|q1")
    reranked = [reranked[2], reranked[0], reranked[1]]
    assert reorder_rows(rows, reranked) == [rows[2], rows[0], rows[1]]


def test_a_dropped_row_raises_rather_than_shortening_the_measurement():
    rows = [{"source_id": "1"}, {"source_id": "2"}]
    reranked = rows_to_search_results(rows, window_id="A|semantic|q1")[:1]
    with pytest.raises(ValueError, match="not a permutation"):
        reorder_rows(rows, reranked)
