"""Scoring: blending, renormalization, penalties, grades, confidence."""
import pytest

from tldw_chatbook.Evals.skill_eval.models import (
    DimensionScore, JudgeLayerResult, SimLayerResult, SkillEvalDepth,
    StaticLayerResult,
)
from tldw_chatbook.Evals.skill_eval.scoring import (
    CONFIDENCE_BY_DEPTH, DIMENSION_WEIGHTS, LAYER_BLENDS, blend_dimension,
    build_report, grade_for,
)


def _static(**dim):
    base = {k: 0.8 for k in DIMENSION_WEIGHTS}
    base.update({"output_quality": None, "robustness": None}, **dim)
    return StaticLayerResult(sub_scores={}, dimension_scores=base, findings=())


def test_weights_sum_to_one_and_blends_to_one():
    assert sum(DIMENSION_WEIGHTS.values()) == pytest.approx(1.0)
    for name, blend in LAYER_BLENDS.items():
        assert sum(blend) == pytest.approx(1.0), name
        assert name in DIMENSION_WEIGHTS, name


def test_blend_renormalizes_when_sim_missing():
    ds = blend_dimension("triggering_accuracy", static=0.8, judge=0.4, sim=None)
    s, j, _ = LAYER_BLENDS["triggering_accuracy"]
    expect = (s * 0.8 + j * 0.4) / (s + j)
    assert ds.blended == pytest.approx(expect)
    assert ds.available_layers == ("static", "judge")


def test_blend_static_only_quick_depth():
    ds = blend_dimension("tool_surface_sanity", static=0.9, judge=None, sim=None)
    assert ds.blended == pytest.approx(0.9)
    assert ds.available_layers == ("static",)


def test_grade_bands():
    assert grade_for(97) == "A+" and grade_for(96.9) == "A"
    assert grade_for(80) == "B-" and grade_for(59.9) == "F"


def test_build_report_confidence_and_penalty():
    from tldw_chatbook.Evals.skill_eval.models import StaticFinding
    static = StaticLayerResult(
        sub_scores={}, dimension_scores={k: 1.0 for k in DIMENSION_WEIGHTS},
        findings=(StaticFinding("BLOATED_SKILL", 0.05, "fix"),
                  StaticFinding("ORPHAN_REFERENCE", 0.05, "fix")),
    )
    report = build_report({"name": "s"}, SkillEvalDepth.QUICK, static, None, None)
    assert report.confidence == "Estimated"
    assert report.composite == pytest.approx(90.0)  # renormalized 1.0 * 0.90


def test_unmeasurable_dimensions_are_excluded_and_renormalized():
    static = StaticLayerResult(
        sub_scores={}, dimension_scores={
            "triggering_accuracy": 1.0, "instruction_fitness": 1.0,
            "output_quality": None, "scope_calibration": 1.0,
            "progressive_disclosure": 1.0, "tool_surface_sanity": 1.0,
            "token_efficiency": 1.0, "robustness": None,
            "structural_completeness": 1.0}, findings=())
    report = build_report({"name": "s"}, SkillEvalDepth.QUICK, static, None, None)
    # output_quality/robustness have zero static weight -> excluded, weights
    # renormalized over the measurable seven -> perfect static scores hit 100.
    assert report.composite == pytest.approx(100.0)


def test_build_report_degrades_confidence_when_judge_failed():
    static = _static()
    failed = JudgeLayerResult(rubrics={}, artifacts=(), failed=("all"))
    report = build_report({"name": "s"}, SkillEvalDepth.STANDARD, static, failed, None)
    assert report.confidence == "Estimated"
    assert any("judge" in w for w in report.warnings)


def test_confidence_labels_by_depth():
    assert CONFIDENCE_BY_DEPTH[SkillEvalDepth.DEEP] == "Certified"
