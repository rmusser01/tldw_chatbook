# Tests/RAG_Eval/test_regression_gating.py
"""Known-answer tests for the ported regression + quality-gating modules
(always-on; pure -- no network, no external services).

Written against the exact server class APIs (tldw_server2 rag_service's
regression.py / quality_gating.py), read before writing these tests:
`RegressionDetector.check_regression(...)` (not `detect`), `save_baseline`/
`load_baseline`, and `GatingConfig`'s default stable/unstable metric tables.
"""
from __future__ import annotations

from tldw_chatbook.RAG_Search.eval.gating import (
    GatingConfig, GatingEvaluationResult, GatingEvaluator, GatingResult,
    MetricCategory,
)
from tldw_chatbook.RAG_Search.eval.regression import (
    MetricBaseline, RegressionDetector, environment_mismatch,
)


def test_baseline_json_round_trip(tmp_path):
    detector = RegressionDetector(baseline_dir=tmp_path)
    saved = detector.save_baseline(
        metrics={"precision": 0.8, "recall": 0.75},
        pipeline_config={"reranker": "flashrank", "chunk_size": 512},
        metadata={"environment": {"model": "all-MiniLM-L6-v2"}},
        baseline_id="ci-run-1",
    )

    loaded = detector.load_baseline("ci-run-1")

    assert loaded is not None
    assert isinstance(loaded, MetricBaseline)
    assert loaded == saved
    assert loaded.baseline_id == "ci-run-1"
    assert loaded.metrics == {"precision": 0.8, "recall": 0.75}
    assert loaded.pipeline_config == {"reranker": "flashrank", "chunk_size": 512}
    assert loaded.metadata == {"environment": {"model": "all-MiniLM-L6-v2"}}


def test_load_missing_baseline_returns_none(tmp_path):
    detector = RegressionDetector(baseline_dir=tmp_path)
    assert detector.load_baseline("does-not-exist") is None


def test_planted_regression_on_stable_metric_flags_fail(tmp_path):
    # "precision" is a stable metric in GatingConfig's defaults. A drop from
    # 0.8 to 0.5 (37.5% relative degradation) is far outside the detector's
    # default 5% threshold band, so it must be flagged as a hard regression.
    detector = RegressionDetector(baseline_dir=tmp_path)
    detector.save_baseline(metrics={"precision": 0.8}, baseline_id="latest")

    report = detector.check_regression(current_metrics={"precision": 0.5})

    assert report.has_regression is True
    assert report.has_warnings is False
    result = report.results[0]
    assert result.metric_name == "precision"
    assert result.category == MetricCategory.STABLE
    assert result.regressed is True


def test_unstable_metric_regression_flags_warning_not_fail(tmp_path):
    # "faithfulness" is an unstable (LLM-judged) metric in GatingConfig's
    # defaults. A regression there must surface as a warning only --
    # has_warnings True, has_regression False -- never a hard fail.
    detector = RegressionDetector(baseline_dir=tmp_path)
    detector.save_baseline(metrics={"faithfulness": 0.75}, baseline_id="latest")

    report = detector.check_regression(current_metrics={"faithfulness": 0.70})

    assert report.has_warnings is True
    assert report.has_regression is False
    result = report.results[0]
    assert result.metric_name == "faithfulness"
    assert result.category == MetricCategory.UNSTABLE
    assert result.regressed is True


def test_check_regression_no_baseline_found_is_not_a_regression(tmp_path):
    detector = RegressionDetector(baseline_dir=tmp_path)
    report = detector.check_regression(current_metrics={"precision": 0.5}, baseline_id="latest")

    assert report.has_regression is False
    assert report.results == []
    assert "No baseline" in report.summary


def test_gating_evaluator_pass_warn_fail():
    evaluator = GatingEvaluator(GatingConfig())

    passing = evaluator.evaluate({"precision": 0.9})
    assert isinstance(passing, GatingEvaluationResult)
    assert passing.overall_result == GatingResult.PASS
    assert passing.exit_code == 0

    warning = evaluator.evaluate({"precision": 0.9, "faithfulness": 0.5})
    assert warning.overall_result == GatingResult.WARN
    assert warning.exit_code == 2

    failing = evaluator.evaluate({"precision": 0.5})
    assert failing.overall_result == GatingResult.FAIL
    assert failing.exit_code == 1


def test_environment_mismatch_reports_only_differing_keys():
    baseline = MetricBaseline(
        baseline_id="latest",
        created_at="2026-08-08T00:00:00+00:00",
        metrics={"precision": 0.8},
        metadata={
            "environment": {
                "model": "all-MiniLM-L6-v2",
                "sentence_transformers": "2.7.0",
                "platform": "darwin",
            }
        },
    )

    fingerprint = {
        "model": "all-MiniLM-L6-v2",
        "sentence_transformers": "3.0.1",
        "platform": "darwin",
    }

    assert environment_mismatch(baseline, fingerprint) == ["sentence_transformers"]


def test_environment_mismatch_empty_on_match():
    baseline = MetricBaseline(
        baseline_id="latest",
        created_at="2026-08-08T00:00:00+00:00",
        metrics={"precision": 0.8},
        metadata={"environment": {"model": "all-MiniLM-L6-v2", "platform": "darwin"}},
    )

    fingerprint = {"model": "all-MiniLM-L6-v2", "platform": "darwin"}

    assert environment_mismatch(baseline, fingerprint) == []


def test_environment_mismatch_no_environment_metadata_treats_all_as_new():
    baseline = MetricBaseline(
        baseline_id="latest",
        created_at="2026-08-08T00:00:00+00:00",
        metrics={"precision": 0.8},
        metadata={},
    )

    assert environment_mismatch(baseline, {"model": "x"}) == ["model"]
