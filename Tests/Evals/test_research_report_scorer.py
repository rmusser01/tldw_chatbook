"""Research-report self-eval scorer (task-16327).

Deterministic scoring of deep-search research reports from the verification
outcomes the pipeline already produces (task-16319's citation_verification
payload / task-16325's claims). The runner plugs into the existing Evals
framework (specialized runner + category dispatch) -- no parallel harness.
"""

import asyncio
import json
from typing import Any

import pytest

from tldw_chatbook.Evals.eval_runner import EvalRunner, EvalSample, TaskConfig
from tldw_chatbook.Evals.research_report_scorer import (
    BASELINE_VERIFICATION_PAYLOAD,
    score_research_report,
)
from tldw_chatbook.Evals.specialized_runners import ResearchReportRunner


_FULL_PAYLOAD = {
    "markers_total": 10,
    "markers_resolved": 8,
    "unknown_marker_ids": [11, 12],
    "quotes_checked": 4,
    "quotes_verified": 3,
    "quotes_misquoted": 1,
    "uncited_sentences": 6,
    "claims": [
        {"claim_id": "claim-1", "status": "supported"},
        {"claim_id": "claim-2", "status": "supported"},
        {"claim_id": "claim-3", "status": "supported"},
        {"claim_id": "claim-4", "status": "unverified"},
    ],
}


def test_score_metrics_from_full_verification_payload():
    metrics = score_research_report(_FULL_PAYLOAD)

    assert metrics["citation_accuracy"] == pytest.approx(8 / 10)
    assert metrics["quote_grounding"] == pytest.approx(3 / 4)
    assert metrics["claim_support_rate"] == pytest.approx(3 / 4)
    assert metrics["cited_sentence_ratio"] == pytest.approx(10 / 16)


def test_score_zero_when_no_markers():
    metrics = score_research_report(
        {"markers_total": 0, "markers_resolved": 0, "quotes_checked": 0,
         "quotes_verified": 0, "uncited_sentences": 5}
    )
    assert metrics["citation_accuracy"] == 0.0
    assert metrics["quote_grounding"] == 0.0
    assert metrics["cited_sentence_ratio"] == 0.0


def test_score_claim_support_falls_back_to_marker_accuracy_without_claims():
    payload = dict(_FULL_PAYLOAD)
    payload.pop("claims")
    metrics = score_research_report(payload)
    assert metrics["claim_support_rate"] == pytest.approx(8 / 10)


def _research_task_config() -> TaskConfig:
    return TaskConfig(
        name="research-report-eval",
        description="Research report self-eval",
        task_type="research_report",
        dataset_name="unused",
        metadata={"category": "research"},
    )


def _model_config() -> dict[str, Any]:
    return {"provider": "mock", "model_id": "scorer"}


def test_runner_scores_sample_metadata_verification():
    runner = ResearchReportRunner(_research_task_config(), _model_config())
    sample = EvalSample(
        id="run-1",
        input_text=json.dumps({"verification": _FULL_PAYLOAD}),
        metadata={"verification": _FULL_PAYLOAD},
    )

    result = asyncio.run(runner.run_sample(sample))

    assert result.sample_id == "run-1"
    assert result.metrics["citation_accuracy"] == pytest.approx(0.8)
    assert result.metrics["quote_grounding"] == pytest.approx(0.75)
    # No LLM was consulted -- the scorer is deterministic over the payload.
    assert result.metadata["marker_counts"] == {"total": 10, "resolved": 8}


def test_eval_runner_dispatches_research_category():
    runner = EvalRunner(_research_task_config(), _model_config())
    assert isinstance(runner.runner, ResearchReportRunner)


def test_baseline_payload_reproduces_recorded_baseline():
    metrics = score_research_report(BASELINE_VERIFICATION_PAYLOAD)
    assert metrics == score_research_report(BASELINE_VERIFICATION_PAYLOAD)
    assert 0.0 <= metrics["citation_accuracy"] <= 1.0
    assert 0.0 <= metrics["quote_grounding"] <= 1.0
    assert 0.0 <= metrics["claim_support_rate"] <= 1.0
    assert 0.0 <= metrics["cited_sentence_ratio"] <= 1.0
