# TASK-38584: A/B merged views, statistical tests and confidence intervals
# must pair results from the two runs by sample id.
#
# Pre-fix behavior: all three sites zipped `results_a`/`results_b` by
# position, but the two runs execute concurrently and get_run_results
# returns rows in completion order — so model B's answer to a different
# question was compared beside model A's, and count mismatches truncated
# silently.

from contextlib import contextmanager

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.Evals.ab_testing import ABTestConfig, ABTestRunner


@contextmanager
def capture_loguru_warnings():
    """Collect loguru warning messages emitted inside the block."""
    messages = []
    handler_id = loguru_logger.add(
        lambda message: messages.append(str(message)), level="WARNING"
    )
    try:
        yield messages
    finally:
        loguru_logger.remove(handler_id)


def _row(sample_id: str, output: str, accuracy: float) -> dict:
    return {
        "sample_id": sample_id,
        "input_data": {"input": f"question for {sample_id}"},
        "expected_output": "expected",
        "actual_output": output,
        "metrics": {"accuracy": accuracy, "exact_match": accuracy},
    }


class _FakeOrchestrator:
    """Minimal orchestrator double returning canned stored rows."""

    db = type(
        "DB", (), {"get_model": staticmethod(lambda model_id: {"name": model_id})}
    )()

    def __init__(self, results_a, results_b):
        self._results = {"run-a": results_a, "run-b": results_b}

    async def run_evaluation(self, **kwargs):
        return "run-a" if kwargs["model_id"] == "model-a" else "run-b"

    def get_run_results(self, run_id):
        return self._results[run_id]

    def get_run_summary(self, run_id):
        return {"metrics": {}}


def _config() -> ABTestConfig:
    return ABTestConfig(
        name="pairing",
        description="pairing test",
        task_id="task",
        model_a_id="model-a",
        model_b_id="model-b",
        metrics_to_compare=["accuracy"],
    )


class TestPairingBySampleId:
    def test_merge_pairs_same_sample_across_shuffled_orders(self):
        runner = ABTestRunner(orchestrator=None)
        results_a = [
            _row("s1", "a-s1", 1.0),
            _row("s2", "a-s2", 0.0),
            _row("s3", "a-s3", 1.0),
        ]
        # Model B stored in a different completion order.
        results_b = [
            _row("s3", "b-s3", 1.0),
            _row("s1", "b-s1", 1.0),
            _row("s2", "b-s2", 0.0),
        ]

        merged = runner._merge_sample_results(results_a, results_b)

        # Deterministic (sorted) order, and each entry pairs one sample's
        # two answers — not whatever happened to share a list position.
        assert [entry["sample_id"] for entry in merged] == ["s1", "s2", "s3"]
        for entry in merged:
            sample_id = entry["sample_id"]
            assert entry["model_a_output"] == f"a-{sample_id}"
            assert entry["model_b_output"] == f"b-{sample_id}"

    def test_confidence_intervals_pair_by_sample_id(self):
        runner = ABTestRunner(orchestrator=None)
        # Identical per-sample accuracy for both models, stored in
        # different orders: matched differences are all zero.
        results_a = [_row("s1", "a-s1", 1.0), _row("s2", "a-s2", 0.0)]
        results_b = [_row("s2", "b-s2", 0.0), _row("s1", "b-s1", 1.0)]

        intervals = runner._calculate_confidence_intervals(
            results_a, results_b, ["accuracy"]
        )

        assert intervals["accuracy_difference"] == (0.0, 0.0)

    def test_statistical_tests_use_matched_intersection(self):
        runner = ABTestRunner(orchestrator=None)
        # Model B never stored s1; only s2/s3 are comparable.
        results_a = [
            _row("s1", "a-s1", 1.0),
            _row("s2", "a-s2", 0.0),
            _row("s3", "a-s3", 0.0),
        ]
        results_b = [_row("s2", "b-s2", 1.0), _row("s3", "b-s3", 1.0)]

        tests = runner._perform_statistical_tests(results_a, results_b, _config())

        assert tests["accuracy"]["sample_size_a"] == 2
        assert tests["accuracy"]["sample_size_b"] == 2
        # Means over the matched intersection (s2, s3), not over the
        # positionally-zipped prefix (s1, s2) that would give mean 0.5.
        assert tests["accuracy"]["mean_a"] == 0.0
        assert tests["accuracy"]["mean_b"] == 1.0

    def test_pairing_stats_report_mismatched_counts(self):
        runner = ABTestRunner(orchestrator=None)
        results_a = [
            _row("s1", "a-s1", 1.0),
            _row("s2", "a-s2", 0.0),
            _row("s3", "a-s3", 0.0),
        ]
        results_b = [_row("s2", "b-s2", 1.0), _row("s3", "b-s3", 1.0)]

        _, stats = runner._pair_results_by_sample_id(results_a, results_b)

        assert stats == {
            "results_a": 3,
            "results_b": 2,
            "matched_pairs": 2,
            "unmatched_a": 1,
            "unmatched_b": 0,
        }


class TestRunAbTestPairing:
    @pytest.mark.asyncio
    async def test_records_pairing_metadata_and_warns_on_mismatch(self):
        results_a = [
            _row("s1", "a-s1", 1.0),
            _row("s2", "a-s2", 0.0),
            _row("s3", "a-s3", 0.0),
        ]
        # Model B truncated (no s2) and stored out of order.
        results_b = [_row("s3", "b-s3", 1.0), _row("s1", "b-s1", 1.0)]
        runner = ABTestRunner(_FakeOrchestrator(results_a, results_b))

        with capture_loguru_warnings() as messages:
            result = await runner.run_ab_test(_config())

        assert result.sample_size == 2  # matched pairs, not len(results_a)
        assert result.metadata["pairing"] == {
            "results_a": 3,
            "results_b": 2,
            "matched_pairs": 2,
            "unmatched_a": 1,
            "unmatched_b": 0,
        }
        assert any("matched sample ids" in message for message in messages)
        # The merged view contains only matched ids, paired per sample.
        assert [entry["sample_id"] for entry in result.sample_results] == ["s1", "s3"]
        assert result.sample_results[0]["model_b_output"] == "b-s1"
        assert result.sample_results[1]["model_b_output"] == "b-s3"

    @pytest.mark.asyncio
    async def test_no_warning_and_full_sample_size_when_runs_match(self):
        results_a = [
            _row("s1", "a-s1", 1.0),
            _row("s2", "a-s2", 0.0),
            _row("s3", "a-s3", 1.0),
        ]
        results_b = [
            _row("s3", "b-s3", 0.0),
            _row("s1", "b-s1", 1.0),
            _row("s2", "b-s2", 1.0),
        ]
        runner = ABTestRunner(_FakeOrchestrator(results_a, results_b))

        with capture_loguru_warnings() as messages:
            result = await runner.run_ab_test(_config())

        assert result.sample_size == 3
        assert result.metadata["pairing"]["matched_pairs"] == 3
        assert not any("matched sample ids" in message for message in messages)
