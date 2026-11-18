# TASK-38583: LogProbRunner must fail honestly instead of fabricating
# predictions.
#
# Pre-fix behavior (audit live repro): `_get_text_logprob` returned
# float("-inf") for every provider after firing (and discarding) a real paid
# request in the OpenAI branch, so every logprob sample silently predicted
# choice 0 and -Infinity reached the persisted metrics.

import json

import pytest

from tldw_chatbook.Evals.eval_runner import (
    EvalRunner,
    EvalSample,
    EvalSampleResult,
    LogProbRunner,
)
from tldw_chatbook.Evals.task_loader import TaskConfig


def _logprob_task_config() -> TaskConfig:
    return TaskConfig(
        name="logprob_task",
        description="Log probability evaluation",
        task_type="logprob",
        dataset_name="test",
        split="test",
        metric="accuracy",
        generation_kwargs={"temperature": 0.0, "max_tokens": 1},
    )


def _model_config() -> dict:
    return {
        "provider": "openai",
        "model_id": "test-model",
        "api_key": "test-key",
        "retry_attempts": 3,
        "retry_delay": 0.01,
    }


class _RecordingStub:
    """Stub provider returning plain text; records every call."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    async def __call__(self, prompt, system_prompt=None, **kwargs):
        self.calls += 1
        return self.text


class TestLogProbHonestFailure:
    def _runner(self, stub: _RecordingStub) -> LogProbRunner:
        runner = LogProbRunner(
            task_config=_logprob_task_config(), model_config=_model_config()
        )
        runner._call_llm = stub
        return runner

    @pytest.mark.asyncio
    async def test_multiple_choice_sample_is_error_not_choice_zero(self):
        """A plain-text provider yields an ERROR sample, never choice 0."""
        stub = _RecordingStub("The answer is the second option, obviously.")
        runner = self._runner(stub)

        sample = EvalSample(
            id="mc_1",
            input_text="What is 2+2?",
            expected_output="B) 4",
            choices=["A) 3", "B) 4", "C) 5", "D) 6"],
        )

        result = await runner.run_sample(sample)

        assert "error" in result.metrics
        assert result.actual_output is None  # NOT the fabricated choice 0
        assert result.metrics["error"] == 1.0
        assert result.error_info.get("error_code") == "LOGPROBS_UNSUPPORTED"
        assert "logprobs" in result.error_info.get("message", "").lower()
        # No paid request was made (so none was made-and-discarded either).
        assert stub.calls == 0

    @pytest.mark.asyncio
    async def test_single_continuation_sample_is_error(self):
        """Single-continuation logprob samples fail honestly too."""
        stub = _RecordingStub("some plain text")
        runner = self._runner(stub)

        sample = EvalSample(
            id="cont_1",
            input_text="The capital of France is",
            expected_output=" Paris",
        )

        result = await runner.run_sample(sample)

        assert "error" in result.metrics
        assert result.actual_output is None
        assert result.error_info.get("error_code") == "LOGPROBS_UNSUPPORTED"
        assert stub.calls == 0

    @pytest.mark.asyncio
    async def test_unsupported_error_is_not_retried(self):
        """The unsupported signal is non-retryable: no retry sleeps/calls."""
        stub = _RecordingStub("plain text")
        runner = self._runner(stub)

        sample = EvalSample(
            id="mc_retry",
            input_text="Q?",
            expected_output="B) 4",
            choices=["A) 3", "B) 4"],
        )

        result = await runner.run_sample(sample)

        assert result.error_info.get("is_retryable") is False
        assert result.retry_count == 0

    @pytest.mark.asyncio
    async def test_non_openai_provider_also_fails_honestly(self):
        """Providers outside the old OpenAI branch take the same path."""
        stub = _RecordingStub("plain text")
        runner = LogProbRunner(
            task_config=_logprob_task_config(),
            model_config={**_model_config(), "provider": "anthropic"},
        )
        runner._call_llm = stub

        sample = EvalSample(
            id="mc_other",
            input_text="Q?",
            expected_output="B) 4",
            choices=["A) 3", "B) 4"],
        )

        result = await runner.run_sample(sample)

        assert "error" in result.metrics
        assert result.actual_output is None
        assert stub.calls == 0


class TestNonFiniteMetricGuard:
    """Belt-and-braces: -inf/nan never reach persisted metrics."""

    def _result(self, **overrides) -> EvalSampleResult:
        defaults = {
            "sample_id": "s1",
            "input_text": "Q?",
            "expected_output": "A",
            "actual_output": "A",
            "metrics": {"accuracy": 1.0},
        }
        defaults.update(overrides)
        return EvalSampleResult(**defaults)

    def test_sample_result_sanitizes_nonfinite_metrics_and_logprobs(self):
        result = self._result(
            metrics={"logprob": float("-inf"), "accuracy": 1.0},
            logprobs={
                "choice_logprobs": [float("-inf"), float("nan"), -1.5],
                "choice_names": ["A", "B", "C"],
            },
        )

        assert result.metrics["logprob"] is None
        assert result.metrics["accuracy"] == 1.0
        assert result.logprobs["choice_logprobs"] == [None, None, -1.5]
        # Strict-JSON safe at the EvalsDB json.dumps persist boundary.
        json.dumps(result.metrics, allow_nan=False)
        json.dumps(result.logprobs, allow_nan=False)

    def test_aggregate_metrics_skip_nonfinite_values(self):
        runner = EvalRunner(_logprob_task_config(), _model_config())

        good = self._result(metrics={"accuracy": 1.0})
        bad = self._result(sample_id="s2", metrics={"accuracy": float("nan")})
        # Bypass __post_init__ sanitizing to exercise the aggregate guard
        # directly (a metrics dict mutated after construction).
        bad.metrics["accuracy"] = float("-inf")

        aggregate = runner.calculate_aggregate_metrics([good, bad])

        assert aggregate["accuracy_mean"] == 1.0
        assert aggregate["accuracy_count"] == 1
        json.dumps(aggregate, allow_nan=False)
