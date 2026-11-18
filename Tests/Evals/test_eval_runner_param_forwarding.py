# TASK-38587: eval-runner param forwarding and choice extraction.
#
# (a) `stop` was missing from _call_llm's valid_chat_params whitelist, so
#     stop_sequences configured for generation evals never reached
#     chat_api_call (PROVIDER_PARAM_MAP maps "stop" for every provider).
# (b) _extract_choice fabricated choice A whenever extraction failed,
#     biasing accuracy on verbose model replies.
# (c) dead call_kwargs constructions in ClassificationRunner and
#     GenerationRunner _generate_response (built, never used).

from unittest.mock import patch

import pytest

from tldw_chatbook.Evals.eval_runner import (
    ClassificationRunner,
    EvalRunner,
    EvalSample,
    EvalSampleResult,
    GenerationRunner,
    QuestionAnswerRunner,
)
from tldw_chatbook.Evals.task_loader import TaskConfig


def _model_config() -> dict:
    return {
        "provider": "openai",
        "model_id": "test-model",
        "api_key": "test-key",
    }


def _task_config(task_type: str, **overrides) -> TaskConfig:
    defaults = {
        "name": f"{task_type}_task",
        "description": "test",
        "task_type": task_type,
        "dataset_name": "test",
        "split": "test",
        "metric": "accuracy" if task_type == "classification" else "exact_match",
    }
    defaults.update(overrides)
    return TaskConfig(**defaults)


class TestStopParamForwarding:
    @pytest.mark.asyncio
    async def test_stop_sequences_reach_chat_api_call(self):
        """task_config.stop_sequences must arrive at the chat boundary."""
        runner = GenerationRunner(
            task_config=_task_config(
                "generation", stop_sequences=["END"], metric="bleu"
            ),
            model_config=_model_config(),
        )

        with patch(
            "tldw_chatbook.Evals.eval_runner.chat_api_call"
        ) as mock_call:
            mock_call.return_value = {"choices": [{"message": {"content": "ok"}}]}
            response = await runner._generate_response("Write a story.")

        assert response == "ok"
        assert mock_call.call_count == 1
        kwargs = mock_call.call_args.kwargs
        assert kwargs["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_stop_is_none_when_no_stop_sequences(self):
        runner = GenerationRunner(
            task_config=_task_config("generation", metric="bleu"),
            model_config=_model_config(),
        )

        with patch(
            "tldw_chatbook.Evals.eval_runner.chat_api_call"
        ) as mock_call:
            mock_call.return_value = {"choices": [{"message": {"content": "ok"}}]}
            await runner._generate_response("Write.")

        # None is harmless: project_chat_handler_kwargs drops None params,
        # so providers never see a stop override.
        assert mock_call.call_args.kwargs.get("stop") is None

    @pytest.mark.asyncio
    async def test_generation_kwargs_stop_forwarded_for_qa_runner(self):
        """`stop` in generation_kwargs flows through the kwargs passthrough."""
        runner = QuestionAnswerRunner(
            task_config=_task_config(
                "question_answer", generation_kwargs={"stop": ["STOP"]}
            ),
            model_config=_model_config(),
        )

        with patch(
            "tldw_chatbook.Evals.eval_runner.chat_api_call"
        ) as mock_call:
            mock_call.return_value = {"choices": [{"message": {"content": "ok"}}]}
            await runner._generate_response("Question?")

        assert mock_call.call_args.kwargs.get("stop") == ["STOP"]


class TestChoiceExtractionFailure:
    def _classification_runner(self, llm_response: str) -> ClassificationRunner:
        runner = ClassificationRunner(
            task_config=_task_config("classification"),
            model_config={**_model_config(), "retry_delay": 0.01},
        )

        async def stub_call(prompt, system_prompt=None, **kwargs):
            return llm_response

        runner._call_llm = stub_call
        return runner

    def test_extract_choice_returns_none_instead_of_first_choice(self):
        runner = self._classification_runner("unused")

        assert (
            runner._extract_choice(
                "I believe the second option is best.",
                ["A) 3", "B) 4"],
            )
            is None
        )
        # Matching responses still extract.
        assert runner._extract_choice("B) 4", ["A) 3", "B) 4"]) == "B) 4"
        # Free-text samples without choices keep the stripped response.
        assert runner._extract_choice(" plain answer ", []) == "plain answer"

    @pytest.mark.asyncio
    async def test_failed_extraction_counts_as_error_sample(self):
        runner = self._classification_runner(
            "Hmm, this is a nuanced question; let me think out loud..."
        )

        sample = EvalSample(
            id="verbose_1",
            input_text="What is 2+2?",
            expected_output="B) 4",
            choices=["A) 3", "B) 4"],
        )

        result = await runner.run_sample(sample)

        assert "error" in result.metrics
        assert result.actual_output is None  # NOT the fabricated "A) 3"
        assert result.error_info.get("error_code") == "CHOICE_EXTRACTION_FAILED"
        assert result.error_info.get("is_retryable") is False
        assert "accuracy" not in result.metrics
        assert result.retry_count == 0

    def test_failed_extraction_does_not_bias_accuracy(self):
        success = EvalSampleResult(
            sample_id="ok_1",
            input_text="Q",
            expected_output="B) 4",
            actual_output="B) 4",
            metrics={"accuracy": 1.0},
        )
        failed_extraction = EvalSampleResult(
            sample_id="bad_1",
            input_text="Q",
            expected_output="B) 4",
            actual_output=None,
            metrics={"error": 1.0},
        )

        runner = EvalRunner(_task_config("classification"), _model_config())
        aggregate = runner.calculate_aggregate_metrics([success, failed_extraction])

        # Accuracy is computed over the one successful sample only; the
        # failed extraction is an error, never a guessed choice.
        assert aggregate["accuracy_mean"] == 1.0
        assert aggregate["accuracy_count"] == 1
        assert aggregate["error_count"] == 1
        assert aggregate["total_samples"] == 2
