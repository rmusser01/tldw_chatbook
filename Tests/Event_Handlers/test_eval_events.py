"""Evaluation event-handler contract tests."""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tldw_chatbook.Evals.eval_runner import EvalSampleResult
from tldw_chatbook.Event_Handlers import eval_events


@pytest.mark.asyncio
async def test_execute_evaluation_uses_durable_id_and_same_loop_callbacks() -> None:
    app = Mock()
    window = Mock()
    tracker = Mock()
    cost_estimator = Mock()
    cost_estimator.finalize_tracking.return_value = None
    window.query_one.side_effect = lambda selector: {
        "#progress-tracker": tracker,
        "#cost-estimator": cost_estimator,
    }[selector]
    app.query_one.return_value = window

    result = EvalSampleResult(
        sample_id="sample-one",
        input_text="input",
        expected_output="output",
        actual_output="output",
        metrics={"correct": True},
        metadata={"input_tokens": 2, "output_tokens": 3},
    )
    captured_kwargs = {}

    class FakeOrchestrator:
        db = Mock()
        db.get_model.return_value = {
            "provider": "mock",
            "model_id": "contract-model",
        }

        async def run_evaluation(self, **kwargs):
            captured_kwargs.update(kwargs)
            start_result = kwargs["run_started_callback"]("durable-run-id")
            if inspect.isawaitable(start_result):
                await start_result
            await kwargs["progress_callback"](1, 1, result)
            return "durable-run-id"

    adapter = Mock()
    adapter.on_progress = AsyncMock()
    config = {
        "task_id": "task-id",
        "model_id": "model-id",
        "name": "Contract run",
        "max_samples": 1,
    }

    with (
        patch.object(eval_events, "get_orchestrator", return_value=FakeOrchestrator()),
        patch.object(
            eval_events,
            "EvaluationProgressAdapter",
            return_value=adapter,
        ) as adapter_class,
        patch.object(eval_events, "refresh_results_list", new=AsyncMock()),
        patch.object(eval_events, "update_results_table", new=AsyncMock()),
    ):
        await eval_events._execute_evaluation(app, config)

    assert "run_started_callback" in captured_kwargs
    adapter_class.assert_called_once_with(app, "durable-run-id")
    adapter.on_progress.assert_awaited_once()
    progress = adapter.on_progress.await_args.args[0]
    assert (progress.current, progress.total) == (1, 1)
    assert adapter.on_progress.await_args.args[1] is result
    cost_estimator.start_tracking.assert_called_once_with("durable-run-id")
    cost_estimator.update_sample_cost.assert_called_once_with(2, 3, 0)
    window.update_evaluation_progress.assert_called_once()
    app.call_from_thread.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_handler_awaits_cleanup_before_success_notice() -> None:
    app = Mock()
    orchestrator = Mock()
    orchestrator.cancel_evaluation = AsyncMock(return_value=True)

    with patch.object(eval_events, "get_orchestrator", return_value=orchestrator):
        await eval_events.handle_cancel_evaluation(app, "durable-run-id")

    orchestrator.cancel_evaluation.assert_awaited_once_with("durable-run-id")
    app.notify.assert_called_once_with(
        "Evaluation durable-run-id cancelled",
        severity="information",
    )
