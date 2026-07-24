"""Regression tests for bounded evaluation execution contracts."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from tldw_chatbook.Evals.eval_errors import ExecutionError
from tldw_chatbook.Evals.ab_testing import ABTestConfig, ABTestRunner
from tldw_chatbook.Evals.eval_runner import (
    EvalRunner,
    EvalSample,
    EvalSampleResult,
)
from tldw_chatbook.Evals.task_loader import TaskConfig


def _task_config(*, metadata: dict[str, Any] | None = None) -> TaskConfig:
    return TaskConfig(
        name="contract-task",
        description="Evaluation execution contract test",
        task_type="question_answer",
        dataset_name="unused",
        metadata=metadata or {},
    )


def _model_config(**overrides: Any) -> dict[str, Any]:
    return {
        "provider": "mock",
        "model_id": "contract-model",
        **overrides,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_concurrent_requests", True),
        ("max_concurrent_requests", 0),
        ("max_concurrent_requests", -1),
        ("max_concurrent_requests", 1.5),
        ("request_timeout", True),
        ("request_timeout", 0),
        ("request_timeout", -1),
        ("request_timeout", float("nan")),
        ("request_timeout", float("inf")),
        ("retry_attempts", True),
        ("retry_attempts", -1),
        ("retry_attempts", 1.5),
        ("retry_delay", True),
        ("retry_delay", -1),
        ("retry_delay", float("nan")),
        ("retry_delay", float("inf")),
    ],
)
def test_invalid_execution_bounds_fail_during_construction(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        EvalRunner(_task_config(), _model_config(**{field: value}))


@pytest.mark.asyncio
async def test_sync_dispatcher_runs_off_loop_and_returns_string() -> None:
    loop_thread = threading.get_ident()
    provider_threads: list[int] = []
    provider_finished_at = 0.0

    def sync_dispatcher(**_: Any) -> str:
        nonlocal provider_finished_at
        provider_threads.append(threading.get_ident())
        time.sleep(0.05)
        provider_finished_at = time.monotonic()
        return "sync-result"

    runner = EvalRunner(
        _task_config(),
        _model_config(request_timeout=1.0),
    )

    with patch(
        "tldw_chatbook.Evals.eval_runner.chat_api_call",
        side_effect=sync_dispatcher,
    ):
        call = asyncio.create_task(runner.runner._call_llm("hello"))
        await asyncio.sleep(0.01)
        heartbeat_at = time.monotonic()
        result = await call

    assert result == "sync-result"
    assert provider_threads == [provider_threads[0]]
    assert provider_threads[0] != loop_thread
    assert heartbeat_at < provider_finished_at


@pytest.mark.asyncio
async def test_sync_dispatcher_returning_awaitable_is_supported() -> None:
    async def response() -> str:
        await asyncio.sleep(0)
        return "awaited-result"

    def sync_dispatcher(**_: Any):
        return response()

    runner = EvalRunner(_task_config(), _model_config(request_timeout=1.0))
    with patch(
        "tldw_chatbook.Evals.eval_runner.chat_api_call",
        side_effect=sync_dispatcher,
    ):
        assert await runner.runner._call_llm("hello") == "awaited-result"


@pytest.mark.asyncio
async def test_basic_runner_uses_configured_retry_count_delay_and_timeout() -> None:
    calls = 0
    delays: list[float] = []

    async def timeout_call(*_: Any, **__: Any) -> str:
        nonlocal calls
        calls += 1
        raise asyncio.TimeoutError

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    runner = EvalRunner(
        _task_config(),
        _model_config(
            request_timeout=7.5,
            retry_attempts=2,
            retry_delay=0.25,
        ),
    )
    runner.runner._call_llm = timeout_call

    with patch(
        "tldw_chatbook.Evals.eval_runner.asyncio.sleep",
        side_effect=record_sleep,
    ):
        result = await runner.run_single_sample(
            runner.task_config,
            EvalSample(
                id="timed-out",
                input_text="hello",
                expected_output="world",
            ),
        )

    assert calls == 3
    assert delays == [0.25, 0.5]
    assert "7.5 seconds" in result.error_info["message"]


@pytest.mark.asyncio
async def test_specialized_interface_applies_one_configured_retry_loop() -> None:
    calls = 0
    delays: list[float] = []

    async def timeout_call(*_: Any, **__: Any) -> str:
        nonlocal calls
        calls += 1
        raise asyncio.TimeoutError

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    runner = EvalRunner(
        _task_config(metadata={"category": "coding"}),
        _model_config(
            request_timeout=4.25,
            retry_attempts=2,
            retry_delay=0.125,
        ),
    )
    runner.runner._call_llm = timeout_call

    with patch(
        "tldw_chatbook.Evals.eval_runner.asyncio.sleep",
        side_effect=record_sleep,
    ):
        with pytest.raises(ExecutionError) as exc_info:
            await runner.runner.llm_interface.generate("hello")

    assert calls == 3
    assert delays == [0.125, 0.25]
    assert "4.25 seconds" in exc_info.value.context.message


def _samples(count: int) -> list[EvalSample]:
    return [
        EvalSample(
            id=str(index),
            input_text=f"input-{index}",
            expected_output=f"output-{index}",
        )
        for index in range(count)
    ]


def _result(sample: EvalSample) -> EvalSampleResult:
    return EvalSampleResult(
        sample_id=sample.id,
        input_text=sample.input_text,
        expected_output=sample.expected_output,
        actual_output=sample.expected_output or "",
    )


@pytest.mark.asyncio
async def test_samples_are_bounded_callbacks_follow_settlement_and_results_keep_order(
) -> None:
    samples = _samples(3)
    releases = {sample.id: asyncio.Event() for sample in samples}
    started = {sample.id: asyncio.Event() for sample in samples}
    delivered = {sample.id: asyncio.Event() for sample in samples}
    active = 0
    peak_active = 0
    callbacks: list[tuple[int, int, str]] = []

    async def run_sample(sample: EvalSample) -> EvalSampleResult:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        started[sample.id].set()
        try:
            await releases[sample.id].wait()
            return _result(sample)
        finally:
            active -= 1

    async def progress(
        completed: int, total: int, result: EvalSampleResult
    ) -> None:
        await asyncio.sleep(0)
        callbacks.append((completed, total, result.sample_id))
        delivered[result.sample_id].set()

    runner = EvalRunner(
        _task_config(),
        _model_config(max_concurrent_requests=2),
    )
    runner.runner.run_sample = run_sample

    with patch(
        "tldw_chatbook.Evals.eval_runner.DatasetLoader.load_dataset_samples",
        return_value=samples,
    ):
        evaluation = asyncio.create_task(
            runner.run_evaluation(progress_callback=progress)
        )
        await asyncio.wait_for(
            asyncio.gather(started["0"].wait(), started["1"].wait()),
            timeout=1,
        )
        assert not started["2"].is_set()

        releases["1"].set()
        await asyncio.wait_for(delivered["1"].wait(), timeout=1)
        await asyncio.wait_for(started["2"].wait(), timeout=1)
        releases["2"].set()
        await asyncio.wait_for(delivered["2"].wait(), timeout=1)
        releases["0"].set()
        results = await asyncio.wait_for(evaluation, timeout=1)

    assert peak_active == 2
    assert callbacks == [(1, 3, "1"), (2, 3, "2"), (3, 3, "0")]
    assert [result.sample_id for result in results] == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_sync_progress_callback_receives_each_result_once() -> None:
    samples = _samples(3)
    callbacks: list[tuple[int, int, str]] = []
    runner = EvalRunner(_task_config(), _model_config())
    runner.runner.run_sample = lambda sample: asyncio.sleep(
        0, result=_result(sample)
    )

    def progress(completed: int, total: int, result: EvalSampleResult) -> None:
        callbacks.append((completed, total, result.sample_id))

    with patch(
        "tldw_chatbook.Evals.eval_runner.DatasetLoader.load_dataset_samples",
        return_value=samples,
    ):
        await runner.run_evaluation(progress_callback=progress)

    assert len(callbacks) == 3
    assert [completed for completed, _, _ in callbacks] == [1, 2, 3]
    assert all(total == 3 for _, total, _ in callbacks)
    assert sorted(sample_id for _, _, sample_id in callbacks) == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_callback_failure_cancels_and_drains_blocked_siblings() -> None:
    samples = _samples(3)
    second_started = asyncio.Event()
    second_cancelled = asyncio.Event()

    async def run_sample(sample: EvalSample) -> EvalSampleResult:
        if sample.id == "0":
            await second_started.wait()
            return _result(sample)
        if sample.id == "1":
            second_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                second_cancelled.set()
                raise
        return _result(sample)

    def fail_callback(*_: Any) -> None:
        raise RuntimeError("callback failed")

    runner = EvalRunner(
        _task_config(),
        _model_config(max_concurrent_requests=2),
    )
    runner.runner.run_sample = run_sample

    with patch(
        "tldw_chatbook.Evals.eval_runner.DatasetLoader.load_dataset_samples",
        return_value=samples,
    ):
        with pytest.raises(RuntimeError, match="callback failed"):
            await asyncio.wait_for(
                runner.run_evaluation(progress_callback=fail_callback),
                timeout=1,
            )

    assert second_cancelled.is_set()


@pytest.mark.asyncio
async def test_caller_cancellation_cancels_and_drains_sample_tasks() -> None:
    samples = _samples(2)
    started = [asyncio.Event(), asyncio.Event()]
    cancelled = [asyncio.Event(), asyncio.Event()]

    async def run_sample(sample: EvalSample) -> EvalSampleResult:
        index = int(sample.id)
        started[index].set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled[index].set()
            raise

    runner = EvalRunner(
        _task_config(),
        _model_config(max_concurrent_requests=2),
    )
    runner.runner.run_sample = run_sample

    with patch(
        "tldw_chatbook.Evals.eval_runner.DatasetLoader.load_dataset_samples",
        return_value=samples,
    ):
        evaluation = asyncio.create_task(runner.run_evaluation())
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started)),
            timeout=1,
        )
        evaluation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await evaluation

    assert all(event.is_set() for event in cancelled)


@pytest.mark.asyncio
async def test_ab_progress_wrapper_awaits_async_callback() -> None:
    updates: list[tuple[int, int, str]] = []

    async def progress(completed: int, total: int, status: str) -> None:
        await asyncio.sleep(0)
        updates.append((completed, total, status))

    class FakeOrchestrator:
        db = type(
            "DB",
            (),
            {"get_model": staticmethod(lambda model_id: {"name": model_id})},
        )()

        async def run_evaluation(self, **kwargs):
            callback_result = kwargs["progress_callback"](
                1, 1, _result(_samples(1)[0])
            )
            if inspect.isawaitable(callback_result):
                await callback_result
            return kwargs["model_id"]

        def get_run_results(self, _run_id):
            return []

        def get_run_summary(self, _run_id):
            return {"metrics": {}}

    runner = ABTestRunner(FakeOrchestrator())
    result = await runner.run_ab_test(
        ABTestConfig(
            name="contract",
            description="callback contract",
            task_id="task",
            model_a_id="model-a",
            model_b_id="model-b",
        ),
        progress_callback=progress,
    )

    assert result.test_name == "contract"
    assert updates[0] == (0, 2, "Starting model evaluations...")
    assert sorted(updates[1:3]) == [
        (50, 100, "Model A: 1/1"),
        (100, 100, "Model B: 1/1"),
    ]
    assert updates[-1] == (100, 100, "Analyzing results...")
