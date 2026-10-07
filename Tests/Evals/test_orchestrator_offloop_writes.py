"""B5: per-sample result persistence hops off the event loop.

``EvaluationOrchestrator`` used to call ``db.store_result(...)`` inline inside
the awaited ``progress_wrapper`` for every completed sample, so one synchronous
sqlite write stalled the event loop per sample. The write now runs on a worker
thread (``asyncio.to_thread``) for file-backed databases; ``:memory:``
databases stay inline because EvalsDB keeps thread-local connections and each
thread's connection to ``:memory:`` is a private, empty database.

Exception propagation is preserved: a failing ``store_result`` still surfaces
through the run's error path (run marked failed with the storage message).
"""

import threading
from unittest.mock import patch

import pytest

from tldw_chatbook.Evals.eval_errors import EvaluationError
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.Evals.eval_runner import EvalSampleResult


class _ControlledEvalRunner:
    """Minimal EvalRunner stand-in that replays fixed results through the callback."""

    def __init__(self, results: list[EvalSampleResult]):
        self.results = results

    async def run_evaluation(self, *, max_samples=None, progress_callback=None):
        import inspect

        selected = self.results[:max_samples] if max_samples else self.results
        for completed, result in enumerate(selected, 1):
            if progress_callback:
                callback_result = progress_callback(completed, len(selected), result)
                if inspect.isawaitable(callback_result):
                    await callback_result
        return selected

    def calculate_aggregate_metrics(self, results):
        return {
            "total_samples": len(results),
            "error_count": sum(bool(result.error_info) for result in results),
        }


def _result(sample_id: str) -> EvalSampleResult:
    return EvalSampleResult(
        sample_id=sample_id,
        input_text=f"input-{sample_id}",
        expected_output=f"output-{sample_id}",
        actual_output=f"output-{sample_id}",
        metrics={"exact_match": 1.0},
        error_info={},
    )


def _seed_run_inputs(orchestrator: EvaluationOrchestrator) -> tuple[str, str]:
    task_id = orchestrator.db.create_task(
        name="B5 contract task",
        description="Off-loop persistence test",
        task_type="question_answer",
        config_format="custom",
        config_data={
            "name": "B5 contract task",
            "description": "Off-loop persistence test",
            "task_type": "question_answer",
            "dataset_name": "unused",
            "metric": "exact_match",
        },
    )
    model_id = orchestrator.db.create_model(
        name="B5 model",
        provider="mock",
        model_id="b5-model",
    )
    return task_id, model_id


@pytest.fixture
def orchestrator(tmp_path):
    """File-backed orchestrator (file DB is required for the thread hop)."""
    db_path = tmp_path / "b5_evals.db"
    orch = EvaluationOrchestrator(db_path=str(db_path))
    yield orch
    orch.db.close()


@pytest.mark.asyncio
async def test_store_result_runs_off_loop(orchestrator):
    task_id, model_id = _seed_run_inputs(orchestrator)
    seen_threads = []
    real_store = orchestrator.db.store_result

    def recording_store(*args, **kwargs):
        seen_threads.append(threading.current_thread())
        return real_store(*args, **kwargs)

    results = [_result("one"), _result("two")]
    with (
        patch.object(orchestrator.db, "store_result", side_effect=recording_store),
        patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=_ControlledEvalRunner(results),
        ),
    ):
        run_id = await orchestrator.run_evaluation(task_id, model_id)

    assert seen_threads, "store_result was never called"
    loop_thread = threading.current_thread()
    offenders = [t for t in seen_threads if t is loop_thread]
    assert not offenders, (
        f"{len(offenders)}/{len(seen_threads)} store_result calls ran on the event loop thread"
    )
    # The writes still landed.
    assert len(orchestrator.db.get_results_for_run(run_id)) == 2


@pytest.mark.asyncio
async def test_in_memory_db_keeps_inline_writes():
    """:memory: connections are per-thread private, so the write stays on the loop."""
    orch = EvaluationOrchestrator(db_path=":memory:")
    try:
        task_id, model_id = _seed_run_inputs(orch)
        seen_threads = []
        real_store = orch.db.store_result

        def recording_store(*args, **kwargs):
            seen_threads.append(threading.current_thread())
            return real_store(*args, **kwargs)

        results = [_result("only")]
        with (
            patch.object(orch.db, "store_result", side_effect=recording_store),
            patch(
                "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
                return_value=_ControlledEvalRunner(results),
            ),
        ):
            run_id = await orch.run_evaluation(task_id, model_id)

        assert seen_threads, "store_result was never called"
        loop_thread = threading.current_thread()
        assert all(t is loop_thread for t in seen_threads)
        assert len(orch.db.get_results_for_run(run_id)) == 1
    finally:
        orch.db.close()


@pytest.mark.asyncio
async def test_store_result_failure_still_surfaces(orchestrator):
    task_id, model_id = _seed_run_inputs(orchestrator)
    results = [_result("one")]

    def failing_store(*args, **kwargs):
        raise RuntimeError("storage exploded")

    with (
        patch.object(orchestrator.db, "store_result", side_effect=failing_store),
        patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=_ControlledEvalRunner(results),
        ),
        pytest.raises(EvaluationError),
    ):
        await orchestrator.run_evaluation(task_id, model_id)

    run = orchestrator.db.list_runs(limit=1)[0]
    assert run["status"] == "failed"
    assert "storage exploded" in run["error_message"]
