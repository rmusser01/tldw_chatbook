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
from concurrent.futures import Future, ThreadPoolExecutor
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
    orch.close()


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


class _ThreadPerCallExecutor(ThreadPoolExecutor):
    """Run every call on a new thread, as a busy default pool does."""

    def __init__(self) -> None:
        super().__init__()
        self.threads: list[threading.Thread] = []

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()

        def run() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)

        thread = threading.Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()
        return future


@pytest.mark.asyncio
async def test_aclose_leaves_no_worker_thread_connection_open(tmp_path):
    """TASK-38714: EvalsDB keeps one connection per thread and a close
    reaches only the calling thread's. The off-loop writes share one write
    thread and its one connection, which ``aclose()`` closes there."""
    import asyncio
    import sqlite3

    from tldw_chatbook.DB.Evals_DB import EvalsDB

    orch = EvaluationOrchestrator(db_path=str(tmp_path / "close_evals.db"))
    task_id, model_id = _seed_run_inputs(orch)
    opened: list[tuple[threading.Thread, sqlite3.Connection]] = []
    real_get = EvalsDB._get_connection

    def recording_get(self):
        connection = real_get(self)
        if all(connection is not seen for _, seen in opened):
            opened.append((threading.current_thread(), connection))
        return connection

    executor = _ThreadPerCallExecutor()
    asyncio.get_running_loop().set_default_executor(executor)
    with (
        patch.object(EvalsDB, "_get_connection", recording_get),
        patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=_ControlledEvalRunner(
                [_result("one"), _result("two"), _result("three")]
            ),
        ),
    ):
        run_id = await orch.run_evaluation(task_id, model_id)
        await orch.aclose()
    for thread in executor.threads:
        thread.join(5)

    worker_handles = [
        connection
        for thread, connection in opened
        if thread is not threading.current_thread()
    ]
    # A connection open costs ~45 ms (a private-SQLite helper process), so
    # three writes must share one.
    assert len(worker_handles) == 1, worker_handles
    for connection in worker_handles:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.in_transaction.__get__(connection)
    reopened = EvalsDB(str(tmp_path / "close_evals.db"))
    try:
        assert len(reopened.get_results_for_run(run_id)) == 3
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retirement_entry", ["deadline", "cancel_drain", "cancel_close"]
)
async def test_cancelled_native_write_refuses_drain_at_deadline_then_retires(
    tmp_path, retirement_entry
):
    """Cancellation and deadline refusal retain one physical retirement across resume."""
    import asyncio
    import sqlite3
    import time

    orch = EvaluationOrchestrator(db_path=str(tmp_path / "held_evals.db"))
    task_id, model_id = _seed_run_inputs(orch)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    retirement_entered = asyncio.Event()
    second_submission = asyncio.Event()
    retirement_starts = []
    handles = []
    real_store = orch.db.store_result
    real_retire = orch._retire_write_thread
    real_submission = orch._store_result_off_loop

    async def observed_submission(**kwargs):
        if entered.is_set():
            second_submission.set()
        return await real_submission(**kwargs)

    def observed_retire():
        retirement_starts.append(threading.current_thread())
        loop.call_soon_threadsafe(retirement_entered.set)
        return real_retire()

    def held_store(**kwargs):
        if handles:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                sqlite3.Connection.in_transaction.__get__(handles[-1])
        handles.append(orch.db._get_connection())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "native write was never released"
        return real_store(**kwargs)

    run = drain = second_run = None
    try:
        with (
            patch.object(orch.db, "store_result", side_effect=held_store),
            patch.object(orch, "_retire_write_thread", side_effect=observed_retire),
            patch.object(
                orch, "_store_result_off_loop", side_effect=observed_submission
            ),
            patch(
                "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
                return_value=_ControlledEvalRunner([_result("held")]),
            ),
        ):
            run = asyncio.create_task(orch.run_evaluation(task_id, model_id))
            await asyncio.wait_for(entered.wait(), 5)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
            assert not orch._maintenance_runs
            if retirement_entry == "cancel_close":
                drain = asyncio.create_task(orch.aclose())
            else:
                orch._maintenance_close_admission()
                deadline = time.monotonic() + (
                    0.02 if retirement_entry == "deadline" else 2
                )
                drain = asyncio.create_task(orch._maintenance_drain(deadline))
            await asyncio.wait_for(retirement_entered.wait(), 5)
            if retirement_entry != "deadline":
                drain.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await drain
                drain = None
            else:
                done, _ = await asyncio.wait({drain}, timeout=0.2)
                assert drain in done, "native retirement exceeded the supplied deadline"
                assert drain.result() is False
            orch._maintenance_close_admission()
            assert not await orch._maintenance_drain(time.monotonic() + 0.02)
            assert len(retirement_starts) == 1
            assert len(handles) == 1
            sqlite3.Connection.in_transaction.__get__(handles[0])
            with pytest.raises(RuntimeError, match="await aclose"):
                orch.close()
            with pytest.raises(RuntimeError, match="paused_for_maintenance"):
                await orch.run_evaluation(task_id, model_id)
            orch._maintenance_resume()
            second_run = asyncio.create_task(orch.run_evaluation(task_id, model_id))
            await asyncio.wait_for(second_submission.wait(), 5)
            assert len(handles) == 1
            assert len(retirement_starts) == 1
            release.set()
            second_run_id = await second_run
            assert len(handles) == 2
            assert handles[0] is not handles[1]
            orch._maintenance_close_admission()
            assert await orch._maintenance_drain(time.monotonic() + 2)
            assert len(retirement_starts) == 2
            for handle in handles:
                with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                    sqlite3.Connection.in_transaction.__get__(handle)
            assert len(orch.db.get_results_for_run(second_run_id)) == 1
            assert all(
                len(orch.db.get_results_for_run(row["id"])) == 1
                for row in orch.db.list_runs(limit=2)
            )
    finally:
        release.set()
        import sys

        primary = sys.exception()
        tasks = [task for task in (run, second_run, drain) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        outcomes.extend(await asyncio.gather(orch.aclose(), return_exceptions=True))
        errors = [
            error
            for error in outcomes
            if isinstance(error, BaseException)
            and not isinstance(error, asyncio.CancelledError)
        ]
        if primary is not None:
            for error in errors:
                primary.add_note("fixture_cleanup_error:" + type(error).__name__)
        elif errors:
            raise errors[0]


@pytest.mark.asyncio
async def test_failed_native_retirement_retains_its_executor_and_refuses_replacement(
    tmp_path,
):
    """A failed close is an owned failure, never permission for a new writer."""
    import asyncio
    import time

    orch = EvaluationOrchestrator(db_path=str(tmp_path / "failed_close.db"))
    task_id, model_id = _seed_run_inputs(orch)
    real_close = orch.db.close
    try:
        with patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=_ControlledEvalRunner([_result("one")]),
        ):
            await orch.run_evaluation(task_id, model_id)
            executor = orch._write_executor
            orch._maintenance_close_admission()
            with patch.object(
                orch.db, "close", side_effect=RuntimeError("held-close-failure")
            ):
                with pytest.raises(RuntimeError, match="held-close-failure"):
                    await orch._maintenance_drain(time.monotonic() + 2)
                assert orch._write_executor is executor
                with pytest.raises(RuntimeError, match="held-close-failure"):
                    orch.close()
                orch._maintenance_resume()
                with pytest.raises(EvaluationError) as failure:
                    await orch.run_evaluation(task_id, model_id)
                assert isinstance(failure.value.__cause__, RuntimeError)
                assert str(failure.value.__cause__) == "held-close-failure"
                assert orch._write_executor is executor
    finally:
        # Explicitly retire the retained failed owner after restoring native close.
        await asyncio.to_thread(orch._retire_write_thread)
        real_close()
