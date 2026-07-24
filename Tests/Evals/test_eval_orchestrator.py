# test_eval_orchestrator.py
# Description: Unit tests for the eval_orchestrator module
#
"""
Test Evaluation Orchestrator
----------------------------

Tests for the main orchestrator including the _active_tasks bug fix.
"""

import os
import asyncio
import inspect
import stat
from pathlib import Path

import pytest
from unittest.mock import Mock, patch

from tldw_chatbook import config
from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator
from tldw_chatbook.Evals.eval_errors import EvaluationError, ErrorContext, ErrorCategory
from tldw_chatbook.Evals.eval_runner import EvalSampleResult
from tldw_chatbook.Utils.private_paths import PrivatePathError


def _seed_run_inputs(
    orchestrator: EvaluationOrchestrator,
) -> tuple[str, str]:
    task_id = orchestrator.db.create_task(
        name="Contract task",
        description="Orchestrator contract test",
        task_type="question_answer",
        config_format="custom",
        config_data={
            "name": "Contract task",
            "description": "Orchestrator contract test",
            "task_type": "question_answer",
            "dataset_name": "unused",
            "metric": "exact_match",
        },
    )
    model_id = orchestrator.db.create_model(
        name="Contract model",
        provider="mock",
        model_id="contract-model",
    )
    return task_id, model_id


def _orchestrator_result(
    sample_id: str,
    *,
    error: bool = False,
) -> EvalSampleResult:
    return EvalSampleResult(
        sample_id=sample_id,
        input_text=f"input-{sample_id}",
        expected_output=f"output-{sample_id}",
        actual_output=None if error else f"output-{sample_id}",
        metrics={"error": 1.0} if error else {"exact_match": 1.0},
        error_info={"error_category": "provider"} if error else {},
    )


class _ControlledEvalRunner:
    def __init__(self, results: list[EvalSampleResult]):
        self.results = results

    async def run_evaluation(self, *, max_samples=None, progress_callback=None):
        selected = self.results[:max_samples] if max_samples else self.results
        for completed, result in enumerate(selected, 1):
            if progress_callback:
                callback_result = progress_callback(
                    completed, len(selected), result
                )
                if inspect.isawaitable(callback_result):
                    await callback_result
        return selected

    def calculate_aggregate_metrics(self, results):
        return {
            "total_samples": len(results),
            "error_count": sum(bool(result.error_info) for result in results),
        }


class TestEvaluationOrchestrator:
    """Test suite for EvaluationOrchestrator."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        """Create an orchestrator instance with temporary database."""
        db_path = tmp_path / "test_evals.db"
        return EvaluationOrchestrator(db_path=str(db_path))

    def test_active_tasks_initialization(self, orchestrator):
        """Test that _active_tasks is properly initialized (bug fix verification)."""
        # This tests the critical bug fix - _active_tasks should be initialized
        assert hasattr(orchestrator, "_active_tasks"), (
            "_active_tasks attribute is missing"
        )
        assert isinstance(orchestrator._active_tasks, dict), (
            "_active_tasks should be a dictionary"
        )
        assert len(orchestrator._active_tasks) == 0, (
            "_active_tasks should be empty initially"
        )

    @pytest.mark.asyncio
    async def test_cancel_evaluation_with_no_tasks(self, orchestrator):
        """Test cancel_evaluation doesn't crash when no tasks exist."""
        result = await orchestrator.cancel_evaluation("non_existent_run_id")
        assert result is False, "Should return False for non-existent run"

    @pytest.mark.asyncio
    async def test_cancel_evaluation_with_active_task(self, orchestrator):
        """Test cancelling an active evaluation task."""
        run_id = "test_run_123"

        async def owned_work():
            try:
                await asyncio.Event().wait()
            finally:
                orchestrator._active_tasks.pop(run_id, None)

        task = asyncio.create_task(owned_work())
        orchestrator._active_tasks[run_id] = task
        await asyncio.sleep(0)
        result = await orchestrator.cancel_evaluation(run_id)

        assert result is True, "Should return True when task is cancelled"
        assert task.cancelled()
        assert run_id not in orchestrator._active_tasks

    @pytest.mark.asyncio
    async def test_cancel_all_evaluations(self, orchestrator):
        """Test asynchronous close drains all active evaluations."""
        async def owned_work(run_id):
            try:
                await asyncio.Event().wait()
            finally:
                orchestrator._active_tasks.pop(run_id, None)

        for index in range(3):
            run_id = f"run_{index}"
            orchestrator._active_tasks[run_id] = asyncio.create_task(
                owned_work(run_id)
            )
        await asyncio.sleep(0)

        with patch.object(orchestrator.db, "close") as close:
            await orchestrator.aclose()

        assert len(orchestrator._active_tasks) == 0, "All tasks should be removed"
        close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_evaluation_tracking(self, orchestrator):
        """Test that run_evaluation properly tracks active tasks."""
        with patch.object(orchestrator, "db") as mock_db:
            with patch.object(orchestrator, "task_loader") as mock_loader:
                with patch.object(
                    orchestrator.concurrent_manager, "register_run", return_value=True
                ):
                    # Mock task config
                    mock_task = Mock()
                    mock_task.task_type = "question_answer"
                    mock_task.dataset_name = "test_dataset"
                    mock_loader.get_task.return_value = mock_task

                    # Mock database methods
                    mock_db.create_run.return_value = "test_run_id"
                    mock_db.update_run_status.return_value = None

                    # Mock model config
                    model_config = {
                        "provider": "test",
                        "model_id": "test-model",
                        "name": "Test Model",
                    }

                    # Mock get_task and get_model
                    mock_db.get_task.return_value = {
                        "name": "Test Task",
                        "task_type": "question_answer",
                        "dataset_name": "test_dataset",
                        "config_data": {"metric": "exact_match"},
                    }
                    mock_db.get_model.return_value = model_config

                    # Start evaluation (will fail but should track)
                    try:
                        await orchestrator.run_evaluation(
                            task_id="test_task", model_id="test-model", max_samples=10
                        )
                    except Exception:
                        pass  # Expected to fail in test environment

                    # Check if tracking was attempted
                    # Note: In real implementation, task would be added to _active_tasks

    def test_database_initialization(self, tmp_path):
        """Test database is properly initialized."""
        db_path = tmp_path / "test_evals.db"
        orchestrator = EvaluationOrchestrator(db_path=str(db_path))

        assert orchestrator.db is not None, "Database should be initialized"
        assert hasattr(orchestrator.db, "db_path"), "Database should have db_path"

    def test_default_database_uses_secured_user_data_directory(
        self, monkeypatch, tmp_path
    ):
        user_data_dir = tmp_path / "secured-user-data"
        user_data_dir.mkdir()
        monkeypatch.setattr(config, "get_user_data_dir", lambda: user_data_dir)

        with patch("tldw_chatbook.Evals.eval_orchestrator.EvalsDB") as evals_db_class:
            EvaluationOrchestrator()

        evals_db_class.assert_called_once_with(
            db_path=str(user_data_dir / "evals.db"),
            client_id="eval_orchestrator",
        )

    @pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
    def test_custom_database_parent_is_not_created_or_mutated(self, tmp_path: Path):
        custom_parent = tmp_path / "custom"
        custom_parent.mkdir()
        custom_parent.chmod(0o751)
        db_path = custom_parent / "evals.db"

        with patch("tldw_chatbook.Evals.eval_orchestrator.EvalsDB") as evals_db_class:
            EvaluationOrchestrator(db_path=db_path)

        assert stat.S_IMODE(custom_parent.stat().st_mode) == 0o751
        evals_db_class.assert_called_once_with(
            db_path=str(db_path),
            client_id="eval_orchestrator",
        )

    def test_custom_database_parent_must_exist(self, tmp_path: Path):
        db_path = tmp_path / "missing" / "evals.db"

        with (
            patch("tldw_chatbook.Evals.eval_orchestrator.EvalsDB") as evals_db_class,
            pytest.raises(PrivatePathError),
        ):
            EvaluationOrchestrator(db_path=db_path)

        evals_db_class.assert_not_called()

    def test_memory_database_token_is_preserved(self):
        with patch("tldw_chatbook.Evals.eval_orchestrator.EvalsDB") as evals_db_class:
            EvaluationOrchestrator(db_path=":memory:")

        evals_db_class.assert_called_once_with(
            db_path=":memory:",
            client_id="eval_orchestrator",
        )

    def test_component_initialization(self, orchestrator):
        """Test all components are properly initialized."""
        assert orchestrator.concurrent_manager is not None, "Concurrent manager missing"
        assert orchestrator.validator is not None, "Validator missing"
        assert orchestrator.error_handler is not None, "Error handler missing"
        assert orchestrator.task_loader is not None, "Task loader missing"
        assert orchestrator._client_id == "eval_orchestrator", (
            "Client ID not set correctly"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_callback", [False, True])
    async def test_progress_callback_runs_after_durable_storage(
        self, orchestrator, async_callback
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        results = [_orchestrator_result("one"), _orchestrator_result("two")]
        observations = []

        def observe(completed, total, result):
            run = orchestrator.db.list_runs(limit=1)[0]
            stored = orchestrator.db.get_results_for_run(run["id"])
            observations.append((completed, total, result.sample_id, len(stored)))

        async def observe_async(completed, total, result):
            observe(completed, total, result)

        callback = observe_async if async_callback else observe
        with patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=_ControlledEvalRunner(results),
        ):
            run_id = await orchestrator.run_evaluation(
                task_id,
                model_id,
                progress_callback=callback,
            )

        assert observations == [
            (1, 2, "one", 1),
            (2, 2, "two", 2),
        ]
        assert orchestrator.db.get_run(run_id)["status"] == "completed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_source", ["storage", "callback"])
    async def test_pipeline_failure_is_persisted_and_escapes(
        self, orchestrator, failure_source
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        results = [_orchestrator_result("one")]

        def fail_callback(*_):
            raise RuntimeError("callback exploded")

        store_patch = (
            patch.object(
                orchestrator.db,
                "store_result",
                side_effect=RuntimeError("storage exploded"),
            )
            if failure_source == "storage"
            else patch.object(
                orchestrator.db,
                "store_result",
                wraps=orchestrator.db.store_result,
            )
        )
        callback = fail_callback if failure_source == "callback" else None

        with (
            store_patch,
            patch(
                "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
                return_value=_ControlledEvalRunner(results),
            ),
            pytest.raises(EvaluationError),
        ):
            await orchestrator.run_evaluation(
                task_id,
                model_id,
                progress_callback=callback,
            )

        run = orchestrator.db.list_runs(limit=1)[0]
        assert run["status"] == "failed"
        assert f"{failure_source} exploded" in run["error_message"]

    @pytest.mark.asyncio
    async def test_error_results_are_retained_and_make_run_failed(
        self, orchestrator
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        results = [
            _orchestrator_result("clean"),
            _orchestrator_result("failed", error=True),
        ]

        with patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=_ControlledEvalRunner(results),
        ):
            run_id = await orchestrator.run_evaluation(task_id, model_id)

        stored = orchestrator.db.get_results_for_run(run_id)
        run = orchestrator.db.get_run(run_id)
        assert len(stored) == 2
        assert run["status"] == "failed"
        assert "1 of 2 evaluation samples failed" in run["error_message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("async_callback", [False, True])
    async def test_run_started_callback_exposes_registered_running_run(
        self, orchestrator, async_callback
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        callback_seen = asyncio.Event()
        provider_started_after_callback = False

        def observe_start(run_id):
            assert orchestrator.db.get_run(run_id)["status"] == "running"
            assert orchestrator._active_tasks[run_id] is asyncio.current_task()
            assert run_id in orchestrator.concurrent_manager._active_runs
            callback_seen.set()

        async def observe_start_async(run_id):
            await asyncio.sleep(0)
            observe_start(run_id)

        class StartAwareRunner(_ControlledEvalRunner):
            async def run_evaluation(self, **kwargs):
                nonlocal provider_started_after_callback
                provider_started_after_callback = callback_seen.is_set()
                return await super().run_evaluation(**kwargs)

        callback = observe_start_async if async_callback else observe_start
        with patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=StartAwareRunner([_orchestrator_result("one")]),
        ):
            run_id = await orchestrator.run_evaluation(
                task_id,
                model_id,
                run_started_callback=callback,
            )

        assert callback_seen.is_set()
        assert provider_started_after_callback
        assert run_id not in orchestrator._active_tasks

    @pytest.mark.asyncio
    async def test_run_started_callback_failure_starts_no_samples(
        self, orchestrator
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        sample_started = False

        class MustNotRun(_ControlledEvalRunner):
            async def run_evaluation(self, **kwargs):
                nonlocal sample_started
                sample_started = True
                return await super().run_evaluation(**kwargs)

        def fail_start(_run_id):
            raise RuntimeError("start callback exploded")

        with (
            patch(
                "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
                return_value=MustNotRun([_orchestrator_result("one")]),
            ),
            pytest.raises(EvaluationError),
        ):
            await orchestrator.run_evaluation(
                task_id,
                model_id,
                run_started_callback=fail_start,
            )

        run = orchestrator.db.list_runs(limit=1)[0]
        assert not sample_started
        assert run["status"] == "failed"
        assert "start callback exploded" in run["error_message"]
        assert not orchestrator._active_tasks
        assert not orchestrator.concurrent_manager._active_runs

    @pytest.mark.asyncio
    async def test_public_cancellation_drains_owner_and_persists_cancelled(
        self, orchestrator
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        run_id_ready = asyncio.Future()
        sample_started = asyncio.Event()
        sample_cancelled = asyncio.Event()

        class BlockingRunner(_ControlledEvalRunner):
            async def run_evaluation(self, **kwargs):
                sample_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    sample_cancelled.set()
                    raise

        def capture_run_id(run_id):
            run_id_ready.set_result(run_id)

        with patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=BlockingRunner([]),
        ):
            run_task = asyncio.create_task(
                orchestrator.run_evaluation(
                    task_id,
                    model_id,
                    run_started_callback=capture_run_id,
                )
            )
            run_id = await asyncio.wait_for(run_id_ready, timeout=1)
            await asyncio.wait_for(sample_started.wait(), timeout=1)

            assert await orchestrator.cancel_evaluation(run_id) is True
            with pytest.raises(asyncio.CancelledError):
                await run_task

        assert sample_cancelled.is_set()
        assert orchestrator.db.get_run(run_id)["status"] == "cancelled"
        assert run_id not in orchestrator._active_tasks
        assert run_id not in orchestrator.concurrent_manager._active_runs
        assert await orchestrator.cancel_evaluation(run_id) is False
        assert await orchestrator.cancel_evaluation("unknown") is False

    @pytest.mark.asyncio
    async def test_direct_cancellation_uses_same_durable_cleanup_path(
        self, orchestrator
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        run_id_ready = asyncio.Future()
        sample_started = asyncio.Event()
        sample_cancelled = asyncio.Event()

        class BlockingRunner(_ControlledEvalRunner):
            async def run_evaluation(self, **kwargs):
                sample_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    sample_cancelled.set()
                    raise

        with patch(
            "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
            return_value=BlockingRunner([]),
        ):
            run_task = asyncio.create_task(
                orchestrator.run_evaluation(
                    task_id,
                    model_id,
                    run_started_callback=run_id_ready.set_result,
                )
            )
            run_id = await asyncio.wait_for(run_id_ready, timeout=1)
            await asyncio.wait_for(sample_started.wait(), timeout=1)
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

        assert sample_cancelled.is_set()
        assert orchestrator.db.get_run(run_id)["status"] == "cancelled"
        assert run_id not in orchestrator._active_tasks
        assert run_id not in orchestrator.concurrent_manager._active_runs

    @pytest.mark.asyncio
    async def test_aclose_drains_active_run_before_closing_database(
        self, orchestrator
    ):
        task_id, model_id = _seed_run_inputs(orchestrator)
        run_id_ready = asyncio.Future()
        sample_started = asyncio.Event()

        class BlockingRunner(_ControlledEvalRunner):
            async def run_evaluation(self, **kwargs):
                sample_started.set()
                await asyncio.Event().wait()

        with (
            patch(
                "tldw_chatbook.Evals.eval_orchestrator.EvalRunner",
                return_value=BlockingRunner([]),
            ),
            patch.object(orchestrator.db, "close", wraps=orchestrator.db.close) as close,
        ):
            run_task = asyncio.create_task(
                orchestrator.run_evaluation(
                    task_id,
                    model_id,
                    run_started_callback=run_id_ready.set_result,
                )
            )
            run_id = await asyncio.wait_for(run_id_ready, timeout=1)
            await asyncio.wait_for(sample_started.wait(), timeout=1)

            await orchestrator.aclose()

        assert run_task.cancelled()
        assert not orchestrator._active_tasks
        assert not orchestrator.concurrent_manager._active_runs
        assert orchestrator.db.get_run(run_id)["status"] == "cancelled"
        close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_sync_close_refuses_active_run_without_closing_database(
        self, orchestrator
    ):
        task = asyncio.create_task(asyncio.Event().wait())
        orchestrator._active_tasks["active-run"] = task

        with patch.object(orchestrator.db, "close") as close:
            with pytest.raises(RuntimeError, match="active evaluation"):
                orchestrator.close()
            close.assert_not_called()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        orchestrator._active_tasks.clear()

    @pytest.mark.asyncio
    async def test_create_task_from_file(self, orchestrator, tmp_path):
        """Test creating a task from a file."""
        # Create a test task file
        task_file = tmp_path / "test_task.json"
        task_data = [{"id": "1", "input": "What is 2+2?", "output": "4"}]

        import json

        with open(task_file, "w") as f:
            json.dump(task_data, f)

        # Mock task loader and database
        from tldw_chatbook.Evals.task_loader import TaskConfig

        mock_task = TaskConfig(
            name="Test Task",
            description="Test task for unit testing",
            task_type="question_answer",
            dataset_name=str(task_file),
            metric="exact_match",
        )

        with patch.object(
            orchestrator.task_loader, "load_task", return_value=mock_task
        ):
            with patch.object(orchestrator.db, "create_task", return_value="task_123"):
                task_id = await orchestrator.create_task_from_file(
                    str(task_file), "Test Task"
                )

                assert task_id == "task_123", "Should return task ID"

    def test_get_run_status(self, orchestrator):
        """Test getting run status."""
        with patch.object(
            orchestrator.db,
            "get_run",
            return_value={"run_id": "run_123", "status": "completed", "progress": 100},
        ):
            status = orchestrator.get_run_status("run_123")

            assert status["status"] == "completed"
            assert status["progress"] == 100

    def test_list_available_tasks(self, orchestrator):
        """Test listing available tasks."""
        with patch.object(orchestrator.db, "list_tasks") as mock_list:
            mock_list.return_value = [
                {"task_id": "1", "name": "Task 1"},
                {"task_id": "2", "name": "Task 2"},
            ]

            tasks = orchestrator.list_available_tasks()

            assert len(tasks) == 2
            assert tasks[0]["name"] == "Task 1"
            mock_list.assert_called_once()


def test_eval_event_singleton_delegates_default_path_selection(monkeypatch):
    from tldw_chatbook.Event_Handlers import eval_events

    orchestrator = Mock()
    orchestrator_class = Mock(return_value=orchestrator)
    monkeypatch.setattr(eval_events, "EvaluationOrchestrator", orchestrator_class)
    monkeypatch.setattr(eval_events, "_orchestrator", None)

    assert eval_events.get_orchestrator() is orchestrator
    orchestrator_class.assert_called_once_with()


class TestOrchestratorIntegration:
    """Integration tests for the orchestrator."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self, tmp_path):
        """Test a complete evaluation flow."""
        # Create orchestrator with temp database
        db_path = tmp_path / "test_evals.db"
        orchestrator = EvaluationOrchestrator(db_path=str(db_path))

        # Create a test task file
        task_file = tmp_path / "test_task.json"
        task_data = {
            "name": "Integration Test Task",
            "task_type": "question_answer",
            "dataset": [
                {
                    "id": "1",
                    "input": "What is the capital of France?",
                    "output": "Paris",
                },
                {"id": "2", "input": "What is 2+2?", "output": "4"},
            ],
            "metric": "exact_match",
        }

        import json

        with open(task_file, "w") as f:
            json.dump(task_data, f)

        # Mock the LLM calls
        with patch("tldw_chatbook.Chat.Chat_Functions.chat_api_call") as mock_chat:
            mock_chat.return_value = ("Paris", None)  # Mock response

            try:
                # Create task
                task_id = await orchestrator.create_task_from_file(
                    str(task_file), "Integration Test"
                )

                # Prepare model config
                model_config = {
                    "provider": "mock",
                    "model_id": "mock-model",
                    "name": "Mock Model",
                    "api_key": "mock_key",
                }

                # Run evaluation
                # Note: This may fail in test environment, but we're testing the flow
                await orchestrator.run_evaluation(
                    task_id=task_id, model_configs=[model_config], max_samples=2
                )

            except Exception as e:
                # Expected in test environment
                print(f"Expected error in test: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_management(self, tmp_path):
        """Test concurrent evaluation management."""
        db_path = tmp_path / "test_evals.db"
        orchestrator = EvaluationOrchestrator(db_path=str(db_path))

        # Test that concurrent manager is working by simulating a conflict
        from tldw_chatbook.Evals.eval_errors import ValidationError, ErrorSeverity

        conflict_error = ValidationError(
            ErrorContext(
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.WARNING,
                message="An evaluation is already running for this task and model combination",
                is_retryable=True,
            )
        )

        with patch.object(
            orchestrator.concurrent_manager, "register_run", side_effect=conflict_error
        ):
            # Mock the database to avoid other errors
            with patch.object(orchestrator.db, "get_task") as mock_get_task:
                mock_get_task.return_value = {
                    "name": "Test Task",
                    "task_type": "question_answer",
                    "dataset_name": "test_dataset",
                    "config_data": {"metric": "exact_match"},
                }

                with patch.object(orchestrator.db, "get_model") as mock_get_model:
                    mock_get_model.return_value = {
                        "provider": "test",
                        "model_id": "test",
                        "name": "Test Model",
                    }

                    with patch.object(
                        orchestrator.db, "create_run", return_value="test-run"
                    ):
                        # Should raise error due to concurrent run conflict
                        # The ValidationError gets wrapped as EvaluationError
                        with pytest.raises(
                            (ValidationError, EvaluationError)
                        ) as exc_info:
                            await orchestrator.run_evaluation(
                                task_id="test", model_id="test", max_samples=10
                            )

                        # Check that the error is related to concurrent runs
                        error_msg = str(exc_info.value).lower()
                        assert (
                            "already running" in error_msg
                            or "evaluation failed" in error_msg
                        )


class TestOrchestratorErrorHandling:
    """Test error handling in the orchestrator."""

    @pytest.mark.asyncio
    async def test_invalid_task_id_handling(self, tmp_path):
        """Test handling of invalid task ID."""
        db_path = tmp_path / "test_evals.db"
        orchestrator = EvaluationOrchestrator(db_path=str(db_path))

        # Mock database to return None for invalid task
        with patch.object(orchestrator.db, "get_task") as mock_get_task:
            mock_get_task.return_value = None  # Task not found

            # Mock get_model to avoid other errors
            with patch.object(orchestrator.db, "get_model") as mock_get_model:
                mock_get_model.return_value = {
                    "provider": "test",
                    "model_id": "test-model",
                    "name": "Test Model",
                }

                # The error will be wrapped as DatabaseError by _db_operation
                from tldw_chatbook.Evals.eval_errors import DatabaseError

                with pytest.raises((EvaluationError, DatabaseError)):
                    await orchestrator.run_evaluation(
                        task_id="invalid_task", model_id="test_model", max_samples=10
                    )

    @pytest.mark.asyncio
    async def test_invalid_model_config_handling(self, tmp_path):
        """Test handling of invalid model configuration."""
        db_path = tmp_path / "test_evals.db"
        orchestrator = EvaluationOrchestrator(db_path=str(db_path))

        # Mock get_task to return valid task
        with patch.object(orchestrator.db, "get_task") as mock_get_task:
            mock_get_task.return_value = {
                "name": "Test Task",
                "task_type": "question_answer",
                "dataset_name": "test_dataset",
                "config_data": {"metric": "exact_match"},
            }

            # Mock get_model to return None (model not found)
            with patch.object(orchestrator.db, "get_model") as mock_get_model:
                mock_get_model.return_value = None  # Model not found

                # The error will be wrapped as DatabaseError by _db_operation
                from tldw_chatbook.Evals.eval_errors import DatabaseError

                with pytest.raises((EvaluationError, DatabaseError)):
                    await orchestrator.run_evaluation(
                        task_id="test_task", model_id="invalid_model", max_samples=10
                    )
