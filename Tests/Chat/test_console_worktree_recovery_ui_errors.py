"""Mounted recovery failures remain local to the owning Console view."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest
from loguru import logger
from textual.app import App
from textual.worker import WorkerFailed

import tldw_chatbook.Chat.console_worktree_recovery as recovery_module
from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
from tldw_chatbook.UI.Console_Modules.worktree import open_recovery
from tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog import _RecoveryButton


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, stale_view",
    [("database", False), ("database", True), ("submission", False)],
)
async def test_picker_failure_is_contained(monkeypatch, tmp_path, failure, stale_view):
    """A selected failure leaves the app alive and the recovery owner settled.

    Args:
        monkeypatch: Replaces only the selected failure boundary.
        tmp_path: Isolated path for the manual recovery database.
        failure: Database-open or executor-submission failure to inject.
        stale_view: Whether the originating Console view has been replaced.
    """
    intent = SimpleNamespace(persisted_conversation_id="conversation")
    controller = SimpleNamespace(
        store=SimpleNamespace(active_session_id="session"),
        capture_worktree_recovery_intent=lambda session: intent,
        request_worktree_merge_confirm=lambda *args, **kwargs: {"allow": True},
    )
    capacity = RuntimeCapacity()
    helper = recovery_module.ConsoleWorktreeRecovery(
        controller,
        SimpleNamespace(
            runs_db=SimpleNamespace(db_path_str=str(tmp_path / "runs.db")),
            runtime_capacity=capacity,
        ),
    )
    row = {
        "run_id": "run",
        "child_path": str(tmp_path / "child"),
        "writer_state": "drained",
        "run_status": "done",
        "mutation_state": "unresolved",
    }

    async def listed(*args):
        return recovery_module.RecoveryPage(
            rows=(row,), conversation_id="conversation", repository=str(tmp_path)
        )

    monkeypatch.setattr(helper, "list_work", listed)
    monkeypatch.setattr(recovery_module, "validate_intent", lambda *args: object())
    engine_entries = []
    monkeypatch.setattr(
        recovery_module,
        "recover_agent_worktree",
        lambda *args, **kwargs: engine_entries.append(True),
    )
    entered = []

    def failing_database(*args):
        entered.append(True)
        raise sqlite3.OperationalError("injected database-open failure")

    if failure == "database":
        monkeypatch.setattr(recovery_module, "AgentRunsDB", failing_database)
    else:
        loop = asyncio.get_running_loop()
        original_submit = loop.run_in_executor

        def reject_recovery_submission(executor, func, *args):
            submitted = getattr(func, "args", (None,))[0]
            if getattr(submitted, "__name__", None) == "worker":
                entered.append(True)
                raise RuntimeError("injected executor-submission failure")
            return original_submit(executor, func, *args)

        monkeypatch.setattr(loop, "run_in_executor", reject_recovery_submission)
    notifications = []
    diagnostics = []
    log_sink = logger.add(
        lambda message: diagnostics.append(str(message)),
        format="{message}",
        level="WARNING",
    )

    class RecoveryApp(App):
        """Minimal mounted owner for the real recovery picker."""

        def _console_runtime(self):
            return runtime

        def notify(self, message, **kwargs):
            """Record the bounded notification without rendering a toast.

            Args:
                message: User-visible recovery outcome.
                **kwargs: Optional Textual notification arguments.
            """
            notifications.append(message)

    app = RecoveryApp()
    runtime = SimpleNamespace(
        chat_controller=controller, worktree_recovery=helper, view=app
    )
    caught = None
    try:
        try:
            async with app.run_test(size=(100, 35)) as pilot:
                await open_recovery(app)
                await pilot.pause()
                app.screen.query(_RecoveryButton).first().press()
                if stale_view:
                    runtime.view = object()
                for _ in range(100):
                    if "conversation" in helper.receipts:
                        break
                    await pilot.pause(0.02)
                await asyncio.sleep(0)
        except WorkerFailed as exc:
            caught = exc
        assert entered, "probe never reached the selected recovery failure seam"
        assert helper.receipts["conversation"].reason_code == "recovery_failed"
        assert not capacity.snapshot().executions
        assert not engine_entries, "failed admission must not attempt a Git effect"
        failure_logs = [
            line for line in diagnostics if "agent_worktree_recovery_failed:" in line
        ]
        assert len(failure_logs) == 1
        assert (
            "OperationalError" if failure == "database" else "RuntimeError"
        ) in failure_logs[0]
        assert "injected" not in failure_logs[0]
        assert caught is None, f"Textual terminated the app: {caught!r}"
        assert notifications == (
            []
            if stale_view
            else ["Recovery failed; inspect recorded work before retrying."]
        )
    finally:
        await helper.close()
        capacity.close()
        logger.remove(log_sink)
