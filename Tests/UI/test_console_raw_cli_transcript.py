"""Focused contracts for live raw-CLI transcript markers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import asyncio
from pathlib import Path
import threading
import time
from time import monotonic
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat import console_chat_models
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
    RawCliPresentation,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService
from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.Tools.raw_cli_executor import (
    MAX_RAW_PREVIEW_BYTES,
    RawCliRequest,
    RawCliResult,
    RawCliStreamEvent,
)
from tldw_chatbook.UI.Console_Modules.raw_cli import ConsoleRawCliController
from tldw_chatbook.UI.Console_Modules import wiring as wiring_module
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash
from tldw_chatbook.Widgets.Console import console_assistant_turn as assistant_turn_module
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


def _presentation(**changes: object) -> RawCliPresentation:
    values = {
        "invocation_id": "raw-1",
        "caller": "user",
        "lifecycle_state": "running",
        "command": "printf '[bold]literal[/bold]'",
        "shell": "/bin/bash",
        "cwd": "/tmp/project",
        "started_at_monotonic": 10.0,
        "elapsed_seconds": 0.0,
        "exit_code": None,
        "truncated": False,
        "cleanup_proven": None,
    }
    return RawCliPresentation(**(values | changes))  # type: ignore[arg-type]


def test_raw_cli_presentation_is_frozen_and_strictly_bounded() -> None:
    """The marker carries data, never callbacks or unbounded free-form state."""
    presentation_type = getattr(console_chat_models, "RawCliPresentation", None)
    assert presentation_type is not None, "Task 7 presentation contract is missing"

    presentation = presentation_type(
        invocation_id="raw-1",
        caller="user",
        lifecycle_state="running",
        command="printf '[bold]literal[/bold]'",
        shell="/bin/bash",
        cwd="/tmp/project",
        started_at_monotonic=10.0,
        elapsed_seconds=0.0,
        exit_code=None,
        truncated=False,
        cleanup_proven=None,
    )

    assert presentation.invocation_id == "raw-1"
    with pytest.raises(FrozenInstanceError):
        presentation.lifecycle_state = "exited"

    invalid = {
        "invocation_id": "",
        "caller": "system",
        "lifecycle_state": "unknown",
        "command": "",
        "shell": "bash\nunsafe",
        "cwd": "cwd\runsafe",
        "started_at_monotonic": -1.0,
        "elapsed_seconds": float("inf"),
        "exit_code": True,
        "truncated": 1,
        "cleanup_proven": "yes",
    }
    defaults = {
        "invocation_id": "raw-1",
        "caller": "user",
        "lifecycle_state": "running",
        "command": "pwd",
        "shell": "bash",
        "cwd": "/tmp",
        "started_at_monotonic": 10.0,
        "elapsed_seconds": 0.0,
        "exit_code": None,
        "truncated": False,
        "cleanup_proven": None,
    }
    for field, value in invalid.items():
        with pytest.raises((TypeError, ValueError), match=field.replace("_", " ")):
            presentation_type(**(defaults | {field: value}))
    for field in ("command", "shell", "cwd"):
        with pytest.raises(ValueError, match="UTF-8"):
            presentation_type(**(defaults | {field: "\ud800"}))

    assert _presentation(
        lifecycle_state="starting",
        started_at_monotonic=None,
    ).started_at_monotonic is None
    with pytest.raises(ValueError, match="started at monotonic"):
        _presentation(lifecycle_state="starting")
    with pytest.raises(ValueError, match="started at monotonic"):
        _presentation(lifecycle_state="running", started_at_monotonic=None)


def test_store_rejects_unbounded_raw_marker_writes_and_trajectory_bypass() -> None:
    store = ConsoleChatStore()
    session = store.create_session(title="raw bounds")
    oversized = "x" * (64 * 1024 + 1)

    for field in ("content", "tool_output_full"):
        values = {
            "content": "bounded",
            "tool_output_full": "bounded",
        }
        values[field] = oversized
        with pytest.raises(ValueError, match="64 KiB"):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.TOOL,
                raw_cli_presentation=_presentation(),
                record_trajectory=False,
                **values,
            )

    marker = store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content="bounded",
        tool_output_full="bounded",
        raw_cli_presentation=_presentation(),
        record_trajectory=False,
    )
    for field in ("content", "tool_output_full"):
        with pytest.raises(ValueError, match="64 KiB"):
            store.update_tool_marker(session.id, marker.id, **{field: oversized})

    with pytest.raises(ValueError, match="raw CLI presentation"):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.TOOL,
            content="trajectory bypass",
            record_trajectory=False,
        )


def test_store_updates_one_display_only_raw_marker_without_trajectory() -> None:
    """Streaming replaces a marker snapshot, never a tree or trajectory row."""
    store = ConsoleChatStore()
    session = store.create_session(title="raw")
    anchor = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="ordinary tree node",
    )
    trajectory_calls: list[tuple] = []
    store._record_trajectory_tool_marker = (  # type: ignore[method-assign]
        lambda *args: trajectory_calls.append(args)
    )

    started = store.append_message(
        session.id,
        role=ConsoleMessageRole.TOOL,
        content="stdout:\n(first chunk)",
        tool_output_full="stdout:\n(first chunk)",
        activity_presentation=ConsoleActivityPresentation("tool", "Raw CLI", "done"),
        raw_cli_presentation=_presentation(),
        record_trajectory=False,
        message_id="raw-marker-1",
    )
    prior_snapshot = store.messages_for_session(session.id)[-1]
    terminal = replace(
        started.raw_cli_presentation,
        lifecycle_state="exited",
        elapsed_seconds=1.25,
        exit_code=7,
        truncated=True,
        cleanup_proven=True,
    )

    updated = store.update_tool_marker(
        session.id,
        started.id,
        content="stdout:\n(first chunk)\n\nstderr:\n[red]literal[/red]",
        tool_output_full=(
            "stdout:\n(first chunk)\n\nstderr:\n[red]literal[/red]"
        ),
        raw_cli_presentation=terminal,
    )

    assert trajectory_calls == []
    assert updated.id == started.id == "raw-marker-1"
    assert updated.raw_cli_presentation == terminal
    assert prior_snapshot.content == "stdout:\n(first chunk)"
    assert prior_snapshot.raw_cli_presentation.lifecycle_state == "running"
    assert store.messages_for_session(session.id)[-1] == updated
    assert store.active_leaf(session.id) == anchor.id
    assert started.id not in store._nodes_by_session[session.id]
    stale_running = replace(
        terminal,
        lifecycle_state="running",
        exit_code=None,
        cleanup_proven=None,
    )
    with pytest.raises(ValueError, match="lifecycle"):
        store.update_tool_marker(
            session.id,
            started.id,
            raw_cli_presentation=stale_running,
        )
    with pytest.raises(ValueError, match="invocation"):
        store.update_tool_marker(
            session.id,
            started.id,
            raw_cli_presentation=replace(
                terminal,
                invocation_id="different-invocation",
            ),
        )
    assert (
        store.messages_for_session(session.id)[-1].raw_cli_presentation.lifecycle_state
        == "exited"
    )
    with pytest.raises(ValueError, match="bounded"):
        store.update_tool_marker(session.id, started.id, role=ConsoleMessageRole.USER)
    with pytest.raises(KeyError):
        store.update_tool_marker(session.id, "missing", content="no")


def _marker(presentation: RawCliPresentation) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="Command:\ntrue\n\nstdout:\n(no output)\n\nstderr:\n(no output)",
        id="raw-marker-1",
        tool_output_full="stdout:\n(no output)\n\nstderr:\n(no output)",
        activity_presentation=ConsoleActivityPresentation(
            "tool", "Raw CLI", "done"
        ),
        raw_cli_presentation=presentation,
    )


def test_raw_cli_actions_are_bounded_to_one_active_invocation() -> None:
    service = ConsoleMessageActionService()
    running = _marker(_presentation())

    starting = replace(
        running,
        raw_cli_presentation=replace(
            running.raw_cli_presentation,
            lifecycle_state="starting",
            started_at_monotonic=None,
        ),
    )
    assert [action.action_id for action in service.available_actions(starting)] == [
        "raw-cli-stop",
        "tool-output",
    ]
    assert service.dispatch("raw-cli-stop", starting).status == "completed"

    actions = service.available_actions(running)
    assert [(action.action_id, action.label, action.enabled) for action in actions] == [
        ("raw-cli-stop", "Stop", True),
        ("tool-output", "Full output", True),
    ]
    dispatched = service.dispatch("raw-cli-stop", running)
    assert dispatched.status == "completed"
    assert dispatched.target_message_id == running.id
    assert dispatched.target_invocation_id == "raw-1"

    stopping = replace(
        running,
        raw_cli_presentation=replace(
            running.raw_cli_presentation,
            lifecycle_state="stopping",
        ),
    )
    actions = service.available_actions(stopping)
    assert [(action.action_id, action.label, action.enabled) for action in actions] == [
        ("raw-cli-stop", "Stopping…", False),
        ("tool-output", "Full output", True),
    ]
    assert service.dispatch("raw-cli-stop", stopping).status == "blocked"

    terminal = replace(
        running,
        raw_cli_presentation=replace(
            running.raw_cli_presentation,
            lifecycle_state="cancelled",
            elapsed_seconds=0.5,
            cleanup_proven=True,
        ),
    )
    assert [
        action.action_id for action in service.available_actions(terminal)
    ] == ["tool-output"]
    assert service.dispatch("raw-cli-stop", terminal).status == "blocked"
    assert (
        service.dispatch("raw-cli-stop", replace(running, raw_cli_presentation=None)).status
        == "blocked"
    )


class _PreAdmissionExecutor:
    """Prove cancellation is visible before any process-tree admission."""

    def __init__(self) -> None:
        self.cancelled_before_admission = False
        self.worker_admitted = False
        self.shell_committed = False
        self.calls = 0

    def execute(
        self,
        request: RawCliRequest,
        *,
        cancel_event: threading.Event,
        on_event: Any,
        admit_worker: Any,
    ) -> RawCliResult:
        del on_event
        self.calls += 1
        self.cancelled_before_admission = cancel_event.is_set()
        owner = self

        class Tree:
            def admit(self) -> None:
                owner.worker_admitted = True

        admit_worker(Tree(), lambda: setattr(owner, "shell_committed", True))
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.0,
            stdout_preview="",
            stderr_preview="",
            record_output="",
            exit_code=None,
            terminal_state="cancelled",
            truncated=False,
            cleanup_proven=True,
        )


def test_runtime_registration_callback_closes_stop_before_admission_race(
    tmp_path: Path,
) -> None:
    executor = _PreAdmissionExecutor()
    runtime = RawCliRuntime(lambda: True, executor=executor)
    assert runtime.arm().armed is True
    request = RawCliRequest(
        invocation_id="raw-before-admission",
        caller="user",
        command="true",
        shell="auto",
        initial_directory=tmp_path,
        timeout_seconds=30.0,
        console_session_id="session-1",
    )
    cancel_results: list[bool] = []

    result = runtime.execute(
        request,
        lambda _event: None,
        on_registered=lambda: cancel_results.append(runtime.cancel(request.invocation_id)),
    )

    assert cancel_results == [True]
    assert executor.calls == 1
    assert executor.cancelled_before_admission is True
    assert executor.worker_admitted is False
    assert executor.shell_committed is False
    assert result.terminal_state == "cancelled"


def _raw_stash(command: str) -> ConsoleDraftStash:
    return ConsoleDraftStash(
        segments=[],
        text=f"! {command}",
        has_paste=False,
        raw_cli_prefix_typed=True,
    )


class _StreamingRuntime:
    permitted = True
    armed = True

    def __init__(self, *, after_registered: Any | None = None) -> None:
        self.after_registered = after_registered
        self.cancelled: list[str] = []

    def execute(
        self,
        request: RawCliRequest,
        on_event: Any,
        *,
        on_registered: Any,
        on_started: Any,
    ) -> RawCliResult:
        on_registered()
        if self.after_registered is not None:
            self.after_registered()
        on_started(25.0)
        on_event(
            RawCliStreamEvent(
                "stdout",
                "x" * 5_000,
                total_bytes=5_000,
                truncated=False,
            )
        )
        on_event(
            RawCliStreamEvent(
                "stderr",
                "[red]literal[/red]\x1b]8;;https://invalid.example\x07link",
                total_bytes=56,
                truncated=True,
            )
        )
        time.sleep(0.08)
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="/bin/bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=1.25,
            stdout_preview="x" * 5_000,
            stderr_preview=(
                "[red]literal[/red]\x1b]8;;https://invalid.example\x07link"
            ),
            record_output="bounded record",
            exit_code=7,
            terminal_state="exited",
            truncated=True,
            cleanup_proven=True,
        )

    def cancel(self, invocation_id: str) -> bool:
        self.cancelled.append(invocation_id)
        return True


def _controller_with_store(
    tmp_path: Path,
    runtime: Any,
    *,
    schedule_projection: Any,
) -> tuple[
    ConsoleRawCliController,
    ConsoleChatStore,
    str,
    list[Any],
    list[ConsoleChatMessage],
    list[ConsoleChatMessage],
]:
    store = ConsoleChatStore()
    session = store.create_session(title="raw")
    runs_db = AgentRunsDB(tmp_path / "agent-runs.db")
    workers: list[Any] = []
    appended: list[ConsoleChatMessage] = []
    updated: list[ConsoleChatMessage] = []

    def append_marker(*args: Any, **kwargs: Any) -> ConsoleChatMessage:
        marker = store.append_message(*args, **kwargs)
        appended.append(marker)
        return marker

    def update_marker(*args: Any, **kwargs: Any) -> ConsoleChatMessage:
        marker = store.update_tool_marker(*args, **kwargs)
        updated.append(marker)
        return marker

    controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: runtime,
        active_session_id=lambda: session.id,
        persist_session_if_needed=lambda _session_id: "conversation-1",
        active_leaf_anchor=lambda _session_id: None,
        persisted_leaf_anchor=lambda _session_id, _leaf_id: None,
        selected_local_root=lambda _session_id: tmp_path,
        private_scratch_root=lambda _session_id: tmp_path,
        refusal_stash_bank={},
        accepts_raw_cli_refusal_callbacks=lambda: True,
        restore_stash=lambda _session_id, _stash: True,
        append_local_error=lambda _session_id, _text: None,
        append_store_marker=append_marker,
        update_store_marker=update_marker,
        agent_runs_db=lambda: runs_db,
        run_log_access=lambda: tmp_path / "app-data",
        start_worker=lambda work, **_kwargs: workers.append(work),
        marshal_to_ui=lambda callback, *args: callback(*args),
        schedule_projection=schedule_projection,
    )
    return controller, store, session.id, workers, appended, updated


def test_controller_streams_one_stable_marker_and_never_loses_terminal_update(
    tmp_path: Path,
) -> None:
    visible = True
    projections: list[str] = []

    def schedule_projection(session_id: str) -> None:
        if visible:
            projections.append(session_id)

    def navigate_away() -> None:
        nonlocal visible
        visible = False

    runtime = _StreamingRuntime(after_registered=navigate_away)
    controller, store, session_id, workers, appended, updated = (
        _controller_with_store(
            tmp_path,
            runtime,
            schedule_projection=schedule_projection,
        )
    )

    assert controller.start_user_command(
        _raw_stash("printf '[bold]literal[/bold]\x1b]0;title\x07'")
    )
    workers[0]()

    marker = store.messages_for_session(session_id)[-1]
    assert len(appended) == 1
    assert appended[0].raw_cli_presentation.lifecycle_state == "starting"
    assert marker.id == appended[0].id
    assert {snapshot.id for snapshot in updated} == {marker.id}
    assert [
        snapshot.raw_cli_presentation.lifecycle_state for snapshot in updated
    ] == ["running", "running", "exited"]
    assert "x" * 100 in updated[-2].tool_output_full
    assert "[red]literal[/red]" in updated[-2].tool_output_full
    assert projections == [session_id], "away screens must not be repainted"
    assert marker.raw_cli_presentation.lifecycle_state == "exited"
    assert marker.raw_cli_presentation.caller == "user"
    assert marker.raw_cli_presentation.started_at_monotonic == 25.0
    assert marker.raw_cli_presentation.elapsed_seconds == 1.25
    assert marker.raw_cli_presentation.shell == "/bin/bash"
    assert marker.raw_cli_presentation.exit_code == 7
    assert marker.raw_cli_presentation.truncated is True
    assert marker.raw_cli_presentation.cleanup_proven is True
    assert "Command:" in marker.content
    assert "Caller: User" in marker.content
    assert "Shell: /bin/bash" in marker.content
    assert f"CWD: {tmp_path}" in marker.content
    assert "Exit code: 7" in marker.content
    assert "Truncated: Yes" in marker.content
    assert "Cleanup: Proven" in marker.content
    assert "\\x1b]0;title\\x07" in marker.content
    assert len(marker.content.encode("utf-8")) < len(
        marker.tool_output_full.encode("utf-8")
    )
    assert marker.tool_output_full.startswith("stdout:\n")
    assert "\n\nstderr:\n[red]literal[/red]" in marker.tool_output_full
    assert "\x1b" not in marker.tool_output_full


class _RegistrationBlockingRuntime(_StreamingRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.registered = threading.Event()
        self.release = threading.Event()

    def execute(
        self,
        request: RawCliRequest,
        on_event: Any,
        *,
        on_registered: Any,
        on_started: Any,
    ) -> RawCliResult:
        del on_event, on_started
        on_registered()
        self.registered.set()
        assert self.release.wait(2.0), "test did not release runtime admission"
        return RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="/bin/bash",
            initial_directory=request.initial_directory,
            elapsed_seconds=0.1,
            stdout_preview="",
            stderr_preview="",
            record_output="",
            exit_code=None,
            terminal_state="cancelled",
            truncated=False,
            cleanup_proven=True,
        )


def test_marker_stop_before_runtime_admission_cancels_once_and_disables(
    tmp_path: Path,
) -> None:
    runtime = _RegistrationBlockingRuntime()
    controller, store, session_id, workers, _appended, _updated = (
        _controller_with_store(
            tmp_path,
            runtime,
            schedule_projection=lambda _session_id: None,
        )
    )
    assert controller.start_user_command(_raw_stash("sleep 30"))
    worker = threading.Thread(target=workers[0])
    worker.start()
    assert runtime.registered.wait(2.0), "marker was not published after registration"
    marker = store.messages_for_session(session_id)[-1]
    assert marker.raw_cli_presentation.lifecycle_state == "starting"
    assert marker.raw_cli_presentation.started_at_monotonic is None

    assert controller.stop_user_command(marker) is True
    stopping = store.messages_for_session(session_id)[-1]
    assert stopping.raw_cli_presentation.lifecycle_state == "stopping"
    assert stopping.raw_cli_presentation.started_at_monotonic is None
    assert (
        assistant_turn_module.raw_cli_status_copy(
            stopping.raw_cli_presentation,
            now=10_000.0,
        )
        == "Stopping… · 0.0s"
    )
    assert controller.stop_user_command(stopping) is False
    assert runtime.cancelled == [marker.raw_cli_presentation.invocation_id]

    runtime.release.set()
    worker.join(2.0)
    assert not worker.is_alive()
    terminal = store.messages_for_session(session_id)[-1]
    assert terminal.id == marker.id
    assert terminal.raw_cli_presentation.lifecycle_state == "cancelled"


def test_stale_running_updates_do_not_regress_stopping_or_terminal(
    tmp_path: Path,
) -> None:
    projections: list[str] = []
    runtime = _StreamingRuntime()
    controller, store, session_id, _workers, _appended, _updated = (
        _controller_with_store(
            tmp_path,
            runtime,
            schedule_projection=projections.append,
        )
    )
    request = RawCliRequest(
        invocation_id="raw-cas",
        caller="user",
        command="sleep 30",
        shell="auto",
        initial_directory=tmp_path,
        timeout_seconds=30.0,
        console_session_id=session_id,
    )
    controller._append_starting_marker(request, session_id)
    marker = store.messages_for_session(session_id)[-1]
    assert controller.stop_user_command(marker) is True

    projections.clear()
    controller._update_running_marker(
        request,
        session_id,
        25.0,
        "stale stdout",
        "",
        False,
    )
    assert projections == []
    assert (
        store.messages_for_session(session_id)[-1].raw_cli_presentation.lifecycle_state
        == "stopping"
    )

    controller._finish_marker(
        request,
        session_id,
        25.0,
        RawCliResult(
            invocation_id=request.invocation_id,
            caller=request.caller,
            resolved_shell="/bin/bash",
            initial_directory=tmp_path,
            elapsed_seconds=1.0,
            stdout_preview="terminal stdout",
            stderr_preview="",
            record_output="",
            exit_code=None,
            terminal_state="cancelled",
            truncated=False,
            cleanup_proven=True,
        ),
    )
    projections.clear()
    controller._update_running_marker(
        request,
        session_id,
        25.0,
        "later stale stdout",
        "",
        False,
    )
    assert projections == []
    terminal = store.messages_for_session(session_id)[-1]
    assert terminal.raw_cli_presentation.lifecycle_state == "cancelled"
    assert "terminal stdout" in terminal.tool_output_full


class _ExceptionAfterTruncationRuntime(_StreamingRuntime):
    def execute(
        self,
        request: RawCliRequest,
        on_event: Any,
        *,
        on_registered: Any,
        on_started: Any,
    ) -> RawCliResult:
        del request
        on_registered()
        on_started(25.0)
        on_event(
            RawCliStreamEvent(
                "stdout",
                "x" * (MAX_RAW_PREVIEW_BYTES + 1),
                total_bytes=MAX_RAW_PREVIEW_BYTES + 1,
                truncated=False,
            )
        )
        raise RuntimeError("runtime failed after bounded output")


def test_runtime_exception_preserves_stream_truncation_on_failed_marker(
    tmp_path: Path,
) -> None:
    controller, store, session_id, workers, _appended, _updated = (
        _controller_with_store(
            tmp_path,
            _ExceptionAfterTruncationRuntime(),
            schedule_projection=lambda _session_id: None,
        )
    )

    assert controller.start_user_command(_raw_stash("printf flood")) is True
    workers[0]()

    marker = store.messages_for_session(session_id)[-1]
    assert marker.raw_cli_presentation.lifecycle_state == "failed"
    assert marker.raw_cli_presentation.truncated is True
    assert "Truncated: Yes" in marker.content


@pytest.mark.asyncio
async def test_raw_cli_projection_coalesces_to_one_worker_with_trailing_refresh() -> None:
    schedule = getattr(wiring_module, "_schedule_raw_cli_projection", None)
    assert schedule is not None, "bounded raw CLI projection scheduler is missing"

    class Screen:
        def __init__(self) -> None:
            self.app = SimpleNamespace(screen=self)
            self._closing = False
            self._closed = False
            self.workers: list[tuple[Any, dict[str, Any]]] = []
            self.sync_calls = 0
            self.first_sync_started = asyncio.Event()
            self.release_first_sync = asyncio.Event()

        def _ensure_console_chat_store(self) -> Any:
            return SimpleNamespace(
                active_session_id="session-1",
                ensure_session=lambda: SimpleNamespace(id="session-1"),
            )

        async def _sync_native_console_chat_ui(self) -> None:
            self.sync_calls += 1
            if self.sync_calls == 1:
                self.first_sync_started.set()
                await self.release_first_sync.wait()

        def run_worker(self, work: Any, **kwargs: Any) -> None:
            self.workers.append((work, kwargs))

    screen = Screen()
    screen.app.screen = object()
    schedule(screen, "session-1")
    assert screen.workers == []

    screen.app.screen = screen
    schedule(screen, "other-session")
    assert screen.workers == []

    schedule(screen, "session-1")
    assert len(screen.workers) == 1
    work, options = screen.workers[0]
    assert options == {
        "group": "console-raw-cli-projection",
        "exit_on_error": False,
    }
    task = asyncio.create_task(work)
    await screen.first_sync_started.wait()
    schedule(screen, "session-1")
    schedule(screen, "session-1")
    assert len(screen.workers) == 1
    screen.release_first_sync.set()
    await task
    assert screen.sync_calls == 2

    schedule(screen, "session-1")
    assert len(screen.workers) == 2
    screen.workers[-1][0].close()

    screen._closing = True
    schedule(screen, "session-1")
    assert len(screen.workers) == 2

    class FailingScreen(Screen):
        def run_worker(self, work: Any, **kwargs: Any) -> None:
            del work, kwargs
            raise RuntimeError("worker admission failed")

    failing = FailingScreen()
    schedule(failing, "session-1")
    assert failing._raw_cli_projection_in_flight is False
    assert failing._raw_cli_projection_dirty is False


def test_raw_cli_callbacks_after_session_removal_are_teardown_safe(
    tmp_path: Path,
) -> None:
    controller: ConsoleRawCliController
    store: ConsoleChatStore
    session_id: str

    def remove_session() -> None:
        store.close_session(session_id)

    runtime = _StreamingRuntime(after_registered=remove_session)
    controller, store, session_id, workers, _appended, _updated = (
        _controller_with_store(
            tmp_path,
            runtime,
            schedule_projection=lambda _session_id: None,
        )
    )

    assert controller.start_user_command(_raw_stash("printf teardown")) is True
    workers[0]()
    assert store.sessions() == []


def test_initial_raw_marker_append_failure_prevents_launch(tmp_path: Path) -> None:
    class Runtime:
        permitted = True
        armed = True

        def __init__(self) -> None:
            self.launched = False

        def execute(
            self,
            request: RawCliRequest,
            on_event: Any,
            *,
            on_registered: Any,
            on_started: Any,
        ) -> RawCliResult:
            del on_event, on_started
            on_registered()
            self.launched = True
            raise AssertionError("launch passed failed marker append")

    runtime = Runtime()
    controller, _store, _session_id, workers, _appended, _updated = (
        _controller_with_store(
            tmp_path,
            runtime,
            schedule_projection=lambda _session_id: None,
        )
    )
    controller._append_store_marker = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("append failed")
    )

    assert controller.start_user_command(_raw_stash("true")) is True
    workers[0]()
    assert runtime.launched is False


@pytest.mark.asyncio
async def test_chat_screen_routes_only_the_bounded_raw_stop_action() -> None:
    marker = _marker(_presentation())
    stopped: list[ConsoleChatMessage] = []

    class Event:
        def __init__(self) -> None:
            self.button = SimpleNamespace(
                id=f"console-message-action-raw-cli-stop-{marker.id}",
                console_action_id="raw-cli-stop",
                console_message_id=marker.id,
            )
            self.was_stopped = False

        def stop(self) -> None:
            self.was_stopped = True

    event = Event()
    screen = SimpleNamespace(
        _raw_cli=SimpleNamespace(
            stop_user_command=lambda message: stopped.append(message) or True
        ),
        _ensure_console_chat_store=lambda: SimpleNamespace(
            active_session_id="session-1",
            messages_for_session=lambda _session_id: (marker,),
        ),
        _message=SimpleNamespace(
            handle_console_message_action=lambda _event: pytest.fail(
                "raw CLI Stop reached the generic message action controller"
            )
        ),
    )

    handled = await ChatScreen.handle_console_message_action(screen, event)

    assert handled is True
    assert event.was_stopped is True
    assert stopped == [marker]


def test_raw_cli_status_copy_distinguishes_every_required_terminal_state() -> None:
    status_copy = getattr(assistant_turn_module, "raw_cli_status_copy", None)
    assert status_copy is not None, "raw CLI lifecycle copy helper is missing"
    assert (
        status_copy(
            _presentation(
                lifecycle_state="starting",
                started_at_monotonic=None,
            ),
            now=10.5,
        )
        == "Starting · 0.0s"
    )
    assert status_copy(_presentation(), now=10.5) == "Running · 0.5s"
    assert (
        status_copy(
            _presentation(lifecycle_state="stopping"),
            now=10.5,
        )
        == "Stopping… · 0.5s"
    )
    assert "Stopped" in status_copy(
        _presentation(
            lifecycle_state="cancelled",
            elapsed_seconds=1.0,
            cleanup_proven=True,
        )
    )
    assert "Timed out" in status_copy(
        _presentation(
            lifecycle_state="timed_out",
            elapsed_seconds=1.0,
            cleanup_proven=True,
        )
    )
    assert "Cleanup unproven" in status_copy(
        _presentation(
            lifecycle_state="cancelled",
            elapsed_seconds=1.0,
            cleanup_proven=False,
        )
    )


class RawCliTranscriptHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


class RawCliActivityHeaderHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield assistant_turn_module.ConsoleActivityHeader(
            "activity-prelaunch",
            "Raw CLI",
            "done",
            raw_cli_presentation=_presentation(
                lifecycle_state="stopping",
                started_at_monotonic=None,
            ),
        )
        yield assistant_turn_module.ConsoleActivityHeader(
            "activity-terminal",
            "Raw CLI",
            "done",
            raw_cli_presentation=_presentation(
                lifecycle_state="cancelled",
                elapsed_seconds=1.0,
                cleanup_proven=True,
            ),
        )
        yield assistant_turn_module.ConsoleActivityHeader(
            "activity-running",
            "Raw CLI",
            "done",
            raw_cli_presentation=_presentation(),
        )


def _rendered_static_text(widget: Any, selector: str) -> str:
    return str(widget.query_one(selector, Static).renderable)


@pytest.mark.asyncio
async def test_mounted_prelaunch_and_terminal_rows_hold_no_elapsed_timer() -> None:
    app = RawCliTranscriptHarness()
    stopping = _marker(
        _presentation(
            lifecycle_state="stopping",
            started_at_monotonic=None,
            elapsed_seconds=0.0,
        )
    )
    terminal = replace(
        _marker(
            _presentation(
                lifecycle_state="cancelled",
                elapsed_seconds=1.0,
                cleanup_proven=True,
            )
        ),
        id="raw-terminal-marker",
    )

    async with app.run_test(size=(110, 24)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([stopping, terminal])
        await transcript.refresh_messages()
        await pilot.pause()

        header_selector = (
            f"#console-message-header-{stopping.id} "
            ".console-transcript-speaker-label"
        )
        before = _rendered_static_text(transcript, header_selector)
        assert "Stopping… · 0.0s" in before
        message_header = transcript.query_one(
            f"#console-message-header-{stopping.id}"
        )
        terminal_header = transcript.query_one(
            f"#console-message-header-{terminal.id}"
        )
        assert message_header._raw_cli_elapsed_timer is None
        assert terminal_header._raw_cli_elapsed_timer is None

        await pilot.pause(0.2)
        assert _rendered_static_text(transcript, header_selector) == before


@pytest.mark.asyncio
async def test_activity_headers_allocate_only_active_elapsed_timer() -> None:
    app = RawCliActivityHeaderHarness()

    async with app.run_test(size=(80, 12)) as pilot:
        prelaunch = app.query_one("#console-activity-header-activity-prelaunch")
        terminal = app.query_one("#console-activity-header-activity-terminal")
        running = app.query_one("#console-activity-header-activity-running")
        assert prelaunch._raw_cli_elapsed_timer is None
        assert terminal._raw_cli_elapsed_timer is None
        running_timer = running._raw_cli_elapsed_timer
        assert running_timer is not None

        running.sync_header(
            "Raw CLI",
            "done",
            expanded=False,
            expandable=False,
            selected=False,
            raw_cli_presentation=_presentation(
                lifecycle_state="exited",
                elapsed_seconds=1.0,
                exit_code=0,
                cleanup_proven=True,
            ),
        )
        await pilot.pause()
        assert running._raw_cli_elapsed_timer is None
        assert running_timer._task is None


@pytest.mark.asyncio
async def test_mounted_raw_cli_row_is_literal_focusable_and_lifecycle_complete() -> (
    None
):
    app = RawCliTranscriptHarness()
    started_at = monotonic() - 1.0
    running = _marker(
        _presentation(
            command="printf '[bold]literal[/bold]\\x1b]0;title\\x07'",
            cwd="/tmp/[link=https://invalid.example]literal[/link]",
            started_at_monotonic=started_at,
        )
    )
    running = replace(
        running,
        content=(
            "Command:\nprintf '[bold]literal[/bold]\\x1b]0;title\\x07\n\n"
            "Caller: User\n"
            "Shell: /bin/bash\n"
            "CWD: /tmp/[link=https://invalid.example]literal[/link]\n"
            "Elapsed: 0.0s\nExit code: Pending\nTruncated: No\n"
            "Cleanup: Pending\n\n"
            "stdout:\n[bold]literal[/bold]\n\nstderr:\n[red]literal[/red]"
        ),
        tool_output_full=(
            "stdout:\n[bold]literal[/bold]\n\nstderr:\n[red]literal[/red]"
        ),
    )

    async with app.run_test(size=(110, 36)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([running])
        transcript.select_message(running.id)
        await transcript.refresh_messages()
        await pilot.pause()

        header_selector = (
            f"#console-message-header-{running.id} "
            ".console-transcript-speaker-label"
        )
        before = _rendered_static_text(transcript, header_selector)
        assert "Running" in before
        message_header = transcript.query_one(
            f"#console-message-header-{running.id}"
        )
        message_timer = message_header._raw_cli_elapsed_timer
        assert message_timer is not None
        stop = transcript.query_one(
            f"#console-message-action-raw-cli-stop-{running.id}", Button
        )
        assert stop.disabled is False
        assert stop.can_focus is True
        for _attempt in range(20):
            await pilot.press("tab")
            await pilot.pause()
            if stop.has_focus:
                break
        assert stop.has_focus is True

        body = transcript.query_one(
            f"#console-message-{running.id} .console-transcript-message-body",
            Static,
        )
        body_text = str(body.renderable)
        assert "Caller: User" in body_text
        assert "[bold]literal[/bold]" in body_text
        assert "[red]literal[/red]" in body_text
        assert "[link=https://invalid.example]literal[/link]" in body_text
        await pilot.pause(0.2)
        after = _rendered_static_text(transcript, header_selector)
        assert after != before, "the mounted command row must own its elapsed timer"

        stopping = replace(
            running,
            raw_cli_presentation=replace(
                running.raw_cli_presentation,
                lifecycle_state="stopping",
                elapsed_seconds=1.2,
            ),
        )
        transcript.set_messages([stopping])
        await transcript.refresh_messages()
        await pilot.pause()
        stop = transcript.query_one(
            f"#console-message-action-raw-cli-stop-{running.id}", Button
        )
        assert stop.disabled is True
        assert str(stop.label) == "Stopping…"
        assert "Stopping…" in _rendered_static_text(transcript, header_selector)

        timed_out = replace(
            stopping,
            content=stopping.content.replace(
                "Exit code: Pending\nTruncated: No\nCleanup: Pending",
                "Exit code: Pending\nTruncated: Yes\nCleanup: Unproven",
            ),
            raw_cli_presentation=replace(
                stopping.raw_cli_presentation,
                lifecycle_state="timed_out",
                elapsed_seconds=2.0,
                truncated=True,
                cleanup_proven=False,
            ),
        )
        transcript.set_messages([timed_out])
        await transcript.refresh_messages()
        await pilot.pause()

        assert not list(
            transcript.query(
                f"#console-message-action-raw-cli-stop-{running.id}"
            )
        )
        terminal_header = _rendered_static_text(transcript, header_selector)
        assert "Timed out" in terminal_header
        assert "Cleanup unproven" in terminal_header
        assert message_header._raw_cli_elapsed_timer is None
        assert message_timer._task is None
        terminal_text = transcript.to_plain_text()
        for expected in (
            "Command:",
            "Caller: User",
            "Shell: /bin/bash",
            "CWD: /tmp/",
            "Elapsed:",
            "Exit code: Pending",
            "Truncated: Yes",
            "Cleanup: Unproven",
            "stdout:",
            "stderr:",
        ):
            assert expected in terminal_text
