from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Terminal.contracts import (
    TERMINAL_DISCLOSURE_LINES,
    TerminalLaunchRequest,
    TerminalLifecycle,
    TerminalProjection,
    TerminalReason,
)
from tldw_chatbook.Terminal.launch import ShellChoice, discover_shell_choices
from tldw_chatbook.Terminal.screen_model import (
    SafeTerminalCell,
    SafeTerminalLine,
    SafeTerminalRun,
    SafeTerminalStyle,
    TerminalScreenSnapshot,
)
from tldw_chatbook.Terminal.session_manager import (
    TerminalArmResult,
    TerminalCreateResult,
    TerminalSessionView,
    TerminalSubscriptionToken,
    TerminalViewState,
    TerminalViewToken,
)
from tldw_chatbook.UI.Console_Modules.terminal import ConsoleTerminalController
from tldw_chatbook.UI.Screens import settings_screen
from tldw_chatbook.Widgets.Console.console_terminal_workspace import (
    ConsoleTerminalWorkspace,
    TerminalViewport,
)
from tldw_chatbook.Widgets.Console.console_terminal_session_modal import (
    TerminalSessionFormResult,
)


def _line(
    text: str,
    *,
    style: SafeTerminalStyle = SafeTerminalStyle(),
) -> SafeTerminalLine:
    return SafeTerminalLine(
        runs=(
            SafeTerminalRun(
                cells=tuple(SafeTerminalCell(character, 1) for character in text),
                style=style,
            ),
        )
    )


def _shell_choices() -> tuple[ShellChoice, ...]:
    paths = {"bash": "/bin/bash", "zsh": "/bin/zsh", "sh": "/bin/sh"}
    return discover_shell_choices(
        platform_name="posix",
        account_shell="/bin/zsh",
        executable_lookup=paths.get,
        executable_is_file=lambda _path: True,
    )


def _session(
    session_id: str,
    name: str,
    *,
    lifecycle: TerminalLifecycle = TerminalLifecycle.RUNNING,
    generation: int = 1,
    lines: tuple[SafeTerminalLine, ...] | None = None,
    cleanup_receipt: Any = None,
) -> TerminalSessionView:
    return TerminalSessionView(
        projection=TerminalProjection(
            session_id=session_id,
            name=name,
            lifecycle=lifecycle,
            reason=(
                TerminalReason.CLEANUP_UNPROVEN
                if lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
                else None
            ),
        ),
        screen=TerminalScreenSnapshot(
            lines=lines or (_line("prompt> "),),
            generation=generation,
            dirty_lines=(0,),
        ),
        shell="default",
        start_directory="/work/project",
        columns=80,
        rows=24,
        cleanup_receipt=cleanup_receipt,
    )


class _WorkspaceApp(ConsolidatedCSSApp):
    def __init__(self, workspace: ConsoleTerminalWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        yield self.workspace


@pytest.mark.asyncio
async def test_workspace_renders_locked_unarmed_and_armed_authority_states() -> None:
    workspace = ConsoleTerminalWorkspace()
    app = _WorkspaceApp(workspace)

    async with app.run_test(size=(120, 40)) as pilot:
        workspace.project(
            permitted=False,
            armed=False,
            view_state=TerminalViewState(),
        )
        await pilot.pause()
        assert (
            "locked"
            in str(
                workspace.query_one("#console-terminal-access", Static).renderable
            ).lower()
        )
        assert (
            workspace.query_one("#console-terminal-open-settings", Button).display
            is True
        )
        assert workspace.query_one("#console-terminal-arm", Button).display is False
        assert workspace.query_one("#console-terminal-danger", Static).display is False

        workspace.project(
            permitted=True,
            armed=False,
            view_state=TerminalViewState(),
        )
        await pilot.pause()
        assert (
            "not armed"
            in str(
                workspace.query_one("#console-terminal-access", Static).renderable
            ).lower()
        )
        assert workspace.query_one("#console-terminal-arm", Button).display is True

        workspace.project(
            permitted=True,
            armed=True,
            view_state=TerminalViewState(
                selected_session_id="one",
                sessions=(_session("one", "Build shell"),),
            ),
        )
        await pilot.pause()
        danger = workspace.query_one("#console-terminal-danger", Static)
        assert danger.display is True
        assert "HOST TERMINAL - FULL USER ACCESS" in str(danger.renderable)
        metadata = str(
            workspace.query_one("#console-terminal-metadata", Static).renderable
        )
        assert "Build shell" in metadata
        assert "running" in metadata
        assert "default" in metadata
        assert "/work/project" in metadata
        assert "80×24" in metadata


@pytest.mark.asyncio
async def test_workspace_renders_four_records_and_state_specific_actions() -> None:
    workspace = ConsoleTerminalWorkspace()
    sessions = (
        _session("one", "One"),
        _session("two", "Two", lifecycle=TerminalLifecycle.EXITED),
        _session(
            "three",
            "Three",
            lifecycle=TerminalLifecycle.CLEANUP_UNPROVEN,
            cleanup_receipt=SimpleNamespace(action="retry"),
        ),
        _session("four", "Four", lifecycle=TerminalLifecycle.DRAINING),
    )
    workspace.project(
        permitted=True,
        armed=True,
        view_state=TerminalViewState(selected_session_id="one", sessions=sessions),
    )
    app = _WorkspaceApp(workspace)

    async with app.run_test(size=(140, 44)) as pilot:
        await pilot.pause()
        labels = [
            str(workspace.query_one(f"#console-terminal-session-{index}", Button).label)
            for index in range(4)
        ]
        assert labels == ["One", "Two", "Three", "Four"]
        assert workspace.query_one("#console-terminal-new", Button).disabled is True
        assert workspace.query_one("#console-terminal-close", Button).display is True
        assert workspace.query_one("#console-terminal-retry", Button).display is False

        workspace.project(
            permitted=True,
            armed=True,
            view_state=replace(
                TerminalViewState(selected_session_id="one", sessions=sessions),
                selected_session_id="three",
            ),
        )
        await pilot.pause()
        assert workspace.query_one("#console-terminal-close", Button).display is False
        assert workspace.query_one("#console-terminal-retry", Button).display is True


@pytest.mark.parametrize("permitted", [False, True])
@pytest.mark.asyncio
async def test_cleanup_receipt_stays_actionable_while_locked_or_unarmed(
    permitted: bool,
) -> None:
    cleanup = _session(
        "cleanup",
        "Cleanup required",
        lifecycle=TerminalLifecycle.CLEANUP_UNPROVEN,
        cleanup_receipt=SimpleNamespace(action="retry"),
    )
    workspace = ConsoleTerminalWorkspace()
    workspace.project(
        permitted=permitted,
        armed=False,
        view_state=TerminalViewState(
            selected_session_id="cleanup",
            sessions=(cleanup,),
        ),
    )
    app = _WorkspaceApp(workspace)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        session = workspace.query_one("#console-terminal-session-0", Button)
        assert session.display is True
        assert str(session.label) == "Cleanup required"
        assert workspace.query_one("#console-terminal-retry", Button).display is True
        assert workspace.query_one("#console-terminal-new", Button).display is False
        assert workspace.query_one("#console-terminal-rename", Button).display is False
        assert workspace.query_one("#console-terminal-focus", Button).display is False
        assert workspace.query_one("#console-terminal-close", Button).display is False
        assert (
            workspace.query_one("#console-terminal-jump-live", Button).display is False
        )
        metadata = str(
            workspace.query_one("#console-terminal-metadata", Static).renderable
        )
        assert "cleanup_unproven" in metadata


@pytest.mark.asyncio
async def test_workspace_discloses_when_visible_allocation_is_clamped() -> None:
    selected = _session("one", "One")
    workspace = ConsoleTerminalWorkspace()
    workspace.project(
        permitted=True,
        armed=True,
        view_state=TerminalViewState(
            selected_session_id="one",
            sessions=(selected,),
        ),
    )
    app = _WorkspaceApp(workspace)

    async with app.run_test(size=(400, 180)) as pilot:
        viewport = workspace.query_one("#console-terminal-viewport", TerminalViewport)
        viewport.styles.width = 360
        viewport.styles.height = 130
        await pilot.pause()
        workspace.project(
            permitted=True,
            armed=True,
            view_state=TerminalViewState(
                selected_session_id="one",
                sessions=(selected,),
            ),
        )

        assert workspace.terminal_size() == (300, 120)
        metadata = str(
            workspace.query_one("#console-terminal-metadata", Static).renderable
        )
        assert "viewport capped at 300×120" in metadata


@pytest.mark.asyncio
async def test_viewport_receives_only_literal_safe_cells_not_rich_markup() -> None:
    style = SafeTerminalStyle(
        fg="red",
        bg="default",
        bold=True,
        underscore=True,
    )
    literal = "[bold red]not markup[/]"
    workspace = ConsoleTerminalWorkspace()
    workspace.project(
        permitted=True,
        armed=True,
        view_state=TerminalViewState(
            selected_session_id="one",
            sessions=(_session("one", "One", lines=(_line(literal, style=style),)),),
        ),
    )
    app = _WorkspaceApp(workspace)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        viewport = workspace.query_one("#console-terminal-viewport", TerminalViewport)
        assert isinstance(viewport.renderable, Text)
        assert viewport.renderable.plain == literal
        assert "backend" not in vars(workspace)
        assert all(
            not isinstance(value, (bytes, bytearray, memoryview))
            for value in vars(workspace).values()
        )


class _ProjectionSink:
    def __init__(self) -> None:
        self.projected: list[tuple[bool, bool, TerminalViewState]] = []
        self.jump_live_calls = 0
        self.focus_calls = 0
        self.statuses: list[str] = []

    def project(
        self,
        *,
        permitted: bool,
        armed: bool,
        view_state: TerminalViewState,
    ) -> None:
        self.projected.append((permitted, armed, view_state))

    def terminal_size(self) -> tuple[int, int]:
        return (80, 24)

    def jump_live(self) -> None:
        self.jump_live_calls += 1

    def focus_terminal(self) -> None:
        self.focus_calls += 1

    def set_status(self, message: str) -> None:
        self.statuses.append(message)


class _Runtime:
    def __init__(self, state: TerminalViewState) -> None:
        self.permitted = True
        self.armed = False
        self.disclosure_acknowledged = False
        self.state = state
        self._generation = 0
        self.callbacks: dict[TerminalSubscriptionToken, Callable[[], None]] = {}
        self.created: list[TerminalLaunchRequest] = []
        self.closed: list[str] = []
        self.renamed: list[tuple[str, str]] = []
        self.resized: list[tuple[str, int, int, TerminalViewToken]] = []
        self.applied_resizes: list[tuple[str, TerminalViewToken]] = []
        self.focused: list[str] = []
        self.retried: list[str] = []
        self.keys: list[tuple[str, bytes]] = []
        self.pastes: list[tuple[str, str, bool]] = []

    def arm(self, *, acknowledge_disclosure: bool = False) -> TerminalArmResult:
        if not self.permitted:
            self.armed = False
            return TerminalArmResult(reason=TerminalReason.LOCKED)
        if not self.disclosure_acknowledged and not acknowledge_disclosure:
            return TerminalArmResult(disclosure_required=True)
        self.disclosure_acknowledged = True
        self.armed = True
        return TerminalArmResult(armed=True)

    def attach_view(self) -> TerminalViewToken:
        self._generation += 1
        return TerminalViewToken(self._generation)

    def detach_view(self, _view: TerminalViewToken) -> bool:
        return True

    def subscribe(self, callback: Callable[[], None]) -> TerminalSubscriptionToken:
        token = TerminalSubscriptionToken(len(self.callbacks) + 1)
        self.callbacks[token] = callback
        return token

    def unsubscribe(self, token: TerminalSubscriptionToken) -> bool:
        return self.callbacks.pop(token, None) is not None

    def view_state(self, _view: TerminalViewToken) -> TerminalViewState:
        return self.state

    def projections(self) -> tuple[TerminalProjection, ...]:
        return tuple(session.projection for session in self.state.sessions)

    def projection(self, session_id: str) -> TerminalProjection | None:
        return next(
            (
                session.projection
                for session in self.state.sessions
                if session.projection.session_id == session_id
            ),
            None,
        )

    def create_session(self, request: TerminalLaunchRequest) -> TerminalCreateResult:
        self.created.append(request)
        return TerminalCreateResult(admitted=True)

    def close_session(self, session_id: str, *, view: TerminalViewToken) -> object:
        del view
        self.closed.append(session_id)
        return object()

    def rename_session(
        self, session_id: str, name: str, *, view: TerminalViewToken
    ) -> bool:
        del view
        self.renamed.append((session_id, name))
        return True

    def resize_session(
        self,
        session_id: str,
        *,
        columns: int,
        rows: int,
        view: TerminalViewToken,
    ) -> bool:
        self.resized.append((session_id, columns, rows, view))
        return True

    async def apply_pending_resize(
        self,
        session_id: str,
        *,
        view: TerminalViewToken,
    ) -> bool:
        self.applied_resizes.append((session_id, view))
        return True

    def focus_session(self, session_id: str, *, view: TerminalViewToken) -> bool:
        del view
        self.focused.append(session_id)
        self.state = replace(self.state, selected_session_id=session_id)
        return True

    def retry_cleanup(self, session_id: str, *, view: TerminalViewToken) -> object:
        del view
        self.retried.append(session_id)
        return object()

    def send_key(
        self,
        session_id: str,
        data: bytes,
        *,
        view: TerminalViewToken,
    ) -> object:
        del view
        self.keys.append((session_id, data))
        return SimpleNamespace(accepted=True)

    def send_paste(
        self,
        session_id: str,
        text: str,
        *,
        bracketed: bool,
        view: TerminalViewToken,
    ) -> object:
        del view
        self.pastes.append((session_id, text, bracketed))
        return SimpleNamespace(accepted=True)

    def emit(self) -> None:
        for callback in tuple(self.callbacks.values()):
            callback()


def _controller(
    runtime: _Runtime,
    sink: _ProjectionSink,
    *,
    scheduled: list[Callable[[], None]] | None = None,
    marshalled: list[Callable[[], None]] | None = None,
    confirmations: list[tuple[str, str]] | None = None,
    confirmation_result: bool = True,
    modal_result: TerminalSessionFormResult | None = None,
    selected_root: Callable[[], Path | None] = lambda: Path.cwd(),
    account_home: Callable[[], Path] = Path.home,
    settings_calls: list[None] | None = None,
    async_runs: list[Callable[[], Awaitable[Any]]] | None = None,
    shell_choices: Callable[[], tuple[ShellChoice, ...]] = lambda: (),
) -> ConsoleTerminalController:
    scheduled = [] if scheduled is None else scheduled
    marshalled = [] if marshalled is None else marshalled
    confirmations = [] if confirmations is None else confirmations
    settings_calls = [] if settings_calls is None else settings_calls
    async_runs = [] if async_runs is None else async_runs

    async def confirm(title: str, message: str) -> bool:
        confirmations.append((title, message))
        return confirmation_result

    async def present_modal(_modal: object) -> TerminalSessionFormResult | None:
        return modal_result

    return ConsoleTerminalController(
        terminal_runtime=lambda: runtime,
        workspace_accessor=lambda: sink,
        selected_local_root=selected_root,
        account_home=account_home,
        open_privacy_settings=lambda: settings_calls.append(None),
        confirm=confirm,
        present_session_modal=present_modal,
        marshal_to_ui=marshalled.append,
        schedule_frame=scheduled.append,
        shell_choices=shell_choices,
        run_async=async_runs.append,
    )


def _run_one(callbacks: list[Callable[[], None]]) -> None:
    callbacks.pop(0)()


def test_controller_coalesces_selected_repaint_and_ignores_hidden_output() -> None:
    selected = _session("one", "One", generation=1)
    hidden = _session("two", "Two", generation=1)
    runtime = _Runtime(
        TerminalViewState(selected_session_id="one", sessions=(selected, hidden))
    )
    sink = _ProjectionSink()
    scheduled: list[Callable[[], None]] = []
    marshalled: list[Callable[[], None]] = []
    controller = _controller(
        runtime,
        sink,
        scheduled=scheduled,
        marshalled=marshalled,
    )

    controller.open_workspace()
    _run_one(scheduled)
    assert len(sink.projected) == 1

    runtime.state = replace(
        runtime.state,
        sessions=(
            selected,
            replace(hidden, screen=replace(hidden.screen, generation=2)),
        ),
    )
    runtime.emit()
    _run_one(marshalled)
    _run_one(scheduled)
    assert len(sink.projected) == 1

    runtime.state = replace(
        runtime.state,
        sessions=(
            replace(selected, screen=replace(selected.screen, generation=2)),
            hidden,
        ),
    )
    runtime.emit()
    runtime.emit()
    assert len(marshalled) == 2
    _run_one(marshalled)
    _run_one(marshalled)
    assert len(scheduled) == 1
    _run_one(scheduled)
    assert len(sink.projected) == 2


def test_controller_drops_stale_generation_callbacks_after_detach_and_remount() -> None:
    runtime = _Runtime(
        TerminalViewState(
            selected_session_id="one",
            sessions=(_session("one", "One"),),
        )
    )
    sink = _ProjectionSink()
    scheduled: list[Callable[[], None]] = []
    marshalled: list[Callable[[], None]] = []
    controller = _controller(
        runtime,
        sink,
        scheduled=scheduled,
        marshalled=marshalled,
    )

    controller.open_workspace()
    stale_refresh = scheduled.pop()
    controller.detach_workspace()
    controller.open_workspace()
    stale_refresh()
    assert sink.projected == []
    _run_one(scheduled)
    assert len(sink.projected) == 1

    old_callback = next(iter(runtime.callbacks.values()))
    controller.detach_workspace()
    old_callback()
    assert marshalled == []


def test_controller_fails_closed_when_subscribe_and_cleanup_raise() -> None:
    class BrokenSubscriptionRuntime(_Runtime):
        def subscribe(self, callback: Callable[[], None]) -> TerminalSubscriptionToken:
            del callback
            raise RuntimeError("subscribe failed")

        def detach_view(self, view: TerminalViewToken) -> bool:
            del view
            raise RuntimeError("cleanup failed")

    runtime = BrokenSubscriptionRuntime(TerminalViewState())
    sink = _ProjectionSink()
    controller = _controller(runtime, sink)

    assert controller.open_workspace() is False
    assert controller.is_open is False
    assert sink.statuses == ["Terminal view is unavailable."]


@pytest.mark.asyncio
async def test_controller_resizes_once_on_remount_and_selected_session_change() -> None:
    first = _session("one", "One")
    second = _session("two", "Two")
    runtime = _Runtime(
        TerminalViewState(selected_session_id="one", sessions=(first, second))
    )
    runtime.armed = True
    sink = _ProjectionSink()
    scheduled: list[Callable[[], None]] = []
    marshalled: list[Callable[[], None]] = []
    async_runs: list[Callable[[], Awaitable[Any]]] = []
    controller = _controller(
        runtime,
        sink,
        scheduled=scheduled,
        marshalled=marshalled,
        async_runs=async_runs,
    )

    controller.open_workspace()
    _run_one(scheduled)
    assert len(async_runs) == 1
    stale_resize = async_runs.pop()
    controller.detach_workspace()
    assert await stale_resize() is False
    assert runtime.resized == []

    controller.open_workspace()
    _run_one(scheduled)
    assert len(async_runs) == 1
    assert await async_runs.pop()() is True
    assert [(item[0], item[1], item[2]) for item in runtime.resized] == [
        ("one", 80, 24)
    ]

    runtime.state = replace(runtime.state, selected_session_id="two")
    runtime.emit()
    _run_one(marshalled)
    _run_one(scheduled)
    assert len(async_runs) == 1
    assert await async_runs.pop()() is True
    assert [(item[0], item[1], item[2]) for item in runtime.resized] == [
        ("one", 80, 24),
        ("two", 80, 24),
    ]


@pytest.mark.asyncio
async def test_controller_routes_successful_session_and_input_actions(
    tmp_path: Path,
) -> None:
    running = _session("one", "One")
    cleanup = _session(
        "cleanup",
        "Cleanup",
        lifecycle=TerminalLifecycle.CLEANUP_UNPROVEN,
    )
    runtime = _Runtime(
        TerminalViewState(selected_session_id="one", sessions=(running, cleanup))
    )
    runtime.armed = True
    sink = _ProjectionSink()
    scheduled: list[Callable[[], None]] = []
    controller = _controller(
        runtime,
        sink,
        scheduled=scheduled,
        modal_result=TerminalSessionFormResult("Renamed", None, None),
    )
    controller.open_workspace()
    _run_one(scheduled)

    assert await controller.request_rename("one") is True
    assert controller.request_focus("one") is True
    assert controller.request_retry_cleanup("cleanup") is True
    assert controller.send_key(b"x") is True
    assert controller.send_paste("hello", bracketed=True) is True
    assert await controller.request_resize(100, 40) is True

    assert runtime.renamed == [("one", "Renamed")]
    assert runtime.focused == ["one"]
    assert runtime.retried == ["cleanup"]
    assert runtime.keys == [("one", b"x")]
    assert runtime.pastes == [("one", "hello", True)]
    assert sink.focus_calls == 1
    assert [(item[0], item[1], item[2]) for item in runtime.resized] == [
        ("one", 100, 40)
    ]


@pytest.mark.asyncio
async def test_controller_creates_from_revalidated_allowlisted_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    runtime = _Runtime(TerminalViewState())
    runtime.armed = True
    sink = _ProjectionSink()
    controller = _controller(
        runtime,
        sink,
        modal_result=TerminalSessionFormResult("Terminal 1", "default", root),
        selected_root=lambda: root,
        account_home=lambda: tmp_path,
        shell_choices=_shell_choices,
    )

    assert await controller.request_new_session() is True
    assert runtime.created == [
        TerminalLaunchRequest(
            name="Terminal 1",
            shell="default",
            start_directory=str(root),
            columns=80,
            rows=24,
        )
    ]


@pytest.mark.asyncio
async def test_controller_routes_locked_arm_to_settings_and_shares_disclosure() -> None:
    runtime = _Runtime(TerminalViewState())
    runtime.permitted = False
    sink = _ProjectionSink()
    settings_calls: list[None] = []
    confirmations: list[tuple[str, str]] = []
    controller = _controller(
        runtime,
        sink,
        settings_calls=settings_calls,
        confirmations=confirmations,
    )

    assert await controller.request_arm() is False
    assert settings_calls == [None]
    assert confirmations == []

    runtime.permitted = True
    assert await controller.request_arm() is True
    assert confirmations == [
        ("Arm Terminal for this launch?", "\n\n".join(TERMINAL_DISCLOSURE_LINES))
    ]
    assert settings_screen.TERMINAL_DISCLOSURE_LINES is TERMINAL_DISCLOSURE_LINES


@pytest.mark.asyncio
async def test_controller_confirms_running_close_but_not_exited_close() -> None:
    running = _session("running", "Running")
    exited = _session("exited", "Exited", lifecycle=TerminalLifecycle.EXITED)
    runtime = _Runtime(
        TerminalViewState(selected_session_id="running", sessions=(running, exited))
    )
    runtime.armed = True
    sink = _ProjectionSink()
    confirmations: list[tuple[str, str]] = []
    scheduled: list[Callable[[], None]] = []
    controller = _controller(
        runtime,
        sink,
        scheduled=scheduled,
        confirmations=confirmations,
    )
    controller.open_workspace()
    _run_one(scheduled)

    assert await controller.request_close("running") is True
    assert len(confirmations) == 1
    assert "terminate" in confirmations[0][1].lower()
    assert await controller.request_close("exited") is True
    assert len(confirmations) == 1
    assert runtime.closed == ["running", "exited"]


@pytest.mark.asyncio
async def test_controller_revalidates_start_directory_after_modal_returns(
    tmp_path: Path,
) -> None:
    launch_root = tmp_path / "root"
    launch_root.mkdir()
    runtime = _Runtime(TerminalViewState())
    runtime.armed = True
    sink = _ProjectionSink()
    result = TerminalSessionFormResult(
        name="Terminal 1",
        shell="default",
        start_directory=launch_root,
    )

    async def present_and_remove(_modal: object) -> TerminalSessionFormResult:
        launch_root.rmdir()
        return result

    controller = ConsoleTerminalController(
        terminal_runtime=lambda: runtime,
        workspace_accessor=lambda: sink,
        selected_local_root=lambda: launch_root,
        account_home=lambda: tmp_path,
        open_privacy_settings=lambda: None,
        confirm=lambda _title, _message: pytest.fail("unexpected confirmation"),
        present_session_modal=present_and_remove,
        marshal_to_ui=lambda callback: callback(),
        schedule_frame=lambda callback: callback(),
        shell_choices=lambda: (),
    )

    created = await controller.request_new_session()

    assert created is False
    assert runtime.created == []
