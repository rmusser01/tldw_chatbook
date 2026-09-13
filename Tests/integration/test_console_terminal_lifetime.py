"""Mounted real-POSIX lifetime qualification for Console Terminal."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Callable

import pytest

from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook import config as config_module
from tldw_chatbook.Terminal.contracts import (
    TerminalLaunchRequest,
    TerminalLifecycle,
)
from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_terminal_workspace import (
    ConsoleTerminalWorkspace,
    TerminalViewport,
)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.name != "posix", reason="requires a POSIX PTY"),
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TERMINAL_CHILD = REPOSITORY_ROOT / "Tests/fixtures/terminal/terminal_child.py"


def _selected_terminal_text(
    manager: TerminalSessionManager,
    console: ChatScreen,
) -> str:
    view = console._terminal._view
    if view is None:
        return ""
    state = manager.view_state(view)
    if state is None or state.selected_session_id is None:
        return ""
    selected = next(
        (
            session
            for session in state.sessions
            if session.projection.session_id == state.selected_session_id
        ),
        None,
    )
    if selected is None:
        return ""
    return "\n".join(
        line.text for line in (*selected.screen.scrollback, *selected.screen.lines)
    )


async def _wait_until(
    pilot,
    predicate: Callable[[], bool],
    *,
    timeout: float = 8.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.01)
    assert predicate()


async def _send_terminal_command(
    console: ChatScreen,
    pilot,
    command: str,
) -> None:
    assert console._terminal.send_paste(command, bracketed=False) is True
    await pilot.press("enter")


@pytest.mark.asyncio
async def test_posix_mounted_real_terminal_focus_input_and_navigation(
    tmp_path: Path,
) -> None:
    """Prove real PTY I/O survives mounted Console and app lifetimes."""
    assert TERMINAL_CHILD.is_file()
    manager_source = Path(inspect.getfile(TerminalSessionManager)).resolve()
    assert manager_source.is_relative_to(REPOSITORY_ROOT)
    assert TERMINAL_CHILD.resolve().is_relative_to(REPOSITORY_ROOT)

    working = tmp_path / "retained cwd"
    working.mkdir()
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    assert config_module.save_settings_to_cli_config(
        {"console": {"raw_cli_permitted": True}}
    )
    app.app_config = config_module.load_settings(force_reload=True)
    gateway = CapturingGateway(chunks=("model-independent",))
    app.console_provider_gateway_factory = lambda: gateway
    manager = app.terminal_session_manager
    assert manager.arm(acknowledge_disclosure=True).armed is True
    created = manager.create_session(
        TerminalLaunchRequest(
            name="Mounted real shell",
            shell="default",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        )
    )
    assert created.admitted is True
    assert created.projection is not None
    session_id = created.projection.session_id

    try:
        async with app.run_test(size=(160, 48)) as pilot:
            console = ChatScreen(app)
            await app.push_screen(console)
            app._initial_screen_pushed = True
            app.current_tab = "chat"
            await _wait_for_selector(console, pilot, "#console-terminal-open")
            assert await pilot.click("#console-terminal-open")
            await _wait_for_selector(console, pilot, "#console-terminal-viewport")
            await pilot.pause()
            viewport = console.query_one("#console-terminal-viewport", TerminalViewport)
            assert pilot.app.focused is viewport
            assert viewport.input_focused is True

            probe_command = " ; ".join(
                (
                    f"cd {shlex.quote(str(working))}",
                    "export TERMINAL_CHILD_VALUE='café-終'",
                    f"{shlex.quote(sys.executable)} "
                    f"{shlex.quote(str(TERMINAL_CHILD))} probe",
                    "printf 'HISTORY-OLDEST\\n'",
                    f"{shlex.quote(sys.executable)} -c "
                    + shlex.quote(
                        "for index in range(80): print(f'HISTORY-{index:03d}')"
                    ),
                    f"{shlex.quote(sys.executable)} "
                    f"{shlex.quote(str(TERMINAL_CHILD))} alternate",
                    "printf 'PRIMARY-RESTORED\\n'",
                )
            )
            await _send_terminal_command(console, pilot, probe_command)
            await _wait_until(
                pilot,
                lambda: "PRIMARY-RESTORED" in _selected_terminal_text(manager, console),
            )
            compact = _selected_terminal_text(manager, console).replace("\n", "")
            assert str(working) in compact
            assert '"stdin_tty": true' in compact
            assert '"stdout_tty": true' in compact
            assert '"stderr_tty": true' in compact
            assert f'"value": {json.dumps("café-終")}' in compact
            assert "HISTORY-OLDEST" in compact
            assert "PRIMARY-RESTORED" in compact
            state = manager.view_state(console._terminal._view)
            assert state is not None
            selected = next(
                session
                for session in state.sessions
                if session.projection.session_id == session_id
            )
            assert selected.screen.in_alternate is False
            assert selected.screen.scrollback

            winch_command = (
                f"{shlex.quote(sys.executable)} "
                f"{shlex.quote(str(TERMINAL_CHILD))} winch"
            )
            await _send_terminal_command(console, pilot, winch_command)
            await _wait_until(
                pilot,
                lambda: "WINCH_READY" in _selected_terminal_text(manager, console),
            )
            assert await console._terminal.request_resize(91, 31) is True
            await pilot.press("x")
            await _wait_until(
                pilot,
                lambda: "WINCH:91x31" in _selected_terminal_text(manager, console),
            )
            await pilot.press("enter")
            await _send_terminal_command(
                console,
                pilot,
                "printf 'WINCH-DONE\\n'",
            )
            await _wait_until(
                pilot,
                lambda: "WINCH-DONE" in _selected_terminal_text(manager, console),
            )

            await pilot.press("ctrl+right_square_bracket")
            await pilot.pause()
            assert viewport.input_focused is False
            before_page_up = viewport.renderable.plain
            await pilot.press("pageup")
            await pilot.pause()
            assert viewport.history_offset > 0
            assert viewport.renderable.plain != before_page_up
            assert "HISTORY-" in viewport.renderable.plain

            assert await pilot.click("#console-terminal-return")
            await _wait_for_selector(console, pilot, "#console-native-transcript")
            before_turn = manager.projection(session_id)
            await console._submission._submit_console_native_draft("ordinary model turn")
            await _wait_until(pilot, lambda: bool(gateway.sent_messages))
            assert manager.projection(session_id) == before_turn

            console.action_open_console_terminal()
            await _wait_for_selector(console, pilot, "#console-terminal-viewport")
            before_recompose = manager.projection(session_id)
            before_processes = manager.managed_process_inventory_for_tests()
            console.refresh(recompose=True)
            await _wait_for_selector(console, pilot, "#console-main-column")
            await _wait_for_selector(console, pilot, "#console-terminal-viewport")
            assert isinstance(
                console.query_one("#console-main-column"),
                ConsoleTerminalWorkspace,
            )
            assert manager.projection(session_id) == before_recompose
            assert manager.managed_process_inventory_for_tests() == before_processes
            assert "PRIMARY-RESTORED" in _selected_terminal_text(manager, console)

            await app.handle_screen_navigation(NavigateToScreen("library"))
            await pilot.pause()
            assert manager.projection(session_id) is not None
            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await pilot.pause()
            reopened = app.screen
            assert isinstance(reopened, ChatScreen)
            await _wait_for_selector(reopened, pilot, "#console-terminal-open")
            reopened.action_open_console_terminal()
            await _wait_for_selector(reopened, pilot, "#console-terminal-viewport")
            assert "PRIMARY-RESTORED" in _selected_terminal_text(manager, reopened)
            assert await pilot.click("#console-terminal-viewport")
            await pilot.pause()

            await _send_terminal_command(reopened, pilot, "exit 23")
            await _wait_until(
                pilot,
                lambda: (
                    manager.projection(session_id) is not None
                    and manager.projection(session_id).lifecycle
                    is TerminalLifecycle.EXITED
                ),
            )
            exited = manager.projection(session_id)
            assert exited is not None
            assert exited.exit_code == 23
            assert exited.stream_closed is True
            assert exited.output_complete is True
            assert "exit 23" in _selected_terminal_text(manager, reopened)

            view = reopened._terminal._view
            assert view is not None
            assert manager.close_session(session_id, view=view) is not None
            assert await asyncio.to_thread(
                manager.wait_for_cleanup,
                session_id,
                timeout_seconds=5.0,
            )
            assert manager.projection(session_id) is None

            disarm_result = manager.create_session(
                TerminalLaunchRequest(
                    name="Disarm cleanup",
                    shell="default",
                    start_directory=str(tmp_path),
                    columns=80,
                    rows=24,
                )
            )
            assert disarm_result.admitted is True
            assert disarm_result.projection is not None
            disarm_id = disarm_result.projection.session_id
            manager.disarm()
            assert await asyncio.to_thread(
                manager.wait_for_cleanup,
                disarm_id,
                timeout_seconds=6.0,
            )

            assert manager.arm().armed is True
            shutdown_result = manager.create_session(
                TerminalLaunchRequest(
                    name="App shutdown cleanup",
                    shell="default",
                    start_directory=str(tmp_path),
                    columns=80,
                    rows=24,
                )
            )
            assert shutdown_result.admitted is True
            assert shutdown_result.projection is not None
            await asyncio.wait_for(
                app._shutdown_terminal_session_manager(), timeout=6.0
            )
            assert manager.projections() == ()
    finally:
        await manager.shutdown(deadline_seconds=5.0)
        manager.finalize_shutdown()
