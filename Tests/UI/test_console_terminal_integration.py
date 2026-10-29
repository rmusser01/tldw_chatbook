"""Integration contract for the app-owned, user-only Console Terminal."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shlex
import sys
import time

import pytest
from textual import on
from textual.widgets import Button, Label, Static

from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Terminal.contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    MAX_COLUMNS,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
    TerminalLaunchRequest,
    TerminalLifecycle,
)
from tldw_chatbook.Terminal.session_manager import TerminalSessionManager
from tldw_chatbook.UI.Console_Modules.transcript import ConsoleTranscriptRegion
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.Console.console_terminal_workspace import (
    ConsoleTerminalWorkspace,
    TerminalViewport,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

# These tests mount the real ChatScreen, which reads app config at build time.
pytestmark = pytest.mark.bootstrap_profile


class RecordingTerminalBackend:
    """Complete inert backend whose counters expose accidental lifecycle work."""

    def __init__(self, *, cleanup_proof: CleanupProof | None = None) -> None:
        self.cleanup_proof = cleanup_proof or CleanupProof(True, True, True)
        self.started = []
        self.writes: list[bytes] = []
        self.resizes: list[tuple[int, int]] = []
        self.priority_close_requests = 0
        self.cleanup_attempts: list[CleanupAttempt] = []
        self.finalize_calls = 0

    def start(self, request, admission: AdmissionGate) -> BackendIdentity:
        self.started.append((request, admission))
        return BackendIdentity(session_id=admission.token)

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def resize(self, columns: int, rows: int) -> None:
        self.resizes.append((columns, rows))

    def request_priority_close(self) -> None:
        self.priority_close_requests += 1

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        self.cleanup_attempts.append(attempt)
        return self.cleanup_proof

    def cleanup_raw_drain(self, attempt: CleanupAttempt) -> CleanupProof:
        return self.cleanup(attempt)

    def finalize_shutdown(self) -> None:
        self.finalize_calls += 1


class TerminalNavigationHarness(ConsoleHarness):
    """Console harness that records, rather than follows, navigation."""

    def __init__(self, app_instance) -> None:
        super().__init__(app_instance)
        self.navigation_messages: list[NavigateToScreen] = []

    @on(NavigateToScreen)
    def capture_navigation(self, message: NavigateToScreen) -> None:
        self.navigation_messages.append(message)
        message.stop()


def _terminal_app(
    *,
    permitted: bool | Callable[[], bool],
    cleanup_proof: CleanupProof | None = None,
) -> tuple[object, TerminalSessionManager, list[RecordingTerminalBackend]]:
    read_permitted = permitted if callable(permitted) else lambda: permitted
    app = _build_test_app(
        config_overrides={"console": {"raw_cli_permitted": read_permitted() is True}}
    )
    _configure_native_ready_console(app)
    app.terminal_session_manager.finalize_shutdown()
    backends: list[RecordingTerminalBackend] = []

    def backend_factory() -> RecordingTerminalBackend:
        backend = RecordingTerminalBackend(cleanup_proof=cleanup_proof)
        backends.append(backend)
        return backend

    manager = TerminalSessionManager(read_permitted, backend_factory)
    app.terminal_session_manager = manager
    return app, manager, backends


def _create_running_terminal(
    manager: TerminalSessionManager,
    root: Path,
    *,
    name: str = "Persistent shell",
) -> str:
    assert manager.arm(acknowledge_disclosure=True).armed is True
    created = manager.create_session(
        TerminalLaunchRequest(
            name=name,
            shell="default",
            start_directory=str(root),
            columns=80,
            rows=24,
        )
    )
    assert created.admitted is True
    assert created.projection is not None
    return created.projection.session_id


async def _wait_until(
    pilot,
    predicate: Callable[[], bool],
    *,
    detail: Callable[[], object] | None = None,
    timeout: float = 4.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.01)
    assert predicate(), detail() if detail is not None else None


def _surrounding_console_widgets(console: ChatScreen) -> dict[str, object]:
    """Return the shell widgets that a center-only Terminal swap must preserve."""
    return {
        "header": console.query_one("#console-workbench-header"),
        "context": console.query_one("#console-left-rail"),
        "inspector": console.query_one("#console-right-rail"),
        "context-handle": console.query_one("#console-context-rail-handle"),
        "inspector-handle": console.query_one("#console-inspector-rail-handle"),
        "control-bar": console.query_one("#console-control-bar"),
        "composer": console.query_one("#console-native-composer"),
        "navigation": console.query_one(MainNavigationBar),
        "workspace-grid": console.query_one("#console-workspace-grid"),
    }


@pytest.mark.asyncio
async def test_locked_terminal_entry_routes_to_privacy_without_open_or_launch() -> None:
    app, manager, backends = _terminal_app(permitted=False)
    host = TerminalNavigationHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console.action_open_console_terminal()
        await pilot.pause()

        assert [message.screen_name for message in host.navigation_messages] == [
            "settings"
        ]
        assert host.navigation_messages[0].screen_context == {
            "category": SettingsCategoryId.PRIVACY_SECURITY.value,
        }
        assert console._terminal.is_open is False
        assert console.query_one("#console-native-transcript")
        assert backends == []

    manager.finalize_shutdown()


@pytest.mark.parametrize("locked", [True, False], ids=["locked", "unarmed"])
@pytest.mark.asyncio
async def test_cleanup_receipt_and_retry_remain_usable_without_launch_authority(
    tmp_path: Path,
    locked: bool,
) -> None:
    permission = {"value": True}
    app, manager, backends = _terminal_app(
        permitted=lambda: permission["value"],
        cleanup_proof=CleanupProof(),
    )
    session_id = _create_running_terminal(manager, tmp_path)
    view = manager.attach_view()
    assert manager.close_session(session_id, view=view) is not None
    assert manager.wait_for_cleanup(session_id, timeout_seconds=1)
    assert manager.detach_view(view) is True
    if locked:
        permission["value"] = False
    else:
        manager.disarm()
        assert manager.wait_for_cleanup(session_id, timeout_seconds=1)
    host = TerminalNavigationHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        console.action_open_console_terminal()
        await _wait_for_selector(console, pilot, "#console-terminal-retry")
        await pilot.pause()

        retry = console.query_one("#console-terminal-retry", Button)
        access = str(
            console.query_one("#console-terminal-access", Static).renderable
        ).lower()
        assert retry.display is True
        assert pilot.app.focused is retry
        assert ("locked" if locked else "not armed") in access
        assert host.navigation_messages == []
        assert len(backends) == 1
        attempts_before_retry = len(backends[0].cleanup_attempts)

        assert await pilot.click(retry)
        await _wait_until(
            pilot,
            lambda: len(backends[0].cleanup_attempts) > attempts_before_retry,
        )
        assert manager.wait_for_cleanup(session_id, timeout_seconds=1)
        assert (
            manager.projection(session_id).lifecycle
            is TerminalLifecycle.CLEANUP_UNPROVEN
        )
        assert console.query_one("#console-terminal-retry", Button).display is True
        assert backends[0].started[0][0].name == "Persistent shell"
        bottom_actions = [
            console.query_one(f"#console-terminal-{name}", Button)
            for name in ("rename", "focus", "close", "retry", "jump-live", "return")
            if console.query_one(f"#console-terminal-{name}", Button).display
        ]
        assert retry in bottom_actions
        assert len({button.region.y for button in bottom_actions}) == 1

    manager.finalize_shutdown()


@pytest.mark.asyncio
async def test_terminal_session_survives_conversation_recompose_and_navigation(
    tmp_path: Path,
) -> None:
    app, manager, backends = _terminal_app(permitted=True)
    session_id = _create_running_terminal(manager, tmp_path)
    assert manager.offer_output(session_id, b"persisted-screen").accepted
    assert manager.process_output(session_id, visible=True) is not None

    async with app.run_test(size=(160, 48)) as pilot:
        chat = ChatScreen(app)
        await app.push_screen(chat)
        app._initial_screen_pushed = True
        app.current_tab = "chat"
        await _wait_for_selector(chat, pilot, "#console-native-composer")

        chat.action_open_console_terminal()
        await _wait_for_selector(chat, pilot, "#console-terminal-viewport")
        await pilot.pause()
        workspace = chat.query_one("#console-main-column", ConsoleTerminalWorkspace)
        first_view = chat._terminal._view
        original_conversation = chat._ensure_console_chat_store().active_session_id
        assert original_conversation is not None
        assert (
            "persisted-screen"
            in workspace.query_one(
                "#console-terminal-viewport", TerminalViewport
            ).renderable.plain
        )
        assert str(tmp_path) in str(
            workspace.query_one("#console-terminal-metadata", Static).renderable
        )

        chat.action_new_console_tab()
        await _wait_until(
            pilot,
            lambda: (
                chat._ensure_console_chat_store().active_session_id
                != original_conversation
            ),
        )
        assert manager.selected_session_id == session_id
        assert len(manager.projections()) == 1
        assert len(backends[0].started) == 1

        chat.refresh(recompose=True)
        await _wait_for_selector(chat, pilot, "#console-terminal-viewport")
        await pilot.pause()
        assert chat._terminal._view is first_view
        assert chat.query_one("#console-main-column") is workspace
        assert len(manager._subscriptions) == 1

        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat not in app.screen_stack
        assert manager._current_view is None
        assert manager._subscriptions == {}
        assert manager.selected_session_id == session_id
        assert len(manager.projections()) == 1
        assert len(backends[0].started) == 1
        assert backends[0].cleanup_attempts == []
        assert backends[0].priority_close_requests == 0

        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await pilot.pause()
        reopened = app.screen
        assert isinstance(reopened, ChatScreen)
        await _wait_for_selector(reopened, pilot, "#console-native-composer")
        reopened.action_open_console_terminal()
        await _wait_for_selector(reopened, pilot, "#console-terminal-viewport")
        await pilot.pause()
        reopened_workspace = reopened.query_one(
            "#console-main-column", ConsoleTerminalWorkspace
        )
        assert (
            "persisted-screen"
            in reopened_workspace.query_one(
                "#console-terminal-viewport", TerminalViewport
            ).renderable.plain
        )
        assert str(tmp_path) in str(
            reopened_workspace.query_one(
                "#console-terminal-metadata", Static
            ).renderable
        )
        assert manager.selected_session_id == session_id
        assert len(manager.projections()) == 1
        assert manager._view_generation == 2
        assert len(manager._subscriptions) == 1
        assert len(backends[0].started) == 1
        assert backends[0].cleanup_attempts == []
        assert backends[0].priority_close_requests == 0
        assert backends[0].finalize_calls == 0


@pytest.mark.parametrize(
    ("terminal_size", "expect_clamp"),
    [
        ((160, 45), False),
        ((100, 32), False),
        ((80, 24), False),
        ((420, 150), True),
    ],
    ids=["standard", "100-columns", "narrow", "capped"],
)
@pytest.mark.asyncio
async def test_terminal_uses_painted_center_geometry_without_overflow(
    tmp_path: Path,
    terminal_size: tuple[int, int],
    expect_clamp: bool,
) -> None:
    app, manager, backends = _terminal_app(permitted=True)
    session_id = _create_running_terminal(manager, tmp_path)
    host = ConsoleHarness(app)

    async with host.run_test(size=terminal_size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-terminal-open")
        console.action_open_console_terminal()
        await _wait_for_selector(console, pilot, "#console-terminal-viewport")
        await pilot.pause()

        workspace = console.query_one("#console-main-column", ConsoleTerminalWorkspace)
        viewport = workspace.query_one("#console-terminal-viewport", TerminalViewport)
        grid = console.query_one("#console-workspace-grid")
        assert workspace.layout.name == "grid"
        assert grid.content_region.contains_region(workspace.region)
        assert workspace.content_region.contains_region(viewport.region), (
            terminal_size,
            workspace.content_region,
            [(row.value, row.unit.name) for row in workspace.styles.grid_rows or ()],
            viewport.styles.height,
            [(child.id, child.region) for child in workspace.children if child.display],
        )
        for child in workspace.children:
            if child.display:
                assert workspace.content_region.contains_region(child.region), (
                    terminal_size,
                    child.id,
                    workspace.content_region,
                    child.region,
                )

        session = workspace.query_one("#console-terminal-session-0", Button)
        new = workspace.query_one("#console-terminal-new", Button)
        assert session.display is True
        assert new.display is True
        assert session.region.y == new.region.y
        bottom_actions = [
            workspace.query_one(f"#console-terminal-{name}", Button)
            for name in ("rename", "focus", "close", "retry", "jump-live", "return")
            if workspace.query_one(f"#console-terminal-{name}", Button).display
        ]
        assert len({button.region.y for button in bottom_actions}) == 1

        expected = (
            min(MAX_COLUMNS, max(MIN_COLUMNS, viewport.size.width)),
            min(MAX_ROWS, max(MIN_ROWS, viewport.size.height)),
        )
        assert workspace.terminal_size() == expected
        await _wait_until(
            pilot,
            lambda: bool(backends[0].resizes) and backends[0].resizes[-1] == expected,
            detail=lambda: (
                terminal_size,
                viewport.size,
                expected,
                backends[0].resizes,
            ),
        )
        assert backends[0].resizes[-1] == expected, (
            terminal_size,
            viewport.size,
            expected,
            backends[0].resizes,
        )
        view = console._terminal._view
        assert view is not None
        state = manager.view_state(view)
        assert state is not None
        assert state.selected_session_id == session_id
        selected = state.sessions[0]
        assert (selected.columns, selected.rows) == expected
        metadata = str(
            workspace.query_one("#console-terminal-metadata", Static).renderable
        )
        if expect_clamp:
            assert viewport.size.width > MAX_COLUMNS
            assert viewport.size.height > MAX_ROWS
            assert "viewport capped at 300×120" in metadata
        else:
            assert viewport.size.width <= MAX_COLUMNS
            assert viewport.size.height <= MAX_ROWS
            assert "viewport capped" not in metadata

    manager.finalize_shutdown()


# The app factory reads the real collection-bound config source.
@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_terminal_rail_focus_capture_release_tab_and_return(
    tmp_path: Path,
) -> None:
    app, manager, _backends = _terminal_app(permitted=True)
    session_id = _create_running_terminal(manager, tmp_path)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-terminal-open")

        assert await pilot.click("#console-terminal-open")
        await _wait_for_selector(console, pilot, "#console-terminal-viewport")
        await pilot.pause()
        viewport = console.query_one("#console-terminal-viewport", TerminalViewport)
        assert pilot.app.focused is viewport
        assert viewport.input_focused is True

        await pilot.press("tab")
        await pilot.pause()
        input_event = manager.take_input(session_id)
        assert input_event is not None
        assert input_event.data == b"\t"
        assert pilot.app.focused is viewport

        await pilot.press("ctrl+right_square_bracket")
        await pilot.pause()
        assert viewport.input_focused is False
        assert pilot.app.focused is viewport

        await pilot.press("tab")
        await pilot.pause()
        assert (pilot.app.focused.id or "").startswith("console-terminal-")
        assert manager.take_input(session_id) is None

        assert await pilot.click("#console-terminal-return")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        # TASK-33622.8 AC#3: the return lands in the composer, not the transcript.
        composer = console.query_one("#console-native-composer")
        await _wait_until(pilot, lambda: pilot.app.focused is composer)
        assert console._terminal.is_open is False

    manager.finalize_shutdown()


@pytest.mark.asyncio
async def test_unarmed_terminal_replaces_only_center_and_return_restores_transcript() -> (
    None
):
    app, manager, backends = _terminal_app(permitted=True)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        surrounding = _surrounding_console_widgets(console)
        assert all(widget.is_mounted for widget in surrounding.values())
        preserved_ids = {
            "console-workbench-header",
            "console-control-bar",
            "console-context-rail-handle",
            "console-left-rail",
            "console-right-rail",
            "console-inspector-rail-handle",
            "console-native-composer",
        }

        console.action_open_console_terminal()
        await _wait_for_selector(console, pilot, "#console-terminal-arm")
        await pilot.pause()

        workspace = console.query_one("#console-main-column", ConsoleTerminalWorkspace)
        assert _surrounding_console_widgets(console) == surrounding
        assert all(widget.is_mounted for widget in surrounding.values())
        assert not console.query("#console-native-transcript")
        assert all(console.query_one(f"#{widget_id}") for widget_id in preserved_ids)
        assert console.query_one(MainNavigationBar)
        assert [
            child.id for child in console.query_one("#console-workspace-grid").children
        ] == [
            "console-context-rail-handle",
            "console-left-rail",
            "console-main-column",
            "console-right-rail",
            "console-inspector-rail-handle",
        ]
        arm = workspace.query_one("#console-terminal-arm", Button)
        assert arm.display is True
        assert pilot.app.focused is arm
        assert manager.armed is False
        assert backends == []

        assert await pilot.click(arm)
        await _wait_for_selector(pilot.app.screen, pilot, "#confirmation-dialog")
        disclosure = str(
            pilot.app.screen.query_one(".dialog-message", Label).renderable
        )
        assert "same OS permissions as Chatbook" in disclosure
        assert "not sent to a model" in disclosure
        assert backends == []
        assert await pilot.click("#cancel-button")
        await pilot.pause()
        assert manager.armed is False

        assert await pilot.click("#console-terminal-return")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await pilot.pause()
        assert _surrounding_console_widgets(console) == surrounding
        assert all(widget.is_mounted for widget in surrounding.values())
        assert console._terminal.is_open is False
        # TASK-33622.8 AC#3: the return lands in the composer.
        await _wait_until(
            pilot, lambda: pilot.app.focused is surrounding["composer"]
        )
        assert isinstance(
            console.query_one("#console-main-column"), ConsoleTranscriptRegion
        )
        assert backends == []

    manager.finalize_shutdown()


def _drain_terminal_input(manager: TerminalSessionManager, session_id: str) -> bytes:
    """Return every input byte the session's actor has queued, in order."""
    received = b""
    while (event := manager.take_input(session_id)) is not None:
        received += event.data
    return received


def _composer_draft(console: ChatScreen) -> str:
    return console.query_one("#console-native-composer").draft_text()


def _focused_in_terminal_workspace(console: ChatScreen) -> bool:
    focused = console.app.focused
    workspace = console.query_one("#console-main-column")
    return (
        isinstance(workspace, ConsoleTerminalWorkspace)
        and focused is not None
        and (focused is workspace or workspace in focused.ancestors)
        and focused.display
    )


async def _create_session_through_new_modal(console: ChatScreen, pilot) -> None:
    """Press New and Create exactly as the user does, through the real modal."""
    assert await pilot.click("#console-terminal-new")
    await _wait_for_selector(pilot.app.screen, pilot, "#console-terminal-session-save")
    assert await pilot.click("#console-terminal-session-save")


@pytest.mark.asyncio
async def test_arm_and_create_hand_the_keyboard_to_the_new_terminal_session(
    tmp_path: Path,
) -> None:
    """TASK-33622.5 AC#1/#3/#4/#5: after Arm and New, typing reaches the shell.

    On dev the Arm dialog and the New modal both returned focus to the chat
    composer, so typed shell commands became a chat draft (and Enter sent
    them to the model) while the footer still offered "Enter send / queue".
    """
    app, manager, backends = _terminal_app(permitted=True)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-terminal-open")
        draft_before = _composer_draft(console)

        assert await pilot.click("#console-terminal-open")
        await _wait_for_selector(console, pilot, "#console-terminal-arm")
        assert await pilot.click("#console-terminal-arm")
        await _wait_for_selector(pilot.app.screen, pilot, "#confirm-button")
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: manager.armed is True)

        # AC#3: confirming Arm keeps focus in the Terminal center.
        await _wait_until(
            pilot,
            lambda: _focused_in_terminal_workspace(console)
            and pilot.app.focused.id == "console-terminal-new",
            detail=lambda: pilot.app.focused,
        )

        await _create_session_through_new_modal(console, pilot)
        await _wait_until(pilot, lambda: len(manager.projections()) == 1)
        session_id = manager.projections()[0].session_id
        viewport = console.query_one("#console-terminal-viewport", TerminalViewport)
        # AC#1: the new session is selected and its viewport owns the keys.
        await _wait_until(
            pilot,
            lambda: manager.selected_session_id == session_id
            and pilot.app.focused is viewport,
            detail=lambda: (manager.selected_session_id, pilot.app.focused),
        )
        assert backends[0].started[0][0].name

        await pilot.press("e", "c", "h", "o", "space", "h", "i", "enter")
        await pilot.pause()
        assert _drain_terminal_input(manager, session_id) == b"echo hi\r"
        assert _composer_draft(console) == draft_before
        assert pilot.app.focused is viewport

        # AC#4: the composer is collapsed out of view and takes no keys, and
        # the footer names Terminal keys instead of the chat send.
        composer = console.query_one("#console-native-composer")
        assert composer.display is False
        assert composer.focusable is False
        source, shortcuts = console._footer_shortcut_registration
        keys = [shortcut[0] for shortcut in shortcuts]
        assert source == "console"
        assert "Ctrl+]" in keys
        assert "Enter" not in keys
        assert "Y" not in keys

        # The release chord parks the viewport; printable keys must not
        # tunnel into the hidden composer from there either.
        await pilot.press("ctrl+right_square_bracket")
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert _composer_draft(console) == draft_before

    manager.finalize_shutdown()


@pytest.mark.asyncio
async def test_creating_another_terminal_session_selects_and_focuses_it(
    tmp_path: Path,
) -> None:
    """TASK-33622.5 AC#2: Terminal 2 takes the keys, not the older Terminal 1."""
    app, manager, _backends = _terminal_app(permitted=True)
    first_id = _create_running_terminal(manager, tmp_path, name="Terminal 1")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-terminal-open")
        composer = console.query_one("#console-native-composer")
        # Let the Console's activation focus restore (refresh + 0.2 s timer)
        # land first, so this test measures the create path, not that race.
        await _wait_until(pilot, lambda: pilot.app.focused is composer)
        await pilot.pause(0.3)
        assert await pilot.click("#console-terminal-open")
        await _wait_for_selector(console, pilot, "#console-terminal-viewport")
        viewport = console.query_one("#console-terminal-viewport", TerminalViewport)
        await _wait_until(
            pilot,
            lambda: pilot.app.focused is viewport,
            detail=lambda: pilot.app.focused,
        )
        assert manager.selected_session_id == first_id
        draft_before = _composer_draft(console)

        await _create_session_through_new_modal(console, pilot)
        await _wait_until(pilot, lambda: len(manager.projections()) == 2)
        second_id = next(
            projection.session_id
            for projection in manager.projections()
            if projection.session_id != first_id
        )
        await _wait_until(
            pilot,
            lambda: manager.selected_session_id == second_id
            and pilot.app.focused is viewport,
            detail=lambda: (manager.selected_session_id, pilot.app.focused),
        )
        await pilot.press("l", "s")
        await pilot.pause()
        assert _drain_terminal_input(manager, second_id) == b"ls"
        assert _drain_terminal_input(manager, first_id) == b""
        assert _composer_draft(console) == draft_before

        # F6 leaves the Terminal and Shift+F6 comes back to it: the center
        # pane is the Terminal while it owns the center. The line under the
        # terminal follows real focus both ways (live, it kept claiming
        # "Ctrl+] Release input" after F6 had moved the keys to a rail).
        status = console.query_one("#console-terminal-status")
        assert str(status.renderable).startswith("Ctrl+] Release input")
        await pilot.press("f6")
        await pilot.pause()
        assert pilot.app.focused is not viewport
        assert str(status.renderable).startswith(
            "Focus or click the terminal to type in it"
        )
        await pilot.press("shift+f6")
        await _wait_until(
            pilot,
            lambda: pilot.app.focused is viewport,
            detail=lambda: pilot.app.focused,
        )
        await pilot.pause()
        assert str(status.renderable).startswith("Ctrl+] Release input")

    manager.finalize_shutdown()


def _session_tab_labels(console: ChatScreen) -> list[str]:
    strip = console.query_one("#console-native-tab-strip")
    return [str(button.label) for button in strip.query(Button)]


def _rendered_message_ids(console: ChatScreen) -> frozenset[str]:
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    return transcript.mounted_message_content_ids()


@pytest.mark.asyncio
async def test_terminal_round_trip_restores_tabs_transcript_and_composer_focus() -> (
    None
):
    """TASK-33622.8 AC#1-#4: Return to transcript restores what the user left.

    On dev the swap back mounted a fresh transcript center and nothing
    re-synced it: the tab strip showed only "New tab", the transcript said
    "No messages yet." for a populated chat, and focus sat on the transcript
    surface, where a typed "y" opened Trace.
    """
    app, manager, _backends = _terminal_app(permitted=True)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        store = console._ensure_console_chat_store()
        first = store.ensure_session()
        for index in range(12):
            store.append_message(
                first.id,
                role=ConsoleMessageRole.USER,
                content=f"alpha question {index}",
            )
            store.append_message(
                first.id,
                role=ConsoleMessageRole.ASSISTANT,
                content=f"alpha answer {index}\n\nwith a second paragraph",
            )
        second = store.create_session(title="Beta thread")
        store.append_message(
            second.id, role=ConsoleMessageRole.USER, content="beta question"
        )
        store.switch_session(first.id)
        await console._sync_native_console_chat_ui()
        await _wait_until(
            pilot,
            lambda: len(_rendered_message_ids(console)) >= 4,
            detail=lambda: _rendered_message_ids(console),
        )
        labels_before = _session_tab_labels(console)
        # Two chat tabs (each with its close button), New tab and Temporary.
        assert labels_before.count("✕") == 2, labels_before
        assert "Beta thread" in labels_before, labels_before
        assert "Temporary" in labels_before, labels_before
        ids_before = _rendered_message_ids(console)

        # The reader scrolled up into the history before opening the Terminal.
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        await _wait_until(pilot, lambda: transcript.max_scroll_y > 4)
        transcript.release_anchor()
        transcript.scroll_to(y=2, animate=False)
        await _wait_until(pilot, lambda: transcript.scroll_y == 2)

        assert await pilot.click("#console-terminal-open")
        await _wait_for_selector(console, pilot, "#console-terminal-return")
        assert await pilot.click("#console-terminal-return")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        restored = console.query_one("#console-native-transcript", ConsoleTranscript)
        assert restored is not transcript  # the swap really remounted the center

        composer = console.query_one("#console-native-composer")
        await _wait_until(
            pilot,
            lambda: _session_tab_labels(console) == labels_before
            and _rendered_message_ids(console) == ids_before
            and restored.scroll_y == 2
            and pilot.app.focused is composer,
            detail=lambda: (
                _session_tab_labels(console),
                len(_rendered_message_ids(console)),
                restored.scroll_y,
                pilot.app.focused,
            ),
        )
        # Never the quiet "No messages yet." body for this populated chat.
        assert not [
            body
            for body in restored.query(".console-transcript-empty-body")
            if body.display and body.region.area
        ]
        assert composer.display is True

        draft_before = _composer_draft(console)
        await pilot.press("y")
        await pilot.pause()
        assert _composer_draft(console) == draft_before + "y"
        assert pilot.app.screen is console  # "y" did not open Trace

    manager.finalize_shutdown()


#: Prints, from the PTY's own ioctl size, one row per visible line but the
#: prompt's: a leading-space status pair, a full-width A....Z line, then rows
#: numbered to the bottom. Every cell the PTY thinks it has is accounted for.
_FULL_SCREEN_PROBE = (
    "import os, sys; "
    "c, r = os.get_terminal_size(sys.stdout.fileno()); "
    "rows = [' M a', '?? b', 'A' + '.' * (c - 2) + 'Z']; "
    "rows += [f'R{i:03d}' for i in range(len(rows) + 1, r)]; "
    "sys.stdout.write(chr(10).join(rows) + chr(10))"
)


def _pty_size(manager: TerminalSessionManager, console: ChatScreen) -> tuple[int, int]:
    view = console._terminal._view
    state = manager.view_state(view) if view is not None else None
    if state is None or state.selected_session_id is None:
        return (0, 0)
    selected = next(
        session
        for session in state.sessions
        if session.projection.session_id == state.selected_session_id
    )
    return (selected.columns, selected.rows)


def _pty_screen_lines(manager: TerminalSessionManager, console: ChatScreen) -> list[str]:
    view = console._terminal._view
    state = manager.view_state(view) if view is not None else None
    if state is None:
        return []
    selected = next(
        (
            session
            for session in state.sessions
            if session.projection.session_id == state.selected_session_id
        ),
        None,
    )
    return [] if selected is None else [line.text for line in selected.screen.lines]


def _content_size(widget) -> tuple[int, int]:
    region = widget.content_region
    return (region.width, region.height)


def _painted_rows(console: ChatScreen, widget) -> list[str]:
    """The compositor's cells inside ``widget``'s content region, row by row."""
    region = widget.content_region
    strips = list(console._compositor.render_strips())
    return [strips[y].crop(region.x, region.right).text for y in range(region.y, region.bottom)]


def _assert_full_screen_probe_visible(rows: list[str], state: str) -> None:
    width, height = len(rows[0]), len(rows)
    assert rows[0].startswith(" M a"), (state, rows[:3])
    assert rows[1].startswith("?? b"), (state, rows[:3])
    assert rows[2] == "A" + "." * (width - 2) + "Z", (state, rows[2])
    assert rows[height - 2].startswith(f"R{height - 1:03d}"), (state, rows[-3:])


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX PTY")
@pytest.mark.asyncio
async def test_real_pty_cells_stay_visible_and_sized_in_both_focus_states(
    tmp_path: Path,
) -> None:
    """TASK-33625.3 AC#1/#2/#4 with a real shell in a real PTY.

    On dev the focused viewport drew a focus outline OVER its outermost cells:
    column 0 (" M" read "M", "??" read "?"), the last column and the top and
    bottom rows vanished while the PTY was sized to the full widget.
    """
    from tldw_chatbook.app_service_wiring import _build_terminal_backend

    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    _configure_native_ready_console(app)
    app.terminal_session_manager.finalize_shutdown()
    manager = TerminalSessionManager(lambda: True, _build_terminal_backend)
    app.terminal_session_manager = manager
    assert manager.arm(acknowledge_disclosure=True).armed is True
    created = manager.create_session(
        TerminalLaunchRequest(
            name="Real shell",
            shell="default",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        )
    )
    assert created.admitted is True
    host = ConsoleHarness(app)
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-terminal-open")
            assert await pilot.click("#console-terminal-open")
            await _wait_for_selector(console, pilot, "#console-terminal-viewport")
            viewport = console.query_one("#console-terminal-viewport", TerminalViewport)
            await _wait_until(pilot, lambda: pilot.app.focused is viewport, timeout=8)
            await _wait_until(
                pilot,
                lambda: _pty_size(manager, console) == _content_size(viewport),
                detail=lambda: (_pty_size(manager, console), _content_size(viewport)),
                timeout=8,
            )
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote(_FULL_SCREEN_PROBE)}"
            assert console._terminal.send_paste(command, bracketed=False) is True
            await pilot.press("enter")
            height = viewport.content_region.height
            # Wait on the PTY's own screen, then judge what the user sees.
            await _wait_until(
                pilot,
                lambda: any(
                    line.startswith(f"R{height - 1:03d}")
                    for line in _pty_screen_lines(manager, console)
                )
                and any(
                    line.startswith(f"R{height - 1:03d}")
                    for line in viewport.renderable.plain.splitlines()
                ),
                detail=lambda: _pty_screen_lines(manager, console),
                timeout=15,
            )

            # AC#1: focused -- every PTY cell is on screen, column 0 included.
            assert pilot.app.focused is viewport
            _assert_full_screen_probe_visible(_painted_rows(console, viewport), "focused")
            assert _pty_size(manager, console) == _content_size(viewport)

            # Unfocused: nothing shifts and nothing reappears or vanishes.
            console.query_one("#console-terminal-return").focus()
            await _wait_until(pilot, lambda: not viewport.has_focus)
            _assert_full_screen_probe_visible(_painted_rows(console, viewport), "unfocused")
            assert _pty_size(manager, console) == _content_size(viewport)

            viewport.focus()
            await _wait_until(pilot, lambda: viewport.has_focus)
            _assert_full_screen_probe_visible(_painted_rows(console, viewport), "refocused")

            # AC#2: the PTY follows the visible content through a rail toggle
            # and a terminal resize.
            before = _content_size(viewport)
            console.action_toggle_console_context_rail()
            await _wait_until(
                pilot,
                lambda: _content_size(viewport) != before
                and _pty_size(manager, console) == _content_size(viewport),
                detail=lambda: (before, _pty_size(manager, console), _content_size(viewport)),
                timeout=8,
            )
            await pilot.resize_terminal(140, 40)
            await _wait_until(
                pilot,
                lambda: _pty_size(manager, console) == _content_size(viewport)
                and viewport.content_region.right <= 140,
                detail=lambda: (_pty_size(manager, console), _content_size(viewport)),
                timeout=8,
            )
    finally:
        await manager.shutdown(deadline_seconds=5.0)
        manager.finalize_shutdown()


def _visible_terminal_buttons(workspace: ConsoleTerminalWorkspace) -> list[Button]:
    return [
        button
        for button in workspace.query(Button)
        if button.display and str(button.label).strip()
    ]


@pytest.mark.parametrize(
    ("terminal_size", "explicit_inspector", "rails_yield"),
    [
        ((80, 24), False, True),
        ((80, 24), True, True),
        ((100, 30), True, False),
    ],
    ids=["80x24", "80x24-inspector-opened", "100-columns-inspector-opened"],
)
@pytest.mark.asyncio
async def test_narrow_terminal_gets_the_center_and_unclipped_labels(
    tmp_path: Path,
    terminal_size: tuple[int, int],
    explicit_inspector: bool,
    rails_yield: bool,
) -> None:
    """TASK-33625.3 AC#3: rails yield below 100 columns, labels never clip.

    On dev an Inspector the user had opened stayed open at 80x24, leaving the
    Terminal about 44x6 with tabs reading "Ter" and actions reading
    "Ren / Focu / Clos / Jum / Return to".
    """
    app, manager, _backends = _terminal_app(permitted=True)
    _create_running_terminal(manager, tmp_path, name="Terminal 1")
    second = manager.create_session(
        TerminalLaunchRequest(
            name="Terminal 2",
            shell="default",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        )
    )
    assert second.admitted is True
    host = ConsoleHarness(app)

    async with host.run_test(size=terminal_size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        left = console.query_one("#console-left-rail")
        right = console.query_one("#console-right-rail")
        if explicit_inspector:
            console._set_console_rail_preference(right_open=True)
            await _wait_until(pilot, lambda: right.display is True)
        preferred = (
            console._current_console_rail_state().preferred_left_open,
            console._current_console_rail_state().preferred_right_open,
        )

        console.action_open_console_terminal()
        await _wait_for_selector(console, pilot, "#console-terminal-viewport")
        workspace = console.query_one("#console-main-column", ConsoleTerminalWorkspace)
        grid = console.query_one("#console-workspace-grid")
        if rails_yield:
            await _wait_until(
                pilot,
                lambda: not left.display
                and not right.display
                and workspace.region.width == grid.content_region.width,
                detail=lambda: (left.display, right.display, workspace.region, grid.content_region),
            )
        await pilot.pause()

        # The user's rail preference is untouched by the yield.
        state = console._current_console_rail_state()
        assert (state.preferred_left_open, state.preferred_right_open) == preferred

        # Every visible label is painted whole: never a word cut short.
        buttons = _visible_terminal_buttons(workspace)
        labels = [str(button.label) for button in buttons]
        assert "Terminal 1" in labels and "Terminal 2" in labels, labels
        strips = list(console._compositor.render_strips())
        for button in buttons:
            label = str(button.label)
            region = button.content_region
            painted = strips[region.y].crop(region.x, region.right).text.strip()
            assert painted == label, (terminal_size, button.id, label, painted, region)

        assert await pilot.click("#console-terminal-return")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        if explicit_inspector and terminal_size[0] < 100:
            # Returning gives the rail the user opened back to them.
            await _wait_until(pilot, lambda: right.display is True)

    manager.finalize_shutdown()


@pytest.mark.asyncio
async def test_rail_toggles_still_open_a_rail_over_a_narrow_terminal(
    tmp_path: Path,
) -> None:
    """TASK-33625.3: the yield is a default, never a lock on Alt+I / Alt+C.

    Without this, a rail toggle under a narrow Terminal wrote a preference
    and showed nothing -- the Alt+I "way back" (TASK-24604) went dead.
    """
    app, manager, _backends = _terminal_app(permitted=True)
    _create_running_terminal(manager, tmp_path, name="Terminal 1")
    host = ConsoleHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        left = console.query_one("#console-left-rail")
        right = console.query_one("#console-right-rail")
        console.action_open_console_terminal()
        await _wait_for_selector(console, pilot, "#console-terminal-viewport")
        await _wait_until(pilot, lambda: not left.display and not right.display)

        await console.action_toggle_console_inspector_rail()
        await _wait_until(
            pilot,
            lambda: right.display is True,
            detail=lambda: (left.display, right.display),
        )
        await console.action_toggle_console_inspector_rail()
        await _wait_until(pilot, lambda: right.display is False)

        console.action_toggle_console_context_rail()
        await _wait_until(
            pilot,
            lambda: left.display is True,
            detail=lambda: (left.display, right.display),
        )

        # Opening the Terminal again starts from the yield once more.
        assert await pilot.click("#console-terminal-return")
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        console.action_open_console_terminal()
        await _wait_for_selector(console, pilot, "#console-terminal-viewport")
        await _wait_until(
            pilot,
            lambda: not left.display and not right.display,
            detail=lambda: (left.display, right.display),
        )

    manager.finalize_shutdown()
