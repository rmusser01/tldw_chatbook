"""Shell-wide workbench pane focus convention tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App
from textual.widgets import Button

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.notes_screen import NotesScreen
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class _ConsoleHarness(App[None]):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


def _notes_app_instance() -> Mock:
    app = Mock()
    service = Mock()
    service.list_notes.return_value = []
    service.get_keywords_for_note.return_value = []
    app.notes_service = service
    app.notes_user_id = "default_user"
    app.notes_scope_service = Mock()
    app.server_notes_workspace_service = Mock()
    app.notify = Mock()
    app.open_study_screen = Mock()
    app.open_notes_workspace = Mock()
    app.push_screen = Mock()
    app.push_screen_wait = AsyncMock(return_value=True)
    app.call_from_thread = Mock()
    app.loguru_logger = Mock()
    app.current_selected_note_id = None
    app.current_selected_note_version = None
    app.current_selected_note_title = ""
    app.current_selected_note_content = ""
    return app


class _NotesHarness(App[None]):
    def __init__(self, screen: NotesScreen) -> None:
        super().__init__()
        self.screen_under_test = screen
        self.app_instance = screen.app_instance

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


class _PersonasHarness(App[None]):
    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    def compose(self):
        yield AppFooterStatus(id="app-footer-status")

    async def on_mount(self) -> None:
        await self.push_screen(PersonasScreen(self.app_instance))


async def _wait_for_focused_id(app: App[None], pilot, widget_id: str) -> None:
    for _ in range(40):
        if getattr(app.focused, "id", None) == widget_id:
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Expected focus on {widget_id!r}, found {getattr(app.focused, 'id', None)!r}"
    )


@pytest.mark.asyncio
async def test_console_f6_cycles_between_workbench_panes_and_wraps_backward():
    app_instance = _build_test_app()
    host = _ConsoleHarness(app_instance)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        console._set_console_rail_preference(
            left_open=True,
            right_open=True,
            notify_on_failure=False,
        )
        await pilot.pause()
        console.query_one("#console-native-composer").focus()

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-context-rail-collapse")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-native-transcript")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-inspector-rail-collapse")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "console-native-composer")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(host, pilot, "console-inspector-rail-collapse")


@pytest.mark.asyncio
async def test_notes_f6_cycles_between_workbench_panes_without_using_tab_order():
    screen = NotesScreen(_notes_app_instance())
    host = _NotesHarness(screen)

    async with host.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        screen.query_one("#notes-editor-area").focus()

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "notes-inspector-rail-collapse")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "notes-search-input")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "notes-editor-area")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(host, pilot, "notes-search-input")


@pytest.mark.asyncio
async def test_personas_f6_cycles_between_workbench_panes_from_text_input():
    app_instance = _build_test_app()
    host = _PersonasHarness(app_instance)

    async with host.run_test(size=(140, 42)) as pilot:
        personas = host.screen_stack[-1]
        await _wait_for_selector(personas, pilot, "#personas-library-search")
        personas.query_one("#personas-library-search").focus()

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "personas-preview-toggle")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "personas-conversations-list")

        await pilot.press("f6")
        await _wait_for_focused_id(host, pilot, "personas-library-search")

        await pilot.press("shift+f6")
        await _wait_for_focused_id(host, pilot, "personas-conversations-list")


def test_workbench_screens_expose_f6_bindings_without_ctrl_arrow_conflicts():
    screen_classes = (ChatScreen, NotesScreen, PersonasScreen)
    for screen_class in screen_classes:
        bindings = getattr(screen_class, "BINDINGS", ())
        keys = {binding[0] if isinstance(binding, tuple) else binding.key for binding in bindings}
        assert "f6" in keys
        assert "shift+f6" in keys
        assert "ctrl+left" not in keys
        assert "ctrl+right" not in keys
