"""Regression tests for the round-4 UX batch (UX-069, 070, 072-076, 079, 080)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Select, Static

from tldw_chatbook.UI.Logs_Window import LogRecord, LogsWindow, _styled_line
from tldw_chatbook.UI.Navigation.main_navigation import nav_button_label
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm


# UX-072 -----------------------------------------------------------------
def test_every_destination_has_a_hotkey_route() -> None:
    from tldw_chatbook.app import TldwCli

    actions = {
        binding.action
        for binding in TldwCli.BINDINGS
        if binding.action.startswith("shell_destination(")
    }
    expected = {
        f"shell_destination({index})" for index in range(len(SHELL_DESTINATION_ORDER))
    }
    assert actions == expected


def test_fkey_labels_on_late_destinations() -> None:
    assert nav_button_label(10, "Lab") == "F7 Lab"
    assert nav_button_label(11, "Logs") == "F8 Logs"
    assert nav_button_label(12, "Settings") == "F9 Settings"


# UX-069 -----------------------------------------------------------------
class _FormHarness(App[None]):
    def __init__(self):
        super().__init__()
        self.dismissed_with = "NOT_DISMISSED"

    def compose(self) -> ComposeResult:
        yield Static("harness")


@pytest.mark.asyncio
async def test_preset_fills_cron_and_preview_humanizes() -> None:
    app = _FormHarness()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = app.screen
        assert isinstance(form, ReminderForm)
        kind = form.query_one("#reminder-kind", Select)
        kind.value = "recurring"
        await pilot.pause()

        preset = form.query_one("#reminder-cron-preset", Select)
        preset.value = "0 9 * * 1"  # Every Monday at 09:00
        await pilot.pause()

        assert form.query_one("#reminder-cron", Input).value == "0 9 * * 1"
        preview = str(form.query_one("#reminder-cron-preview", Static).render())
        assert "Weekly on Monday" in preview


@pytest.mark.asyncio
async def test_custom_cron_invalid_shows_preview_guidance() -> None:
    app = _FormHarness()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = app.screen
        kind = form.query_one("#reminder-kind", Select)
        kind.value = "recurring"
        await pilot.pause()
        preset = form.query_one("#reminder-cron-preset", Select)
        preset.value = "custom"
        cron = form.query_one("#reminder-cron", Input)
        cron.value = "not a cron"
        await pilot.pause()
        preview = str(form.query_one("#reminder-cron-preview", Static).render())
        assert "Not a valid cron" in preview
        assert preset.value == "custom"


# UX-080 (dirty guard) ----------------------------------------------------
@pytest.mark.asyncio
async def test_clean_form_escape_dismisses_without_dialog() -> None:
    app = _FormHarness()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm(), callback=lambda r: None)
        await pilot.pause()
        assert isinstance(app.screen, ReminderForm)
        app.screen.action_dismiss()
        await pilot.pause()
        assert not isinstance(app.screen, ReminderForm)


@pytest.mark.asyncio
async def test_dirty_form_escape_asks_before_discarding() -> None:
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    app = _FormHarness()
    async with app.run_test() as pilot:
        await app.push_screen(ReminderForm())
        await pilot.pause()
        form = app.screen
        form.query_one("#reminder-title", Input).value = "typed something"
        await pilot.pause()
        form.action_dismiss()
        await pilot.pause()
        assert isinstance(app.screen, ConfirmationDialog)
        # Keep editing returns to the form.
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert isinstance(app.screen, ReminderForm)


# UX-070 -----------------------------------------------------------------
def test_lab_default_view_is_ollama() -> None:
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    window = LLMManagementWindow(None)
    assert window.active_view == "ollama"


def test_ollama_prereq_reports_detection() -> None:
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    window = LLMManagementWindow(None)
    with patch("shutil.which", return_value="/usr/local/bin/ollama"):
        assert "found: /usr/local/bin/ollama" in window._ollama_prereq_text()
    with patch("shutil.which", return_value=None):
        assert "not found on PATH" in window._ollama_prereq_text()


# UX-075 -----------------------------------------------------------------
def test_log_lines_styled_by_level_with_text_intact() -> None:
    error = _styled_line(LogRecord("ERROR", "m", "an error line"))
    assert error.plain == "an error line"
    assert error.style == "bold bright_red"
    warning = _styled_line(LogRecord("WARNING", "m", "a warning line"))
    assert warning.style == "bright_yellow"
    info = _styled_line(LogRecord("INFO", "m", "an info line"))
    assert info.style == ""


# UX-076 -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_token_display_hidden_on_non_chat_screens() -> None:
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

    class FooterApp(App[None]):
        def compose(self) -> ComposeResult:
            yield AppFooterStatus(show_token_count=False)

    app = FooterApp()
    async with app.run_test(size=(140, 24)) as pilot:
        await pilot.pause()
        footer = app.query_one(AppFooterStatus)
        assert footer._token_count_display.display is False


# UX-079 -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_copy_all_uses_full_session_buffer() -> None:
    from collections import deque

    buffer = deque(["line one", "line two", "line three"])
    copied = {}

    class LogsApp(App[None]):
        def compose(self) -> ComposeResult:
            yield LogsWindow(SimpleNamespace(_log_records=deque(), _log_buffer=buffer))

        def copy_to_clipboard(self, text: str) -> None:
            copied["text"] = text

    app = LogsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.query_one(LogsWindow)._on_copy_all()
        assert copied["text"] == "line one\nline two\nline three"
