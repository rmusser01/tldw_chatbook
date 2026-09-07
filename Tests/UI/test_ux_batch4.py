"""Regression tests for the round-4 UX batch (UX-069, 070, 072-076, 079, 080)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Input, Select, Static

from tldw_chatbook.UI.Logs_Window import LogRecord, LogsWindow, _styled_line
from tldw_chatbook.UI.Navigation.main_navigation import nav_button_label
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm


# UX-072 -----------------------------------------------------------------
def test_every_destination_has_a_hotkey_route() -> None:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.shell_destinations import (
        SHELL_DESTINATION_SHORTCUTS,
    )

    actions = {
        binding.action
        for binding in TldwCli.BINDINGS
        if binding.action.startswith("shell_destination(")
    }
    expected = {
        f"shell_destination({destination_id!r})"
        for destination_id in SHELL_DESTINATION_SHORTCUTS
    }
    assert actions == expected


def test_fkey_labels_on_late_destinations() -> None:
    assert nav_button_label("research", "Research") == "F10 Research"
    assert nav_button_label("lab", "Lab") == "F7 Lab"
    assert nav_button_label("logs", "Logs") == "F8 Logs"
    assert nav_button_label("settings", "Settings") == "F9 Settings"


# UX-069 -----------------------------------------------------------------
class _FormHarness(ConsolidatedCSSApp):
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
        preset.value = "monday"  # Every Monday at... (time defaults to 09:00)
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
def test_lab_default_view_starts_unset_until_children_exist() -> None:
    """`active_view` deliberately starts at "" with init=False (see the
    reactive's own comment: a real default fired the watcher before the
    deferred body mount — ten QueryErrors per arrival). The real initial
    view (llama-cpp since 9dd2374b5) is assigned by `_initialize_view`
    after the children exist — pinned in test_llm_deferred_views.py."""
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    window = LLMManagementWindow(None)
    assert window.active_view == ""


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

    class FooterApp(ConsolidatedCSSApp):
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

    class LogsApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LogsWindow(SimpleNamespace(_log_records=deque(), _log_buffer=buffer))

        def copy_to_clipboard(self, text: str) -> None:
            copied["text"] = text

    app = LogsApp()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.query_one(LogsWindow)._on_copy_all()
        assert copied["text"] == "line one\nline two\nline three"
