"""Regression tests for the round-7 UX batch (UX-071/077/078/054)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Input

from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Logs_Window import LogRecord, LogsWindow, _passes_filter


def _llm_harness():
    app_instance = MagicMock(notify=MagicMock())
    app_instance._llm_server_launch_claims = {}
    app_instance.llamacpp_server_process = None
    app_instance.llamafile_server_process = None

    class Harness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LLMManagementWindow(app_instance)

    return Harness


# UX-078 -----------------------------------------------------------------
def test_discover_binary_via_path_and_common_dirs(tmp_path) -> None:
    with patch("shutil.which", return_value="/usr/local/bin/llama-server"):
        assert LLMManagementWindow._discover_binary(("llama-server",)) == (
            "/usr/local/bin/llama-server"
        )
    with patch("shutil.which", return_value=None):
        fake = tmp_path / "llamafile"
        fake.write_text("#!/bin/sh\n")
        with patch("pathlib.Path.home", return_value=tmp_path.parent.parent):
            # Not on the fake search path; assert the miss path returns None.
            assert LLMManagementWindow._discover_binary(("nope-not-real",)) is None


@pytest.mark.asyncio
async def test_detect_button_fills_input_and_notifies() -> None:
    Harness = _llm_harness()
    app = Harness()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        window = app.query_one(LLMManagementWindow)
        with patch.object(
            LLMManagementWindow,
            "_discover_binary",
            staticmethod(lambda names: "/opt/homebrew/bin/llama-server"),
        ):
            window.query_one("#llamacpp-detect-exec-button").press()
            await pilot.pause()
        value = window.query_one("#llamacpp-exec-path", Input).value
        assert value == "/opt/homebrew/bin/llama-server"
        window.app_instance.notify.assert_called()


# UX-054 -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_lab_form_rows_are_side_by_side() -> None:
    Harness = _llm_harness()
    app = Harness()
    async with app.run_test(size=(140, 42)) as pilot:
        await pilot.pause()
        container = app.query_one("#llamacpp-exec-path").parent
        labels = [w for w in container.children if w.has_class("inline-label")]
        assert labels, "label must live inside the input row"
        assert "Server Executable" in str(labels[0].render())


# UX-077 (regex + saved filters) ------------------------------------------
def test_regex_filter_and_substring_fallback() -> None:
    error_rec = LogRecord("ERROR", "m", "Error 500 on worker 3")
    import re

    pattern = re.compile(r"Error \d+", re.IGNORECASE)
    assert _passes_filter(error_rec, "all", r"Error \d+", pattern)
    assert not _passes_filter(error_rec, "all", r"Error \d+", re.compile("404"))
    # Invalid regex falls back to substring semantics (pattern=None).
    assert _passes_filter(error_rec, "all", "Error 500", None)
    assert not _passes_filter(error_rec, "all", "nope", None)


def test_saved_filter_roundtrip() -> None:
    saved = {}

    class LogsApp(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LogsWindow(SimpleNamespace(_log_records=deque()))

    def _batched_save(section_values):
        # task-21124: save_filter_state persists via ONE batched
        # save_settings_to_cli_config mutation instead of two sequential
        # save_setting_to_cli_config rewrites.
        for section, values in section_values.items():
            for key, value in values.items():
                saved[(section, key)] = value
        return True

    with (
        patch(
            "tldw_chatbook.config.save_settings_to_cli_config",
            _batched_save,
        ),
        patch(
            "tldw_chatbook.config.get_cli_setting",
            lambda section, key, default=None: saved.get((section, key), default),
        ),
    ):
        app = LogsApp()
        # First session: set a filter and save it.
        import asyncio

        async def run():
            async with app.run_test(size=(120, 36)) as pilot:
                window = app.query_one(LogsWindow)
                await pilot.pause()
                window.query_one("#logs-filter-text", Input).value = "tts"
                window._level_chip = "error"
                window.save_filter_state()

        asyncio.run(run())
        assert saved[("logs", "last_filter")] == "tts"
        assert saved[("logs", "last_level_chip")] == "error"


# UX-077 (bulk marks) ------------------------------------------------------
@pytest.mark.asyncio
async def test_bulk_mark_and_toggle() -> None:
    from Tests.UI.test_schedules_workbench import WorkbenchTestAppWithService

    from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
        SchedulesWorkbench,
    )

    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        screen = pilot.app.screen

        screen.action_mark_task()
        await pilot.pause()
        assert len(screen._marked_ids) == 1
        # Marked row carries the dot prefix.
        from textual.widgets import DataTable

        table = screen.query_one("#scheduling-task-table", DataTable)
        # redesign PR-2, Task 2: column 0 is now the glyph, column 1 the
        # title (old single-primitive shape was Title/Type/Status/Next Run).
        first_title = str(table.get_row_at(0)[1])
        assert first_title.startswith("● ")

        screen.action_toggle_enabled()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        service = pilot.app.scheduling_service
        assert service.updated  # bulk path wrote through the service
        assert screen._marked_ids == set()

        screen.action_clear_marks()  # no-op when nothing marked: must not raise
