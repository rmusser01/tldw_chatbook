"""Regression tests for the round-7 UX batch (UX-071/077/078/054)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container as _Container
from textual.widgets import Input

from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Logs_Window import LogRecord, LogsWindow, _passes_filter


def _llm_harness():
    import tldw_chatbook.Widgets.HuggingFace as hf

    class _StubWidget(_Container):
        def __init__(self, *args, **kwargs):
            super().__init__(**{k: v for k, v in kwargs.items() if k == "id"})

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield LLMManagementWindow(MagicMock(notify=MagicMock()))

    monkey = pytest.MonkeyPatch()
    monkey.setattr(hf, "LocalModelsWidget", _StubWidget)
    monkey.setattr(hf, "HuggingFaceModelBrowser", _StubWidget)
    return Harness, monkey


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
    Harness, monkey = _llm_harness()
    try:
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
    finally:
        monkey.undo()


# UX-054 -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_lab_form_rows_are_side_by_side() -> None:
    Harness, monkey = _llm_harness()
    try:
        app = Harness()
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            container = app.query_one("#llamacpp-exec-path").parent
            labels = [w for w in container.children if w.has_class("inline-label")]
            assert labels, "label must live inside the input row"
            assert "Executable Path" in str(labels[0].render())
    finally:
        monkey.undo()


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

    class LogsApp(App[None]):
        def compose(self) -> ComposeResult:
            yield LogsWindow(SimpleNamespace(_log_records=deque()))

    with patch(
        "tldw_chatbook.config.save_setting_to_cli_config",
        lambda section, key, value: saved.__setitem__((section, key), value),
    ), patch(
        "tldw_chatbook.config.get_cli_setting",
        lambda section, key, default=None: saved.get((section, key), default),
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
        first_title = str(table.get_row_at(0)[0])
        assert first_title.startswith("● ")

        screen.action_toggle_enabled()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        service = pilot.app.scheduling_service
        assert service.updated  # bulk path wrote through the service
        assert screen._marked_ids == set()

        screen.action_clear_marks()  # no-op when nothing marked: must not raise
