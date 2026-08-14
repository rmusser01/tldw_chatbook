"""Regression tests for the round-6 UX batch (UX-077 remainder, UX-078)."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Input, Static


# Queue filter (UX-077) ---------------------------------------------------
@pytest.mark.asyncio
async def test_queue_filter_narrows_rows_and_detail_follows_visible() -> None:
    from Tests.UI.test_schedules_workbench import (  # noqa: PLC0415
        WorkbenchTestAppWithService,
    )
    from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
        QUEUE_FILTER_DEBOUNCE_SECONDS,
        SchedulesWorkbench,
    )

    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        screen = pilot.app.screen
        assert len(screen._visible_tasks) == len(screen._tasks)

        screen.query_one("#scheduling-queue-filter", Input).value = "no-such-task"
        # Debounced (task-15476): the queue table only rebuilds once the
        # filter settles, not on every keystroke.
        await pilot.pause(QUEUE_FILTER_DEBOUNCE_SECONDS + 0.1)
        assert screen._visible_tasks == []
        empty = str(
            screen.query_one("#scheduling-task-detail-empty-state", Static).render()
        )
        assert "No tasks match" in empty

        screen.query_one("#scheduling-queue-filter", Input).value = ""
        await pilot.pause(QUEUE_FILTER_DEBOUNCE_SECONDS + 0.1)
        assert len(screen._visible_tasks) == len(screen._tasks)


# Lab position indicator + autofill (UX-077/078) --------------------------
@pytest.mark.asyncio
async def test_ollama_path_autofills_when_found() -> None:
    from unittest.mock import patch

    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    class Harness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LLMManagementWindow(None)

    with patch("shutil.which", return_value="/usr/local/bin/ollama"):
        app = Harness()
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            await pilot.pause(0.5)
            value = app.query_one("#ollama-exec-path", Input).value
            assert value == "/usr/local/bin/ollama"
