"""Regression tests for the round-6 UX batch (UX-077 remainder, UX-078)."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container as _Container
from textual.widgets import Input, Static


# Queue filter (UX-077) ---------------------------------------------------
@pytest.mark.asyncio
async def test_queue_filter_narrows_rows_and_detail_follows_visible() -> None:
    from Tests.UI.test_schedules_workbench import (  # noqa: PLC0415
        WorkbenchTestAppWithService,
    )
    from tldw_chatbook.UI.Screens.scheduling.schedules_workbench import (
        SchedulesWorkbench,
    )

    async with WorkbenchTestAppWithService().run_test() as pilot:
        await pilot.app.push_screen(SchedulesWorkbench(app_instance=pilot.app))
        await pilot.pause()
        screen = pilot.app.screen
        assert len(screen._visible_tasks) == len(screen._tasks)

        screen.query_one("#scheduling-queue-filter", Input).value = "no-such-task"
        await pilot.pause()
        assert screen._visible_tasks == []
        empty = str(
            screen.query_one("#scheduling-task-detail-empty-state", Static).render()
        )
        assert "No tasks match" in empty

        screen.query_one("#scheduling-queue-filter", Input).value = ""
        await pilot.pause()
        assert len(screen._visible_tasks) == len(screen._tasks)


# Lab position indicator + autofill (UX-077/078) --------------------------
@pytest.mark.asyncio
async def test_lab_sidebar_hint_shows_position_and_cycles() -> None:
    import tldw_chatbook.Widgets.HuggingFace as hf
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    class _StubWidget(_Container):
        def __init__(self, *args, **kwargs):
            super().__init__(**{k: v for k, v in kwargs.items() if k == "id"})

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield LLMManagementWindow(None)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(hf, "LocalModelsWidget", _StubWidget)
    monkey.setattr(hf, "HuggingFaceModelBrowser", _StubWidget)
    try:
        app = Harness()
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            window = app.query_one(LLMManagementWindow)
            hint = app.query_one("#llm-sidebar-hint", Static)
            total = len(window.view_mapping)
            assert f"of {total}" in str(hint.render())
            assert "1 of" in str(hint.render())  # Ollama is the default view

            window._cycle_view(1)
            await pilot.pause()
            assert "2 of" in str(hint.render())
            window._cycle_view(-1)
            await pilot.pause()
            assert "1 of" in str(hint.render())
    finally:
        monkey.undo()


@pytest.mark.asyncio
async def test_ollama_path_autofills_when_found() -> None:
    from unittest.mock import patch

    import tldw_chatbook.Widgets.HuggingFace as hf
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow

    class _StubWidget(_Container):
        def __init__(self, *args, **kwargs):
            super().__init__(**{k: v for k, v in kwargs.items() if k == "id"})

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield LLMManagementWindow(None)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(hf, "LocalModelsWidget", _StubWidget)
    monkey.setattr(hf, "HuggingFaceModelBrowser", _StubWidget)
    try:
        with patch("shutil.which", return_value="/usr/local/bin/ollama"):
            app = Harness()
            async with app.run_test(size=(140, 42)) as pilot:
                await pilot.pause()
                await pilot.pause(0.5)
                value = app.query_one("#ollama-exec-path", Input).value
                assert value == "/usr/local/bin/ollama"
    finally:
        monkey.undo()
