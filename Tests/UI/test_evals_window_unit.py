"""Unit tests for the current quick test eval screen."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.widgets import Input, Select, Static

from tldw_chatbook.UI.Evals.navigation.nav_bar import EvalStatus
from tldw_chatbook.UI.Evals.screens.quick_test import QuickTestScreen


class QuickTestHost(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = QuickTestScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


@pytest.fixture
def orchestrator_patch():
    with patch("tldw_chatbook.UI.Evals.screens.quick_test.EvaluationOrchestrator") as mock_cls:
        orchestrator = MagicMock()
        orchestrator.db.list_tasks.return_value = [
            {"id": "task-1", "name": "Summarization"},
            {"id": "task-2", "name": "Classification"},
        ]
        orchestrator.db.list_models.return_value = [
            {"id": "model-1", "name": "GPT-4.1 Mini", "provider": "openai"},
            {"id": "model-2", "name": "Claude Haiku", "provider": "anthropic"},
        ]
        mock_cls.return_value = orchestrator
        yield mock_cls


def _option_values(select: Select) -> list[str]:
    values = []
    for prompt, value in getattr(select, "_options", []):
        if value is Select.BLANK or str(value).startswith("Select."):
            continue
        if value is not Select.BLANK:
            values.append(str(value))
    return values


@pytest.mark.asyncio
async def test_quick_test_screen_loads_tasks_and_models(orchestrator_patch) -> None:
    app = QuickTestHost(SimpleNamespace(notify=MagicMock()))

    async with app.run_test() as pilot:
        await pilot.pause()

        task_select = app.screen.query_one("#task-select", Select)
        model_select = app.screen.query_one("#model-select", Select)

        assert _option_values(task_select) == ["task-1", "task-2"]
        assert _option_values(model_select) == ["model-1", "model-2"]


@pytest.mark.asyncio
async def test_quick_test_screen_selection_updates_state(orchestrator_patch) -> None:
    app = QuickTestHost(SimpleNamespace(notify=MagicMock()))

    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        task_select = screen.query_one("#task-select", Select)
        model_select = screen.query_one("#model-select", Select)

        task_select.value = "task-2"
        model_select.value = "model-1"
        await pilot.pause()

        assert screen.selected_task_id == "task-2"
        assert screen.selected_model_id == "model-1"


@pytest.mark.asyncio
async def test_quick_test_screen_validates_missing_configuration(orchestrator_patch) -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = QuickTestHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        screen.action_run_evaluation()
        app_instance.notify.assert_called_with("Please select a task", severity="error")

        screen.selected_task_id = "task-1"
        app_instance.notify.reset_mock()
        screen.action_run_evaluation()
        app_instance.notify.assert_called_with("Please select a model", severity="error")


@pytest.mark.asyncio
async def test_quick_test_screen_queues_worker_when_valid(orchestrator_patch) -> None:
    app = QuickTestHost(SimpleNamespace(notify=MagicMock()))

    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        screen.selected_task_id = "task-1"
        screen.selected_model_id = "model-1"
        screen.query_one("#samples-input", Input).value = "12"
        screen.query_one("#temp-input", Input).value = "0.5"

        with patch.object(screen, "run_worker") as run_worker:
            screen.action_run_evaluation()

        progress_section = screen.query_one("#progress-section")

        assert screen.evaluation_running is True
        assert "active" in progress_section.classes
        assert screen.nav_bar.status == EvalStatus.RUNNING
        run_worker.assert_called_once()
