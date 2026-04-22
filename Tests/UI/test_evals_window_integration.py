"""Integration tests for the current evals window router."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from tldw_chatbook.UI.Evals.navigation import EvalNavigationScreen, NavigateToEvalScreen
from tldw_chatbook.UI.Evals.screens import QuickTestScreen


class EvalsWindowHost(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._window = EvalsWindow(app_instance=app_instance)

    def compose(self) -> ComposeResult:
        yield self._window


@pytest.mark.asyncio
async def test_evals_window_navigates_to_quick_test_and_resets_home() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = EvalsWindowHost(app_instance)

    with patch("tldw_chatbook.UI.Evals.screens.quick_test.EvaluationOrchestrator") as mock_cls:
        orchestrator = MagicMock()
        orchestrator.db.list_tasks.return_value = []
        orchestrator.db.list_models.return_value = []
        mock_cls.return_value = orchestrator

        async with app.run_test() as pilot:
            await pilot.pause()

            window = app.query_one(EvalsWindow)
            window.handle_navigation(NavigateToEvalScreen("quick_test"))
            await pilot.pause()

            assert isinstance(window.current_screen, QuickTestScreen)
            assert len(window.screen_stack) == 1

            window.reset_to_home()
            await pilot.pause()

            assert isinstance(window.current_screen, EvalNavigationScreen)
            assert window.screen_stack == []


@pytest.mark.asyncio
async def test_evals_window_warns_for_unknown_screen() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = EvalsWindowHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause()

        window = app.query_one(EvalsWindow)
        window.handle_navigation(NavigateToEvalScreen("comparison"))
        await pilot.pause()

        app_instance.notify.assert_called_with(
            "Screen 'comparison' not yet implemented",
            severity="warning",
        )
