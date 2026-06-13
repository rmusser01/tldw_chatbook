"""Current-contract tests for the evals window compatibility export."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.screen import Screen

from tldw_chatbook.UI.evals_window_v2 import EvalsWindow
from tldw_chatbook.UI.Evals.navigation import EvalNavigationScreen, NavigateToEvalScreen
from tldw_chatbook.UI.Evals.screens import EvaluationBrowserScreen, QuickTestScreen


class EvalsWindowHost(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._window = EvalsWindow(app_instance=app_instance)

    def compose(self) -> ComposeResult:
        yield self._window


def test_evals_window_maps_current_routes_to_v3_screens() -> None:
    app_instance = SimpleNamespace(notify=lambda *args, **kwargs: None)
    window = EvalsWindow(app_instance=app_instance)

    assert isinstance(window._create_screen("quick_test"), QuickTestScreen)
    assert isinstance(window._create_screen("tasks"), EvaluationBrowserScreen)
    assert window._create_screen("tasks").view_mode == "manage"
    assert window._create_screen("results").view_mode == "results"


@pytest.mark.asyncio
async def test_evals_window_mounts_navigation_screen_by_default() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = EvalsWindowHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause()

        window = app.query_one(EvalsWindow)
        assert isinstance(window.current_screen, EvalNavigationScreen)
        assert window.screen_stack == []


@pytest.mark.asyncio
async def test_evals_window_navigation_pushes_stack_and_go_back() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = EvalsWindowHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause()

        window = app.query_one(EvalsWindow)

        with patch.object(window, "_create_screen", return_value=Screen(id="dummy-eval-screen")):
            window.handle_navigation(NavigateToEvalScreen("quick_test"))
            await pilot.pause()

        assert len(window.screen_stack) == 1

        window.go_back()
        await pilot.pause()

        assert isinstance(window.current_screen, EvalNavigationScreen)
        assert window.screen_stack == []
