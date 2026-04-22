"""UI tests for the current eval navigation hub."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.UI.Evals.navigation import EvalNavigationScreen, NavigateToEvalScreen


class EvalNavigationHost(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._screen = EvalNavigationScreen(app_instance=app_instance)

    async def on_mount(self) -> None:
        await self.push_screen(self._screen)


def _text(widget: Static) -> str:
    rendered = widget.render()
    return getattr(rendered, "plain", str(rendered))


@pytest.mark.asyncio
async def test_eval_navigation_screen_renders_current_cards_and_status() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = EvalNavigationHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause()

        cards = app.screen.query(".nav-card-button")
        status = app.screen.query_one("#status-text", Static)

        assert len(cards) == 6
        assert app.screen.query_one("#card-quick_test", Button)
        assert app.screen.query_one("#card-results", Button)
        assert "Ready - Choose a workflow" in _text(status)


def test_eval_navigation_screen_posts_message_when_navigating() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    screen = EvalNavigationScreen(app_instance=app_instance)

    with patch.object(screen, "post_message") as post_message:
        screen._navigate_to("quick_test")

    message = post_message.call_args[0][0]
    assert isinstance(message, NavigateToEvalScreen)
    assert message.screen_id == "quick_test"
    app_instance.notify.assert_called_once()


def test_eval_navigation_screen_warns_when_running_last_without_history() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    screen = EvalNavigationScreen(app_instance=app_instance)

    screen.action_run_last()

    app_instance.notify.assert_called_once_with(
        "No previous evaluation to run",
        severity="warning",
    )
