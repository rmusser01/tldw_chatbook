"""Regression tests for the Evals shell seat (dead-end fix).

The Evals destination used to push EvalNavigationScreen on mount, hiding the
shell chrome and stranding users on a permanent "Loading Evaluation Lab..."
placeholder after Escape. The workbench now renders inline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App

from tldw_chatbook.UI.Evals.evals_window_v3 import EvalsWindowV3
from tldw_chatbook.UI.Evals.navigation import EvalNavigationScreen
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen


class EvalsScreenHost(App[None]):
    def __init__(self, app_instance):
        super().__init__()
        self._app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(EvalsScreen(self._app_instance))


def _host() -> EvalsScreenHost:
    return EvalsScreenHost(SimpleNamespace(notify=MagicMock()))


@pytest.mark.asyncio
async def test_evals_screen_mounts_inline_without_pushing() -> None:
    app = _host()

    async with app.run_test() as pilot:
        await pilot.pause()

        # The old flow pushed EvalNavigationScreen over EvalsScreen on mount.
        assert isinstance(app.screen, EvalsScreen)
        assert len(app.screen_stack) == 2  # default screen + EvalsScreen only

        # The permanent placeholder is gone; the hub renders inline instead.
        assert not app.screen.query("#evals-placeholder")
        assert app.screen.query_one("#evals-destination-header")
        window = app.screen.query_one(EvalsWindowV3)
        assert isinstance(window.current_screen, EvalNavigationScreen)
        assert len(app.screen.query(".nav-card-button")) == 6


@pytest.mark.asyncio
async def test_escape_at_hub_keeps_evals_screen() -> None:
    app = _host()

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        # Escape must not pop the shell screen or dead-end on the placeholder:
        # with an empty workbench back stack it is a no-op.
        assert isinstance(app.screen, EvalsScreen)
        window = app.screen.query_one(EvalsWindowV3)
        assert isinstance(window.current_screen, EvalNavigationScreen)


@pytest.mark.asyncio
async def test_digit_shortcut_warns_for_unimplemented_workflow() -> None:
    app_instance = SimpleNamespace(notify=MagicMock())
    app = EvalsScreenHost(app_instance)

    async with app.run_test() as pilot:
        await pilot.pause()

        screen = app.screen
        screen.action_evals_open(
            "comparison"
        )  # card exists, screen not yet implemented
        await pilot.pause()

        window = screen.query_one(EvalsWindowV3)
        assert isinstance(window.current_screen, EvalNavigationScreen)
        assert any(
            "not yet implemented" in str(call.args[0])
            for call in app_instance.notify.call_args_list
            if call.args
        )
