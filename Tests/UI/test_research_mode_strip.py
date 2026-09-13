"""Shared Workspace/Runs navigation for the two real Research screens."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual import on
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


class _ResearchScreenHarness(ConsolidatedCSSApp):
    def __init__(self, route: str) -> None:
        super().__init__()
        self.route = route
        self.navigation: list[str] = []

    async def on_mount(self) -> None:
        from tldw_chatbook.UI.Screens.research_screen import ResearchScreen
        from tldw_chatbook.UI.Screens.research_workspace_screen import (
            ResearchWorkspaceScreen,
        )

        screen_type = (
            ResearchWorkspaceScreen
            if self.route == "research_workspace"
            else ResearchScreen
        )
        await self.push_screen(screen_type(SimpleNamespace()))

    @on(NavigateToScreen)
    def capture_navigation(self, message: NavigateToScreen) -> None:
        self.navigation.append(message.screen_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "active_id", "target_id", "target_route"),
    [
        (
            "research_workspace",
            "research-mode-workspace",
            "research-mode-runs",
            "research",
        ),
        (
            "research",
            "research-mode-runs",
            "research-mode-workspace",
            "research_workspace",
        ),
    ],
)
async def test_real_research_screens_share_mode_strip_and_navigate_real_routes(
    route: str,
    active_id: str,
    target_id: str,
    target_route: str,
) -> None:
    from textual.widgets import Button

    from tldw_chatbook.UI.Research_Workspace_Modules.mode_bar import ResearchModeStrip

    app = _ResearchScreenHarness(route)
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        strips = list(screen.query(ResearchModeStrip))
        assert len(strips) == 1
        assert screen.query_one(f"#{active_id}", Button).has_class("is-active")
        screen.query_one(f"#{target_id}", Button).press()
        await pilot.pause()

    assert app.navigation == [target_route]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["research_workspace", "research"])
async def test_research_screens_do_not_embed_each_other(route: str) -> None:
    from tldw_chatbook.UI.Screens.research_screen import ResearchScreen
    from tldw_chatbook.UI.Screens.research_workspace_screen import (
        ResearchWorkspaceScreen,
    )

    app = _ResearchScreenHarness(route)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen = app.screen_stack[-1]
        assert isinstance(screen, ResearchScreen) == (route == "research")
        assert isinstance(screen, ResearchWorkspaceScreen) == (
            route == "research_workspace"
        )
        if route == "research":
            assert not list(screen.query(ResearchWorkspaceScreen))
        else:
            assert not list(screen.query(ResearchScreen))
