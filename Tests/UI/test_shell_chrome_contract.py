"""Guardrails for shell-owned chrome and primary destination metadata."""

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


@pytest.mark.asyncio
async def test_base_app_screen_mounts_exactly_one_navigation_bar():
    class TestScreen(BaseAppScreen):
        def __init__(self, app_instance):
            super().__init__(app_instance, "home")

    class HostApp(ConsolidatedCSSApp):
        async def on_mount(self):
            await self.push_screen(TestScreen(self))

    app = HostApp()

    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause(0.1)
        assert len(list(pilot.app.screen.query(MainNavigationBar))) == 1


def test_navigation_contract_keeps_context_out_of_top_nav():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    forbidden_local_terms = {
        "approval required",
        "selected source",
        "unsaved changes",
        "provider unavailable",
    }
    joined = " ".join(
        f"{destination.label} {destination.tooltip} {destination.purpose}".lower()
        for destination in SHELL_DESTINATION_ORDER
    )

    for term in forbidden_local_terms:
        assert term not in joined
