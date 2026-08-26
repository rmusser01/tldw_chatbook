"""Guardrails for shell-owned chrome and primary destination metadata."""

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)


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


def test_persona_buddy_shell_bindings_do_not_shadow_global_keys():
    """Buddy's focused controls preserve terminal and shell-owned shortcuts."""

    keys = {binding.key for binding in PersonaBuddyWidget.BINDINGS}
    assert keys == {"h", "j", "k", "l", "H", "J", "K", "L", "0", "c", "x"}
    assert keys.isdisjoint({"ctrl+p", "ctrl+q", "f1", "f6"})


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


def test_research_shell_chrome_names_one_destination_without_leaking_authority():
    from tldw_chatbook.UI.Navigation.shell_destinations import (
        SHELL_DESTINATION_ORDER,
        get_shell_destination,
    )

    research = get_shell_destination("research")
    assert research.label == "Research"
    assert research.primary_route == "research_workspace"
    assert research.related_routes == ("research",)
    assert (
        sum(item.destination_id == "research" for item in SHELL_DESTINATION_ORDER) == 1
    )
    assert "Workspace data:" not in " ".join(
        f"{item.label} {item.tooltip} {item.purpose}"
        for item in SHELL_DESTINATION_ORDER
    )
