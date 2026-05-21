"""Mounted tests for master-shell navigation."""

import pytest
from textual.app import App
from textual.widgets import Button

from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar


def test_compact_navigation_labels_preserve_full_meaning():
    from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination

    wc = get_shell_destination("watchlists_collections")

    assert wc.label == "Watchlists"
    assert wc.full_label == "Watchlists"
    assert "Collections" not in wc.tooltip
    assert wc.navigation_priority < get_shell_destination("settings").navigation_priority


@pytest.mark.asyncio
async def test_master_shell_navigation_order_and_labels():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

        actual = [(button.id, str(button.label).strip()) for button in app.query(".nav-button")]

    assert actual == [
        ("nav-home", "Home"),
        ("nav-console", "Console"),
        ("nav-library", "Library"),
        ("nav-artifacts", "Artifacts"),
        ("nav-personas", "Personas"),
        ("nav-watchlists_collections", "Watchlists"),
        ("nav-schedules", "Schedules"),
        ("nav-workflows", "Workflows"),
        ("nav-mcp", "MCP"),
        ("nav-acp", "ACP"),
        ("nav-skills", "Skills"),
        ("nav-settings", "Settings"),
    ]


@pytest.mark.asyncio
async def test_master_shell_navigation_uses_terminal_tab_rail():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="console")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

        nav_buttons = list(app.query(".nav-button"))
        separators = list(app.query(".nav-separator"))
        active_button = app.query_one("#nav-console", Button)

    assert nav_buttons
    assert all(button.has_class("ascii-nav-tab") for button in nav_buttons)
    assert separators == []
    assert active_button.has_class("is-active")


@pytest.mark.asyncio
async def test_home_and_console_remain_first_primary_destinations():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)
        buttons = list(app.query(".nav-button"))

    assert [(button.id, str(button.label).strip()) for button in buttons[:2]] == [
        ("nav-home", "Home"),
        ("nav-console", "Console"),
    ]


@pytest.mark.asyncio
async def test_master_shell_navigation_routes_to_primary_route():
    events = []

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

        def on_mount(self):
            self.query_one("#nav-console", Button).press()

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

    assert events == ["chat"]


@pytest.mark.asyncio
async def test_active_destination_subroute_can_return_to_primary_route():
    events = []

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="library", active_route="study")

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        app.query_one("#nav-library", Button).press()
        await pilot.pause(0.1)

    assert events == ["library"]


@pytest.mark.asyncio
async def test_active_destination_primary_route_still_noops():
    events = []

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="library", active_route="library")

        def on_navigate_to_screen(self, message):
            events.append(message.screen_name)

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        app.query_one("#nav-library", Button).press()
        await pilot.pause(0.1)

    assert events == []


@pytest.mark.asyncio
async def test_every_visible_master_shell_nav_destination_resolves():
    from Tests.UI.test_screen_navigation import _build_test_app
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    app = _build_test_app()

    for destination in SHELL_DESTINATION_ORDER:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(destination.primary_route)
        assert screen_class is not None, destination.primary_route
