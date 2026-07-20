"""Mounted tests for master-shell navigation."""

from types import SimpleNamespace

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
    assert (
        wc.navigation_priority < get_shell_destination("settings").navigation_priority
    )


def test_master_shell_navigation_uses_compact_spacing_for_full_destination_rail():
    css = MainNavigationBar.DEFAULT_CSS

    assert "margin: 0;" in css
    assert "padding: 0;" in css
    assert ".nav-overflow-hint" in css


@pytest.mark.asyncio
async def test_master_shell_navigation_order_and_labels():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)

        actual = [
            (button.id, str(button.label).strip())
            for button in app.query(".nav-button")
        ]

    assert actual == [
        ("nav-home", "1 Home"),
        ("nav-console", "2 Console"),
        ("nav-library", "3 Library"),
        ("nav-artifacts", "4 Artifacts"),
        ("nav-personas", "5 Personas"),
        ("nav-watchlists_collections", "6 Watchlists"),
        ("nav-schedules", "7 Schedules"),
        ("nav-workflows", "8 Workflows"),
        ("nav-mcp", "9 MCP"),
        ("nav-acp", "0 ACP"),
        ("nav-lab", "Lab"),
        ("nav-settings", "Settings"),
    ]


def test_nav_button_label_numbering_scheme():
    from tldw_chatbook.UI.Navigation.main_navigation import nav_button_label

    # ctrl+1..ctrl+9 cover the first nine destinations, ctrl+0 the tenth;
    # the remaining destinations (Lab, Settings) stay unnumbered.
    digits = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    for index, digit in enumerate(digits):
        assert nav_button_label(index, "Label") == f"{digit} Label"
    assert nav_button_label(10, "Lab") == "Lab"
    assert nav_button_label(11, "Settings") == "Settings"


@pytest.mark.asyncio
async def test_master_shell_navigation_scrolls_active_destination_into_view_on_mount():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="settings")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)

        nav = app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        active_button = app.query_one("#nav-settings", Button)

        assert strip.scroll_offset.x > 0
        assert active_button.region.width > 0
        assert active_button.region.x >= strip.region.x
        assert active_button.region.right <= strip.region.right


@pytest.mark.asyncio
async def test_master_shell_navigation_scrolls_when_active_destination_changes():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="home")

        def on_navigate_to_screen(self, message):
            pass

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)

        nav = app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        assert strip.scroll_offset.x == 0

        app.query_one("#nav-settings", Button).press()
        await pilot.pause(0.1)

        active_button = app.query_one("#nav-settings", Button)
        assert active_button.has_class("is-active")
        assert strip.scroll_offset.x > 0
        assert active_button.region.width > 0
        assert active_button.region.right <= strip.region.right


@pytest.mark.asyncio
async def test_master_shell_navigation_docks_overflow_hint_outside_scroll_strip():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="settings")

    app = TestApp()

    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause(0.1)

        nav = app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        hint = app.query_one("#nav-overflow-hint")

        assert hint.parent is nav
        assert hint not in strip.children
        # Even with the strip scrolled to the last destination, the hint
        # stays visible at the bar's right edge.
        assert strip.scroll_offset.x > 0
        assert hint.region.width > 0
        assert hint.region.right == nav.region.right


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
    assert ".nav-separator" not in MainNavigationBar.DEFAULT_CSS
    assert "background: $primary-darken-1;" in MainNavigationBar.DEFAULT_CSS


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
        ("nav-home", "1 Home"),
        ("nav-console", "2 Console"),
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
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(
            destination.primary_route
        )
        assert screen_class is not None, destination.primary_route


def test_folded_routes_highlight_owning_destination():
    folded = {
        "search": ("library", "search"),
        "media": ("library", "media"),
        "study": ("library", "study"),
        "writing": ("library", "writing"),
        "research": ("library", "research"),
        "ingest": ("library", "ingest"),
        "llm": ("lab", "llm"),
        "stts": ("lab", "stts"),
        "evals": ("lab", "evals"),
        "logs": ("settings", "logs"),
        "stats": ("settings", "stats"),
        # The retired Coding screen folds into Console.
        "coding": ("console", "chat"),
    }

    for route, (destination_id, canonical_route) in folded.items():
        nav = MainNavigationBar(active=route)
        assert nav.active_destination_id == destination_id, route
        assert nav.active_route == canonical_route, route


@pytest.mark.asyncio
async def test_folded_screen_boxes_owning_destination_button():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="search", active_route="search")

    app = TestApp()

    async with app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        assert app.query_one("#nav-library", Button).has_class("is-active")
        assert not app.query_one("#nav-lab", Button).has_class("is-active")

    class LabApp(App):
        def compose(self):
            yield MainNavigationBar(active="llm", active_route="llm")

    lab_app = LabApp()

    async with lab_app.run_test(size=(180, 20)) as pilot:
        await pilot.pause(0.1)
        assert lab_app.query_one("#nav-lab", Button).has_class("is-active")
        assert not lab_app.query_one("#nav-library", Button).has_class("is-active")


def test_shell_destination_hotkeys_follow_destination_order():
    """Ctrl+1..9 then Ctrl+0 map onto SHELL_DESTINATION_ORDER, in order."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    hotkey_bindings = [
        binding
        for binding in TldwCli.BINDINGS
        if binding.action.startswith("shell_destination(")
    ]

    expected_keys = list(TldwCli.SHELL_DESTINATION_HOTKEYS)
    assert expected_keys == [
        "ctrl+1",
        "ctrl+2",
        "ctrl+3",
        "ctrl+4",
        "ctrl+5",
        "ctrl+6",
        "ctrl+7",
        "ctrl+8",
        "ctrl+9",
        "ctrl+0",
    ]
    # One binding per hotkey, zipped against the destination order; the layer
    # never invents keys beyond ctrl+0 and never skips a destination.
    assert len(hotkey_bindings) == min(len(expected_keys), len(SHELL_DESTINATION_ORDER))
    for index, binding in enumerate(hotkey_bindings):
        destination = SHELL_DESTINATION_ORDER[index]
        assert binding.key == expected_keys[index]
        assert binding.action == f"shell_destination({index})"
        assert destination.accessible_label in binding.description
        # Index numbers belong to the key layer, not the nav labels.
        assert str(index + 1) not in destination.label


def test_action_shell_destination_posts_primary_route():
    """The single hotkey action navigates to each destination's primary route."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    posted = []
    fake_app = SimpleNamespace(post_message=posted.append)

    for index, destination in enumerate(SHELL_DESTINATION_ORDER):
        TldwCli.action_shell_destination(fake_app, index)
        message = posted[-1]
        assert isinstance(message, NavigateToScreen)
        assert message.screen_name == destination.primary_route, (
            destination.destination_id
        )

    # Textual binding actions pass the argument as a string.
    posted.clear()
    TldwCli.action_shell_destination(fake_app, "0")
    assert isinstance(posted[-1], NavigateToScreen)
    assert posted[-1].screen_name == SHELL_DESTINATION_ORDER[0].primary_route

    # Out-of-range indices are a safe no-op.
    posted.clear()
    TldwCli.action_shell_destination(fake_app, len(SHELL_DESTINATION_ORDER))
    TldwCli.action_shell_destination(fake_app, -1)
    TldwCli.action_shell_destination(fake_app, "not-a-number")
    assert posted == []
