"""TASK-32592: every More destination must paint while keyboard focused."""

import pytest
from textual.app import ComposeResult
from textual.widgets import Button

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)
from tldw_chatbook.UI.Navigation.nav_overflow_menu import NavOverflowMenu
from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER


class NavigationHost(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET

    def __init__(self):
        super().__init__()
        self.routes = []

    def compose(self) -> ComposeResult:
        yield MainNavigationBar(active="console")

    def on_navigate_to_screen(self, message: NavigateToScreen) -> None:
        self.routes.append(message.screen_name)


def assert_focused_destination_paints(menu, button):
    """Inspect clipped compositor output, not the widget's content property."""
    assert menu.focused is button
    strips = menu._compositor.render_strips()
    assert 0 <= button.region.y < len(strips)
    row = strips[button.region.y].text
    label = button.label
    assert (label.plain if hasattr(label, "plain") else str(label)) in row
    assert button.styles.text_style.bold or button.styles.text_style.underline
    viewport = menu.query_one("#nav-overflow-menu").scrollable_content_region
    assert viewport.contains_region(button.region)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_all_overflow_destinations_paint_and_activate_at_80_columns(theme):
    app = NavigationHost()
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        opener = app.query_one("#nav-overflow-hint", Button)
        for destination in SHELL_DESTINATION_ORDER:
            opener.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            menu = app.screen
            assert isinstance(menu, NavOverflowMenu)
            button = menu.query_one(
                f"#nav-overflow-{destination.destination_id}", Button
            )
            first = menu.query_one(".nav-overflow-destination", Button)
            first.focus()
            await pilot.pause()
            for _ in range(len(SHELL_DESTINATION_ORDER)):
                if menu.focused is button:
                    break
                await pilot.press("tab")
            await pilot.pause()
            assert_focused_destination_paints(menu, button)
            await pilot.press("enter")
            await pilot.pause()
            assert not isinstance(app.screen, NavOverflowMenu)
            assert app.routes[-1] == destination.primary_route
            assert app.focused is opener


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_open_menu_keeps_last_destination_visible_through_resize(theme):
    app = NavigationHost()
    app.theme = theme
    async with app.run_test(size=(120, 40)) as pilot:
        opener = app.query_one("#nav-overflow-hint", Button)
        opener.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        menu = app.screen
        first = menu.query_one(".nav-overflow-destination", Button)
        first.focus()
        await pilot.pause()
        await pilot.press("shift+tab")
        await pilot.pause()
        last = menu.query_one("#nav-overflow-meetings", Button)
        assert menu.focused is last
        for size in [(120, 40), (80, 24), (120, 40)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert app.screen is menu
            assert_focused_destination_paints(menu, last)
        await pilot.press("escape")
        await pilot.pause()
        assert app.focused is opener
        assert not app.routes
