"""Regression tests for chrome UX fixes (UX-039, UX-041, UX-006).

Footer: global hints (F1/F6/palette/quit) are always present, screen
context prepends instead of replacing, and narrow widths degrade
gracefully without mid-word clipping. Nav bar: overflow indicators track
real scroll state, and all 13 destination labels fit at 140 columns.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.shortcut_context import ShortcutAction, ShortcutContext
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus


class _FooterHarness(App[None]):
    def __init__(self):
        super().__init__()
        self.footer = AppFooterStatus()

    def compose(self) -> ComposeResult:
        yield Static("body")
        yield self.footer


def _shown_text(footer: AppFooterStatus) -> str:
    return str(footer._shortcut_display.render())


@pytest.mark.asyncio
async def test_footer_defaults_to_global_hints() -> None:
    app = _FooterHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        footer = app.footer
        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT
        shown = _shown_text(footer)
        for hint in ("F1 help", "F6 panes", "Ctrl+P palette", "Ctrl+Q quit"):
            assert hint in shown


@pytest.mark.asyncio
async def test_footer_context_prepends_but_never_replaces_globals() -> None:
    app = _FooterHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        footer = app.footer
        footer.set_workbench_shortcuts(
            source="test", shortcuts=(("c", "create task"), ("d", "delete"))
        )
        await pilot.pause()
        assert footer.shortcut_text.startswith("c create task")
        shown = _shown_text(footer)
        assert "c create task" in shown
        for hint in ("F1 help", "Ctrl+P palette", "Ctrl+Q quit"):
            assert hint in shown


@pytest.mark.asyncio
async def test_footer_narrow_width_drops_context_before_globals() -> None:
    app = _FooterHarness()
    async with app.run_test(size=(64, 24)) as pilot:
        await pilot.pause()
        footer = app.footer
        footer.set_workbench_shortcuts(
            source="test",
            shortcuts=(("c", "create task"), ("d", "delete"), ("s", "sync now")),
        )
        await pilot.pause()
        shown = _shown_text(footer)
        # Context must yield to the globals; the globals never clip mid-word.
        assert "Ctrl+Q" in shown
        assert "create tas" not in shown or "create task" in shown


@pytest.mark.asyncio
async def test_footer_clear_restores_globals() -> None:
    app = _FooterHarness()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        footer = app.footer
        footer.set_workbench_shortcuts(source="test", shortcuts=(("x", "thing"),))
        footer.clear_shortcut_context(source="test")
        await pilot.pause()
        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


class _NavHarness(App[None]):
    def __init__(self, active: str = "home"):
        super().__init__()
        self.active = active

    def compose(self) -> ComposeResult:
        yield MainNavigationBar(active=self.active)


@pytest.mark.asyncio
async def test_nav_bar_fits_all_destinations_at_156() -> None:
    app = _NavHarness()
    async with app.run_test(size=(156, 24)) as pilot:
        await pilot.pause()
        strip = app.query_one("#nav-destination-strip")
        # No overflow -> no indicators, and the strip fits every label.
        assert strip.max_scroll_x == 0
        await pilot.pause(0.6)  # let the hint interval fire
        assert app.query_one("#nav-overflow-hint").display is False
        assert app.query_one("#nav-overflow-hint-left").display is False
        # Rightmost destination fully inside the strip's visible window.
        settings = app.query_one("#nav-settings")
        assert settings.region.right <= strip.content_region.right


@pytest.mark.asyncio
async def test_nav_bar_overflow_at_140_shows_indicators_not_clipped_labels() -> None:
    # "^N " key prefixes made labels ~10 cells wider; at 140 the bar scrolls
    # slightly. The contract: indicators appear, labels never clip mid-word,
    # and the active destination scrolls fully into view.
    app = _NavHarness(active="settings")
    async with app.run_test(size=(140, 24)) as pilot:
        await pilot.pause()
        strip = app.query_one("#nav-destination-strip")
        assert strip.max_scroll_x > 0
        await pilot.pause(0.6)
        assert app.query_one("#nav-overflow-hint").display is True
        assert app.query_one("#nav-overflow-hint-left").display is True
        settings = app.query_one("#nav-settings")
        assert settings.region.right <= strip.content_region.right


@pytest.mark.asyncio
async def test_nav_bar_overflow_indicators_at_80() -> None:
    app = _NavHarness(active="logs")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        strip = app.query_one("#nav-destination-strip")
        assert strip.max_scroll_x > 0
        await pilot.pause(0.6)
        # Active destination scrolled into view -> something hidden left.
        assert app.query_one("#nav-overflow-hint-left").display is True


@pytest.mark.asyncio
async def test_nav_bar_right_hint_appears_only_with_more_content_right() -> None:
    app = _NavHarness(active="home")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.6)
        # Home is leftmost: nothing to the left, plenty to the right.
        assert app.query_one("#nav-overflow-hint-left").display is False
        assert app.query_one("#nav-overflow-hint").display is True
