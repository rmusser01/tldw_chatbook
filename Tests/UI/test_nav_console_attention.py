"""task-31385: the Console nav button carries a badge while interrupts pend."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Button

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Navigation.main_navigation import (
    CONSOLE_ATTENTION_ATTR,
    CONSOLE_ATTENTION_GLYPH,
    MainNavigationBar,
    set_console_attention,
)


class _App(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield MainNavigationBar(active="home")


@pytest.mark.asyncio
async def test_badge_appears_on_a_mounted_bar_and_clears_at_zero():
    app = _App()
    async with app.run_test(size=(140, 24)) as pilot:
        await pilot.pause()
        button = app.query_one("#nav-console", Button)
        base = str(button.label)
        assert CONSOLE_ATTENTION_GLYPH not in base
        set_console_attention(app, 2)
        assert str(button.label) == f"{base} {CONSOLE_ATTENTION_GLYPH}"
        set_console_attention(app, 0)
        assert str(button.label) == base


@pytest.mark.asyncio
async def test_a_bar_composed_after_the_round_armed_shows_the_badge_on_mount():
    app = _App()
    setattr(app, CONSOLE_ATTENTION_ATTR, 1)
    async with app.run_test(size=(140, 24)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#nav-console", Button).label).endswith(CONSOLE_ATTENTION_GLYPH)
