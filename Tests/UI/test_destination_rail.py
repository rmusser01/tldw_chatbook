"""Shared destination rail widgets: the Chat-free base behind ConsoleRailHandle."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.destination_rail import DestinationRailHandle


class _HandleHarness(App[None]):
    def __init__(self, handle: DestinationRailHandle) -> None:
        super().__init__()
        self._handle = handle

    def compose(self) -> ComposeResult:
        yield self._handle


@pytest.mark.asyncio
async def test_base_handle_renders_label_and_badge_verbatim():
    """The base applies no vocabulary of its own -- Console's lives in its subclass."""
    handle = DestinationRailHandle(
        label="Catalog",
        badge="3 servers",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert str(app.query_one("#lab-rail-open", Button).label) == "Catalog"
        assert str(app.query_one("#lab-rail-badge", Static).renderable) == "3 servers"


@pytest.mark.asyncio
async def test_base_handle_default_tooltip_names_the_rail():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Catalog rail"


@pytest.mark.asyncio
async def test_base_handle_accepts_an_explicit_tooltip():
    handle = DestinationRailHandle(
        label="Whatever",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="right",
        open_tooltip="Open Inspector rail",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail-open", Button).tooltip == "Open Inspector rail"


@pytest.mark.asyncio
async def test_base_handle_keeps_the_existing_css_class_names():
    """Class names are deliberately unchanged so the CSS bundle sees no diff."""
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert "console-rail-handle" in handle.classes
        assert "console-rail-handle-left" in handle.classes


@pytest.mark.asyncio
async def test_base_handle_omits_the_badge_when_empty():
    handle = DestinationRailHandle(
        label="Catalog",
        button_id="lab-rail-open",
        badge_id="lab-rail-badge",
        side="left",
    )
    app = _HandleHarness(handle)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        assert not app.query("#lab-rail-badge")


def test_shared_glyphs_match_the_console_originals():
    """Guard the deliberate duplication of the glyph literals.

    ``destination_rail`` redeclares these rather than importing from
    ``Chat.console_glyphs``, so the shared widget stays free of the Chat
    layer. That duplication would otherwise drift silently if either side
    changed.
    """
    from tldw_chatbook.Chat import console_glyphs
    from tldw_chatbook.Widgets.destination_rail import (
        GLYPH_COLLAPSED,
        GLYPH_EXPANDED,
    )

    assert GLYPH_EXPANDED == console_glyphs.GLYPH_EXPANDED
    assert GLYPH_COLLAPSED == console_glyphs.GLYPH_COLLAPSED
