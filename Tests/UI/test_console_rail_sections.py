"""Console rail section header widget contracts."""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)


class _HeaderApp(App):
    def compose(self):
        yield ConsoleRailSectionHeader(
            "Details",
            section_id="details",
            open=False,
            id="header-under-test",
        )


@pytest.mark.asyncio
async def test_rail_section_header_renders_title_and_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        title = app.query_one("#console-rail-section-title-details", Static)
        assert str(getattr(title.renderable, "plain", title.renderable)) == "Details"
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "+"
        assert toggle.tooltip == "Expand Details"


@pytest.mark.asyncio
async def test_rail_section_header_sync_open_flips_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        header = app.query_one("#header-under-test", ConsoleRailSectionHeader)
        header.sync_open(True)
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "-"
        assert toggle.tooltip == "Collapse Details"
