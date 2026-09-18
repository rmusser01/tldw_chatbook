"""Token-backed replacements preserve explicit geometry against ID rules."""

import pytest
from textual.containers import Horizontal
from textual.widgets import Static

from .consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp


class UtilityApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    CSS = """
    #utility-probe { width: 12; height: 8; padding: 2; margin: 2; }
    """

    def compose(self):
        with Horizontal():
            yield Static("value", id="utility-probe", classes="w-fill h-1 p-0 m-0")


@pytest.mark.asyncio
async def test_explicit_utility_geometry_wins_id_rules_and_can_be_released():
    app = UtilityApp()
    async with app.run_test(size=(80, 24)) as pilot:
        probe = app.query_one("#utility-probe")
        assert probe.size.height == 1
        assert probe.size.width == 80
        assert tuple(probe.styles.padding) == (0, 0, 0, 0)
        assert tuple(probe.styles.margin) == (0, 0, 0, 0)
        probe.remove_class("w-fill", "h-1", "p-0", "m-0")
        await pilot.pause()
        assert str(probe.styles.width) == "12"
        assert str(probe.styles.height) == "8"
