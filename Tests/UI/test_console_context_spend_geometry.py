"""Actual pre-extraction Context/spend geometry and public Inspector route."""

import pytest

from Tests.UI.test_console_cost_chip_screen import (
    ConsoleHarness,
    _build_test_app,
    _configure_anthropic_ready_console,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_conversation_inspector import MODAL_ID
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleCostChip

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
async def test_context_spend_chip_keeps_real_geometry_and_inspector_route(size):
    app = _build_test_app()
    _configure_anthropic_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("Context and cost display characterization")
        console._sync_console_settings_summary()
        console._sync_console_cost_chip()
        await pilot.pause()
        chip = console.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.is_mounted and chip.display and not chip.disabled
        assert chip.region.width > 0 and chip.region.height > 0
        chip.scroll_visible(immediate=True)
        await pilot.pause()
        assert console.get_widget_at(*chip.region.center)[0] is chip
        assert composer.draft_text() == "Context and cost display characterization"
        assert await pilot.click(chip)
        await _wait_for_selector(host.screen, pilot, f"#{MODAL_ID}")
        assert composer.draft_text() == "Context and cost display characterization"
        await pilot.press("escape")
