"""Actual pre-extraction Context/spend geometry and public Inspector route."""

import pytest
from textual.containers import HorizontalScroll

from Tests.UI.test_console_cost_chip_screen import (
    ConsoleHarness,
    _build_test_app,
    _configure_anthropic_ready_console,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_conversation_inspector import MODAL_ID
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _cost_span(line: ConsoleStatusText) -> tuple[int, int]:
    for name, start, end in line.segment_spans:
        if name == "cost":
            return start, end
    raise AssertionError("cost segment not on the line")


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
        # TASK-35160: the cost chip is the strip line's trailing "cost"
        # segment; the geometry contract moves to the line and the segment's
        # own clickable cells.
        strip = console.query_one("#console-status-chips", ConsoleStatusChips)
        line = console.query_one("#console-status-text", ConsoleStatusText)
        assert line.is_mounted and line.display and not line.disabled
        assert line.region.width > 0 and line.region.height > 0
        assert strip.segment_visible("cost")
        start, end = _cost_span(line)
        assert end > start
        # Bring the trailing cost segment into the strip viewport (the line
        # scrolls when it overflows; the chip era scrolled the chip itself).
        scroller = console.query_one(
            "#console-status-chip-scroll", HorizontalScroll
        )
        scroller.scroll_to(x=scroller.max_scroll_x, animate=False)
        await pilot.pause()
        center_column = (start + end) // 2
        center_x = line.region.x + center_column
        center_y = line.region.y + line.region.height // 2
        assert console.get_widget_at(center_x, center_y)[0] is line
        assert composer.draft_text() == "Context and cost display characterization"
        assert await pilot.click(line, offset=(center_column, 0))
        await _wait_for_selector(host.screen, pilot, f"#{MODAL_ID}")
        assert composer.draft_text() == "Context and cost display characterization"
        await pilot.press("escape")
