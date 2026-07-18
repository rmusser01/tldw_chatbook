"""Console left-rail title tests."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app


@pytest.mark.asyncio
async def test_console_rail_title_reads_console_context() -> None:
    """The left-rail title reads "Console context" rather than "Session & Context"."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        title = console.query_one("#console-context-rail-title", Static)
        assert "Console context" in str(title.renderable)
