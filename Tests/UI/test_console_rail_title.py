"""Console left-rail header tests."""

from __future__ import annotations

import pytest
from textual.containers import Horizontal
from textual.widgets import Button

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app


@pytest.mark.asyncio
async def test_console_context_rail_header_uses_the_full_collapse_button() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        button = console.query_one("#console-context-rail-collapse", Button)
        header = button.parent

        assert isinstance(header, Horizontal)
        assert list(header.children) == [button]
        assert not console.query("#console-context-rail-title")
        # TASK-23195: still one full-width collapse button, now with a
        # readable name instead of an 18-column ASCII arrow.
        assert "Context" in str(button.label)
        assert "<---------" not in str(button.label)
        assert button.tooltip == "Collapse Console context rail"
