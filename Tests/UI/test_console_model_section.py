"""Console left-rail Model section tests."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.test_screen_navigation import _build_test_app


@pytest.mark.asyncio
async def test_model_section_renders_four_rows() -> None:
    """The Model rail body shows provider, model, temperature, and max-tokens rows."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        assert console.query_one("#console-model-section-provider")
        assert console.query_one("#console-model-section-model")
        assert console.query_one("#console-model-section-temperature")
        assert console.query_one("#console-model-section-max-tokens")


@pytest.mark.asyncio
async def test_model_sync_updates_rows() -> None:
    """Refreshing the settings summary updates the new row value widgets."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        console._sync_console_settings_summary()
        await pilot.pause(0.2)
        provider = console.query_one(
            "#console-model-section-provider .console-model-section-value", Static
        )
        assert str(provider.renderable).strip()
