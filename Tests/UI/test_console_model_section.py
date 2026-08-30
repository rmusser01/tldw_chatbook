"""Console left-rail Model section tests."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from Tests.UI.app_factory import _build_test_app


@pytest.mark.asyncio
async def test_model_section_renders_the_parameters_only_it_shows() -> None:
    """The Model rail body shows the sampling rows and nothing duplicated.

    TASK-23196: it used to show Provider and Model too, which the persistent
    status bar and the Inspector's run recipe were both already rendering at
    the same moment -- three copies of two values, and this was the copy
    costing scarce rail rows. What remains is what is shown nowhere else.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        await pilot.pause(0.2)
        console = host.screen_stack[-1]
        assert console.query_one("#console-model-section-temperature")
        assert console.query_one("#console-model-section-max-tokens")
        assert not console.query("#console-model-section-provider")
        assert not console.query("#console-model-section-model")


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
        temperature = console.query_one(
            "#console-model-section-temperature .console-model-section-value", Static
        )
        assert str(temperature.renderable).strip()
