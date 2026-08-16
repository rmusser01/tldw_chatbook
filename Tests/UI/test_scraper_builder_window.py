"""TASK-15991: ScraperBuilderWindow must open.

Born red at 11646bba0: `_compose_options` called the nonexistent
`FormBuilder.create_switch`, so pushing the screen died in compose with
AttributeError and the window had never been openable.

The harness loads the shipped stylesheet (bare harnesses measure fiction)
and waits on conditions, not pause counts.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import Select, Switch

from tldw_chatbook.UI.ScraperBuilderWindow import ScraperBuilderWindow

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = str(BUNDLE)

    def on_mount(self) -> None:
        self.push_screen(ScraperBuilderWindow())


async def _wait_for(pilot, predicate, what: str, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
async def test_scraper_builder_window_opens_and_switch_value_reaches_options():
    """Opening the window composes without raising (AC #1/#2), and a switch's
    value actually reaches its consumer (_format_options_code)."""
    app = _Harness()
    async with app.run_test(size=(160, 48)) as pilot:
        screen = await _wait_for(
            pilot,
            lambda: (
                app.screen if isinstance(app.screen, ScraperBuilderWindow) else None
            ),
            "scraper builder screen",
        )

        # The Options tab's switches exist with their consumer-facing ids.
        remove_scripts = await _wait_for(
            pilot,
            lambda: (
                (
                    screen.query("#remove-scripts")
                    and screen.query_one("#remove-scripts", Switch)
                )
                or None
            ),
            "#remove-scripts switch",
        )
        assert screen.query_one("#remove-styles", Switch).value is True
        assert screen.query_one("#preserve-links", Switch).value is True
        assert screen.query_one("#wait-javascript", Switch).value is False

        # The options Selects carry machine tokens as values, not labels
        # (the tuples were (value, label)-reversed, an open-crash at compose).
        assert screen.query_one("#text-processing", Select).value == "clean"

        # Default flows through to the consumer...
        assert remove_scripts.value is True
        assert '"remove_scripts": True' in screen._format_options_code()

        # ...and a driven change flows through too (not a constant).
        remove_scripts.value = False
        await pilot.pause()
        assert '"remove_scripts": False' in screen._format_options_code()
