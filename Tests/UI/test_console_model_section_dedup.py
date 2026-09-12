"""The Model section must not repeat what the status bar already says.

TASK-23196. The 2026-08-29 UX audit found provider and model rendered in
three places at once in a single screenshot: the Context rail's Model
section, the persistent status bar, and the Inspector's run recipe. The
rail's copy is the one that costs scarce vertical space -- the status bar is
always visible and costs the rail nothing, and it carries both values at
every width where the rail is shown at all (below 100 columns the rail
force-collapses).

What stays in the Model section is what is NOT duplicated elsewhere: the
sampling parameters, the system-prompt row, and Configure.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.console_rail_section_helpers import open_rail_section
from Tests.UI.test_console_left_rail import make_console_pilot


def _section_text(screen) -> str:
    body = screen.query_one("#console-rail-section-body-model")
    return " ".join(
        str(getattr(widget, "renderable", ""))
        for widget in body.query("*")
        if widget.display
    )


@pytest.mark.asyncio
async def test_model_section_does_not_repeat_provider_and_model() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await open_rail_section(screen, pilot, "model")
        await pilot.pause(0.4)

        assert not screen.query("#console-model-section-provider"), (
            "the Model section still repeats the provider the status bar shows"
        )
        assert not screen.query("#console-model-section-model"), (
            "the Model section still repeats the model the status bar shows"
        )


@pytest.mark.asyncio
async def test_model_section_keeps_what_is_not_shown_elsewhere() -> None:
    """De-duplicating must not gut the section."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await open_rail_section(screen, pilot, "model")
        await pilot.pause(0.4)

        assert screen.query("#console-model-section-temperature")
        assert screen.query("#console-model-section-max-tokens")
        assert screen.query_one("#console-rail-system-line", Static)
        assert screen.query_one("#console-model-section-configure", Button)

        text = _section_text(screen)
        assert "Temperature" in text
        assert "Max tokens" in text


@pytest.mark.asyncio
async def test_the_status_bar_still_carries_provider_and_model() -> None:
    """The surviving copy must actually survive."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        await pilot.pause(0.3)

        painted = " ".join(
            str(getattr(widget, "renderable", ""))
            for widget in screen.query("*")
            if widget.display
        )
        assert "llama_cpp" in painted, "the provider vanished from the Console chrome"
        assert "local-model" in painted, "the model vanished from the Console chrome"
