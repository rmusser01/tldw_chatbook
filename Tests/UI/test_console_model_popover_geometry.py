"""Geometry of Switch model, the Console Alt+M popover (TASK-33004.4).

Rewritten on purpose: the 85%/170-column wide tier (PR #2672) and its
Python toggle are gone. Spec §6 sets one width token instead:
``$ds-model-switcher-width`` (120) in ``css/core/_variables.tcss``, with
auto height up to ``$ds-model-switcher-max-height`` (80%).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_model_switcher import Recorder, build_switcher

VARIABLES = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/core/_variables.tcss"
)


class PopoverGeometryHarness(ConsolidatedCSSApp):
    """Isolated app that loads the same consolidated CSS as production."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


def test_switcher_size_comes_from_tokens() -> None:
    """The width and the height cap are tokens, not literals in the widget."""
    tokens = VARIABLES.read_text(encoding="utf-8")
    assert re.search(r"^\$ds-model-switcher-width: 120;$", tokens, re.MULTILINE)
    assert re.search(r"^\$ds-model-switcher-max-height: 80%;$", tokens, re.MULTILINE)


@pytest.mark.parametrize(
    ("size", "width"),
    (((211, 44), 120), ((235, 52), 120), ((150, 40), 120), ((90, 30), 90)),
    ids=["211x44", "235x52", "150", "narrow-clamps"],
)
@pytest.mark.asyncio
async def test_switcher_is_120_columns_and_never_wider_than_the_screen(
    size: tuple[int, int], width: int
) -> None:
    """120 columns at 211x44 and 235x52; a narrower terminal clamps to it,
    and no tier class is toggled from Python any more."""
    app = PopoverGeometryHarness()
    async with app.run_test(size=size) as pilot:
        switcher = build_switcher(Recorder())
        await app.push_screen(switcher)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

        container = switcher.query_one("#console-model-popover")
        assert container.region.width == width
        assert container.region.height <= int(size[1] * 0.8)
        assert not container.has_class("-console-popover-wide")

        await pilot.resize_terminal(235, 52)
        await pilot.pause()
        await pilot.pause()
        assert container.region.width == 120
