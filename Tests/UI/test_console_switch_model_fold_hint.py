"""TASK-33004.8: Switch model carries no fold hint, and none renders stale.

The Alt+M popover's old '▼ more — scroll for conversation settings' hint
stayed visible at the bottom of its scroll (a Phase 2 capture finding).
TASK-33004.4 replaced that form with the Switch model pair list, which has
no fold hint (spec mockup (a)), so this pins AC#2: at the owner's
full-screen sizes no fold hint renders, at rest or after scrolling.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_switch_model_entry_points import (
    _open_console,
    _switcher_ready,
)
from Tests.UI.test_console_switch_model_keys import _console_app, _Harness

_RETIRED_HINT_ID = "console-popover-fold-hint"
_HINT_FRAGMENTS = ("more — scroll", "scroll for")


def _hint_texts(switcher) -> list[str]:
    """Return every fold-hint-like text the open switcher renders.

    Args:
        switcher: The mounted Switch model screen.

    Returns:
        The rendered Static texts that read like a fold hint.
    """
    found = []
    for static in switcher.query(Static):
        text = str(static.render())
        if any(fragment in text for fragment in _HINT_FRAGMENTS):
            found.append(text)
    return found


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(211, 44), (235, 52)])
@private_profile_test
async def test_switch_model_renders_no_fold_hint(request, size) -> None:
    """AC#2: no fold hint at rest, at the end of the scroll, or back at top."""
    app = _console_app()
    harness = _Harness(app)
    async with harness.run_test(size=size) as pilot:
        await _open_console(harness, pilot)
        await pilot.press("alt+m")
        switcher = await _switcher_ready(harness, pilot)

        assert not switcher.query(f"#{_RETIRED_HINT_ID}"), size
        assert _hint_texts(switcher) == [], size

        for key in ("end", "home"):
            await pilot.press(key)
            await pilot.pause()
            assert _hint_texts(switcher) == [], (size, key)

        await pilot.press("escape")
        await pilot.pause()
