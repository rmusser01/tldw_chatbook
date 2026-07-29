"""Speech Recognition in the Console grammar.

Measured on the running view before this change: 30 of 34 controls sat below
a 26-row fold. The transcript -- the thing the view exists to show -- was not
the primary region; it competed with a sidebar of switches, a stats block
and two four-row notices.

This is a `compose()` rewrite only. All 36 methods stay where they are, and
every one of the 25 control ids must survive, because the behaviour queries
them by id.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen

#: Ids composed unconditionally. Frozen as the yardstick: the behaviour
#: queries these by id, so a dropped one is a dead handler.
#:
#: `history-list` and `clear-history-button` are deliberately NOT here. They
#: are composed only `if settings["privacy"]["save_history"]`, and that
#: whole feature is a shell -- `_add_to_history` is a `pass` stub, so nothing
#: is ever recorded. See task-1331; asserting they mount would be asserting
#: that a non-feature is present.
RECOGNITION_CONTROLS = (
    "auto-clear-switch",
    "buffer-duration-input",
    "commands-switch",
    "copy-button",
    "dictation-clear-btn",
    "dictation-pause-btn",
    "dictation-toggle-btn",
    "duration-display",
    "language-select",
    "local-only-switch",
    "privacy-status",
    "provider-select",
    "punctuation-switch",
    "save-history-switch",
    "save-md-button",
    "save-text-button",
    "speed-display",
    "state-display",
    "stats-display",
    "status-container",
    "transcript-display",
    "troubleshoot-btn",
    "word-count",
)


async def _open(app, pilot):
    screen = STTSScreen(app)
    await app.push_screen(screen)
    await pilot.pause()
    row = next(
        b for b in screen.query(Button)
        if getattr(b, "lab_view_key", None) == "dictation"
    )
    row.press()
    for _ in range(8):
        await pilot.pause()
    return screen


@pytest.mark.asyncio
async def test_no_control_is_lost_in_the_rewrite():
    """The behaviour queries all 25 by id; a missing one is a dead handler
    that fails at the moment the user presses it."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open(app, pilot)
        present = {w.id for w in screen.query("*") if w.id}

    missing = sorted(set(RECOGNITION_CONTROLS) - present)
    assert not missing, f"lost in the rewrite: {missing}"


@pytest.mark.asyncio
async def test_the_transcript_is_the_primary_region():
    """The view exists to show what was said. Before, the transcript shared
    the screen with a sidebar of switches and a stats block."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open(app, pilot)
        transcript = screen.query_one("#transcript-display")
        body = screen.query_one("#lab-body")
        assert transcript.region.height >= body.container_size.height // 3, (
            f"transcript is {transcript.region.height} rows of "
            f"{body.container_size.height}"
        )


@pytest.mark.asyncio
async def test_the_primary_action_is_above_the_fold():
    """Start/stop dictation is why the user opened this view."""
    app = _build_test_app()
    async with app.run_test(size=(200, 60)) as pilot:
        screen = await _open(app, pilot)
        body = screen.query_one("#lab-body")
        toggle = screen.query_one("#dictation-toggle-btn")
        assert body.region.contains_region(toggle.region), (
            f"start/stop below the fold: y={toggle.region.y}"
        )
