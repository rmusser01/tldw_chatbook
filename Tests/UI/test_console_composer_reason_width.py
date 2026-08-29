"""TASK-24415: the Send disabled-reason strip must not starve the draft.

Live finding (2026-08-29, real app in tmux, 80-column terminal, blocked
provider): the expanded composer row gives ``#console-send-disabled-reason``
an ``auto`` width capped at ``SEND_REASON_MAX_WIDTH`` (52), while the ``1fr``
visible draft has ``min_width: 0`` -- so at narrow widths the strip consumed
the row and the draft collapsed to ZERO columns: no typed text, no caret, no
placeholder, while the slash-command popup filtered against invisible input
above it. The width sweep measured: draft visible at >=100 app columns,
truncated mid-word at 95, invisible at <=90 with the 42-cell blocked copy.

These tests assert the *laid-out geometry* of the real mounted Console
composer (``region`` widths, not ``.value``), per the geometry-assertions
lesson in ``backlog/docs/lessons-testing-evidence.md``.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

#: App size whose composer row cannot hold left cluster + actions row + a
#: legible reason strip + the 32-cell draft floor (TASK-24415's live case).
APP_NARROW = (80, 30)
#: App size where the full 52-cell strip and the draft floor coexist.
APP_WIDE = (160, 42)

#: The draft floor the actions-row budget has promised since TASK-2154.14
#: ("the draft keeps its 32-cell floor" -- the arithmetic the layout never
#: enforced). The visible draft must never be starved below it while a wider
#: row could satisfy it.
DRAFT_FLOOR = 32


async def _composer_parts(host, pilot):
    """Mount the ready Console and return (composer, draft, reason strip)."""
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    draft = composer.query_one("#console-command-visible-text", Static)
    strip = composer.query_one("#console-send-disabled-reason", Static)
    # The empty draft of a freshly mounted ready console shows the muted
    # idle reason ("type a message") -- the strip is live, not theoretical.
    await pilot.pause()
    await pilot.pause()
    return composer, draft, strip


@pytest.mark.asyncio
async def test_narrow_composer_keeps_the_draft_visible_beside_a_reason():
    """At 80 columns the draft must lay out with visible width, not zero.

    The strip may hide or ellipsize -- it is advisory copy and the Send
    tooltip carries the same reason -- but the draft the user is typing
    into (and the `/` slash-trigger that filters on it) must never be
    allocated zero columns.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_NARROW) as pilot:
        composer, draft, strip = await _composer_parts(host, pilot)
        assert str(strip.renderable).strip(), "idle reason copy missing"
        assert draft.region.width >= 8, (
            f"visible draft collapsed to {draft.region.width} columns at "
            f"{APP_NARROW[0]}-column app width"
        )


@pytest.mark.asyncio
async def test_wide_composer_keeps_reason_strip_capped_and_draft_floor():
    """At wide sizes the strip stays within its 52-cell cap and the draft
    keeps its promised 32-cell floor."""
    _, host = _ready_host()
    async with host.run_test(size=APP_WIDE) as pilot:
        composer, draft, strip = await _composer_parts(host, pilot)
        assert strip.display is True
        assert strip.region.width <= ConsoleComposerBar.SEND_REASON_MAX_WIDTH
        assert draft.region.width >= DRAFT_FLOOR, (
            f"draft laid out at {draft.region.width} columns; the "
            f"{DRAFT_FLOOR}-cell floor the actions budget promises was "
            "starved by the reason strip"
        )


@pytest.mark.asyncio
async def test_resize_from_wide_to_narrow_reapplies_the_strip_budget():
    """The clamp is a function of the live row width: shrinking the terminal
    must retract the strip rather than the draft."""
    _, host = _ready_host()
    async with host.run_test(size=APP_WIDE) as pilot:
        composer, draft, strip = await _composer_parts(host, pilot)
        assert draft.region.width >= DRAFT_FLOOR

        await pilot.resize_terminal(APP_NARROW[0], APP_NARROW[1])
        await pilot.pause()
        await pilot.pause()

        assert draft.region.width >= 8, (
            "after resizing to a narrow terminal the draft collapsed to "
            f"{draft.region.width} columns while the reason strip still held "
            "layout space"
        )
