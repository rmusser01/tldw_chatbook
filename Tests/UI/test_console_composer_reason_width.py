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

from types import SimpleNamespace

import pytest
from textual.geometry import Region
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


# ---------------------------------------------------------------------------
# Direct unit coverage for _send_reason_width_cap's own branches (qodo
# finding 2 on PR #2214): the mounted geometry tests above pin the
# integration, these pin the arithmetic with the row widget stubbed, so
# every branch is reached deterministically -- no-mount NoMatches, the
# pre-layout zero-width fallback, the hide threshold, intermediate
# budgets, and the SEND_REASON_MAX_WIDTH cap.
# ---------------------------------------------------------------------------


def _bare_composer() -> ConsoleComposerBar:
    return ConsoleComposerBar()


def _cap_with_row_width(composer, row_width: int, monkeypatch) -> int:
    row = SimpleNamespace(content_region=Region(0, 0, row_width, 1))
    monkeypatch.setattr(composer, "query_one", lambda *args, **kwargs: row)
    return composer._send_reason_width_cap()


def _chip_cap_with_row_width(composer, row_width: int, monkeypatch) -> int:
    row = SimpleNamespace(content_region=Region(0, 0, row_width, 1))
    monkeypatch.setattr(composer, "query_one", lambda *args, **kwargs: row)
    return composer._voice_chip_width_cap()


def test_cap_without_a_mounted_row_falls_back_to_the_static_max():
    """An unmounted composer has no row to query: the static cap applies
    until the next resize re-derives against a real width."""
    assert _bare_composer()._send_reason_width_cap() == (
        ConsoleComposerBar.SEND_REASON_MAX_WIDTH
    )


def test_cap_with_unlaid_out_row_falls_back_to_the_static_max(monkeypatch):
    assert (
        _cap_with_row_width(_bare_composer(), 0, monkeypatch)
        == ConsoleComposerBar.SEND_REASON_MAX_WIDTH
    )


def test_cap_is_the_static_max_when_the_row_can_spare_it(monkeypatch):
    # reserved = LEFT_CLUSTER_WIDTH(18) + actions row(25) = 43; budget
    # beyond the 32-cell draft floor = 200 - 75 = 125, far over the 52 cap.
    assert _cap_with_row_width(_bare_composer(), 200, monkeypatch) == 52


def test_cap_is_the_spare_budget_at_intermediate_widths(monkeypatch):
    assert _cap_with_row_width(_bare_composer(), 100, monkeypatch) == 23


def test_cap_hides_at_the_legibility_boundary(monkeypatch):
    # 89 - 77 = 12: exactly the legibility floor -- still shown.
    assert _cap_with_row_width(_bare_composer(), 89, monkeypatch) == 12
    # 88 - 77 = 11: below it -- hide (0).
    assert _cap_with_row_width(_bare_composer(), 88, monkeypatch) == 0
    # Far below: negative budget -- hide.
    assert _cap_with_row_width(_bare_composer(), 40, monkeypatch) == 0


def test_cap_accounts_for_attachment_actions_width(monkeypatch):
    """A staged attachment widens the actions row by 4; the strip's budget
    shrinks by exactly that."""
    composer = _bare_composer()
    composer._pending_attachment_label = "image.png"
    assert _cap_with_row_width(composer, 100, monkeypatch) == 19


# ---------------------------------------------------------------------------
# TASK-24620: the dictation voice chip must not starve the draft either.
# Same starvation class TASK-24415 fixed for the send-disabled reason strip:
# set_voice_status sized the chip against the composer's FULL width with
# only VOICE_CHIP_MIN_WIDTH (24) reserved, ignoring the left cluster, the
# actions row, and the draft -- a long state message took up to 53 cells
# and the 1fr draft (min_width 0) got the remainder.
# ---------------------------------------------------------------------------

#: A realistic long chip copy (error states and executor-wait copy run this
#: long; the preparing copy is 51 cells).
CHIP_MESSAGE = "Microphone unavailable — check permissions and retry"


@pytest.mark.asyncio
async def test_narrow_composer_keeps_draft_visible_beside_voice_chip():
    _, host = _ready_host()
    async with host.run_test(size=APP_NARROW) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.query_one("#console-command-visible-text", Static)
        chip = composer.query_one("#console-voice-status", Static)
        await pilot.pause()
        await pilot.pause()

        composer.set_voice_status("error", message=CHIP_MESSAGE)
        await pilot.pause()
        await pilot.pause()
        # Below the legible budget the chip hides -- the draft floor wins,
        # and the Dictate button's own label still carries the mic state.
        assert chip.display is False
        assert draft.region.width >= 8, (
            f"visible draft collapsed to {draft.region.width} columns beside "
            "the voice chip at 80-column app width"
        )


@pytest.mark.asyncio
async def test_wide_composer_keeps_voice_chip_capped_and_draft_floor():
    _, host = _ready_host()
    async with host.run_test(size=APP_WIDE) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.query_one("#console-command-visible-text", Static)
        chip = composer.query_one("#console-voice-status", Static)
        await pilot.pause()
        await pilot.pause()

        composer.set_voice_status("error", message=CHIP_MESSAGE)
        await pilot.pause()
        await pilot.pause()
        assert chip.display is True
        assert chip.region.width <= ConsoleComposerBar.VOICE_CHIP_MAX_WIDTH
        assert draft.region.width >= DRAFT_FLOOR


@pytest.mark.asyncio
async def test_resize_from_wide_to_narrow_retracts_the_voice_chip():
    _, host = _ready_host()
    async with host.run_test(size=APP_WIDE) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        draft = composer.query_one("#console-command-visible-text", Static)
        composer.set_voice_status("error", message=CHIP_MESSAGE)
        await pilot.pause()
        await pilot.pause()
        assert draft.region.width >= DRAFT_FLOOR

        await pilot.resize_terminal(APP_NARROW[0], APP_NARROW[1])
        await pilot.pause()
        await pilot.pause()

        assert draft.region.width >= 8, (
            "after resizing to a narrow terminal the draft collapsed to "
            f"{draft.region.width} columns while the voice chip still held "
            "layout space"
        )


# ---------------------------------------------------------------------------
# Direct unit coverage for _voice_chip_width_cap's branches (TASK-24620):
# same stub-row pattern as the reason-cap tests above. Reserved = left
# cluster (18) + actions (25) + ADVISORY_MARGIN_ALLOWANCE (2) = 45, so the
# budget is row_width - 45 - DRAFT_MIN_RENDER_WIDTH(32) = row_width - 77.
# ---------------------------------------------------------------------------


def test_chip_cap_without_a_mounted_row_falls_back_to_the_ceiling():
    assert _bare_composer()._voice_chip_width_cap() == (
        ConsoleComposerBar.VOICE_CHIP_MAX_WIDTH
    )


def test_chip_cap_branches(monkeypatch):
    composer = _bare_composer()
    # Pre-layout row: fallback ceiling.
    assert _chip_cap_with_row_width(composer, 0, monkeypatch) == 53
    # Wide row: the ceiling binds, not the budget.
    assert _chip_cap_with_row_width(composer, 200, monkeypatch) == 53
    # Intermediate: the spare budget binds (100 - 77 = 23).
    assert _chip_cap_with_row_width(composer, 100, monkeypatch) == 23
    # Legibility boundary: 89 - 77 = 12 shows; 88 - 77 = 11 hides.
    assert _chip_cap_with_row_width(composer, 89, monkeypatch) == 12
    assert _chip_cap_with_row_width(composer, 88, monkeypatch) == 0
    # Far below: hide.
    assert _chip_cap_with_row_width(composer, 40, monkeypatch) == 0


def test_reason_cap_subtracts_a_displayed_chip(monkeypatch):
    """The chip has priority; the reason strip budgets around its cached
    width (TASK-24620: without this the two strips jointly starved the
    draft by each claiming the full remainder)."""
    composer = _bare_composer()
    composer._voice_chip_last_width = 40
    # 160 - 45 - 32 - 40 = 43, under the 52 cap.
    assert _cap_with_row_width(composer, 160, monkeypatch) == 43
    composer._voice_chip_last_width = 90
    # 160 - 45 - 32 - 90 < 0 -> hide.
    assert _cap_with_row_width(composer, 160, monkeypatch) == 0
