"""Vertical (up/down) caret movement in the Console composer.

The composer already grows to `MAX_DRAFT_ROWS` (4) wrapped rows, but the
caret could only move left/right/home/end -- never up/down -- inside a
wrapped or multi-line draft. `ConsoleComposerBar.move_cursor_up`/
`move_cursor_down` add row-stepping caret movement; `ChatScreen.on_key`
routes the "up"/"down" keys into them, consuming the event only when the
move actually happened (the composer has no goal-column memory across
moves, and the first/last visual row is a real boundary, not a wrap-around).

All widget-level and routed tests below run at app size (120, 30), the
exact fixture size `test_console_composer_overflow.py` establishes a
visible-draft wrap width of 57 cells for (`ConsoleComposerBar.
_draft_render_width()` reads the mounted Static's own region, so tests that
call `move_cursor_up`/`move_cursor_down` -- not just the pure wrap
classmethods -- need a real mount at a known size to get a deterministic
width).
"""

from __future__ import annotations

import pytest
from textual.events import Key
from textual.widgets import Static

from Tests.UI.test_console_composer_overflow import _CssTrueConsoleHarness, WIDTH
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

APP_SIZE = (120, 30)
assert WIDTH == 57  # pins the premise every hand-computed offset below relies on.


# ---------------------------------------------------------------------------
# Widget-level: explicit multi-line drafts (real `\n` line breaks)
# ---------------------------------------------------------------------------


async def _focused_composer(host, pilot, text: str) -> ConsoleComposerBar:
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.load_draft(text)
    composer.focus()
    await pilot.pause()
    return composer


@pytest.mark.asyncio
async def test_up_from_mid_row_two_lands_on_the_same_column_in_row_one():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(
            host, pilot, "first line here\nsecond row of text\nthird line content"
        )
        # row0 "first line here"        (15 chars) -> [0, 15)
        # row1 "second row of text"     (18 chars) -> [16, 34)
        # row2 "third line content"     (18 chars) -> [35, 53)
        composer.position_cursor_from_display_index(35 + 5)
        assert composer.cursor_index == 40

        moved = composer.move_cursor_up()
        await pilot.pause()

        assert moved is True
        assert composer.cursor_index == 16 + 5


@pytest.mark.asyncio
async def test_down_mirrors_up_back_to_the_same_column_in_row_two():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(
            host, pilot, "first line here\nsecond row of text\nthird line content"
        )
        # See row offsets in the mirrored Up test above.
        composer.position_cursor_from_display_index(16 + 5)
        assert composer.cursor_index == 21

        moved = composer.move_cursor_down()
        await pilot.pause()

        assert moved is True
        assert composer.cursor_index == 35 + 5


@pytest.mark.asyncio
async def test_up_from_the_first_row_returns_false_and_moves_nothing():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "only one row\nsecond row")
        composer.position_cursor_from_display_index(4)
        before = composer.cursor_index

        moved = composer.move_cursor_up()
        await pilot.pause()

        assert moved is False
        assert composer.cursor_index == before


@pytest.mark.asyncio
async def test_down_from_the_last_row_returns_false_and_moves_nothing():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "first row\nlast row here")
        # cursor lands at the end (last row) by default via `load_draft`.
        before = composer.cursor_index
        assert before == len("first row\nlast row here")

        moved = composer.move_cursor_down()
        await pilot.pause()

        assert moved is False
        assert composer.cursor_index == before


@pytest.mark.asyncio
async def test_up_from_a_long_rows_column_fifty_clamps_into_a_twenty_char_row():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        row0 = "12345678901234567890"  # 20 chars -> [0, 20)
        row1 = "A" * 55  # -> [21, 76)
        composer = await _focused_composer(host, pilot, f"{row0}\n{row1}")
        composer.position_cursor_from_display_index(21 + 50)
        assert composer.cursor_index == 71

        moved = composer.move_cursor_up()
        await pilot.pause()

        assert moved is True
        # Clamped to row0's own length (20), i.e. row0's end -- not the raw
        # column-50 offset carried over unclamped.
        assert composer.cursor_index == 20


# ---------------------------------------------------------------------------
# Widget-level: a single soft-wrapped (no explicit `\n`) long line
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_up_across_a_soft_wrapped_line_lands_on_the_same_column():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        # No whitespace at all -> `_cell_wrap_line` hard-breaks by width with
        # nothing to greedily fill around, so this wraps to exactly
        # [0,57) [57,114) [114,150) at WIDTH == 57 -- three soft-wrapped rows,
        # no real `\n` anywhere in the source text.
        text = "".join(str(i % 10) for i in range(150))
        composer = await _focused_composer(host, pilot, text)
        assert len(ConsoleComposerBar._wrap_draft_line_slices(text, WIDTH)) == 3

        composer.position_cursor_from_display_index(57 + 30)
        assert composer.cursor_index == 87

        moved = composer.move_cursor_up()
        await pilot.pause()

        assert moved is True
        assert composer.cursor_index == 30


# ---------------------------------------------------------------------------
# Routed: ChatScreen.on_key -> composer.move_cursor_up/move_cursor_down
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pilot_press_up_and_down_move_the_real_caret():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(
            host, pilot, "line0\nline1\nline2"
        )
        # row0 "line0" -> [0, 5); row1 "line1" -> [6, 11); row2 "line2" -> [12, 17)
        composer.position_cursor_from_display_index(12 + 2)
        assert composer.cursor_index == 14

        await pilot.press("up")
        await pilot.pause()
        assert composer.cursor_index == 6 + 2

        await pilot.press("up")
        await pilot.pause()
        assert composer.cursor_index == 0 + 2

        await pilot.press("down")
        await pilot.pause()
        assert composer.cursor_index == 6 + 2


@pytest.mark.asyncio
async def test_pilot_press_up_on_the_first_row_leaves_the_caret_untouched():
    """Side-effect signal only (fragile to prove non-consumption this way
    alone -- the unit-style test below asserts the actual conditional-stop
    routing behavior directly)."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "only one row")
        before = composer.cursor_index

        await pilot.press("up")
        await pilot.pause()

        assert composer.cursor_index == before


@pytest.mark.asyncio
async def test_on_key_only_consumes_up_and_down_when_the_move_actually_happened(
    monkeypatch,
):
    """Unit-style verification of `ChatScreen.on_key`'s conditional-consume
    branch, independent of any cursor side effect: the routing must call
    `event.stop()`/`event.prevent_default()` if and only if
    `move_cursor_up`/`move_cursor_down` returned True.

    A mutation that makes the routing consume unconditionally (`event.stop()`
    called regardless of the move's return value) fails the False-return
    assertions here.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("some draft text")
        composer.focus()
        await pilot.pause()

        monkeypatch.setattr(composer, "move_cursor_up", lambda: False)
        not_consumed_up = Key("up", None)
        console.on_key(not_consumed_up)
        assert not_consumed_up._stop_propagation is False
        assert not_consumed_up._no_default_action is False

        monkeypatch.setattr(composer, "move_cursor_down", lambda: False)
        not_consumed_down = Key("down", None)
        console.on_key(not_consumed_down)
        assert not_consumed_down._stop_propagation is False
        assert not_consumed_down._no_default_action is False

        monkeypatch.setattr(composer, "move_cursor_up", lambda: True)
        consumed_up = Key("up", None)
        console.on_key(consumed_up)
        assert consumed_up._stop_propagation is True
        assert consumed_up._no_default_action is True

        monkeypatch.setattr(composer, "move_cursor_down", lambda: True)
        consumed_down = Key("down", None)
        console.on_key(consumed_down)
        assert consumed_down._stop_propagation is True
        assert consumed_down._no_default_action is True


# ---------------------------------------------------------------------------
# Windowed drafts (>4 rows): the caret-following window must re-window on Up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_moving_up_above_the_visible_window_scrolls_it_to_follow_the_caret():
    """>MAX_DRAFT_ROWS-row draft; walk the caret up from the last row past
    the initially-visible window and confirm the PAINTED first row (CSS-true
    harness, `render_line`) shows earlier content the window did not
    originally include -- the bare harness loads no CSS and lies about
    painted geometry (see `_CssTrueConsoleHarness`'s own docstring).
    """
    app, _ = _ready_host()
    host = _CssTrueConsoleHarness(app)
    async with host.run_test(size=APP_SIZE) as pilot:
        text = "\n".join(f"LINE{i}" for i in range(8))
        composer = await _focused_composer(host, pilot, text)
        assert composer.cursor_index == len(text)  # row7 (LINE7), the tail.

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        initial_row0 = visible_draft.render_line(0).text
        assert "LINE0" not in initial_row0

        for _ in range(7):
            assert composer.move_cursor_up() is True
            await pilot.pause()

        # Every "LINEn" row is exactly 5 characters, so climbing straight up
        # preserves the starting column (5, the tail of "LINE7") the whole
        # way -- no clamp is ever needed -- landing at the tail of "LINE0",
        # not column 0.
        assert composer.cursor_index == 5
        final_row0 = visible_draft.render_line(0).text
        assert "LINE0" in final_row0

        # One more Up: already on the first row -- False, nothing moves, and
        # the window (already showing row0 first) stays exactly as painted.
        assert composer.move_cursor_up() is False
        await pilot.pause()
        assert composer.cursor_index == 5
        assert "LINE0" in visible_draft.render_line(0).text
