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
    # De-flake (established pattern, test_console_composer_cursor.py): own
    # every blink phase so a painted-caret-glyph assertion can never race a
    # periodic blink tick that hides it.
    composer._cursor_blink_timer.pause()
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


# ---------------------------------------------------------------------------
# Fix round 2 (review): the splice-based mapping drifted on soft-wrapped
# rows. Coverage below cross-checks `cursor_index` against the INDEPENDENTLY
# PAINTED caret glyph (`_draft_renderable`'s own splice, an entirely
# different code path) rather than trusting the same arithmetic under test
# to also grade itself -- the review's own blind-spot list.
# ---------------------------------------------------------------------------


def _painted_caret_rowcol(visible_draft: Static) -> tuple[int, int]:
    """Return (row, column) of the painted caret glyph, or raise if absent."""
    for row in range(visible_draft.size.height):
        text = visible_draft.render_line(row).text
        column = text.find(ConsoleComposerBar.CURSOR_GLYPH)
        if column != -1:
            return row, column
    raise AssertionError("caret glyph not painted in any visible row")


@pytest.mark.asyncio
async def test_down_across_soft_wrapped_rows_matches_painted_caret_including_column_zero():
    """HIGH fix: Down across soft-wrapped rows, at several columns including
    0, must land at the same column (clamped to the row below's length) --
    verified two ways: against `_wrap_draft_line_slices` (the row/offset
    authority) AND against the independently-painted caret glyph. The
    reviewer's differential sweep found 114/150 wrong positions on this
    exact shape (digits, no whitespace, no explicit `\\n`) under the old
    splice-based mapping, including a degenerate case where Down from
    column 0 didn't change rows at all -- covered explicitly below.
    """
    app, _ = _ready_host()
    host = _CssTrueConsoleHarness(app)
    async with host.run_test(size=APP_SIZE) as pilot:
        text = "".join(str(i % 10) for i in range(150))
        composer = await _focused_composer(host, pilot, text)
        width = composer._draft_render_width()
        slices = ConsoleComposerBar._wrap_draft_line_slices(text, width)
        assert len(slices) == 3  # still soft-wrapped into 3 rows at this width.
        row0, row1, row2 = slices

        visible_draft = composer.query_one("#console-command-visible-text", Static)

        for column in (0, 1, 10, len(row0.text) // 2, len(row0.text) - 1):
            composer.position_cursor_from_display_index(row0.start + column)
            await pilot.pause()
            assert _painted_caret_rowcol(visible_draft) == (0, column), column

            moved = composer.move_cursor_down()
            await pilot.pause()

            expected_column = min(column, len(row1.text))
            expected_index = row1.start + expected_column
            assert moved is True, column
            assert composer.cursor_index == expected_index, column
            assert (
                _painted_caret_rowcol(visible_draft) == (1, expected_column)
            ), column

        # row1 -> row2: row2 is the SHORT remainder row, so a late column in
        # row1 exercises the clamp on a genuinely soft-wrapped transition
        # (the existing clamp test only covers explicit-`\n` rows).
        assert len(row2.text) < len(row1.text)
        late_column = len(row1.text) - 1
        composer.position_cursor_from_display_index(row1.start + late_column)
        await pilot.pause()
        assert _painted_caret_rowcol(visible_draft) == (1, late_column)

        moved = composer.move_cursor_down()
        await pilot.pause()

        expected_column = len(row2.text)  # clamped to row2's own length.
        assert moved is True
        assert composer.cursor_index == row2.start + expected_column
        assert _painted_caret_rowcol(visible_draft) == (2, expected_column)


@pytest.mark.asyncio
async def test_up_from_the_first_column_after_a_whitespace_wrap_boundary_ascends():
    """MEDIUM fix: a caret sitting right at a whitespace wrap boundary (the
    first column of a soft-wrapped row) must resolve Up to the row ABOVE,
    matching where the caret is actually PAINTED -- the old space-splice
    model kept it on the row below (a space extends the trailing whitespace
    run and stays on the earlier row; the real `CURSOR_GLYPH` attaches to
    the following word and wraps down), so `move_cursor_up()` returned
    False on a caret the user can plainly see sitting on the row below.
    """
    app, _ = _ready_host()
    host = _CssTrueConsoleHarness(app)
    async with host.run_test(size=APP_SIZE) as pilot:
        text = "the quick brown fox jumps over the lazy dog by the winding river " * 3
        composer = await _focused_composer(host, pilot, text)
        width = composer._draft_render_width()
        slices = ConsoleComposerBar._wrap_draft_line_slices(text, width)
        assert len(slices) >= 2
        # A genuine soft-wrap boundary: row0's end IS row1's start (no
        # separator between them -- this is prose with no explicit `\n`).
        assert slices[0].end == slices[1].start

        boundary_index = slices[1].start  # column 0 of row 1.
        composer.position_cursor_from_display_index(boundary_index)
        await pilot.pause()

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        # Pins the review's premise: the caret paints on row 1, column 0 --
        # not row 0 -- before any move happens.
        assert _painted_caret_rowcol(visible_draft) == (1, 0)

        moved = composer.move_cursor_up()
        await pilot.pause()

        assert moved is True
        assert composer.cursor_index == slices[0].start  # column 0 of row 0.
        assert _painted_caret_rowcol(visible_draft) == (0, 0)


@pytest.mark.asyncio
async def test_windowed_draft_with_unequal_row_lengths_clamps_while_climbing():
    """Windowed (>4-row) draft with genuinely UNEQUAL row lengths -- the
    original windowed test used 8 equal-length rows, which the report
    itself noted meant "no clamp is ever needed"; this one forces a clamp
    mid-climb (row2 is 1 character) while also exercising two no-clamp
    transitions, cross-checked against hand-derived offsets from the same
    `_wrap_draft_line_slices` rows the implementation itself walks.
    """
    app, _ = _ready_host()
    host = _CssTrueConsoleHarness(app)
    async with host.run_test(size=APP_SIZE) as pilot:
        rows = ["SENT0", "B" * 20, "C", "D" * 15, "EE"]
        text = "\n".join(rows)
        composer = await _focused_composer(host, pilot, text)
        assert composer.cursor_index == len(text)  # tail: row4 ("EE"), col 2.

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        initial_painted = "".join(
            visible_draft.render_line(row).text
            for row in range(visible_draft.size.height)
        )
        assert "SENT0" not in initial_painted  # row0 starts outside the window.

        # Hand-derived from the same row boundaries `_wrap_draft_line_slices`
        # produces for this text (verified independently: row0 "SENT0" [0,5),
        # row1 "B"*20 [6,26), row2 "C" [27,28), row3 "D"*15 [29,44), row4
        # "EE" [45,47)). Column carried from row4 (2) clamps to row2's
        # single character (up#2, 2 -> 1); every other step carries its
        # column unclamped.
        expected_after_each_up = [31, 28, 7, 1]

        for expected in expected_after_each_up:
            assert composer.move_cursor_up() is True
            await pilot.pause()
            assert composer.cursor_index == expected

        # Row0 is now inside the (re-centered) window, and painted at the
        # very top since nothing is scrolled off above it. The caret glyph
        # lands mid-word (column 1 of "SENT0"), splitting the literal
        # substring -- strip it before checking row content.
        row0_stripped = visible_draft.render_line(0).text.replace(
            ConsoleComposerBar.CURSOR_GLYPH, ""
        )
        assert "SENT0" in row0_stripped
        assert _painted_caret_rowcol(visible_draft) == (0, 1)

        # One more Up: already on row 0 -- False, nothing moves further.
        assert composer.move_cursor_up() is False
        assert composer.cursor_index == 1


# ---------------------------------------------------------------------------
# Fix round 2 (review): LOW findings -- a no-op boundary Up/Down must still
# collapse a full-draft selection and break undo-coalescing, exactly like
# every other move method's own boundary case (`move_cursor_left` at index
# 0, `move_cursor_right` at the draft's end). Pure unmounted-widget tests,
# following `test_console_composer_undo.py`'s own established convention.
# ---------------------------------------------------------------------------


def test_noop_up_collapses_a_full_draft_selection_so_typing_inserts_not_replaces():
    composer = ConsoleComposerBar()
    composer.load_draft("hello")
    composer.select_all_draft()
    assert composer.has_full_draft_selection() is True

    moved = composer.move_cursor_up()  # single row -- boundary, no-op.

    assert moved is False
    assert composer.has_full_draft_selection() is False
    composer.insert_text("X")
    assert composer.draft_text() == "helloX"  # inserted at the caret, not a replace.


def test_noop_down_breaks_typed_run_coalescing_like_every_other_cursor_key():
    composer = ConsoleComposerBar()
    for character in "ab":
        composer.insert_text(character)
    assert composer.draft_text() == "ab"

    moved = composer.move_cursor_down()  # single row -- boundary, no-op.

    assert moved is False
    for character in "cd":
        composer.insert_text(character)
    assert composer.draft_text() == "abcd"

    # Two separate undo arcs ("cd" then "ab"), not one combined "abcd" --
    # proves the no-op broke coalescing between the two typed runs.
    assert composer.undo() is True
    assert composer.draft_text() == "ab"
    assert composer.undo() is True
    assert composer.draft_text() == ""
    assert composer.undo() is False


# ---------------------------------------------------------------------------
# Fix round 3 (re-review): LOW-A -- clamping to a soft-wrap-CONTIGUOUS target
# row's own full length lands the offset on the NEXT row's column 0 (the two
# rows share that exact source offset with no separator between them), which
# `_row_index_for_canonical_offset` -- and the painted glyph -- both resolve
# to the next row, not the intended target. Zero on explicit-newline drafts,
# where a row's own separator gives "end of row" a distinct offset.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clamp_into_a_soft_wrap_contiguous_row_stays_on_that_row_not_the_next():
    """Reviewer's live reproduction (re-review), reproduced generically: a
    column near the end of row 1 -- at or past the length of BOTH
    neighboring rows -- used to make Up read as a no-op-that-still-consumes
    (painted row unchanged, stuck at the boundary's column 0 instead of
    ascending to row 0) and Down skip row 2 entirely and land on row 3's
    column 0. Both directions must instead land on the LAST valid column
    of the actual target row (length - 1), painted there, not bumped past
    it onto the next row.
    """
    app, _ = _ready_host()
    host = _CssTrueConsoleHarness(app)
    async with host.run_test(size=APP_SIZE) as pilot:
        text = "the quick brown fox jumps over the lazy dog by the winding river " * 3
        composer = await _focused_composer(host, pilot, text)
        width = composer._draft_render_width()
        slices = ConsoleComposerBar._wrap_draft_line_slices(text, width)
        assert len(slices) >= 4
        row0, row1, row2, row3 = slices[:4]
        # All soft-wrap contiguous -- this prose has no explicit `\n` at all.
        assert row0.end == row1.start
        assert row1.end == row2.start
        assert row2.end == row3.start

        # A column near row1's own end, at or past BOTH neighboring rows'
        # lengths -- the shape the reviewer's live reproduction used (row1
        # columns 51-53 of a 54-char row, flanked by two 51-char rows).
        column = len(row1.text) - 1
        assert column >= len(row0.text)
        assert column >= len(row2.text)

        visible_draft = composer.query_one("#console-command-visible-text", Static)

        composer.position_cursor_from_display_index(row1.start + column)
        await pilot.pause()
        moved_up = composer.move_cursor_up()
        await pilot.pause()
        assert moved_up is True
        assert _painted_caret_rowcol(visible_draft)[0] == 0
        assert composer.cursor_index == row0.start + len(row0.text) - 1

        composer.position_cursor_from_display_index(row1.start + column)
        await pilot.pause()
        moved_down = composer.move_cursor_down()
        await pilot.pause()
        assert moved_down is True
        assert _painted_caret_rowcol(visible_draft)[0] == 2
        assert composer.cursor_index == row2.start + len(row2.text) - 1
