"""Characterisation of the Console composer keymap (decomposition wave 5).

These tests exist to pin the *observable* behaviour of the composer key
table -- the resulting `draft_text()` and `cursor_index` after a REAL key
press driven through `pilot` -- so that moving the composer-only branches
out of `ChatScreen.on_key` and into `ConsoleComposerBar.handle_console_key`
can be proven byte-for-byte behaviour-preserving. They deliberately assert
end state, never internal calls: a characterisation test that asserts
"`delete_left` was called" would pass just as happily against a broken
move.

The screen-level routing is pinned here too, because it is exactly what the
move must NOT change: `ChatScreen.on_key`'s stated policy is to "treat the
Console composer as the default printable text target", so a printable key
pressed while *nothing* is focused still has to land in the composer. That
policy stays on the screen (`_should_capture_console_input` is the gate);
only the per-key branch bodies move.

The host is the real Console screen (`test_console_dictation._ready_host`),
not a hand-built screen fixture -- the keymap is only observable through
the screen's `on_key` before the move, so anything lighter would be
characterising a reimplementation of the thing under test rather than the
thing itself.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

APP_SIZE = (140, 42)


async def _focused_composer(host, pilot, text: str = "") -> ConsoleComposerBar:
    """Mount the ready Console, load `text`, and focus the composer."""
    console = await _mounted_console(host, pilot)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    if text:
        composer.load_draft(text)
    composer.focus()
    await pilot.pause()
    return composer


# ---------------------------------------------------------------------------
# Deletion keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backspace_deletes_the_character_left_of_the_caret():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "hello")
        assert composer.cursor_index == 5

        await pilot.press("backspace")
        await pilot.pause()

        assert composer.draft_text() == "hell"
        assert composer.cursor_index == 4


@pytest.mark.asyncio
async def test_delete_removes_the_character_right_of_the_caret():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "hello")
        composer.position_cursor_from_display_index(2)
        await pilot.pause()
        assert composer.cursor_index == 2

        await pilot.press("delete")
        await pilot.pause()

        assert composer.draft_text() == "helo"
        assert composer.cursor_index == 2


@pytest.mark.asyncio
async def test_ctrl_w_deletes_the_word_left_of_the_caret():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "hello world")

        await pilot.press("ctrl+w")
        await pilot.pause()

        assert composer.draft_text() == "hello "
        assert composer.cursor_index == len("hello ")


@pytest.mark.asyncio
async def test_ctrl_u_clears_the_whole_draft():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "throw me away")

        await pilot.press("ctrl+u")
        await pilot.pause()

        assert composer.draft_text() == ""
        assert composer.cursor_index == 0


# ---------------------------------------------------------------------------
# Caret movement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_left_and_right_step_the_caret_without_changing_the_draft():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "abcdef")

        await pilot.press("left")
        await pilot.press("left")
        await pilot.pause()
        assert composer.cursor_index == 4

        await pilot.press("right")
        await pilot.pause()
        assert composer.cursor_index == 5
        assert composer.draft_text() == "abcdef"


@pytest.mark.asyncio
async def test_home_and_end_jump_the_caret_to_the_draft_bounds():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "abcdef")

        await pilot.press("home")
        await pilot.pause()
        assert composer.cursor_index == 0

        await pilot.press("end")
        await pilot.pause()
        assert composer.cursor_index == 6
        assert composer.draft_text() == "abcdef"


@pytest.mark.asyncio
async def test_up_and_down_step_visual_rows_inside_a_multiline_draft():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "first\nsecond\nthird")
        # rows: "first" [0,5)  "second" [6,12)  "third" [13,18)
        composer.position_cursor_from_display_index(13 + 2)
        await pilot.pause()
        assert composer.cursor_index == 15

        # Not on the last visual row after the first Up, so history recall
        # declines and ordinary caret movement gets its chance.
        await pilot.press("up")
        await pilot.pause()
        assert composer.cursor_index == 6 + 2

        await pilot.press("down")
        await pilot.pause()
        assert composer.cursor_index == 13 + 2
        assert composer.draft_text() == "first\nsecond\nthird"


@pytest.mark.asyncio
async def test_up_on_the_first_visual_row_never_moves_the_caret():
    """Row 0 hands Up to prompt-history recall, which never moves the caret."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "first\nsecond")
        composer.position_cursor_from_display_index(3)
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()

        assert composer.cursor_index == 3


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_a_selects_the_entire_draft():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "select all of me")
        assert composer.has_full_draft_selection() is False

        await pilot.press("ctrl+a")
        await pilot.pause()

        assert composer.has_full_draft_selection() is True
        assert composer.draft_text() == "select all of me"


# ---------------------------------------------------------------------------
# Undo / redo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_z_undoes_a_typed_run_and_ctrl_shift_z_redoes_it():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)

        await pilot.press("h", "i")
        await pilot.pause()
        assert composer.draft_text() == "hi"

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == ""

        await pilot.press("ctrl+shift+z")
        await pilot.pause()
        assert composer.draft_text() == "hi"


@pytest.mark.asyncio
async def test_enter_uses_expand_confirmation_copy_for_a_collapsed_paste():
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"Pasted text | {len(pasted)} characters | Expand"
        )

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == "Expand?"
        assert composer.draft_text() == pasted


# ---------------------------------------------------------------------------
# Screen-level routing that the move must NOT change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_printable_key_reaches_the_composer_while_nothing_is_focused():
    """`on_key`'s policy: the composer is the default printable text target."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        host.set_focus(None)
        await pilot.pause()
        assert host.focused is None

        await pilot.press("z")
        await pilot.pause()

        assert composer.draft_text() == "z"
        assert composer.cursor_index == 1


@pytest.mark.asyncio
async def test_an_edit_key_reaches_the_composer_while_nothing_is_focused():
    """The same default-target policy covers the edit keys, not just typing."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")
        host.set_focus(None)
        await pilot.pause()
        assert host.focused is None

        await pilot.press("left")
        await pilot.press("backspace")
        await pilot.pause()

        assert composer.draft_text() == "ac"
        assert composer.cursor_index == 1
