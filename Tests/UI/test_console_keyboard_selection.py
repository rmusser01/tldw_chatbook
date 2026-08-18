"""Keyboard text-selection mode state machine (console selection phase 5).

``s`` arms keyboard-driven text selection on the currently j/k-selected
message row -- a keyboard-only entry into the same ``SelectionManager`` the
mouse drag already drives (phase 1). Task 2 built the mode's skeleton
(enter/exit, Escape layering ahead of the existing clear-selection binding,
mouse takeover, and the row-destruction guard). Task 3 (this file's newer
tests) wires the h/l/w/b/0/$/j/k/o motion keys that move the selection, and
Task 4 wires the copy/quote actions on Enter.
"""

import pytest
from textual.app import App, ComposeResult
from textual.events import MouseDown

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_selection_menu import ConsoleSelectionMenu
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleMarkdownMessage,
    ConsoleToolDiffRow,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)

_DIFF_PATH = "/tmp/a.py"
_DIFF_OLD = "alpha\nbeta\ngamma\n"
_DIFF_NEW = "alpha\nBETA\ngamma\ndelta\n"


class _KeyboardSelectionApp(App[None]):
    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT, content="answer text", id="m1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content="second message",
                    id="m2",
                ),
                # USER rows never take the assistant-markdown path (see
                # ``ConsoleTranscript._render_row``), so these are always
                # ``ConsoleTranscriptMessage`` (plain) regardless of the
                # ``[chat_defaults] assistant_markdown`` toggle -- Task 3's
                # plain-row motion tests need that guarantee.
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="answer text", id="p1"
                ),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content="line one\nline two\nline three",
                    id="p2",
                ),
            ]
        )
        # A summary-boundary banner is a non-interactive, non-addressable
        # (never has a ``console-message-*`` id) render-derived row --
        # present here so mode entry on an eligible row elsewhere is
        # unaffected by a protected row sharing the transcript.
        transcript.summary_boundary_message_id = "m2"
        yield transcript


def _mouse_event(
    event_cls, widget, *, screen_x: int, screen_y: int, button: int = 1
) -> object:
    """Build a mouse event addressed to ``widget`` with absolute coordinates.

    Copied from ``test_console_selection_transcript.py``'s helper of the
    same name/shape (kept local -- these are private test helpers, not a
    shared module).
    """
    return event_cls(
        widget=widget,
        x=screen_x - widget.region.x,
        y=screen_y - widget.region.y,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


async def _mounted_row(
    pilot, message_id: str
) -> ConsoleTranscriptMessage | ConsoleMarkdownMessage:
    transcript = pilot.app.query_one(ConsoleTranscript)
    transcript.set_messages(transcript._messages)
    await transcript.refresh_messages()
    await pilot.pause()
    return pilot.app.query_one(
        f"#console-message-{message_id}",
        (ConsoleTranscriptMessage, ConsoleMarkdownMessage),
    )


@pytest.mark.asyncio
async def test_s_enters_mode_on_selected_eligible_row():
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        transcript.selected_message_id = row.message_id

        await pilot.press("s")
        await pilot.pause()

        assert transcript._kb_selection_row is row
        sel = transcript.selection_manager.state.selection
        assert sel is not None and (sel.start, sel.end) == (0, 1)
        assert row.get_selection_text() == row.get_display_text()[0:1]
        hint = transcript.query_one("#console-kb-selection-hint")
        assert hint.display is True


@pytest.mark.asyncio
async def test_s_without_selection_or_on_protected_row_is_a_toast_not_a_mode():
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")
        hint = transcript.query_one("#console-kb-selection-hint")

        # No message selection at all: there is nothing to arm a keyboard
        # text-selection cursor on.
        assert transcript.selected_message_id is None
        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_selection_row is None
        assert hint.display is False

        # A selected id with no currently-addressable, eligible row mounted
        # (here: hidden inside the height-watermark prune window -- the same
        # "no row to arm on" outcome a protected/non-addressable row like the
        # summary banner produces) is still a toast, not a mode.
        transcript.selected_message_id = "m1"
        transcript._pruned_message_ids.add("m1")
        await transcript.refresh_messages()
        await pilot.pause()

        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_selection_row is None
        assert hint.display is False


@pytest.mark.asyncio
async def test_escape_layering_mode_first_message_selection_second():
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        transcript.selected_message_id = row.message_id
        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_selection_row is row

        # First Escape: mode off, text selection cleared, message selection
        # (Enter's j/k selection) untouched.
        await pilot.press("escape")
        await pilot.pause()

        assert transcript._kb_selection_row is None
        assert transcript.selection_manager.state.selection is None
        assert transcript.selected_message_id == row.message_id
        hint = transcript.query_one("#console-kb-selection-hint")
        assert hint.display is False

        # Second Escape: the existing clear-selection binding, now free to
        # fire because the mode's on_key branch no longer intercepts it.
        await pilot.press("escape")
        await pilot.pause()

        assert transcript.selected_message_id is None


@pytest.mark.asyncio
async def test_mouse_down_exits_mode_before_arming_a_drag():
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        transcript.selected_message_id = row.message_id
        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_selection_row is row

        # The row's own region works for either mounted widget type (plain
        # or markdown) -- no need for the body Static's class, which only
        # the plain row exposes.
        rr = row.region
        row.post_message(
            _mouse_event(MouseDown, row, screen_x=rr.x + 2, screen_y=rr.bottom - 1)
        )
        await pilot.pause()

        assert transcript._kb_selection_row is None


@pytest.mark.asyncio
async def test_row_destruction_guard_exits_mode_without_crash():
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        transcript.selected_message_id = row.message_id
        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_selection_row is row

        transcript.set_messages([])
        await transcript.refresh_messages()
        await pilot.pause()

        # The armed row is gone; a key that means nothing to the mode itself
        # (Task 3 has not wired motions yet) must not crash, and the
        # destruction guard must have exited the mode.
        await pilot.press("l")
        await pilot.pause()

        assert transcript._kb_selection_row is None
        hint = transcript.query_one("#console-kb-selection-hint")
        assert hint.display is False


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["down", "up"])
async def test_message_nav_keys_are_no_ops_in_mode(key):
    """`down`/`up`/`enter` stay inert in the mode -- `j`/`k` graduated to real
    line motions in Task 3 (see ``test_j_k_move_by_line_preserving_column``
    and ``test_j_is_inert_at_text_end_on_a_single_line_row``) and are no
    longer part of this no-op contract.

    Before Task 2's fix, only `escape` was intercepted by the mode branch,
    so `down`/`up`/`enter` fell through to the pre-existing BINDINGS chain:
    `down`/`up` moved `selected_message_id` to a different message while
    `_kb_selection_row` (and the manager state, and the hint) stayed pinned
    to the OLD row -- a silent mode/message-selection desync. `enter`
    toggled message selection (and would open the selection menu once one
    exists), which must not fire while keyboard text-selection mode owns
    the keyboard -- Task 4 wires the mode's own Enter-finish action; until
    then it stays a no-op here too.
    """
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        transcript.selected_message_id = row.message_id
        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_selection_row is row

        await pilot.press(key)
        await pilot.pause()

        assert transcript.selected_message_id == row.message_id
        assert transcript._kb_selection_row is row
        sel = transcript.selection_manager.state.selection
        assert sel is not None
        assert (sel.row_key, sel.start, sel.end) == (row.id, 0, 1)
        hint = transcript.query_one("#console-kb-selection-hint")
        assert hint.display is True
        assert not app.query(ConsoleSelectionMenu)


# --- Task 3: motion keys -----------------------------------------------------


@pytest.mark.asyncio
async def test_l_and_h_extend_and_shrink_by_char_on_plain_rows():
    """`l`/`h` walk the active end one character at a time, floored at 1 unit
    away from the anchor -- repeated `h` presses past the floor stop moving
    rather than reaching (or crossing) it."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "p1")
        assert isinstance(row, ConsoleTranscriptMessage)
        transcript.selected_message_id = "p1"
        await pilot.press("s")
        await pilot.pause()

        await pilot.press("l", "l", "l")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 4)

        await pilot.press("h")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 3)

        # Past the floor: several more `h` presses must stop at (0, 1), not
        # cross the anchor (which would make the selection empty/reversed).
        await pilot.press("h", "h", "h", "h", "h")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 1)
        assert transcript._kb_anchor == 0
        assert transcript._kb_end == 1


@pytest.mark.asyncio
async def test_w_b_0_dollar_move_the_active_end():
    """`w`/`b` jump by word, `0`/`$` jump to the current line's bounds -- all
    move the active end, floored away from the fixed anchor exactly like
    `h`/`l` (`0` from a position that maps back to the anchor's own line
    start is floored to `anchor + 1` rather than landing on the anchor)."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "p1")
        transcript.selected_message_id = "p1"
        await pilot.press("s")
        await pilot.pause()

        await pilot.press("w")  # "answer |text" -> start of "text"
        await pilot.pause()
        assert transcript.selection_manager.state.selection.end == 7

        await pilot.press("w")  # already the last word -> end of text
        await pilot.pause()
        assert transcript.selection_manager.state.selection.end == 11

        await pilot.press("b")  # back to the start of "text"
        await pilot.pause()
        assert transcript.selection_manager.state.selection.end == 7

        await pilot.press("0")  # line start == anchor (0) -> floored to 1
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 1)

        await pilot.press("$")  # line end -> end of text
        await pilot.pause()
        assert transcript.selection_manager.state.selection.end == 11


@pytest.mark.asyncio
async def test_j_k_move_by_line_preserving_column():
    """`j`/`k` walk the active end by whole lines on multi-line text,
    preserving the column where the target line allows it."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "p2")
        transcript.selected_message_id = "p2"
        await pilot.press("s")
        await pilot.pause()
        assert transcript._kb_end == 1  # column 1 on "line one"

        await pilot.press("j")
        await pilot.pause()
        assert transcript._kb_end == 10  # column 1 on "line two"

        await pilot.press("j")
        await pilot.pause()
        assert transcript._kb_end == 19  # column 1 on "line three"

        await pilot.press("k")
        await pilot.pause()
        assert transcript._kb_end == 10

        await pilot.press("k")
        await pilot.pause()
        assert transcript._kb_end == 1


@pytest.mark.asyncio
async def test_j_is_inert_at_text_end_on_a_single_line_row():
    """`j` on a row with no next line clamps forward to the text's end (the
    line-motion helper's designed last-line behaviour, task-1 tested); once
    the active end is already there, a further `j` is a genuine no-op --
    there is nowhere left to go."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "p1")
        transcript.selected_message_id = "p1"
        await pilot.press("s")
        await pilot.pause()

        await pilot.press("$")
        await pilot.pause()
        assert transcript.selection_manager.state.selection.end == 11

        await pilot.press("j")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 11)
        assert transcript._kb_anchor == 0
        assert transcript._kb_end == 11


@pytest.mark.asyncio
async def test_o_swaps_anchor_and_end_so_mid_text_spans_are_reachable():
    """`o` swaps anchor and end so a later forward motion can move what WAS
    the anchor, reaching a selection whose start is off the row's origin --
    unreachable with the anchor permanently pinned at 0."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "p1")
        transcript.selected_message_id = "p1"
        await pilot.press("s")
        await pilot.pause()

        await pilot.press("l", "l", "l")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 4)

        await pilot.press("o")
        await pilot.pause()
        assert (transcript._kb_anchor, transcript._kb_end) == (4, 0)
        # The sorted selection is unchanged by the swap itself.
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 4)

        await pilot.press("w")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert sel.start > 0
        assert (sel.start, sel.end) == (3, 4)


@pytest.mark.asyncio
async def test_markdown_rows_take_char_motions():
    """Live-spike fact (Task 2): markdown rows store the selection range as
    raw character offsets, not snapped to whole source lines -- `l`/`w`
    motions land exactly where the pure helpers say, unlike diff rows."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        assert isinstance(row, ConsoleMarkdownMessage)
        transcript.selected_message_id = "m1"
        await pilot.press("s")
        await pilot.pause()

        await pilot.press("l", "l", "l")
        await pilot.pause()
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 4)
        # Stored as-is on the row -- no whole-line snapping.
        assert row._selection_line_range == (0, 4)


@pytest.mark.asyncio
async def test_mode_keys_do_not_leak_to_bindings():
    """An unrecognized single character (e.g. `c`, the Copy binding) is
    claimed and dropped by the mode -- it must not fall through to its
    normal BINDING while keyboard text-selection mode owns the keyboard."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "p1")
        transcript.selected_message_id = "p1"
        await pilot.press("s")
        await pilot.pause()

        calls: list[str] = []
        transcript.action_invoke_selected_action = calls.append

        await pilot.press("c")
        await pilot.pause()

        assert calls == []
        assert transcript._kb_selection_row is row
        sel = transcript.selection_manager.state.selection
        assert (sel.start, sel.end) == (0, 1)


@pytest.mark.asyncio
async def test_char_keys_are_inert_on_diff_rows():
    """Diff rows only take line-granularity motions (`j`/`k`/`o`); char/word
    motions leave the selection untouched. Mode entry cannot reach a diff
    row today (out of scope for Task 2's entry -- see
    ``_enter_keyboard_selection``'s docstring), so this wires the mode state
    directly onto a mounted ``ConsoleToolDiffRow`` and drives
    ``_kb_apply_motion`` the same way the mode's ``on_key`` branch would."""

    class _DiffApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ConsoleTranscript(id="console-native-transcript")

    app = _DiffApp()
    async with app.run_test(size=(60, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        diff_row = ConsoleToolDiffRow("d1", (_DIFF_PATH, _DIFF_OLD, _DIFF_NEW))
        await transcript.mount(diff_row)
        await pilot.pause()

        transcript._kb_selection_row = diff_row
        transcript._kb_anchor, transcript._kb_end = 0, 1
        transcript.selection_manager.begin_drag(diff_row.id, 0)
        transcript.selection_manager.extend_drag(diff_row.id, 1)
        diff_row.set_selection_range(0, 1)
        before_range = diff_row._selection_range

        transcript._kb_apply_motion("l")
        assert transcript._kb_anchor == 0
        assert transcript._kb_end == 1
        assert diff_row._selection_range == before_range

        transcript._kb_apply_motion("j")
        assert transcript._kb_end == 15
        # Grows by one whole (snapped) diff line -- both unified-diff header
        # lines are now in the projection's stored range.
        assert diff_row._selection_range == (0, 27)
        assert diff_row.get_selection_text() == "--- /tmp/a.py\n+++ /tmp/a.py"

        transcript._kb_apply_motion("o")
        assert (transcript._kb_anchor, transcript._kb_end) == (15, 0)


# --- Task 4: Enter finishes the selection and opens the real menu ------------


@pytest.mark.asyncio
async def test_enter_opens_the_same_menu_with_feedback_gating():
    """Enter in mode = mouse-release parity: the SAME TranscriptTextSelected
    path mounts the SAME menu, with feedback buttons present for assistant
    prose and Request/LGTM run-gated (bare harness reports no active run)."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.selected_message_id = "m1"
        await pilot.pause()
        transcript.focus()
        await pilot.press("s")
        await pilot.press("l", "l")
        await pilot.press("enter")
        await pilot.pause()

        menu = app.screen.query_one(ConsoleSelectionMenu)
        assert not menu.query_one("#console-selection-comment").disabled
        assert menu.query_one("#console-selection-request-changes").disabled
        assert menu.query_one("#console-selection-lgm").disabled
        # Mode is over; the highlight and manager state survive for the menu.
        assert transcript._kb_selection_row is None
        assert transcript.selection_manager.state.selection is not None
        hint = transcript.query_one("#console-kb-selection-hint")
        assert hint.display is False


@pytest.mark.asyncio
async def test_keyboard_finish_drains_the_release_click_token():
    """finish_drag() arms a one-shot release-click suppression token for the
    mouse path; keyboard has no release Click to consume it, and a stale
    token would eat the NEXT genuine row click's selection toggle."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.selected_message_id = "m1"
        await pilot.pause()
        transcript.focus()
        await pilot.press("s")
        await pilot.press("l")
        await pilot.press("enter")
        await pilot.pause()

        assert transcript.selection_manager.consume_release_click() is False
        assert transcript.selection_manager.just_finished is False


@pytest.mark.asyncio
async def test_menu_anchor_derives_from_row_region_and_stays_in_transcript():
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.selected_message_id = "m1"
        await pilot.pause()
        transcript.focus()
        row = transcript.query_one("#console-message-m1")
        row_region = row.region
        await pilot.press("s")
        await pilot.press("enter")
        await pilot.pause()

        menu = app.screen.query_one(ConsoleSelectionMenu)
        anchor_x, anchor_y = menu._anchor
        bounds = transcript.region
        assert bounds.x <= anchor_x <= bounds.right
        assert bounds.y <= anchor_y <= bounds.bottom
        # The menu can hop entirely above the row because the keyboard path
        # handed it the row's top, exactly like the mouse path does.
        assert menu._selection_top == row_region.y


@pytest.mark.asyncio
async def test_enter_with_no_active_selection_row_opens_nothing():
    """Enter outside the mode keeps its old meaning (message-selection
    toggle) -- pinned so the mode branch's Enter never leaks out of mode."""
    app = _KeyboardSelectionApp()
    async with app.run_test(size=(100, 40)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.selected_message_id = "m1"
        await pilot.pause()
        transcript.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not app.screen.query(ConsoleSelectionMenu)
