"""Keyboard text-selection mode state machine (console selection phase 5).

``s`` arms keyboard-driven text selection on the currently j/k-selected
message row -- a keyboard-only entry into the same ``SelectionManager`` the
mouse drag already drives (phase 1). This is the mode's SKELETON only:
enter/exit, Escape layering ahead of the existing clear-selection binding,
mouse takeover, and the row-destruction guard. Entry seeds a one-character
selection at the row's start; Task 3 wires the h/l/w/b/0/$/j/k motion keys
that move it, and Task 4 wires the copy/quote actions.
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
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


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
@pytest.mark.parametrize("key", ["j", "k", "down", "up", "enter"])
async def test_message_nav_and_confirm_keys_are_no_ops_in_mode(key):
    """Task 3 owns real motions; until then these keys must not desync the mode.

    Before this fix, only `escape` was intercepted by the mode branch, so
    `j`/`k`/`down`/`up`/`enter` fell through to the pre-existing BINDINGS
    chain: `j`/`k`/`down`/`up` moved `selected_message_id` to a different
    message while `_kb_selection_row` (and the manager state, and the hint)
    stayed pinned to the OLD row -- a silent mode/message-selection desync.
    `enter` toggled message selection (and would open the selection menu
    once one exists), which must not fire while keyboard text-selection
    mode owns the keyboard.
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
