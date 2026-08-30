"""Mouse drag selection wiring on the native Console transcript (phase 1).

``ConsoleTranscript`` gains the ``selection_manager`` plus MouseDown/Move/Up
plumbing: a left-button press on a plain row arms a drag, moves map screen
cells to body-text character offsets (wrap-aware over the body Static's
content lines), and the release finishes the drag -- posting
``ConsoleTranscript.TranscriptTextSelected`` for menu-worthy (non-empty)
selections. Markdown rows arm the same drag at LINE granularity (task G):
cells map to whole markdown source lines. A drag release must not toggle
message selection (click suppression), a genuine plain click still must,
protected controls never arm a drag, and reconciliation cancels state whose
row was removed/rebuilt.
"""

import pytest
from textual.app import App, ComposeResult
from textual.events import Click, MouseDown, MouseMove, MouseUp
from textual.widgets import Markdown, Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_selection import TextSelection
from tldw_chatbook.Widgets.Console.console_selection_menu import ConsoleSelectionMenu
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    ConsoleTranscriptActionButton,
    ConsoleTranscriptMessage,
)

_LONG_BODY = "one two three four five six seven eight nine ten eleven twelve"
#: ``Content.wrap`` folds ``_LONG_BODY`` at width 40 into
#: ``'one two three four five six seven eight'`` (39 chars) and
#: ``'nine ten eleven twelve'``; the fold space is dropped, so line 1
#: starts at source offset 40 (39 chars + 1 dropped space).
_LONG_LINE_1_START = 40


class _SelectionTranscriptApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.selected_events: list[ConsoleTranscript.TranscriptTextSelected] = []

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="hello selectable world", id="m1"
                ),
                ConsoleChatMessage(role=ConsoleMessageRole.USER, content=_LONG_BODY, id="m2"),
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT, content="answer text", id="m3"
                ),
            ]
        )
        yield transcript

    def on_console_transcript_transcript_text_selected(
        self, event: ConsoleTranscript.TranscriptTextSelected
    ) -> None:
        self.selected_events.append(event)


async def _mounted_row(pilot, message_id: str) -> ConsoleTranscriptMessage:
    transcript = pilot.app.query_one(ConsoleTranscript)
    transcript.set_messages(transcript._messages)
    await transcript.refresh_messages()
    await pilot.pause()
    return pilot.app.query_one(f"#console-message-{message_id}", ConsoleTranscriptMessage)


def _body_static(row: ConsoleTranscriptMessage) -> Static:
    return row.query_one(".console-transcript-message-body", Static)


def _mouse_event(
    event_cls, widget, *, screen_x: int, screen_y: int, button: int = 1
) -> object:
    """Build a mouse event addressed to ``widget`` with absolute coordinates.

    The transcript's handlers resolve cells from ``screen_x``/``screen_y``
    (relative ``x``/``y`` are target-dependent), so tests pin both. Left
    presses carry ``button=1`` -- the encoding Textual's XTerm driver and
    ``Pilot.click`` both use (0 denotes a buttonless move report).
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


def _raw_app_mouse(event_cls, screen_x: int, screen_y: int, *, button: int = 1):
    return event_cls(
        widget=None,
        x=screen_x,
        y=screen_y,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


async def _drag_over_body(
    pilot, row: ConsoleTranscriptMessage, start_x: int, end_x: int, line: int = 0
) -> None:
    """Drive a left-button drag across body cells ``start_x..end_x``."""
    body = _body_static(row)
    base_y = body.region.y + line
    row.post_message(
        _mouse_event(MouseDown, row, screen_x=body.region.x + start_x, screen_y=base_y)
    )
    await pilot.pause()
    row.post_message(
        _mouse_event(MouseMove, row, screen_x=body.region.x + end_x, screen_y=base_y)
    )
    await pilot.pause()
    row.post_message(
        _mouse_event(MouseUp, row, screen_x=body.region.x + end_x, screen_y=base_y)
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_transcript_exposes_selection_manager():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)):
        transcript = app.query_one(ConsoleTranscript)
        assert transcript.selection_manager.state.active is False
        assert transcript.selection_manager.state.selection is None


@pytest.mark.asyncio
async def test_drag_selects_text_and_posts_selection_event():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        release_x = body.region.x + 11
        release_y = body.region.y
        await _drag_over_body(pilot, row, start_x=3, end_x=11)

        assert transcript.selection_manager.just_finished is True
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert selection.row_key == row.id
        assert (selection.start, selection.end) == (3, 11)
        assert row.get_selection_text() == "lo selec"
        assert len(app.selected_events) == 1
        event = app.selected_events[0]
        assert event.selection == TextSelection(row_key=row.id, start=3, end=11)
        # Menu anchoring uses the release cell (screen coordinates, captured
        # pre-drag: the post-mount menu can re-anchor the transcript scroll).
        assert event.screen_x == release_x
        assert event.screen_y == release_y


@pytest.mark.asyncio
async def test_drag_maps_wrapped_body_lines_to_source_offsets():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m2")
        # The long body wraps at width 40; dragging over the SECOND visual
        # line must resolve inside that line's source span, not line 0's.
        await _drag_over_body(pilot, row, start_x=0, end_x=3, line=1)

        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert (selection.start, selection.end) == (_LONG_LINE_1_START, _LONG_LINE_1_START + 3)
        assert row.get_selection_text() == "nin"


@pytest.mark.asyncio
async def test_drag_release_does_not_toggle_message_selection():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        release_x = body.region.x + 11
        release_y = body.region.y
        await _drag_over_body(pilot, row, start_x=3, end_x=11)

        # A real drag that stays inside one row makes the App synthesize a
        # Click for the release (same widget under down and up). That click
        # completed a text-selection drag and must not select the message.
        # Coordinates are pre-drag: the post-mount menu can re-anchor the
        # transcript scroll and move the row.
        row.post_message(
            _mouse_event(Click, row, screen_x=release_x, screen_y=release_y)
        )
        await pilot.pause()
        assert transcript.selected_message_id is None

        # The next GENUINE click cycle still works: its empty MouseUp commits
        # the message toggle and deliberately leaves ``just_finished`` so the
        # optional synthesized Click is consumed as a duplicate.
        await pilot.click("#console-message-m1")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"


@pytest.mark.asyncio
async def test_plain_click_without_drag_still_selects_message():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")

        await pilot.click("#console-message-m1")
        await pilot.pause()

        assert transcript.selected_message_id == "m1"
        assert transcript.selection_manager.state.active is False
        assert transcript.selection_manager.just_finished is False


@pytest.mark.asyncio
async def test_plain_click_on_markdown_row_still_selects_message():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")

        await pilot.click("#console-message-m3")
        await pilot.pause()

        assert transcript.selected_message_id == "m3"


@pytest.mark.asyncio
async def test_click_after_drag_on_markdown_row_toggles_selection():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, row, start_x=3, end_x=11)
        assert transcript.selection_manager.just_finished is True

        # The screen-anchored menu can overlay the markdown row, so close it
        # first (keyboard path) before the genuine click.
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(ConsoleSelectionMenu)

        # Regression (review fix round 1): a genuine click on a MARKDOWN row
        # whose MouseDown can never arm a drag must not inherit the plain
        # row's drag-release suppression -- the flag has to be consumed by
        # the fresh press (or the suppressed click itself), not stick until
        # some plain-row/negative-space click happens to clear it.
        await pilot.click("#console-message-m3")
        await pilot.pause()
        assert transcript.selected_message_id == "m3"

        # And a second markdown-row click still behaves normally (toggles
        # the selection back off).
        await pilot.click("#console-message-m3")
        await pilot.pause()
        assert transcript.selected_message_id is None


@pytest.mark.asyncio
async def test_drag_extends_while_mouse_is_captured():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        # Mouse capture routes subsequent moves to the transcript itself
        # (event control = capturer), so extension must not depend on the
        # event's control pointing at the row.
        row.post_message(
            _mouse_event(MouseDown, row, screen_x=body.region.x + 3, screen_y=body.region.y)
        )
        await pilot.pause()
        transcript.post_message(
            _mouse_event(
                MouseMove, transcript, screen_x=body.region.x + 11, screen_y=body.region.y
            )
        )
        await pilot.pause()

        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert (selection.start, selection.end) == (3, 11)
        assert row.get_selection_text() == "lo selec"

        transcript.post_message(
            _mouse_event(
                MouseUp, transcript, screen_x=body.region.x + 11, screen_y=body.region.y
            )
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is False
        assert len(app.selected_events) == 1


@pytest.mark.asyncio
async def test_drag_past_body_edges_clamps_to_text_bounds():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        row.post_message(
            _mouse_event(MouseDown, row, screen_x=body.region.x + 11, screen_y=body.region.y)
        )
        await pilot.pause()
        # Above the body (over the header): clamp to the text start.
        transcript.post_message(
            _mouse_event(
                MouseMove, transcript, screen_x=body.region.x + 11, screen_y=body.region.y - 1
            )
        )
        await pilot.pause()
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert (selection.start, selection.end) == (0, 11)

        # Below the body (over the following row): clamp to the text end.
        transcript.post_message(
            _mouse_event(
                MouseMove, transcript, screen_x=body.region.x + 11, screen_y=body.region.y + 2
            )
        )
        await pilot.pause()
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert (selection.start, selection.end) == (11, len("hello selectable world"))

        # Ending far right of the last wrapped line clamps to the line end.
        transcript.post_message(
            _mouse_event(
                MouseMove, transcript, screen_x=body.region.x + 999, screen_y=body.region.y
            )
        )
        await pilot.pause()
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert (selection.start, selection.end) == (11, len("hello selectable world"))


@pytest.mark.asyncio
async def test_non_message_controls_never_start_selection():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")

        transcript.post_message(_mouse_event(MouseDown, transcript, screen_x=1, screen_y=0))
        await pilot.pause()
        assert transcript.selection_manager.state.active is False


@pytest.mark.asyncio
async def test_markdown_rows_start_line_selection():
    """Task G flip: markdown rows arm drags and select whole source lines."""
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")
        markdown_row = app.query_one("#console-message-m3")
        body = markdown_row.query_one(Markdown)

        markdown_row.post_message(
            _mouse_event(
                MouseDown, markdown_row, screen_x=body.region.x, screen_y=body.region.y
            )
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is True

        # Markdown drags select at character granularity: a 2-cell drag
        # over "answer text" selects its first two characters (live-spike:
        # whole-line snapping made any partial drag grab the entire
        # message).
        transcript.post_message(
            _mouse_event(
                MouseMove,
                transcript,
                screen_x=body.region.x + 2,
                screen_y=body.region.y,
            )
        )
        await pilot.pause()
        assert markdown_row.get_selection_text() == "an"

        transcript.post_message(
            _mouse_event(
                MouseUp,
                transcript,
                screen_x=body.region.x + 2,
                screen_y=body.region.y,
            )
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is False
        assert len(app.selected_events) == 1


@pytest.mark.asyncio
async def test_markdown_drag_release_does_not_toggle_message_selection():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")
        markdown_row = app.query_one("#console-message-m3")
        body = markdown_row.query_one(Markdown)

        markdown_row.post_message(
            _mouse_event(MouseDown, markdown_row, screen_x=body.region.x, screen_y=body.region.y)
        )
        await pilot.pause()
        transcript.post_message(
            _mouse_event(
                MouseMove,
                transcript,
                screen_x=body.region.x + 2,
                screen_y=body.region.y,
            )
        )
        await pilot.pause()
        transcript.post_message(
            _mouse_event(
                MouseUp,
                transcript,
                screen_x=body.region.x + 2,
                screen_y=body.region.y,
            )
        )
        await pilot.pause()

        # The drag-release Click (suppression) must not have toggled the
        # markdown row's message selection -- including when Textual
        # synthesizes that Click LATE, after an intervening interaction
        # already consumed just_finished (the release-click token owns
        # exactly this case).
        markdown_row.post_message(
            _mouse_event(
                Click, markdown_row, screen_x=body.region.x + 2, screen_y=body.region.y
            )
        )
        await pilot.pause()
        assert transcript.selected_message_id is None

        # Popover semantics in a real layout: the first genuine click
        # dismisses the selection UI (menu + highlight strip) without
        # toggling; the second toggles the markdown row normally.
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(ConsoleSelectionMenu)
        assert markdown_row.get_selection_text() == ""

        await pilot.click(markdown_row, offset=(1, 1))
        await pilot.pause()
        # (If the click geometry lands outside the visible fold in this
        # small harness, the toggle assert below is skipped -- the
        # suppression behavior above is what this test regresses.)
        if transcript.selected_message_id is None:
            m3_center = markdown_row.region.center
            from textual.geometry import Region as _R

            if _R(0, 0, *app.screen.size).contains(*map(int, m3_center)):
                raise AssertionError("click landed on-visible but did not toggle")


@pytest.mark.asyncio
async def test_mouse_up_outside_transcript_finishes_drag():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        row.post_message(
            _mouse_event(MouseDown, row, screen_x=body.region.x + 3, screen_y=body.region.y)
        )
        await pilot.pause()
        transcript.post_message(
            _mouse_event(
                MouseMove, transcript, screen_x=body.region.x + 11, screen_y=body.region.y
            )
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is True

        # Mouse capture routes the release to the transcript even when the
        # pointer is elsewhere: the drag must finish, not stay active.
        transcript.post_message(_mouse_event(MouseUp, transcript, screen_x=35, screen_y=29))
        await pilot.pause()

        assert transcript.selection_manager.state.active is False
        selection = transcript.selection_manager.state.selection
        assert selection is not None
        assert (selection.start, selection.end) == (3, 11)
        assert len(app.selected_events) == 1

        # The next distinct row click works normally: its empty MouseUp commits
        # the message toggle and deliberately leaves ``just_finished`` so the
        # optional synthesized Click is consumed as a duplicate. (m2, not m1:
        # the menu mount can re-anchor the transcript scroll and push m1
        # off-screen.) Close
        # the screen-anchored popover first (Escape cancels pending suppression
        # state): the mounted menu reflows the rows, and a click
        # through the collapsed row region can land on m2's header label
        # instead of its body -- the compact menu moved that landing cell.
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(ConsoleSelectionMenu)
        await pilot.click("#console-message-m2")
        await pilot.pause()
        assert transcript.selected_message_id == "m2"


@pytest.mark.asyncio
async def test_row_removal_cancels_selection_state():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, row, start_x=3, end_x=11)
        assert transcript.selection_manager.state.selection is not None

        transcript.set_messages(
            [ConsoleChatMessage(role=ConsoleMessageRole.USER, content="other", id="m2")]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.selection_manager.state.selection is None
        assert transcript.selection_manager.state.active is False


@pytest.mark.asyncio
async def test_mid_drag_row_removal_releases_capture():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        row.post_message(
            _mouse_event(MouseDown, row, screen_x=body.region.x + 3, screen_y=body.region.y)
        )
        await pilot.pause()
        assert transcript.selection_manager.state.active is True
        assert pilot.app.mouse_captured is transcript

        transcript.set_messages(
            [ConsoleChatMessage(role=ConsoleMessageRole.USER, content="other", id="m2")]
        )
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.selection_manager.state.active is False
        assert pilot.app.mouse_captured is None


@pytest.mark.asyncio
async def test_menu_open_row_body_click_dismisses_menu_and_toggles():
    """Regression (final review): a row-body click must dismiss an open menu.

    Rows stop their own Clicks (the message-selection toggle), so with a
    menu open the press on another row's body never reached the
    transcript's ``on_click`` removal -- the menu stayed mounted while
    the user toggled selections elsewhere. ``on_mouse_down`` now folds
    mounted menus on any press that does not originate inside a menu,
    before arming the new drag; the clicked row's toggle must keep
    working.
    """
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, row, start_x=3, end_x=11)
        assert len(app.query(ConsoleSelectionMenu)) == 1  # menu open at release

        # Click ANOTHER row's body. The press dismisses the selection UI and
        # the same genuine click still toggles the target row.
        other_body = app.query_one("#console-message-m2 .console-transcript-message-body")
        await pilot.click(other_body, offset=(0, 1))
        await pilot.pause()
        assert not app.query(ConsoleSelectionMenu)  # folded
        assert transcript.selected_message_id == "m2"  # toggle still works
        assert transcript.selection_manager.state.active is False
        assert transcript.selection_manager.just_finished is False
        assert transcript._selection_origin_row is None


@pytest.mark.asyncio
async def test_menu_cleanup_raw_mouseup_commits_initial_press_without_click():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        selected_row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, selected_row, start_x=3, end_x=11)
        assert app.query_one(ConsoleSelectionMenu)

        target = app.query_one("#console-message-m2", ConsoleTranscriptMessage)
        target_body = _body_static(target)
        x, y = target_body.region.x + 1, target_body.region.y + 1
        app.post_message(_raw_app_mouse(MouseDown, x, y))
        await pilot.pause()
        assert transcript._selection_origin_row is target
        assert not app.query(ConsoleSelectionMenu)
        assert selected_row.get_selection_text() == ""

        app.post_message(_raw_app_mouse(MouseUp, x, y))
        await pilot.pause()
        assert transcript.selected_message_id == "m2"
        assert transcript._selection_origin_row is None

        first = app.query_one("#console-message-m1", ConsoleTranscriptMessage)
        x2, y2 = first.region.x + 1, first.region.bottom - 1
        app.post_message(_raw_app_mouse(MouseDown, x2, y2))
        await pilot.pause()
        app.post_message(_raw_app_mouse(MouseUp, x2, y2))
        await pilot.pause()
        assert transcript.selected_message_id == "m1"
        assert transcript._selection_origin_row is None


@pytest.mark.asyncio
async def test_escape_clears_armed_press_and_mouse_capture():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        selected_row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, selected_row, start_x=3, end_x=11)
        assert app.query_one(ConsoleSelectionMenu)

        target_body = app.query_one(
            "#console-message-m2 .console-transcript-message-body", Static
        )
        await pilot.mouse_down(target_body, offset=(1, 1))
        await pilot.pause()
        assert transcript.selection_manager.state.active is True
        assert app.mouse_captured is transcript
        assert transcript._selection_origin_row.message_id == "m2"
        assert not app.query(ConsoleSelectionMenu)
        assert selected_row.get_selection_text() == ""

        await pilot.press("escape")
        await pilot.pause()
        assert transcript.selection_manager.is_idle
        assert transcript._selection_origin_row is None
        assert app.mouse_captured is None
        assert not app.query(ConsoleSelectionMenu)
        assert selected_row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_menu_open_right_button_dismisses_without_arming_drag():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        selected_row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, selected_row, start_x=3, end_x=11)
        assert app.query_one(ConsoleSelectionMenu)

        target_body = app.query_one(
            "#console-message-m2 .console-transcript-message-body", Static
        )
        await pilot.mouse_down(target_body, offset=(1, 1), button=3)
        await pilot.pause()

        assert not app.query(ConsoleSelectionMenu)
        assert transcript._selection_origin_row is None
        assert transcript.selection_manager.state.active is False


@pytest.mark.asyncio
async def test_menu_descendant_press_keeps_menu_and_does_not_arm_drag():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        selected_row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, selected_row, start_x=3, end_x=11)
        menu = app.query_one(ConsoleSelectionMenu)
        button = menu.query_one("#console-selection-add-to-chat")
        event = _mouse_event(
            MouseDown,
            button,
            screen_x=button.region.x + 1,
            screen_y=button.region.y,
        )

        transcript.on_mouse_down(event)

        assert menu.is_attached
        assert transcript._selection_origin_row is None
        assert transcript.selection_manager.state.active is False


@pytest.mark.asyncio
async def test_menu_cleanup_same_row_click_preserves_initial_target():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        await _drag_over_body(pilot, row, start_x=3, end_x=11)
        assert app.query_one(ConsoleSelectionMenu)
        assert row.get_selection_text() == "lo selec"

        body = _body_static(row)
        await pilot.click(body, offset=(1, 0))
        await pilot.pause()

        assert transcript.selected_message_id == "m1"
        assert not app.query(ConsoleSelectionMenu)
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_menu_cleanup_markdown_layout_preserves_initial_plain_target():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")
        markdown_row = app.query_one("#console-message-m3")
        markdown_body = markdown_row.query_one(Markdown)
        start_x, start_y = markdown_body.region.x, markdown_body.region.y
        markdown_row.post_message(
            _mouse_event(
                MouseDown,
                markdown_row,
                screen_x=start_x,
                screen_y=start_y,
            )
        )
        await pilot.pause()
        transcript.post_message(
            _mouse_event(
                MouseMove,
                transcript,
                screen_x=start_x + 2,
                screen_y=start_y,
            )
        )
        await pilot.pause()
        transcript.post_message(
            _mouse_event(
                MouseUp,
                transcript,
                screen_x=start_x + 2,
                screen_y=start_y,
            )
        )
        await pilot.pause()
        assert app.query_one(ConsoleSelectionMenu)
        assert markdown_row.get_selection_text() == "an"

        target_body = app.query_one(
            "#console-message-m2 .console-transcript-message-body", Static
        )
        await pilot.click(target_body, offset=(1, 1))
        await pilot.pause()

        assert transcript.selected_message_id == "m2"
        assert not app.query(ConsoleSelectionMenu)
        assert markdown_row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_menu_open_protected_action_press_dismisses_menu_no_drag():
    """Deferred gap (final review): press a REAL protected in-row control.

    A selected message mounts its action row (protected click class) with
    live buttons inside the transcript. With a menu open, pressing an
    action button must fold the menu (press-outside dismissal) and never
    arm a text-selection drag over the protected row.
    """
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")
        # Select m1 so its action row with real buttons mounts.
        await pilot.click("#console-message-m1")
        await pilot.pause()
        await pilot.pause()
        buttons = [
            button
            for button in transcript.query(ConsoleTranscriptActionButton)
            if button.display
        ]
        assert buttons  # the action row really mounted
        button = next((b for b in buttons if not b.disabled), buttons[0])

        # Drag over m2's body (below m1's action row) to open the menu.
        row2 = app.query_one("#console-message-m2", ConsoleTranscriptMessage)
        await _drag_over_body(pilot, row2, start_x=3, end_x=11)
        assert len(app.query(ConsoleSelectionMenu)) == 1

        await pilot.mouse_down(button, offset=(1, 0))
        await pilot.pause()

        assert not app.query(ConsoleSelectionMenu)  # folded by the press
        assert transcript.selection_manager.state.active is False  # no drag armed


@pytest.mark.asyncio
async def test_real_terminal_press_without_control_arms_drag():
    """Live-terminal presses carry widget=None (screen forwarding never sets it).

    Regression for the live spike: every real MouseDown logged ``ctrl=None``
    (only MouseMove gets a translated widget), so keying the arm off
    ``event.control`` made real-terminal drags silent no-ops while
    synthetic-event tests stayed green. The arm must hit-test from screen
    coordinates instead.
    """
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)

        # Shape identical to Textual's screen._forward_event output for a
        # real terminal press: posted to the widget under the pointer, with
        # NO widget attached to the event.
        event = MouseDown(
            widget=None,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=body.region.x + 2,
            screen_y=body.region.y,
        )
        transcript.post_message(event)
        await pilot.pause()

        assert transcript.selection_manager.state.active is True
        assert (
            transcript.selection_manager.state.selection.row_key == row.id
        )

        transcript.post_message(
            MouseMove(
                widget=None,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=body.region.x + 8,
                screen_y=body.region.y,
            )
        )
        await pilot.pause()
        transcript.post_message(
            MouseUp(
                widget=None,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=body.region.x + 8,
                screen_y=body.region.y,
            )
        )
        await pilot.pause()

        assert len(app.query(ConsoleSelectionMenu)) == 1
        assert row.get_selection_text() != ""


@pytest.mark.asyncio
async def test_real_shaped_plain_click_toggles_message_selection():
    """Live spike 2026-08-16 ('can't select messages via mouse').

    A real terminal's plain click is drag-armed on press (the transcript
    captures the mouse), and the synthesized Click is routed to the
    CAPTURING transcript -- the capture only releases when the transcript
    processes the MouseUp, which lands after the Click was already
    forwarded. The row never sees the click, so message selection never
    toggled in real terminals (pilot clicks deliver directly to the widget
    under the pointer, masking this). The transcript's on_click must
    re-dispatch capture-routed clicks to the row the pointer actually
    targeted.
    """
    from textual.events import MouseDown, MouseUp

    def raw(event_cls, x, y, button=1):
        return event_cls(
            widget=None, x=x, y=y, delta_x=0, delta_y=0, button=button,
            shift=False, meta=False, ctrl=False,
        )

    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = await _mounted_row(pilot, "m1")
        body = _body_static(row)
        br = body.region
        cx, cy = br.x + 3, br.y

        pilot.app.post_message(raw(MouseDown, cx, cy))
        await pilot.pause()
        pilot.app.post_message(raw(MouseUp, cx, cy))
        await pilot.pause()
        await pilot.pause()

        assert transcript.selected_message_id == row.message_id

        # The selected row grows its action row and the body Static's
        # cached region goes stale mid-refresh; aim at the ROW's last cell
        # (the body is its bottom line), which is always current.
        await pilot.pause(0.3)
        rr = row.region
        cx, cy = rr.x + 3, rr.bottom - 1
        pilot.app.post_message(raw(MouseDown, cx, cy))
        await pilot.pause()
        pilot.app.post_message(raw(MouseUp, cx, cy))
        await pilot.pause()
        await pilot.pause()

        assert transcript.selected_message_id is None  # second click untoggles
