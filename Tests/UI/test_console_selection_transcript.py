"""Mouse drag selection wiring on the native Console transcript (phase 1).

``ConsoleTranscript`` gains the ``selection_manager`` plus MouseDown/Move/Up
plumbing: a left-button press on a plain row arms a drag, moves map screen
cells to body-text character offsets (wrap-aware over the body Static's
content lines), and the release finishes the drag -- posting
``ConsoleTranscript.TranscriptTextSelected`` for menu-worthy (non-empty)
selections. A drag release must not toggle message selection (click
suppression), a genuine plain click still must, protected controls never
arm a drag, and reconciliation cancels state whose row was removed/rebuilt.
"""

import pytest
from textual.app import App, ComposeResult
from textual.events import Click, MouseDown, MouseMove, MouseUp
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_selection import TextSelection
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
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
    async with app.run_test(size=(40, 30)) as pilot:
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
        # Menu anchoring uses the release cell (screen coordinates).
        assert event.screen_x == body.region.x + 11
        assert event.screen_y == body.region.y


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
        await _drag_over_body(pilot, row, start_x=3, end_x=11)

        # A real drag that stays inside one row makes the App synthesize a
        # Click for the release (same widget under down and up). That click
        # completed a text-selection drag and must not select the message.
        row.post_message(
            _mouse_event(Click, row, screen_x=body.region.x + 11, screen_y=body.region.y)
        )
        await pilot.pause()
        assert transcript.selected_message_id is None

        # The next GENUINE click cycle still works: its button press arms an
        # empty (no-movement) drag whose finish consumes the suppression
        # flag, so its Click toggles the message selection again.
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
async def test_markdown_rows_do_not_start_selection():
    app = _SelectionTranscriptApp()
    async with app.run_test(size=(40, 30)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        await _mounted_row(pilot, "m1")
        markdown_row = app.query_one("#console-message-m3")

        markdown_row.post_message(
            _mouse_event(MouseDown, markdown_row, screen_x=1, screen_y=1)
        )
        await pilot.pause()

        assert transcript.selection_manager.state.active is False


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

        # The manager no longer suppresses subsequent row clicks: a real
        # click is Down+Up+Click; the empty finish consumes the flag so the
        # Click toggles message selection again.
        await pilot.click("#console-message-m1")
        await pilot.pause()
        assert transcript.selected_message_id == "m1"


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
