"""Floating selection menu and transcript wiring (console selection phase 1, task E).

The menu is mounted inside ``ConsoleTranscript`` on selection release, docked
out of the scroll flow and offset to the release cell (tall-transcript
regression), offers "Add to chat" which quotes the active selection up to the
owning screen, and dismisses on Escape / click-outside with no side effects.
Clicks inside the menu stop there so the transcript never treats them as
click-outside.
"""

import pytest
from textual.app import App, ComposeResult
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import Markdown

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_selection import (
    SELECTION_QUOTE_CAP,
    TextSelection,
    cap_quote,
)
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionMenu,
    ConsoleSelectionQuoteRequested,
    ConsoleSideChatRequested,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


class _MenuApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.add_to_chat_events: list[ConsoleSelectionMenu.AddToChat] = []
        self.more_details_events: list[ConsoleSelectionMenu.MoreDetails] = []
        self.ask_side_chat_events: list[ConsoleSelectionMenu.AskInSideChat] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSelectionMenu(screen_x=4, screen_y=6)

    def on_console_selection_menu_add_to_chat(
        self, event: ConsoleSelectionMenu.AddToChat
    ) -> None:
        self.add_to_chat_events.append(event)

    def on_console_selection_menu_more_details(
        self, event: ConsoleSelectionMenu.MoreDetails
    ) -> None:
        self.more_details_events.append(event)

    def on_console_selection_menu_ask_in_side_chat(
        self, event: ConsoleSelectionMenu.AskInSideChat
    ) -> None:
        self.ask_side_chat_events.append(event)


@pytest.mark.asyncio
async def test_menu_offers_add_to_chat_and_posts_message():
    app = _MenuApp()
    async with app.run_test() as pilot:
        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()
        assert len(app.add_to_chat_events) == 1


@pytest.mark.asyncio
async def test_menu_offers_three_stacked_options_in_order():
    """Phase 2: Add to chat, More Details, Ask in Side Chat stack in order."""
    app = _MenuApp()
    async with app.run_test() as pilot:
        del pilot
        buttons = app.query_one(ConsoleSelectionMenu).query("Button")
        assert [button.id for button in buttons] == [
            "console-selection-add-to-chat",
            "console-selection-more-details",
            "console-selection-ask-side-chat",
        ]
        assert [str(button.label) for button in buttons] == [
            "Add to chat",
            "More Details",
            "Ask in Side Chat",
        ]


@pytest.mark.asyncio
async def test_more_details_button_posts_more_details_message():
    app = _MenuApp()
    async with app.run_test() as pilot:
        await pilot.click("#console-selection-more-details")
        await pilot.pause()
        assert len(app.more_details_events) == 1
        assert app.add_to_chat_events == []
        assert app.ask_side_chat_events == []


@pytest.mark.asyncio
async def test_ask_side_chat_button_posts_ask_message():
    app = _MenuApp()
    async with app.run_test() as pilot:
        await pilot.click("#console-selection-ask-side-chat")
        await pilot.pause()
        assert len(app.ask_side_chat_events) == 1
        assert app.add_to_chat_events == []
        assert app.more_details_events == []


@pytest.mark.asyncio
async def test_escape_dismisses_without_side_effects():
    app = _MenuApp()
    async with app.run_test() as pilot:
        assert app.query_one(ConsoleSelectionMenu)  # mounted
        await pilot.press("escape")
        assert not app.query(ConsoleSelectionMenu)  # removed


@pytest.mark.asyncio
async def test_click_inside_menu_does_not_propagate():
    """Clicks inside the menu stop at the menu: no transcript row toggling etc."""
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        # Click lands inside the menu's Add-to-chat button; the transcript's
        # click-outside removal must not run before the button activates.
        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()
        assert len(app.quote_requests) == 1


class _TranscriptMenuApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.quote_requests: list[ConsoleSelectionQuoteRequested] = []
        self.side_chat_requests: list[ConsoleSideChatRequested] = []

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="hello selectable world", id="m1"
                )
            ]
        )
        yield transcript

    def on_console_selection_quote_requested(
        self, event: ConsoleSelectionQuoteRequested
    ) -> None:
        self.quote_requests.append(event)

    def on_console_side_chat_requested(
        self, event: ConsoleSideChatRequested
    ) -> None:
        self.side_chat_requests.append(event)


async def _finish_drag_selection(pilot) -> None:
    transcript = pilot.app.query_one(ConsoleTranscript)
    row = pilot.app.query_one("#console-message-m1", ConsoleTranscriptMessage)
    transcript.selection_manager.begin_drag(row.id, 0)
    transcript.selection_manager.extend_drag(row.id, 5)
    row.set_selection_range(0, 5)
    transcript.selection_manager.finish_drag()
    transcript.post_message(
        ConsoleTranscript.TranscriptTextSelected(selection=TextSelection(row.id, 0, 5), screen_x=4, screen_y=6)
    )
    await pilot.pause()


async def _real_drag(pilot, selector: str) -> None:
    """Perform a real pilot mouse drag over ``selector`` (press, move, release).

    Unlike ``_finish_drag_selection`` this exercises the full mouse path
    (including the release Click pilot synthesizes), so it reproduces the
    message ordering a live terminal produces.
    """
    await pilot.mouse_down(selector, offset=(0, 0))
    await pilot.hover(selector, offset=(5, 0))
    await pilot.mouse_up(selector, offset=(5, 0))
    await pilot.pause()


@pytest.mark.asyncio
async def test_consecutive_selections_remount_exactly_one_menu():
    """Regression: remounting over a still-pruning menu must not hit DuplicateIds.

    ``Widget.remove()`` only SCHEDULES removal; a synchronous same-id remount
    before the prune completes raises Textual's DuplicateIds (app-fatal), so a
    second selection right after the first used to crash the app.
    """
    app = _TranscriptMenuApp()
    body = "#console-message-m1 .console-transcript-message-body"
    async with app.run_test() as pilot:
        await _real_drag(pilot, body)
        assert len(app.query(ConsoleSelectionMenu)) == 1
        await _real_drag(pilot, body)
        assert len(app.query(ConsoleSelectionMenu)) == 1
        assert app.is_running  # no app-fatal DuplicateIds


@pytest.mark.asyncio
async def test_escape_dismisses_menu_in_transcript_context():
    """Escape must dismiss the menu even inside the transcript.

    The transcript's own ``on_key`` intercepts escape (clear-selection +
    stop) during bubbling, so the menu's BINDING never fires; the menu must
    handle escape in its own ``on_key`` (the focused widget's handler runs
    first in the bubble chain).
    """
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        assert app.query_one(ConsoleSelectionMenu)
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(ConsoleSelectionMenu)



@pytest.mark.asyncio
async def test_transcript_mounts_menu_on_selection_release():
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        assert app.query_one(ConsoleSelectionMenu)


@pytest.mark.asyncio
async def test_add_to_chat_quotes_selection_and_cleans_up():
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()
        assert len(app.quote_requests) == 1
        assert app.quote_requests[0].quote == "hello"
        assert not app.query(ConsoleSelectionMenu)
        transcript = app.query_one(ConsoleTranscript)
        assert transcript.selection_manager.state.selection is None
        assert app.query_one("#console-message-m1", ConsoleTranscriptMessage).get_selection_text() == ""


@pytest.mark.asyncio
async def test_click_outside_removes_menu():
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        assert app.query_one(ConsoleSelectionMenu)
        await pilot.click(ConsoleTranscript)
        await pilot.pause()
        assert not app.query(ConsoleSelectionMenu)


class _LongTranscriptMenuApp(_TranscriptMenuApp):
    """Transcript whose single row dwarfs the selection quote cap."""

    _LONG_CONTENT = "x" * (SELECTION_QUOTE_CAP + 1000)

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content=self._LONG_CONTENT, id="m1"
                )
            ]
        )
        yield transcript


async def _select_whole_row(pilot) -> None:
    """Drag-select the entire first row and mount the selection menu."""
    transcript = pilot.app.query_one(ConsoleTranscript)
    row = pilot.app.query_one("#console-message-m1", ConsoleTranscriptMessage)
    length = len(_LongTranscriptMenuApp._LONG_CONTENT)
    transcript.selection_manager.begin_drag(row.id, 0)
    transcript.selection_manager.extend_drag(row.id, length)
    row.set_selection_range(0, length)
    transcript.selection_manager.finish_drag()
    region = transcript.region
    transcript.post_message(
        ConsoleTranscript.TranscriptTextSelected(
            selection=TextSelection(row.id, 0, length),
            screen_x=region.x + 4,
            screen_y=region.y + 2,
        )
    )
    await pilot.pause()
    await pilot.pause()


def _assert_side_chat_cleanup(app: _TranscriptMenuApp) -> None:
    assert not app.query(ConsoleSelectionMenu)  # menu removed
    transcript = app.query_one(ConsoleTranscript)
    assert transcript.selection_manager.state.selection is None
    assert (
        app.query_one(
            "#console-message-m1", ConsoleTranscriptMessage
        ).get_selection_text()
        == ""
    )


@pytest.mark.asyncio
async def test_more_details_requests_side_chat_and_cleans_up():
    """More Details posts ConsoleSideChatRequested(mode="more-details")."""
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        await pilot.click("#console-selection-more-details")
        await pilot.pause()
        assert len(app.side_chat_requests) == 1
        assert app.side_chat_requests[0].mode == "more-details"
        assert app.side_chat_requests[0].quote == "hello"
        assert app.quote_requests == []
        _assert_side_chat_cleanup(app)


@pytest.mark.asyncio
async def test_ask_side_chat_requests_side_chat_and_cleans_up():
    """Ask in Side Chat posts ConsoleSideChatRequested(mode="ask")."""
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        await pilot.click("#console-selection-ask-side-chat")
        await pilot.pause()
        assert len(app.side_chat_requests) == 1
        assert app.side_chat_requests[0].mode == "ask"
        assert app.side_chat_requests[0].quote == "hello"
        assert app.quote_requests == []
        _assert_side_chat_cleanup(app)


@pytest.mark.asyncio
async def test_side_chat_quote_is_capped_more_details():
    """A row longer than the quote cap reaches the side chat capped."""
    app = _LongTranscriptMenuApp()
    async with app.run_test(size=(120, 32)) as pilot:
        await _select_whole_row(pilot)
        await pilot.click("#console-selection-more-details")
        await pilot.pause()
        assert len(app.side_chat_requests) == 1
        expected = cap_quote(_LongTranscriptMenuApp._LONG_CONTENT)
        assert app.side_chat_requests[0].quote == expected
        assert len(app.side_chat_requests[0].quote) <= SELECTION_QUOTE_CAP


@pytest.mark.asyncio
async def test_side_chat_quote_is_capped_ask():
    app = _LongTranscriptMenuApp()
    async with app.run_test(size=(120, 32)) as pilot:
        await _select_whole_row(pilot)
        await pilot.click("#console-selection-ask-side-chat")
        await pilot.pause()
        assert len(app.side_chat_requests) == 1
        assert app.side_chat_requests[0].quote == cap_quote(
            _LongTranscriptMenuApp._LONG_CONTENT
        )


class _TallTranscriptMenuApp(App[None]):
    CSS = """
    ConsoleTranscript {
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content=f"row {i} selectable text", id=f"m{i}"
                )
                for i in range(30)
            ]
        )
        yield transcript


@pytest.mark.asyncio
async def test_menu_anchors_at_release_cell_in_tall_transcript():
    """Regression: the menu overlays the transcript at the release cell.

    The menu is docked (out of scroll flow) and offset relative to the
    transcript, so it must appear near the release point even when the
    scroll content is much taller than the viewport, stay inside the
    transcript's viewport, and not consume a layout slot in the flow.
    """
    app = _TallTranscriptMenuApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-m0", ConsoleTranscriptMessage)
        assert transcript.virtual_size.height > 30  # content taller than viewport
        flow_height_before = transcript.virtual_size.height
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        region = transcript.region
        assert region.height <= 32
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=region.y + 6,
            )
        )
        await pilot.pause()
        menu = app.query_one(ConsoleSelectionMenu)
        # Anchored at the release cell (release_row + 1), within a few cells.
        assert abs(menu.region.x - (region.x + 4)) <= 2
        assert abs(menu.region.y - (region.y + 7)) <= 2
        # Inside the transcript viewport.
        assert region.x <= menu.region.x
        assert menu.region.right <= region.right
        assert region.y <= menu.region.y
        assert menu.region.bottom <= region.bottom
        # Docked out of flow: scroll content height unchanged after mount.
        assert transcript.virtual_size.height == flow_height_before


@pytest.mark.asyncio
async def test_far_right_release_keeps_menu_inside_transcript():
    """Regression: a release near the transcript's right edge must not overhang.

    The transcript pre-clamps the anchor with a small fixed margin, but only
    the menu knows its own extent (border + padding + button); without a
    post-layout clamp a far-right (or near-bottom) release anchored the menu
    with its Add-to-chat button overhanging the transcript -- unreachable.
    """
    app = _TallTranscriptMenuApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-m0", ConsoleTranscriptMessage)
        region = transcript.region
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.right - 2,
                screen_y=region.bottom - 3,
            )
        )
        await pilot.pause()
        await pilot.pause()  # let the post-layout clamp settle
        menu = app.query_one(ConsoleSelectionMenu)
        assert region.x <= menu.region.x
        assert menu.region.right <= region.right
        assert region.y <= menu.region.y
        assert menu.region.bottom <= region.bottom


class _MarkdownTranscriptMenuApp(App[None]):
    """Transcript whose only row is a markdown (ASSISTANT) row (task G)."""

    _MARKDOWN_SOURCE = "line one\nline two\nline three"

    def __init__(self) -> None:
        super().__init__()
        self.quote_requests: list[ConsoleSelectionQuoteRequested] = []

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content=self._MARKDOWN_SOURCE,
                    id="m1",
                )
            ]
        )
        yield transcript

    def on_console_selection_quote_requested(
        self, event: ConsoleSelectionQuoteRequested
    ) -> None:
        self.quote_requests.append(event)


@pytest.mark.asyncio
async def test_markdown_drag_menu_add_to_chat_quotes_whole_lines():
    """Task G: a real markdown-row drag mounts the menu; Add to chat quotes
    whole source lines (the line-snap), never partial lines."""
    app = _MarkdownTranscriptMenuApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(transcript._messages)
        await transcript.refresh_messages()
        await pilot.pause()
        row = app.query_one("#console-message-m1")
        body = row.query_one(Markdown)

        # Press at the body origin, release past the last line's end: the
        # char-level range covers the entire source (no whole-line snap
        # anymore -- live-spike change).
        row.post_message(_markdown_mouse(MouseDown, body, dy=0, dx=0))
        await pilot.pause()
        transcript.post_message(
            _markdown_mouse(
                MouseMove, body, dy=max(0, body.region.height - 1), dx=200
            )
        )
        await pilot.pause()
        transcript.post_message(
            _markdown_mouse(
                MouseUp, body, dy=max(0, body.region.height - 1), dx=200
            )
        )
        await pilot.pause()

        assert app.query_one(ConsoleSelectionMenu)  # menu mounted at release
        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()

        assert len(app.quote_requests) == 1
        assert app.quote_requests[0].quote == _MarkdownTranscriptMenuApp._MARKDOWN_SOURCE
        assert row.get_selection_text() == ""  # cleaned up
        assert not app.query(ConsoleSelectionMenu)


def _markdown_mouse(event_cls, body: Markdown, *, dy: int, dx: int = 2):
    """Build a mouse event over the markdown body at ``dy`` lines down."""
    screen_x = body.region.x + dx
    screen_y = body.region.y + dy
    return event_cls(
        widget=body,
        x=screen_x - body.region.x,
        y=screen_y - body.region.y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=screen_x,
        screen_y=screen_y,
    )


@pytest.mark.asyncio
async def test_add_to_chat_press_does_not_remove_menu_prematurely():
    """Regression (final review): the button's MouseDown must not fold the menu.

    ``ConsoleTranscript.on_mouse_down`` now dismisses mounted menus on any
    press, but a press that originates INSIDE the menu is skipped: the
    Add-to-chat button's MouseDown precedes its Click, so removing the
    menu on the press would unmount the button before its Click can
    activate it.
    """
    app = _TranscriptMenuApp()
    async with app.run_test() as pilot:
        await _finish_drag_selection(pilot)
        assert app.query_one(ConsoleSelectionMenu)

        # The press itself must leave the menu mounted (the button is
        # still alive to receive its Click).
        await pilot.mouse_down("#console-selection-add-to-chat")
        await pilot.pause()
        assert app.query_one(ConsoleSelectionMenu)
        await pilot.mouse_up("#console-selection-add-to-chat")
        await pilot.pause()

        # The full click lands: the quote still reaches the composer seam.
        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()
        assert len(app.quote_requests) == 1
        assert app.quote_requests[0].quote == "hello"


class _FocusTranscriptMenuApp(App[None]):
    """Transcript + composer so focus restore has both destinations."""

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="hello selectable world", id="m1"
                )
            ]
        )
        yield transcript
        yield ConsoleComposerBar(id="console-native-composer")


@pytest.mark.asyncio
async def test_escape_returns_focus_to_previously_focused_transcript():
    """Regression (final review): dismissal must not steal focus into the composer.

    The menu captures ``screen.focused`` on mount (before its own
    ``focus()``); when the transcript held focus before the drag,
    Escape-dismiss must return it there instead of pulling it into the
    composer.
    """
    app = _FocusTranscriptMenuApp()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.focus()
        await pilot.pause()
        await _finish_drag_selection(pilot)
        menu = app.query_one(ConsoleSelectionMenu)
        assert app.focused is app.query_one("#console-selection-add-to-chat")  # first button focused for keyboard nav
        assert menu._previous_focus is transcript  # captured before the grab

        await pilot.press("escape")
        await pilot.pause()

        assert not app.query(ConsoleSelectionMenu)
        assert app.focused is transcript  # NOT the composer


@pytest.mark.asyncio
async def test_escape_with_composer_focus_still_restores_composer():
    """The composer-focused case keeps the composer-focused outcome."""
    app = _FocusTranscriptMenuApp()
    async with app.run_test() as pilot:
        composer = app.query_one(ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        await _finish_drag_selection(pilot)
        assert app.query_one(ConsoleSelectionMenu)

        await pilot.press("escape")
        await pilot.pause()

        assert not app.query(ConsoleSelectionMenu)
        assert app.focused is composer
