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

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_selection import TextSelection
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionMenu,
    ConsoleSelectionQuoteRequested,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleTranscript,
    ConsoleTranscriptMessage,
)


class _MenuApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.add_to_chat_events: list[ConsoleSelectionMenu.AddToChat] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSelectionMenu(local_x=4, local_y=6)

    def on_console_selection_menu_add_to_chat(
        self, event: ConsoleSelectionMenu.AddToChat
    ) -> None:
        self.add_to_chat_events.append(event)


@pytest.mark.asyncio
async def test_menu_offers_add_to_chat_and_posts_message():
    app = _MenuApp()
    async with app.run_test() as pilot:
        await pilot.click("#console-selection-add-to-chat")
        await pilot.pause()
        assert len(app.add_to_chat_events) == 1


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
