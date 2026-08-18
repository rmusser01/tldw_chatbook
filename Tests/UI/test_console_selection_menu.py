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
from textual.widget import Widget
from textual.widgets import Button, Markdown, Static

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
            "console-selection-create-note",
        ]
        assert [str(button.label) for button in buttons] == [
            "Add to chat",
            "More Details",
            "Ask in Side Chat",
            "Create note",
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


class _ShortTranscriptWithComposerApp(App[None]):
    """Transcript that does NOT reach the screen bottom (composer below).

    Live-spike 2026-08-16: the real Console layout puts the composer +
    status bar BELOW the transcript, so the transcript's visible box ends
    above the screen edge. Clamping the menu to SCREEN bounds let a
    bottom-anchored menu paint over the composer.
    """

    CSS = """
    ConsoleTranscript {
        height: 20;
    }
    #composer-standin {
        height: 1fr;
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
        yield Static("composer stand-in", id="composer-standin")


@pytest.mark.asyncio
async def test_last_row_release_keeps_menu_within_transcript_not_composer():
    """Regression (live spike 2026-08-16): clamp to the transcript box, not
    the screen.

    A release on the LAST visible row of a transcript that does not extend
    to the screen bottom must keep the menu inside the transcript's visible
    region -- never overlapping the composer that lives below it.
    """
    app = _ShortTranscriptWithComposerApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-m0", ConsoleTranscriptMessage)
        region = transcript.region
        assert region.bottom < 32  # transcript does NOT reach the screen bottom
        # Fixture shape pin (clamp-fix review): the stand-in must really sit
        # BELOW the transcript, or the test stops regressing the overlap.
        assert app.query_one("#composer-standin").region.y >= region.bottom
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=region.bottom - 1,  # release at the transcript's last row
            )
        )
        await pilot.pause()
        await pilot.pause()  # let the post-layout clamp settle
        menu = app.query_one(ConsoleSelectionMenu)
        assert region.x <= menu.region.x
        assert menu.region.right <= region.right
        assert region.y <= menu.region.y
        assert menu.region.bottom <= region.bottom


class _TinyTranscriptFeedbackApp(App[None]):
    """Transcript box (7 rows) shorter than the compact feedback menu.

    Clamp-fix review: on 24-30 row terminals the transcript box can be
    shorter than even the compact menu; the shrink guard must trade the
    container border + hint line instead of bleeding over the composer.
    """

    CSS = """
    ConsoleTranscript {
        height: 7;
    }
    #composer-standin {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.ASSISTANT,
                    content=f"answer row {i} selectable text",
                    id=f"m{i}",
                )
                for i in range(10)
            ]
        )
        yield transcript
        yield Static("composer stand-in", id="composer-standin")


@pytest.mark.asyncio
async def test_short_owner_box_shrinks_menu_and_keeps_containment():
    """Shrink guard: a box shorter than even the compact feedback menu drops
    the container border and hint line (no actions hidden), and the
    re-measured menu stays inside the transcript box."""
    app = _TinyTranscriptFeedbackApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-m0")
        region = transcript.region
        assert region.height == 7  # box shorter than the 10-row compact menu
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=region.bottom - 1,
            )
        )
        await pilot.pause()
        await pilot.pause()  # measured clamp pass
        await pilot.pause()  # shrink pass re-measures after the class flip
        await pilot.pause()
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.has_class("shrunk-for-short-owner")
        assert not menu.query_one("#console-selection-feedback-hint").display
        # All six actions stay mounted and displayed (no action-hiding).
        assert len([b for b in menu.query("Button") if b.display]) == 7
        assert menu.region.height <= region.height
        assert menu.region.bottom <= region.bottom
        assert menu.region.x >= region.x
        assert menu.region.right <= region.right


@pytest.mark.asyncio
async def test_null_transcript_region_falls_back_to_screen_bounds():
    """Clamp-fix review: a NULL/unmeasured transcript region (textual 8.2.8
    yields NULL_REGION, never None) must not collapse the menu anchor to
    (0, 0) -- the transcript clamps against screen-size bounds instead,
    mirroring the menu-side guard for unmeasured owners."""
    from unittest.mock import PropertyMock, patch

    from textual.geometry import Offset, Region

    app = _TranscriptMenuApp()
    async with app.run_test(size=(80, 24)) as pilot:
        with patch.object(
            ConsoleTranscript,
            "region",
            new_callable=PropertyMock,
            return_value=Region(),
        ):
            await _finish_drag_selection(pilot)  # release at screen (4, 6)
        await pilot.pause()  # let the menu's measured clamp settle
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.absolute_offset == Offset(4, 7)


@pytest.mark.asyncio
async def test_bottom_overflow_menu_hops_above_selected_row():
    """Live spike 2026-08-16 8:48: keep the highlight visible.

    A release near the transcript bottom makes the measured clamp pull the
    menu up from the release point; pinning its bottom to the box bottom
    landed it ON TOP of the just-selected row -- the reverse-video
    highlight strip (the evidence of the selection) hid behind the menu.
    When there is room above, the menu must hop entirely ABOVE the
    selected row instead (and still stay inside the transcript box)."""
    app = _ShortTranscriptWithComposerApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.scroll_end(animate=False)
        await pilot.pause()
        row = app.query_one("#console-message-m29", ConsoleTranscriptMessage)
        region = transcript.region
        # Shape pin: the selected row is the LAST visible row, with room
        # above it for the compact menu (this is the defect screenshot's
        # geometry).
        assert region.bottom - 4 <= row.region.y <= region.bottom - 2
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=row.region.y,
            )
        )
        await pilot.pause()
        await pilot.pause()  # let the post-layout clamp settle
        menu = app.query_one(ConsoleSelectionMenu)
        # The menu sits ENTIRELY above the selected row: its bottom is at
        # most row.y - 1, so the row and its highlight strip stay visible.
        assert menu.region.bottom <= row.region.y - 1
        # ...and still inside the transcript box (never over the composer).
        assert region.y <= menu.region.y
        assert menu.region.bottom <= region.bottom
        assert region.x <= menu.region.x
        assert menu.region.right <= region.right


@pytest.mark.asyncio
async def test_top_row_selection_bottom_overflow_pins_to_box_bottom():
    """Fallback edge: a selection at the very TOP of the box leaves no room
    above -- the menu keeps today's bottom-pinned placement (contained, no
    crash); it may cover the row because there is nowhere better to go."""
    app = _ShortTranscriptWithComposerApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-m0", ConsoleTranscriptMessage)
        region = transcript.region
        assert row.region.y - region.y < 5  # no room for the menu above m0
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        transcript.post_message(
            ConsoleTranscript.TranscriptTextSelected(
                selection=TextSelection(row.id, 0, 5),
                screen_x=region.x + 4,
                screen_y=region.bottom - 1,  # release at the box's last row
            )
        )
        await pilot.pause()
        await pilot.pause()
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.region.bottom <= region.bottom  # pinned inside the box
        assert region.y <= menu.region.y
        assert region.x <= menu.region.x
        assert menu.region.right <= region.right


@pytest.mark.asyncio
async def test_null_selection_row_region_selection_top_none_no_crash_keeps_containment():
    """Fallback edge: when the origin row's region is NULL/unmeasured, the
    transcript passes selection_top=None -- the None plumbing must hold (no
    crash, menu stays contained in the owner box; no above-row placement
    happens without a row top)."""
    from unittest.mock import PropertyMock, patch

    from textual.geometry import Region

    app = _ShortTranscriptWithComposerApp()
    async with app.run_test(size=(80, 32)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        row = app.query_one("#console-message-m29", ConsoleTranscriptMessage)
        region = transcript.region
        transcript.scroll_end(animate=False)
        await pilot.pause()
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript.selection_manager.finish_drag()
        with patch.object(
            ConsoleTranscriptMessage,
            "region",
            new_callable=PropertyMock,
            return_value=Region(),
        ):
            transcript.post_message(
                ConsoleTranscript.TranscriptTextSelected(
                    selection=TextSelection(row.id, 0, 5),
                    screen_x=region.x + 4,
                    screen_y=row.region.y,
                )
            )
            await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu._selection_top is None  # guard really passed None
        assert menu.region.bottom <= region.bottom
        assert region.y <= menu.region.y
        assert region.x <= menu.region.x
        assert menu.region.right <= region.right


class _GeometryOwnerApp(App[None]):
    """Short owner box for direct clamp-geometry tests (no transcript).

    The owner is a plain 10-row Vertical at the screen top; the base menu
    (no feedback entries) measures 5 rows. Placement inputs are exact:
    ``selection_top`` and the anchor, so the above-row placement branches
    are exercised without transcript scroll choreography.
    """

    def __init__(self, *, selection_top: int | None, screen_y: int) -> None:
        super().__init__()
        self._selection_top = selection_top
        self._screen_y = screen_y
        self.owner: Widget | None = None

    CSS = "#owner { height: 10; }"

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        self.owner = Vertical(id="owner")
        yield self.owner

    async def mount_menu(self) -> None:
        """Mount the menu the way the real flow does: onto an
        already-laid-out owner (mounting it in ``compose`` races the clamp
        against the owner's first layout -- the owner's region is still
        NULL when the one-shot clamp runs, and the screen-size fallback
        finds nothing to shift)."""
        assert self.owner is not None
        await self.mount(
            ConsoleSelectionMenu(
                screen_x=2,
                screen_y=self._screen_y,
                owner=self.owner,
                selection_top=self._selection_top,
            )
        )


@pytest.mark.asyncio
async def test_touching_above_row_when_gap_does_not_fit():
    """Review follow-up: a box too short for the one-row gap but tall enough
    for the menu itself must ABDUT the selected row (touching placement)
    rather than pin to the box bottom and land on top of the highlight --
    the reachable corner on small terminals (box <= ~2x menu height)."""
    # selection_top tracks the menu's grown height (4 actions -> 6 rows):
    # the gap row still does not fit (needs top >= 7) but the menu itself
    # exactly does (top == height), which is this test's whole scenario.
    app = _GeometryOwnerApp(selection_top=6, screen_y=10)
    async with app.run_test(size=(80, 24)) as pilot:
        owner = app.query_one("#owner")
        await pilot.pause()  # owner lays out before the menu mounts
        await app.mount_menu()
        menu = app.query_one(ConsoleSelectionMenu)
        await pilot.pause()
        await pilot.pause()  # measured clamp settles
        box = owner.region
        assert menu.region.height == 6  # base-menu geometry pin (4 actions + border)
        # Menu occupies rows 0..5: no gap row, but the row at y6 (and its
        # highlight strip) stays visible below the menu.
        assert menu.region.y == 0
        assert menu.region.bottom == 6
        assert box.contains_region(menu.region)


@pytest.mark.asyncio
async def test_selection_top_below_box_keeps_menu_contained():
    """Defensive bound (review follow-up): a selection_top sampled beyond
    the owner box (stale pre-mount sample, programmatic mounting) must not
    pull the menu outside the box -- the effective row top clamps to the
    box bottom, keeping the containment invariant unconditional."""
    app = _GeometryOwnerApp(selection_top=20, screen_y=10)
    async with app.run_test(size=(80, 24)) as pilot:
        owner = app.query_one("#owner")
        await pilot.pause()  # owner lays out before the menu mounts
        await app.mount_menu()
        menu = app.query_one(ConsoleSelectionMenu)
        await pilot.pause()
        await pilot.pause()  # measured clamp settles
        box = owner.region
        assert menu.region.height == 6  # base-menu geometry pin (4 actions + border)
        assert box.contains_region(menu.region)
        assert menu.region.bottom <= box.bottom


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


class _FrFlowApp(App[None]):
    """Minimal screen-shaped app: docked navbar/footer around a 1fr content
    container, mirroring BaseAppScreen's arrangement.

    Live spike 2026-08-16 (user F12 dump): a ConsoleSelectionMenu mounted on
    the screen shrank the 1fr sibling by exactly the menu's height -- the
    composer floated 9 rows above the footer with dead rows between (the
    "black bar"). Textual 8.2.8's vertical layout excludes
    ``position: absolute`` children from sibling stacking but still feeds
    their height into the fr denominator; ``overlay: screen`` is the style
    that fully removes an overlay from the container's flow math.
    """

    CSS = """
    #navbar {
        dock: top;
        height: 3;
    }
    #screen-content {
        height: 1fr;
    }
    #footer {
        dock: bottom;
        height: 1;
    }
    ConsoleTranscript {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        from textual.containers import Vertical

        transcript = ConsoleTranscript(id="console-native-transcript")
        transcript.set_messages(
            [
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER, content="hello world", id="m1"
                )
            ]
        )
        yield Static("nav", id="navbar")
        with Vertical(id="screen-content"):
            yield transcript
        yield Static("footer", id="footer")


@pytest.mark.asyncio
async def test_screen_mounted_menu_steals_no_flow_height():
    """The screen-mounted menu must not shrink the 1fr content sibling.

    Regression for the live "black bar": mounting the menu consumed its own
    height from the screen's fr budget, pulling the composer up and leaving
    dead rows above the docked footer whenever a selection menu was open.
    """
    app = _FrFlowApp()
    async with app.run_test(size=(80, 40)) as pilot:
        content = app.query_one("#screen-content")
        transcript = app.query_one(ConsoleTranscript)
        await pilot.pause()
        assert content.region.height == 36  # 40 - 3 navbar - 1 footer
        await app.screen.mount(
            ConsoleSelectionMenu(screen_x=4, screen_y=10, owner=transcript)
        )
        await pilot.pause()
        await pilot.pause()  # clamp pass
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.region.y == 10  # still anchored at its cell
        assert (
            content.region.height == 36
        ), "menu stole flow height from the 1fr sibling"


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


# --- Phase 3 task 2: feedback entries + run gating -------------------------


NO_RUN_HINT = "No active run — start a run to send review feedback"

_FEEDBACK_BUTTON_IDS = [
    "console-selection-add-to-chat",
    "console-selection-more-details",
    "console-selection-ask-side-chat",
    "console-selection-create-note",
    "console-selection-request-changes",
    "console-selection-lgm",
    "console-selection-comment",
]


class _FeedbackMenuApp(App[None]):
    """Menu harness with the phase-3 feedback ctor knobs + event capture."""

    def __init__(
        self, *, feedback_available: bool = True, run_active: bool = False
    ) -> None:
        super().__init__()
        self.feedback_available = feedback_available
        self.run_active = run_active
        self.request_changes_events: list[ConsoleSelectionMenu.RequestChanges] = []
        self.lgm_events: list[ConsoleSelectionMenu.Lgm] = []
        self.comment_events: list[ConsoleSelectionMenu.Comment] = []

    def compose(self) -> ComposeResult:
        yield ConsoleSelectionMenu(
            screen_x=4,
            screen_y=6,
            feedback_available=self.feedback_available,
            run_active=self.run_active,
        )

    def on_console_selection_menu_request_changes(
        self, event: ConsoleSelectionMenu.RequestChanges
    ) -> None:
        self.request_changes_events.append(event)

    def on_console_selection_menu_lgm(self, event: ConsoleSelectionMenu.Lgm) -> None:
        self.lgm_events.append(event)

    def on_console_selection_menu_comment(
        self, event: ConsoleSelectionMenu.Comment
    ) -> None:
        self.comment_events.append(event)


@pytest.mark.asyncio
async def test_compact_menu_fits_height_budget():
    """Height budget (clamp-fix review): one row per action button.

    The 3-row library Button chrome (line-pad + tall border) stacked the
    feedback variant to ~24 rows -- taller than a short transcript's whole
    box on 24-30 row terminals, so even the owner-box clamp bled the menu
    over the composer. Compact form: feedback variant <= 10 rows (6
    single-row buttons + 1-row hint + container border), base <= 8.
    """
    app = _FeedbackMenuApp(feedback_available=True, run_active=False)
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.region.height <= 10
    app = _FeedbackMenuApp(feedback_available=False, run_active=True)
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.region.height <= 8


class _AnsiFeedbackMenuApp(App[None]):
    """Feedback menu harness with native ANSI color mode pinned ON.

    Increment review of e2dc272e4: textual 8.2.8's ANSI-mode
    ``Button:ansi.-style-flat:disabled`` border rule (specificity (0,3,1))
    beats the compact rule's ``border: none !important`` ((0,0,2)) -- both
    important, so specificity decides.
    """

    def __init__(self) -> None:
        # textual 8.2.8 supported path: constructor arg -> App.ansi_color
        # reactive; the :ansi pseudo-class on widgets follows
        # app.native_ansi_color.
        super().__init__(ansi_color=True)

    def compose(self) -> ComposeResult:
        yield ConsoleSelectionMenu(
            screen_x=4,
            screen_y=6,
            feedback_available=True,
            run_active=False,
        )


@pytest.mark.asyncio
async def test_ansi_mode_disabled_buttons_stay_borderless_with_labels():
    """Increment review of e2dc272e4: in native ANSI color mode the two
    run-gated buttons re-grew tall borders (2-row border-only boxes, labels
    clipped out, 11-row menu breaking the <=10 budget). Every action must
    stay one row with its label actually rendered, in ANSI mode too."""
    app = _AnsiFeedbackMenuApp()
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        assert app.native_ansi_color is True  # ANSI mode really pinned
        menu = app.query_one(ConsoleSelectionMenu)
        assert menu.region.height <= 10  # height budget holds in ANSI mode
        for selector, label in (
            ("#console-selection-request-changes", "Request changes"),
            ("#console-selection-lgm", "LGTM"),
        ):
            button = menu.query_one(selector, Button)
            assert button.disabled  # the gated pair is the reproducer
            assert button.region.height == 1
            assert button.content_region.height == 1  # a row for the label
            rendered = "".join(
                button.render_line(y).text for y in range(button.region.height)
            )
            assert label in rendered  # label survives, not clipped by border


@pytest.mark.asyncio
async def test_feedback_buttons_absent_without_availability():
    """feedback_available=False: only the three base buttons, no hint.

    (The default-ctor case is guarded by the pre-existing
    ``test_menu_offers_three_stacked_options_in_order``.)
    """
    app = _FeedbackMenuApp(feedback_available=False, run_active=True)
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        menu = app.query_one(ConsoleSelectionMenu)
        ids = [b.id for b in menu.query("Button")]
        assert ids == [
            "console-selection-add-to-chat",
            "console-selection-more-details",
            "console-selection-ask-side-chat",
            "console-selection-create-note",
        ]
        assert not menu.query("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_feedback_buttons_render_after_ask_side_chat_when_available():
    """feedback_available=True renders the three feedback buttons after
    Ask in Side Chat with the exact labels."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=True)
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        menu = app.query_one(ConsoleSelectionMenu)
        buttons = list(menu.query("Button"))
        assert [b.id for b in buttons] == _FEEDBACK_BUTTON_IDS
        assert [str(b.label) for b in buttons[-3:]] == [
            "Request changes",
            "LGTM",
            "Comment",
        ]


@pytest.mark.asyncio
async def test_run_gating_disables_request_and_lg_but_not_comment():
    """No active run: Request changes + LGTM disabled (with the hint as
    tooltip), Comment enabled, and the dim hint line visible."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=False)
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        menu = app.query_one(ConsoleSelectionMenu)
        request = menu.query_one("#console-selection-request-changes", Button)
        lgm = menu.query_one("#console-selection-lgm", Button)
        comment = menu.query_one("#console-selection-comment", Button)
        assert request.disabled
        assert lgm.disabled
        assert not comment.disabled
        assert request.tooltip == NO_RUN_HINT
        assert lgm.tooltip == NO_RUN_HINT
        hint = menu.query_one("#console-selection-feedback-hint", Static)
        assert hint.display
        assert hint.renderable == NO_RUN_HINT


@pytest.mark.asyncio
async def test_run_active_enables_all_feedback_buttons_and_hides_hint():
    """Active run: all three feedback buttons enabled and no hint line."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=True)
    async with app.run_test(size=(80, 40)) as pilot:
        del pilot
        menu = app.query_one(ConsoleSelectionMenu)
        assert not menu.query_one("#console-selection-request-changes").disabled
        assert not menu.query_one("#console-selection-lgm").disabled
        assert not menu.query_one("#console-selection-comment").disabled
        assert not menu.query("#console-selection-feedback-hint")


@pytest.mark.asyncio
async def test_key_navigation_down_cycle_skips_disabled_buttons():
    """Down-cycling must never land on a disabled button (and must still
    reach every enabled one)."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=False)
    async with app.run_test(size=(80, 40)) as pilot:
        menu = app.query_one(ConsoleSelectionMenu)
        buttons = list(menu.query(Button))
        assert [b.id for b in buttons if b.disabled] == [
            "console-selection-request-changes",
            "console-selection-lgm",
        ]
        focused_ids: set[str | None] = set()
        for _ in range(2 * len(buttons)):
            await pilot.press("down")
            focused = app.focused
            assert focused is not None
            assert not focused.disabled, f"focus landed on disabled {focused.id}"
            focused_ids.add(focused.id)
        assert focused_ids == {
            "console-selection-add-to-chat",
            "console-selection-more-details",
            "console-selection-ask-side-chat",
            "console-selection-create-note",
            "console-selection-comment",
        }


@pytest.mark.asyncio
async def test_key_navigation_up_wrap_skips_disabled_buttons():
    """Up from the first button wraps to the LAST enabled button (Comment),
    not the disabled LGTM."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=False)
    async with app.run_test(size=(80, 40)) as pilot:
        menu = app.query_one(ConsoleSelectionMenu)
        del menu
        await pilot.pause()
        assert app.focused.id == "console-selection-add-to-chat"  # mount focuses first
        await pilot.press("up")
        await pilot.pause()
        assert app.focused.id == "console-selection-comment"


@pytest.mark.asyncio
async def test_enabled_feedback_buttons_post_messages():
    """With an active run each feedback button posts its own message."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=True)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.click("#console-selection-request-changes")
        await pilot.pause()
        assert len(app.request_changes_events) == 1
        assert app.lgm_events == []
        assert app.comment_events == []

        await pilot.click("#console-selection-lgm")
        await pilot.pause()
        assert len(app.lgm_events) == 1
        assert len(app.request_changes_events) == 1
        assert app.comment_events == []

        await pilot.click("#console-selection-comment")
        await pilot.pause()
        assert len(app.comment_events) == 1
        assert len(app.request_changes_events) == 1
        assert len(app.lgm_events) == 1


@pytest.mark.asyncio
async def test_comment_posts_when_run_gated():
    """Gated menu still lets Comment through (keyboard users without a run)."""
    app = _FeedbackMenuApp(feedback_available=True, run_active=False)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.click("#console-selection-comment")
        await pilot.pause()
        assert len(app.comment_events) == 1
        assert app.request_changes_events == []
        assert app.lgm_events == []


@pytest.mark.asyncio
async def test_disabled_feedback_buttons_do_not_post():
    """Clicking a run-gated button posts nothing."""
    for selector in ("#console-selection-request-changes", "#console-selection-lgm"):
        app = _FeedbackMenuApp(feedback_available=True, run_active=False)
        async with app.run_test(size=(80, 40)) as pilot:
            await pilot.click(selector)
            await pilot.pause()
            assert app.request_changes_events == []
            assert app.lgm_events == []
            assert app.comment_events == []


class _OwnerCapture(Widget):
    """Sibling-of-menu owner: only direct posting (not bubbling) reaches it."""

    def __init__(self) -> None:
        super().__init__(id="feedback-owner")
        self.received: list[tuple[str, object]] = []

    def on_console_selection_menu_request_changes(self, event) -> None:
        self.received.append(("request-changes", event))

    def on_console_selection_menu_lgm(self, event) -> None:
        self.received.append(("lgm", event))

    def on_console_selection_menu_comment(self, event) -> None:
        self.received.append(("comment", event))


class _OwnerMenuApp(App[None]):
    """Mounts the menu with an explicit owner (transcript-like routing)."""

    def __init__(self, *, run_active: bool = True) -> None:
        super().__init__()
        self.run_active = run_active
        self.owner = _OwnerCapture()

    def compose(self) -> ComposeResult:
        yield self.owner
        yield ConsoleSelectionMenu(
            screen_x=2,
            screen_y=2,
            feedback_available=True,
            run_active=self.run_active,
            owner=self.owner,
        )


@pytest.mark.asyncio
async def test_feedback_messages_post_to_owner_when_provided():
    """Feedback buttons post to the owner when one was passed (the owner is
    a menu SIBLING, so bubbling alone could never reach it)."""
    app = _OwnerMenuApp(run_active=True)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.click("#console-selection-request-changes")
        await pilot.pause()
        await pilot.click("#console-selection-lgm")
        await pilot.pause()
        await pilot.click("#console-selection-comment")
        await pilot.pause()
        assert [kind for kind, _ in app.owner.received] == [
            "request-changes",
            "lgm",
            "comment",
        ]


@pytest.mark.asyncio
async def test_feedback_messages_owner_routed_when_gated_comment():
    """Owner routing works for the always-enabled Comment under gating."""
    app = _OwnerMenuApp(run_active=False)
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.click("#console-selection-comment")
        await pilot.pause()
        assert [kind for kind, _ in app.owner.received] == ["comment"]


# --- Create note (task-18156 Task 6, maintainer request) ---------------------


@pytest.mark.asyncio
async def test_create_note_button_present_for_every_selection():
    """Create note is a base action like Add to chat: present with and
    without feedback availability, never run-gated."""
    for feedback_available in (False, True):
        app = _FeedbackMenuApp(feedback_available=feedback_available, run_active=False)
        async with app.run_test(size=(80, 40)):
            menu = app.query_one(ConsoleSelectionMenu)
            button = menu.query_one("#console-selection-create-note")
            assert not button.disabled
            ids = [b.id for b in menu.query("Button")]
            assert ids.index("console-selection-create-note") == ids.index(
                "console-selection-ask-side-chat"
            ) + 1


@pytest.mark.asyncio
async def test_create_note_button_posts_the_menu_message():
    class _NoteMenuApp(_FeedbackMenuApp):
        def __init__(self) -> None:
            super().__init__(feedback_available=False, run_active=False)
            self.create_note_events: list[ConsoleSelectionMenu.CreateNote] = []

        def on_console_selection_menu_create_note(
            self, event: ConsoleSelectionMenu.CreateNote
        ) -> None:
            self.create_note_events.append(event)

    app = _NoteMenuApp()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.click("#console-selection-create-note")
        await pilot.pause()
    assert len(app.create_note_events) == 1
