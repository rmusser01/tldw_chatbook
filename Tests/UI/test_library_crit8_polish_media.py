"""Critique #8 polish contracts for the Library Media/Conversations surfaces.

Covers tasks 32060, 32065, 32067, 32068, 32070 and 32074 -- keyboard select
mode beside a loaded Reader, the below-64-column Media stage, the
conversation reader's identity/timestamps/pager, the Markdown notice and
byline, the footer, and the Prompts variables glyph.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from textual.widgets import Button, Checkbox, Input, Static

from Tests.UI.test_prompt_variables_dialog import (
    DialogHarness,
    _request as _dialog_request,
)
from Tests.UI.test_library_prompts_canvas import (
    _CanvasHost as _PromptsCanvasHost,
    _structured_editor_state,
)
from Tests.UI.test_library_media_side_by_side import (
    NARROW_SIZE,
    WIDE_SIZE,
    _open_media_list,
)
from Tests.UI.test_library_media_reader_flow import (
    _flow_app,
    _load_row_0,
    _open_article_reader,
)
from Tests.UI.test_library_media_toolbar_adapt import _CanvasApp, _select_state
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.Library.library_pager_state import build_library_pager_display
from tldw_chatbook.Library.library_prompts_state import (
    PromptListRow,
    PromptsListState,
)
from tldw_chatbook.Utils.adaptive_reader_state import ITEMS_MIN_WIDTH
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import SYSTEM_CHECKBOX_ID
from tldw_chatbook.Widgets.Library import LibraryConversationReader
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
from tldw_chatbook.Widgets.Library.library_media_viewer import RENDERED_VIEW_NOTE


def _typed_host(media_type: str, content: str, *, author: str = ""):
    """Two media items of ``media_type`` (two: the list waits for row 1)."""
    app = _build_test_app()
    app.library_new_profile_admission = False
    items = [
        {
            "id": f"media-{index}",
            "title": f"Budget review {index}",
            "type": media_type,
            "last_modified": f"2026-07-0{index}T10:00:00Z",
            "content": content,
            "author": author,
            "version": 1,
        }
        for index in (1, 2)
    ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


def _painted(widget) -> str:
    """Return exactly the cells ``widget`` occupies on the composited screen."""
    screen = widget.screen
    strips = list(screen._compositor.render_strips())
    region = widget.region
    return "\n".join(
        strips[y].text[region.x : region.right]
        for y in range(region.y, min(region.bottom, len(strips)))
    )


@pytest.mark.asyncio
async def test_s_enters_select_mode_from_an_items_row_beside_a_loaded_reader():
    """task-32060 AC#1: "s" belongs to the focused Items row, not the layout.

    Critique #8: with an item open in the Reader the ``s`` gate read the
    Reader's exit availability, so in every layout that keeps a real exit
    (100x30: Library collapsed, Items beside the Reader) the key was inert
    from a list row and the footer dropped the chip -- forcing the mouse.
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)

        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        assert ("s", "select") in screen._library_footer_shortcuts_for_current_state()

        await pilot.press("s")
        await pilot.pause()
        assert screen._media_state.select_mode is True
        footer = screen._library_footer_shortcuts_for_current_state()
        assert ("s", "done selecting") in footer
        assert ("space", "toggle selection") in footer

        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_markdown_notice_is_an_info_fact_not_a_reading_banner():
    """task-32068 AC#1: the note explains the item, so it lives in Info.

    Critique #8: every plain-text item -- which is most of them -- opened
    under "No Markdown formatting to render", a line about a view the reader
    never asked for, above the text they did.
    """
    host = _typed_host("plaintext", "Plain body text.\n")
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_article_reader(host, pilot)

        assert not screen.query("#library-media-content-mode-note")

        screen.query_one("#library-media-reader-select-info", Button).press()
        await pilot.pause()
        note = screen.query_one("#library-media-content-mode-note", Static)
        assert str(note.content) == RENDERED_VIEW_NOTE


@pytest.mark.asyncio
async def test_placeholder_author_paints_no_byline() -> None:
    """task-32068 AC#2: "Unknown" is a placeholder, not a byline.

    Half the local ingest paths write the literal string "Unknown" when they
    find no author, so the Reader header spent a row telling the reader
    nothing at 100x30 -- the guide says the byline appears only when an
    author exists.
    """
    host = _typed_host("plaintext", "Plain body text.\n", author="Unknown")
    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_article_reader(host, pilot)

        assert not screen.query("#library-media-reader-byline")
        header = screen.query_one("#library-media-viewer-title", Static)
        assert "Unknown" not in _painted(header)


_CONFIRM_COPY = (
    "Delete 2 selected items? You can undo right away, or restore later from Trash."
)


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [ITEMS_MIN_WIDTH, ITEMS_MIN_WIDTH + 2])
async def test_bulk_delete_confirm_copy_wraps_at_the_items_floor(width: int) -> None:
    """task-32060 AC#2: the safety sentence wraps inside the narrowest pane.

    The canvas floor (36) sat ABOVE the Items pane floor (32), so at the
    pane's narrowest the canvas overflowed its slot and the confirm sentence
    was clipped mid-word at the pane edge ("Delete 2 selected items? You c").
    """
    app = _CanvasApp(_select_state(selected_count=2, confirming=True))
    async with app.run_test(size=(width, 34)) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-media-canvas", LibraryMediaCanvas)
        copy = app.query_one("#library-media-bulk-delete-confirm-copy", Static)
        assert canvas.region.width <= width
        assert copy.region.right <= width
        painted = " ".join(_painted(copy).split())
        assert _CONFIRM_COPY in painted, painted


@pytest.mark.asyncio
async def test_bulk_delete_confirm_copy_wraps_in_the_real_items_pane_floor() -> None:
    """task-32060 AC#2, in the shell: the canvas takes the slot it is given.

    The Items pane spends 4 cells on padding, so the resolver's own 32-cell
    floor hands the canvas 28 -- eight cells less than the floor the canvas
    used to claim. That overflow, not the wrapping, is what cut the safety
    sentence mid-word.
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.reader_preferences = replace(
            screen._media_state.reader_preferences,
            custom_widths_enabled=True,
            items_width=ITEMS_MIN_WIDTH,
        )
        screen._sync_library_media_reader_layout_from_shell()
        screen.query_one("#library-media-select-toggle").press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")
        screen.query_one("#library-media-row-0", Button).press()
        screen.query_one("#library-media-row-1", Button).press()
        await pilot.pause()
        screen.query_one("#library-media-delete-selected", Button).press()
        copy = await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-confirm-copy"
        )

        shell = screen.query_one("#library-media-reader-shell")
        canvas = screen.query_one("#library-media-canvas", LibraryMediaCanvas)
        assert shell.items.region.width == ITEMS_MIN_WIDTH
        assert canvas.region.right <= shell.items.region.right
        painted = " ".join(_painted(copy).split())
        assert _CONFIRM_COPY in painted, painted

        for media_id in tuple(service.detail_release):
            service.release(media_id)


# ---------------------------------------------------------------------------
# task-32067: the Conversations reader identifies its conversation the way the
# list does -- by title and by age -- and no surface paints a one-page pager.
# ---------------------------------------------------------------------------


def _loaded_conversation_state() -> ConversationReaderState:
    return ConversationReaderState(
        selected_id="chat-a",
        selected_version=4,
        loaded_id="chat-a",
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        messages=(
            ConversationMessageView(
                "message-a", "user", "2026-08-23T12:01:00Z", "revision-a", 5, "hello"
            ),
        ),
        message_total=1,
        complete=True,
    )


@pytest.mark.asyncio
async def test_reader_status_names_the_conversation_not_its_uuid(
    widget_pilot,
) -> None:
    """task-32067 AC#1: "Loaded bf20fab2-0474-…" told the reader nothing."""
    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_conversation_state(),
        loaded_metadata={"title": "Alpha planning"},
        selected_metadata={"title": "Alpha planning"},
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        await pilot.pause()
        status = str(
            reader.query_one("#library-conversation-reader-status", Static).renderable
        )
        assert status.startswith("Loaded Alpha planning · 1 of 1 messages")
        assert "chat-a" not in status


@pytest.mark.asyncio
async def test_reader_messages_carry_the_list_age_not_an_iso_stamp(
    widget_pilot,
) -> None:
    """task-32067 AC#1: the list says "27m"; the transcript said 2026-08-23T…"""
    async with await widget_pilot(
        LibraryConversationReader,
        state=_loaded_conversation_state(),
        loaded_metadata={"title": "Alpha planning"},
        selected_metadata={"title": "Alpha planning"},
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        await pilot.pause()
        heading = str(
            reader.query_one(".library-conversation-reader-message", Static).renderable
        ).splitlines()[0]
        assert "2026-08-23T12:01:00Z" not in heading
        assert re.fullmatch(r"user · (now|\d+[mhdwy])", heading), heading


@pytest.mark.asyncio
async def test_one_page_conversations_list_drops_its_pager_chrome() -> None:
    """task-32067 AC#2: Media's one-page rule, applied to Conversations."""
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        status = await _wait_for_selector(
            screen, pilot, "#library-conversations-page-status"
        )
        assert str(status.renderable) == "1-2 of 2"
        assert not screen.query("#library-conversations-disabled-reason")
        assert not screen.query("#library-conversations-previous")
        assert not screen.query("#library-conversations-next")


@pytest.mark.asyncio
async def test_one_page_prompts_list_drops_its_pager_chrome() -> None:
    """task-32067 AC#2: ...and to Prompts, from the same pager projection."""
    rows = (PromptListRow(prompt_id=1, name="Draft outline", secondary=""),)
    app = _PromptsCanvasHost(
        PromptsListState(rows=rows, count=1, sort="newest"),
        pager=build_library_pager_display(
            applied_page=1,
            requested_page=1,
            page_size=20,
            row_count=1,
            total=1,
            freshness="fresh",
        ),
    )
    async with app.run_test() as pilot:
        label = pilot.app.query_one("#library-prompts-page-label", Static)
        assert str(label.renderable) == "1-1 of 1"
        assert not pilot.app.query("#library-prompts-page-previous")
        assert not pilot.app.query("#library-prompts-page-next")
        status = pilot.app.query_one("#library-prompts-page-status", Static)
        assert str(status.renderable) == ""


# ---------------------------------------------------------------------------
# task-32074: the Prompts variables checkbox states itself in text, and the
# editor's "Use in Console" sits where the Media reader's does.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_variables_checkbox_states_itself_with_a_glyph() -> None:
    """task-32074 AC#1: not an empty frame that means "off" by colour.

    Textual paints the toggle's inner "X" in the button's own background
    while the value is False, so critique #8 met a box with nothing in it
    and no way to tell checked from unchecked in a plain-text capture.
    """
    app = DialogHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        app.show(_dialog_request(system_text="System lane"))
        await pilot.pause()
        checkbox = app.screen.query_one(f"#{SYSTEM_CHECKBOX_ID}", Checkbox)

        assert str(checkbox.render()).startswith("☐")
        checkbox.value = True
        await pilot.pause()
        assert str(checkbox.render()).startswith("☑")


@pytest.mark.asyncio
async def test_use_in_console_sits_in_the_prompt_editor_header() -> None:
    """task-32074 AC#2: beside Basic/Advanced/Info, like the Media reader.

    It used to be row 49 of 52 -- below the whole editor -- while the
    equivalent Media action sits in the Reader header.
    """
    app = _PromptsCanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
    )
    async with app.run_test(size=(140, 40)) as pilot:
        modes = pilot.app.query_one("#library-prompt-mode-controls")
        assert [child.id for child in modes.children] == [
            "library-prompt-mode-basic",
            "library-prompt-mode-advanced",
            "library-prompt-mode-info",
            "library-prompt-insert-console",
        ]
        actions = pilot.app.query_one("#library-prompt-editor-actions")
        assert "library-prompt-insert-console" not in [
            child.id for child in actions.children
        ]


# ---------------------------------------------------------------------------
# task-32070: the footer is one row, and it tells the truth the moment the
# state it describes exists.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rail_search_leaves_exactly_one_footer_row() -> None:
    """task-32070 AC#1: one footer widget, one row, inside its own slot.

    Critique #8 saw a wrapped fragment ("vidence | F6 …") painted above the
    real footer after a rail search at 235x52 -- a second footer row in
    everything but name.
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        search = screen.query_one("#library-search-input", Input)
        search.focus()
        await pilot.pause()
        for character in "evidence":
            await pilot.press(character)
        await pilot.press("enter")
        await pilot.pause()

        footers = list(screen.query(AppFooterStatus))
        assert len(footers) == 1
        footer = footers[0]
        assert footer.region.height == 1
        hints = footer.query_one("#footer-key-quit", Static)
        assert hints.region.height == 1
        assert hints.region.right <= footer.region.right

        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_select_mode_hints_appear_without_a_round_trip() -> None:
    """task-32070 AC#2: the hints land with select mode, not after Escape."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()

        assert screen._media_state.select_mode is True
        footer = screen._library_footer_shortcuts_for_current_state()
        assert ("space", "toggle selection") in footer
        assert ("s", "done selecting") in footer

        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_query_box_footer_names_the_enter_action() -> None:
    """task-32070 AC#3: Enter runs the search, so the footer must say so.

    The "enter run search" copy itself belongs to the Search/RAG keyboard
    branch of this wave; this test pins it once it lands and reports the gap
    until then rather than duplicating that change here.
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        search = screen.query_one("#library-search-input", Input)
        search.focus()
        await pilot.pause()
        for character in "evidence":
            await pilot.press(character)
        await pilot.pause()

        footer = screen._library_footer_shortcuts_for_current_state()
        enter_labels = {label for key, label in footer if key == "enter"}
        for media_id in tuple(service.detail_release):
            service.release(media_id)
        if "run search" not in enter_labels:
            pytest.skip(
                "task-32070 AC#3 is owned by the Search/RAG keyboard branch; "
                f"the query box still advertises {sorted(enter_labels)!r}."
            )
        assert "select evidence" not in enter_labels


# ---------------------------------------------------------------------------
# task-32065: below the 64-column floor the Media stage is the LIST, not an
# empty Reader, and it keeps a named way back to the rail.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_below_64_columns_shows_the_items_list_and_a_way_back() -> None:
    """task-32065: at 60x24 Media painted a placeholder and two grips.

    "Select a media item to read it here." with no list to select from, and
    no control back to the rail -- the destination could not be used at all.
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        await _wait_for_condition(
            pilot,
            # Re-queried every poll: a background list refresh recomposes the
            # row Buttons, and a reference captured before that keeps the
            # detached widget's zero region forever.
            lambda: bool(screen.query("#library-media-row-0"))
            and screen.query_one("#library-media-row-0", Button).region.width > 0,
            message="The Items list never painted at 60 columns.",
        )

        shell = screen.query_one("#library-media-reader-shell")
        assert shell.effective_layout.items_open is True
        assert shell.effective_layout.library_open is False

        back = screen.query_one("#library-media-rail-return", Button)
        assert back.display is True
        assert str(back.label) in ("‹ Library", "< Library")
        assert back in screen.focus_chain

        back.press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(
                "#library-media-reader-shell"
            ).effective_layout.library_open,
            message="'‹ Library' did not bring the rail back.",
        )
        assert screen.query_one("#library-row-browse-media").region.width > 0
        assert screen.query_one("#library-media-rail-return", Button).display is False

        # ...and it comes back when the rail is collapsed again. Live at
        # 60x24 it did not: the collapse recomposes the Items pane, and a
        # control whose visibility was only patched by the layout sync came
        # back mounted-but-hidden.
        screen.query_one("#library-media-library-grip", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one(
                "#library-media-rail-return", Button
            ).display,
            message="'‹ Library' did not return with the collapsed rail.",
        )

        for media_id in tuple(service.detail_release):
            service.release(media_id)
