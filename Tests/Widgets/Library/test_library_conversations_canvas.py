"""Tests for the LibraryConversationsCanvas widget (task-2859 item 1)."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.Library.library_conversations_state import (
    LibraryConversationRow,
    LibraryConversationsCanvasState,
)
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)


pytestmark = pytest.mark.asyncio


def _state(*, rows=(), query: str = "", empty_copy: str = "") -> LibraryConversationsCanvasState:
    return LibraryConversationsCanvasState(
        rows=rows,
        status_copy="",
        empty_copy=empty_copy,
        selected_id="",
        preview_lines=(),
        query=query,
    )


async def test_conversations_canvas_has_a_title_header_matching_the_sibling_pattern(
    widget_pilot,
):
    """Matches the "Media (n)"/"Prompts (n)"/"Skills (n)" sibling convention --
    the Conversations canvas used to have no title Static at all (task-2859
    item 1)."""
    rows = (
        LibraryConversationRow(
            conversation_id="c1", title="Trip planning", secondary="3 messages"
        ),
        LibraryConversationRow(
            conversation_id="c2", title="Recipe ideas", secondary="1 message"
        ),
    )
    async with await widget_pilot(
        LibraryConversationsCanvas, canvas=_state(rows=rows)
    ) as pilot:
        await pilot.pause()
        header = pilot.app.query_one("#library-conversations-title", Static)
        assert str(header.renderable) == "Conversations (2)"


async def test_conversations_canvas_title_reflects_rendered_row_count_not_zero(
    widget_pilot,
):
    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=_state(empty_copy="No conversations yet."),
    ) as pilot:
        await pilot.pause()
        header = pilot.app.query_one("#library-conversations-title", Static)
        assert str(header.renderable) == "Conversations (0)"


async def test_conversations_filter_input_renders_above_the_empty_state_text(
    widget_pilot,
):
    """The filter Input used to render BELOW the empty-state Static -- task-2859
    item 1 moves it above, matching Notes/Prompts (title -> filter -> empty/rows)."""
    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=_state(empty_copy="No conversations yet. Chat in Console and it appears here."),
    ) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        ids_in_order = [
            widget.id
            for widget in screen.walk_children()
            if widget.id in {"library-conversations-filter", "library-conversations-status"}
        ]
        assert ids_in_order == [
            "library-conversations-filter",
            "library-conversations-status",
        ], ids_in_order
        filter_input = pilot.app.query_one("#library-conversations-filter", Input)
        assert filter_input.placeholder == "Filter conversations… (Enter)"
