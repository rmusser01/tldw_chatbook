"""Tests for the LibraryConversationsCanvas widget (task-2859 item 1)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_conversations_state import (
    LibraryConversationRow,
    LibraryConversationsCanvasState,
)
from tldw_chatbook.Library.library_pager_state import build_library_pager_display
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DISABLED_ACTION_MARKER,
    LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
)
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)


pytestmark = pytest.mark.asyncio


def _state(
    *,
    rows=(),
    query: str = "",
    empty_copy: str = "",
    **overrides,
) -> LibraryConversationsCanvasState:
    pager = overrides.pop(
        "pager",
        build_library_pager_display(
            applied_page=None,
            requested_page=1,
            page_size=20,
            row_count=0,
            total=None,
            freshness="uninitialized",
        ),
    )
    values = dict(
        rows=rows,
        status_copy="",
        empty_copy=empty_copy,
        selected_id="",
        preview_lines=(),
        query=query,
        pager=pager,
    )
    values.update(overrides)
    return LibraryConversationsCanvasState(**values)


async def test_conversations_canvas_title_uses_the_authoritative_pager_total(
    widget_pilot,
):
    """A partial final page must not relabel its total as the visible row count."""
    rows = (
        LibraryConversationRow(
            conversation_id="c1", title="Trip planning", secondary="3 messages"
        ),
        LibraryConversationRow(
            conversation_id="c2", title="Recipe ideas", secondary="1 message"
        ),
    )
    pager = build_library_pager_display(
        applied_page=2,
        requested_page=2,
        page_size=20,
        row_count=2,
        total=22,
        freshness="fresh",
    )
    async with await widget_pilot(
        LibraryConversationsCanvas, canvas=_state(rows=rows, pager=pager)
    ) as pilot:
        await pilot.pause()
        header = pilot.app.query_one("#library-conversations-title", Static)
        assert str(header.renderable) == "Conversations (22)"


async def test_conversations_canvas_title_uses_authoritative_zero_total(
    widget_pilot,
):
    pager = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=0,
        freshness="fresh",
    )
    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=_state(empty_copy="No conversations yet.", pager=pager),
    ) as pilot:
        await pilot.pause()
        header = pilot.app.query_one("#library-conversations-title", Static)
        assert str(header.renderable) == "Conversations (0)"


async def test_conversations_canvas_uninitialized_title_omits_unknown_count(
    widget_pilot,
):
    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=_state(empty_copy="Loading conversations."),
    ) as pilot:
        await pilot.pause()
        header = pilot.app.query_one("#library-conversations-title", Static)
        assert str(header.renderable) == "Conversations"


async def test_conversations_canvas_stale_actions_use_source_owned_recovery_reason(
    widget_pilot,
):
    stale_copy = "Source changed again; try again."
    pager = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=2,
        total=None,
        freshness="stale",
        stale_copy=stale_copy,
    )
    rows = (
        LibraryConversationRow(
            conversation_id="c1",
            title="Trip planning",
            secondary="3 messages",
            selected=True,
            checked=True,
        ),
        LibraryConversationRow(
            conversation_id="c2",
            title="Recipe ideas",
            secondary="1 message",
        ),
    )
    canvas = _state(
        rows=rows,
        status_copy=stale_copy,
        selected_id="c1",
        preview_lines=("Trip planning", "Messages: 3"),
        select_mode=True,
        selected_count=1,
        actions_disabled=True,
        pager=pager,
    )

    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=canvas,
    ) as pilot:
        await pilot.pause()
        stale_action_ids = (
            "library-conversations-export",
            "library-conversations-select-toggle",
            "library-conversations-select-all",
            "library-conversations-select-clear",
            "library-conversations-export-selected",
            "library-conversation-open-console",
        )
        for widget_id in stale_action_ids:
            button = pilot.app.query_one(f"#{widget_id}", Button)
            assert button.disabled is True
            assert str(button.tooltip) == stale_copy
            assert button.label.plain.startswith(LIBRARY_DISABLED_ACTION_MARKER)
        for row in pilot.app.query(".library-conversation-row"):
            assert row.disabled is True
            assert str(row.tooltip) == stale_copy

        previous = pilot.app.query_one("#library-conversations-previous", Button)
        next_page = pilot.app.query_one("#library-conversations-next", Button)
        assert str(previous.tooltip) == pager.previous_reason
        assert str(next_page.tooltip) == pager.next_reason
        assert pilot.app.query_one("#library-conversations-retry", Button).disabled is False
        assert pilot.app.query_one("#library-conversations-filter", Input).disabled is False


async def test_conversations_canvas_fresh_disabled_reasons_remain_specific(
    widget_pilot,
):
    empty_pager = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=0,
        total=0,
        freshness="fresh",
    )
    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=_state(empty_copy="No conversations yet.", pager=empty_pager),
    ) as pilot:
        await pilot.pause()
        select = pilot.app.query_one("#library-conversations-select-toggle", Button)
        assert str(select.tooltip) == LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP

    row = LibraryConversationRow(
        conversation_id="c1",
        title="Trip planning",
        secondary="3 messages",
        selected=True,
    )
    fresh_pager = build_library_pager_display(
        applied_page=1,
        requested_page=1,
        page_size=20,
        row_count=1,
        total=1,
        freshness="fresh",
    )
    async with await widget_pilot(
        LibraryConversationsCanvas,
        canvas=_state(
            rows=(row,),
            select_mode=True,
            selected_count=0,
            pager=fresh_pager,
        ),
    ) as pilot:
        await pilot.pause()
        export_selected = pilot.app.query_one(
            "#library-conversations-export-selected", Button
        )
        assert str(export_selected.tooltip) == (
            LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP
        )


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
