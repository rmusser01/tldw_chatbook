"""Pure display-state contracts for the Library Browse ▸ Conversations canvas."""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Library.library_conversations_state import (
    LIBRARY_CONVERSATIONS_EMPTY_COPY,
    LibraryConversationRow,
    LibraryConversationsCanvasState,
    build_library_conversations_state,
)

NOW = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)


def test_rows_are_sorted_by_recency_with_age_labels_and_missing_last():
    records = [
        {
            "id": "conv-b",
            "title": "Beta Chat",
            "updated_at": "2026-07-05T10:00:00+00:00",  # 2h old
            "message_count": 12,
        },
        {
            "id": "conv-c",
            "title": "Gamma Chat",
            "message_count": 3,
            # no updated_at at all
        },
        {
            "id": "conv-a",
            "title": "Alpha Chat",
            "updated_at": "2026-07-05T11:57:00+00:00",  # 3m old
            "message_count": 5,
        },
    ]

    state = build_library_conversations_state(records, now=NOW)

    assert isinstance(state, LibraryConversationsCanvasState)
    assert [row.conversation_id for row in state.rows] == ["conv-a", "conv-b", "conv-c"]
    assert state.rows[0].secondary == "5 messages - 3m"
    assert state.rows[1].secondary == "12 messages - 2h"
    # No age available -> no " - {age}" suffix.
    assert state.rows[2].secondary == "3 messages"
    for row in state.rows:
        assert isinstance(row, LibraryConversationRow)


def test_query_filters_case_insensitively_with_status_copy_singular_and_plural():
    records = [
        {"id": "1", "title": "Alpha Chat", "updated_at": "2026-07-05T11:00:00+00:00"},
        {"id": "2", "title": "Alpha Report", "updated_at": "2026-07-05T10:00:00+00:00"},
        {"id": "3", "title": "Beta Chat", "updated_at": "2026-07-05T09:00:00+00:00"},
    ]

    plural_state = build_library_conversations_state(records, query="alpha", now=NOW)
    assert [row.conversation_id for row in plural_state.rows] == ["1", "2"]
    assert plural_state.status_copy == "2 matches for 'alpha'"
    assert plural_state.empty_copy == ""

    singular_state = build_library_conversations_state(records, query="Beta", now=NOW)
    assert [row.conversation_id for row in singular_state.rows] == ["3"]
    assert singular_state.status_copy == "1 match for 'Beta'"


def test_query_with_no_matches_returns_empty_copy_and_zero_status_copy():
    records = [
        {"id": "1", "title": "Alpha Chat", "updated_at": "2026-07-05T11:00:00+00:00"},
    ]

    state = build_library_conversations_state(records, query="zzz", now=NOW)

    assert state.rows == ()
    assert state.status_copy == "0 matches for 'zzz'"
    assert state.empty_copy == "No conversations match 'zzz'."
    assert state.selected_id == ""
    assert state.preview_lines == ()


def test_no_records_yields_default_empty_copy_and_no_status_copy():
    state = build_library_conversations_state([], now=NOW)

    assert state.rows == ()
    assert state.status_copy == ""
    assert state.empty_copy == LIBRARY_CONVERSATIONS_EMPTY_COPY
    assert state.selected_id == ""
    assert state.preview_lines == ()
    assert state.query == ""


def test_selected_id_not_present_falls_back_to_first_row():
    records = [
        {"id": "conv-a", "title": "Alpha Chat", "updated_at": "2026-07-05T11:00:00+00:00"},
        {"id": "conv-b", "title": "Beta Chat", "updated_at": "2026-07-05T09:00:00+00:00"},
    ]

    state = build_library_conversations_state(records, selected_id="does-not-exist", now=NOW)

    assert state.selected_id == "conv-a"
    assert state.rows[0].selected is True
    assert state.rows[1].selected is False


def test_preview_lines_for_selected_row():
    records = [
        {
            "id": "conv-a",
            "title": "Alpha Chat",
            "updated_at": "2026-07-05T11:57:00+00:00",  # 3m old
            "message_count": 5,
        },
        {
            "id": "conv-b",
            "title": "Beta Chat",
            # No updated_at and no message count -> both unknown in preview.
        },
    ]

    selected_a = build_library_conversations_state(records, selected_id="conv-a", now=NOW)
    assert selected_a.preview_lines == (
        "Alpha Chat",
        "Messages: 5",
        "Updated: 3m",
    )

    selected_b = build_library_conversations_state(records, selected_id="conv-b", now=NOW)
    assert selected_b.preview_lines == (
        "Beta Chat",
        "Messages: unknown",
        "Updated: unknown",
    )


def test_limit_truncates_rows_to_max_after_sorting():
    records = [
        {
            "id": f"conv-{i}",
            "title": f"Chat {i}",
            "updated_at": f"2026-07-05T{11 - i:02d}:00:00+00:00",
        }
        for i in range(5)
    ]

    state = build_library_conversations_state(records, now=NOW, limit=2)

    # Most recent two: conv-0 (11:00) and conv-1 (10:00).
    assert [row.conversation_id for row in state.rows] == ["conv-0", "conv-1"]
    assert len(state.rows) == 2


def test_id_title_count_key_fallbacks_using_conversation_id_and_messages_total():
    records = [
        {
            "conversation_id": "cid-99",
            "title": "  Fallback Chat  ",
            "messages_total": 7,
            "updated_at": "2026-07-05T11:57:00+00:00",
        },
    ]

    state = build_library_conversations_state(records, now=NOW)

    assert len(state.rows) == 1
    row = state.rows[0]
    assert row.conversation_id == "cid-99"
    assert row.title == "Fallback Chat"
    assert row.secondary == "7 messages - 3m"
