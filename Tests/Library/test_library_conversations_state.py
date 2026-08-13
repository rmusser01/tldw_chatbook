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


def test_canvas_state_direct_construction_keeps_safe_pager_defaults():
    state = LibraryConversationsCanvasState(
        rows=(),
        status_copy="",
        empty_copy="",
        selected_id="",
        preview_lines=(),
        query="",
    )

    assert state.range_copy == ""
    assert state.page_copy == ""
    assert state.previous_disabled is True
    assert state.next_disabled is True


def test_canvas_state_positional_construction_keeps_original_optional_slots():
    state = LibraryConversationsCanvasState((), "", "", "", (), "", True, 2)

    assert state.select_mode is True
    assert state.selected_count == 2
    assert state.range_copy == ""
    assert state.page_copy == ""
    assert state.previous_disabled is True
    assert state.next_disabled is True


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


def test_query_uses_supplied_matching_page_with_status_copy_singular_and_plural():
    records = [
        {"id": "1", "title": "Alpha Chat", "updated_at": "2026-07-05T11:00:00+00:00"},
        {"id": "2", "title": "Alpha Report", "updated_at": "2026-07-05T10:00:00+00:00"},
    ]

    plural_state = build_library_conversations_state(
        records, query="alpha", total_count=2, now=NOW
    )
    assert [row.conversation_id for row in plural_state.rows] == ["1", "2"]
    assert plural_state.status_copy == "2 matches for 'alpha'"
    assert plural_state.empty_copy == ""

    singular_state = build_library_conversations_state(
        [{"id": "3", "title": "Beta Chat"}],
        query="Beta",
        total_count=1,
        now=NOW,
    )
    assert [row.conversation_id for row in singular_state.rows] == ["3"]
    assert singular_state.status_copy == "1 match for 'Beta'"


def test_query_keeps_nonmatching_rows_from_the_supplied_service_page():
    state = build_library_conversations_state(
        [
            {"id": "alpha", "title": "Alpha Chat"},
            {"id": "beta", "title": "Beta Chat"},
        ],
        query="alpha",
        total_count=2,
        now=NOW,
    )

    assert [row.conversation_id for row in state.rows] == ["alpha", "beta"]


def test_query_with_no_matches_returns_empty_copy_and_zero_status_copy():
    state = build_library_conversations_state(
        [], query="zzz", total_count=0, now=NOW
    )

    assert state.rows == ()
    assert state.status_copy == "0 matches for 'zzz'"
    assert state.empty_copy == "No conversations match 'zzz'."
    assert state.selected_id == ""
    assert state.preview_lines == ()


def test_empty_copy_matches_console_vocabulary():
    """Console chats appear in Library without an explicit save step, and the
    copy must not promise otherwise (task-179 vocabulary alignment)."""
    assert LIBRARY_CONVERSATIONS_EMPTY_COPY == (
        "No conversations yet. Chat in Console and it appears here."
    )


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
        {
            "id": "conv-a",
            "title": "Alpha Chat",
            "updated_at": "2026-07-05T11:00:00+00:00",
        },
        {
            "id": "conv-b",
            "title": "Beta Chat",
            "updated_at": "2026-07-05T09:00:00+00:00",
        },
    ]

    state = build_library_conversations_state(
        records, selected_id="does-not-exist", now=NOW
    )

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

    selected_a = build_library_conversations_state(
        records, selected_id="conv-a", now=NOW
    )
    assert selected_a.preview_lines == (
        "Alpha Chat",
        "Messages: 5",
        "Updated: 3m",
    )

    selected_b = build_library_conversations_state(
        records, selected_id="conv-b", now=NOW
    )
    assert selected_b.preview_lines == (
        "Beta Chat",
        "Messages: unknown",
        "Updated: unknown",
    )


def test_middle_page_exposes_range_page_and_enabled_navigation():
    records = [
        {
            "id": f"conv-{index}",
            "title": f"Chat {index}",
            "updated_at": f"2026-07-05T{index % 12:02d}:00:00+00:00",
        }
        for index in range(20)
    ]

    state = build_library_conversations_state(
        records,
        page=2,
        page_size=20,
        total_count=47,
        total_known=True,
        has_more=True,
        now=NOW,
    )

    assert len(state.rows) == 20
    assert state.range_copy == "21-40 of 47"
    assert state.page_copy == "Page 2 of 3"
    assert state.previous_disabled is False
    assert state.next_disabled is False


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


def test_final_page_disables_next_without_dropping_supplied_rows():
    records = [
        {"id": f"conv-{index}", "title": f"Chat {index}"}
        for index in range(7)
    ]

    state = build_library_conversations_state(
        records,
        page=3,
        page_size=20,
        total_count=47,
        total_known=True,
        has_more=False,
        now=NOW,
    )

    assert len(state.rows) == 7
    assert state.range_copy == "41-47 of 47"
    assert state.page_copy == "Page 3 of 3"
    assert state.previous_disabled is False
    assert state.next_disabled is True


def test_page_size_does_not_truncate_supplied_service_page_rows():
    state = build_library_conversations_state(
        [
            {"id": "one", "title": "One"},
            {"id": "two", "title": "Two"},
            {"id": "three", "title": "Three"},
        ],
        page_size=2,
        total_count=3,
        now=NOW,
    )

    assert [row.conversation_id for row in state.rows] == ["one", "two", "three"]


def test_empty_filtered_page_reports_zero_matches_and_page_one_of_one():
    state = build_library_conversations_state(
        [],
        query="missing",
        page=1,
        page_size=20,
        total_count=0,
        total_known=True,
        has_more=False,
        now=NOW,
    )

    assert state.status_copy == "0 matches for 'missing'"
    assert state.empty_copy == "No conversations match 'missing'."
    assert state.range_copy == "0 of 0"
    assert state.page_copy == "Page 1 of 1"
    assert state.previous_disabled is True
    assert state.next_disabled is True


def test_query_status_uses_full_service_total_not_current_page_length():
    records = [{"id": f"conv-{index}", "title": "Alpha"} for index in range(20)]

    state = build_library_conversations_state(
        records,
        query="alpha",
        page=2,
        page_size=20,
        total_count=43,
        total_known=True,
        has_more=True,
        now=NOW,
    )

    assert len(state.rows) == 20
    assert state.status_copy == "43 matches for 'alpha'"


def test_loading_and_error_preserve_rows_and_disable_navigation():
    records = [{"id": "conv-1", "title": "Last successful row"}]

    loading = build_library_conversations_state(
        records,
        page=1,
        page_size=20,
        total_count=2,
        total_known=True,
        has_more=True,
        loading=True,
        now=NOW,
    )
    failed = build_library_conversations_state(
        records,
        page=1,
        page_size=20,
        total_count=2,
        total_known=True,
        has_more=True,
        error_copy="Couldn't load conversations. Try again.",
        now=NOW,
    )

    assert [row.conversation_id for row in loading.rows] == ["conv-1"]
    assert loading.status_copy == "Loading conversations…"
    assert loading.previous_disabled is True
    assert loading.next_disabled is True
    assert [row.conversation_id for row in failed.rows] == ["conv-1"]
    assert failed.status_copy == "Couldn't load conversations. Try again."
    assert failed.empty_copy == ""


def test_unknown_total_disables_next_without_explicit_has_more():
    state = build_library_conversations_state(
        [{"id": "conv-1", "title": "One"}],
        page=1,
        page_size=20,
        total_count=1,
        total_known=False,
        has_more=False,
        now=NOW,
    )

    assert state.range_copy == "1-1"
    assert state.page_copy == "Page 1"
    assert state.next_disabled is True


def test_initial_failure_does_not_claim_the_library_is_empty():
    state = build_library_conversations_state(
        [],
        page=1,
        page_size=20,
        total_count=0,
        total_known=False,
        has_more=False,
        error_copy="Couldn't load conversations. Try again.",
        now=NOW,
    )

    assert state.status_copy == "Couldn't load conversations. Try again."
    assert state.empty_copy == ""


def test_secondary_fallback_is_conversation_when_no_message_count():
    """Record with no message_count key has secondary='conversation' (age not appended)."""
    records = [
        {
            "id": "conv-a",
            "title": "Chat With No Count",
            "updated_at": "2026-07-05T11:57:00+00:00",  # Has age, but...
            # No message_count key at all
        },
    ]

    state = build_library_conversations_state(records, now=NOW)

    assert len(state.rows) == 1
    row = state.rows[0]
    # Contract: fallback replaces the whole secondary, age is NOT appended
    assert row.secondary == "conversation"


def test_tolerates_invalid_and_missing_records():
    """Records with None, non-mapping, empty dict, invalid fields are skipped gracefully."""
    records = [
        None,  # Not a mapping
        "not-a-mapping",  # String, not a mapping
        {},  # No id key -> skipped
        {
            "id": "invalid",
            "title": None,  # Invalid title -> uses fallback
            "updated_at": "garbage",  # Invalid timestamp -> parsed as None
            "message_count": "NaN",  # Invalid count -> parsed as None
        },
        {
            "id": "valid",
            "title": "Valid Chat",
            "updated_at": "2026-07-05T11:57:00+00:00",
            "message_count": 5,
        },
    ]

    # Should not raise, should return only the valid record
    state = build_library_conversations_state(records, now=NOW)

    assert len(state.rows) == 2
    valid_ids = [row.conversation_id for row in state.rows]
    assert "invalid" in valid_ids
    assert "valid" in valid_ids
    # Check that "invalid" record was processed with fallbacks
    invalid_row = next(r for r in state.rows if r.conversation_id == "invalid")
    assert invalid_row.title == "Untitled conversation"
    assert invalid_row.secondary == "conversation"
