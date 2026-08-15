"""Pure display-state contracts for the Library Browse ▸ Conversations canvas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import tldw_chatbook.Library.library_conversations_state as conversations_state
from tldw_chatbook.Library.library_conversations_state import (
    LIBRARY_CONVERSATIONS_EMPTY_COPY,
    LibraryConversationRow,
    LibraryConversationsCanvasState,
    build_library_conversations_state,
    validate_library_conversation_page,
)
from tldw_chatbook.Library.library_pager_state import (
    LibraryPagerDisplay,
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


def test_rows_preserve_authoritative_service_order_with_age_labels():
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

    state = build_library_conversations_state(records, total_count=3, now=NOW)

    assert isinstance(state, LibraryConversationsCanvasState)
    assert [row.conversation_id for row in state.rows] == ["conv-b", "conv-c", "conv-a"]
    assert state.rows[0].secondary == "12 messages - 2h"
    # No age available -> no " - {age}" suffix.
    assert state.rows[1].secondary == "3 messages"
    assert state.rows[2].secondary == "5 messages - 3m"
    for row in state.rows:
        assert isinstance(row, LibraryConversationRow)


def test_uninitialized_state_does_not_fabricate_exact_metadata():
    state = build_library_conversations_state(
        [],
        total_known=False,
        freshness="uninitialized",
        now=NOW,
    )

    assert state.pager.title_count is None
    assert state.range_copy == "No page loaded · Total unavailable"
    assert state.page_copy == ""
    assert state.empty_copy == ""


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
    state = build_library_conversations_state([], total_count=0, now=NOW)

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
        records, selected_id="does-not-exist", total_count=2, now=NOW
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
        records, selected_id="conv-a", total_count=2, now=NOW
    )
    assert selected_a.preview_lines == (
        "Alpha Chat",
        "Messages: 5",
        "Updated: 3m",
    )

    selected_b = build_library_conversations_state(
        records, selected_id="conv-b", total_count=2, now=NOW
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

    state = build_library_conversations_state(records, total_count=1, now=NOW)

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


def test_loading_and_error_preserve_last_good_rows_and_metadata():
    records = [
        {"id": f"conv-{index}", "title": f"Last successful row {index}"}
        for index in range(20)
    ]

    loading = build_library_conversations_state(
        records,
        page=2,
        requested_page=3,
        page_size=20,
        total_count=45,
        total_known=True,
        loading=True,
        now=NOW,
    )
    failed = build_library_conversations_state(
        records,
        page=2,
        requested_page=3,
        page_size=20,
        total_count=45,
        total_known=True,
        error_copy="Couldn't load page 3.",
        now=NOW,
    )

    assert len(loading.rows) == 20
    assert loading.status_copy == "Loading page 3…"
    assert loading.range_copy == "21-40 of 45"
    assert loading.page_copy == "Page 2 of 3"
    assert loading.previous_disabled is True
    assert loading.next_disabled is True
    assert len(failed.rows) == 20
    assert failed.status_copy == "Couldn't load page 3."
    assert failed.range_copy == "21-40 of 45"
    assert failed.pager.retry_visible is True
    assert failed.empty_copy == ""


def test_stale_state_suppresses_exact_metadata_and_actions():
    state = build_library_conversations_state(
        [{"id": "conv-1", "title": "One"}],
        page=3,
        page_size=20,
        total_count=None,
        total_known=False,
        freshness="stale",
        stale_copy="Source changed again; try again.",
        selection_notice="Selection cleared.",
        now=NOW,
    )

    assert state.pager.title_count is None
    assert state.range_copy == "List may be out of date"
    assert state.page_copy == ""
    assert state.status_copy == "Source changed again; try again."
    assert state.previous_disabled is True
    assert state.next_disabled is True
    assert state.pager.retry_visible is True
    assert state.actions_disabled is True
    assert state.selection_notice == "Selection cleared."


def test_initial_failure_does_not_claim_the_library_is_empty():
    state = build_library_conversations_state(
        [],
        page=1,
        page_size=20,
        total_count=0,
        total_known=False,
        freshness="uninitialized",
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

    state = build_library_conversations_state(records, total_count=1, now=NOW)

    assert len(state.rows) == 1
    row = state.rows[0]
    # Contract: fallback replaces the whole secondary, age is NOT appended
    assert row.secondary == "conversation"


def test_valid_identified_record_preserves_harmless_field_fallbacks():
    records = [
        {
            "id": "invalid",
            "title": None,
            "updated_at": "garbage",
            "message_count": "NaN",
        },
    ]

    state = build_library_conversations_state(records, total_count=1, now=NOW)

    assert len(state.rows) == 1
    invalid_row = state.rows[0]
    assert invalid_row.title == "Untitled conversation"
    assert invalid_row.secondary == "conversation"


def _conversation_response(
    items: object,
    *,
    limit: object = 20,
    offset: object = 0,
    total: object = 1,
    has_more: object = False,
) -> dict[str, object]:
    return {
        "items": items,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_more": has_more,
        },
    }


def test_validation_rejects_missing_stable_conversation_identity():
    with pytest.raises(ValueError, match="stable conversation identity"):
        validate_library_conversation_page(
            _conversation_response([{"title": "missing id"}]),
            requested_limit=20,
            requested_offset=0,
        )


@pytest.mark.parametrize(
    "items",
    [
        [None],
        ["not-a-mapping"],
        [{"id": ""}],
        [{"id": "   "}],
        [{"id": 7, "conversation_id": "must-not-mask-invalid-id"}],
        [{"id": "duplicate"}, {"id": "duplicate"}],
    ],
)
def test_validation_rejects_nonmapping_invalid_or_duplicate_items(items: list[object]):
    with pytest.raises(ValueError, match="stable conversation identity"):
        validate_library_conversation_page(
            _conversation_response(items, total=len(items)),
            requested_limit=20,
            requested_offset=0,
        )


@pytest.mark.parametrize("items", [None, {}, "rows", ({"id": "one"},)])
def test_validation_requires_items_list(items: object):
    with pytest.raises(ValueError, match="items"):
        validate_library_conversation_page(
            _conversation_response(items),
            requested_limit=20,
            requested_offset=0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", True),
        ("offset", False),
        ("total", True),
        ("limit", 20.0),
        ("offset", "0"),
        ("total", None),
        ("limit", 0),
        ("limit", -1),
        ("offset", -1),
        ("total", -1),
        ("limit", 2**63),
        ("offset", 2**63),
        ("total", 2**63),
    ],
)
def test_validation_rejects_invalid_pagination_integers(field: str, value: object):
    pagination = {"limit": 20, "offset": 0, "total": 1}
    pagination[field] = value
    with pytest.raises(ValueError, match=field):
        validate_library_conversation_page(
            _conversation_response(
                [{"id": "one"}],
                limit=pagination["limit"],
                offset=pagination["offset"],
                total=pagination["total"],
            ),
            requested_limit=20,
            requested_offset=0,
        )


@pytest.mark.parametrize(
    ("requested_limit", "requested_offset"),
    [
        (True, 0),
        (0, 0),
        (-1, 0),
        (2**63, 0),
        (20, True),
        (20, -1),
        (20, 2**63),
    ],
)
def test_validation_rejects_invalid_requested_coordinates(
    requested_limit: object,
    requested_offset: object,
):
    with pytest.raises(ValueError, match="requested_(limit|offset)"):
        validate_library_conversation_page(
            _conversation_response([{"id": "one"}]),
            requested_limit=requested_limit,  # type: ignore[arg-type]
            requested_offset=requested_offset,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("limit", "offset", "requested_limit", "requested_offset"),
    [(10, 0, 20, 0), (20, 20, 20, 0)],
)
def test_validation_rejects_unequal_coordinate_echoes(
    limit: int,
    offset: int,
    requested_limit: int,
    requested_offset: int,
):
    with pytest.raises(ValueError, match="echo"):
        validate_library_conversation_page(
            _conversation_response([], limit=limit, offset=offset, total=0),
            requested_limit=requested_limit,
            requested_offset=requested_offset,
        )


def test_validation_rejects_undersized_nonfinal_page():
    items = [{"id": f"conv-{index}"} for index in range(19)]

    with pytest.raises(ValueError, match="cardinality"):
        validate_library_conversation_page(
            _conversation_response(items, total=45, has_more=True),
            requested_limit=20,
            requested_offset=0,
        )


def test_validation_rejects_oversized_page():
    with pytest.raises(ValueError, match="cardinality"):
        validate_library_conversation_page(
            _conversation_response(
                [{"id": "one"}, {"id": "two"}],
                limit=1,
                total=2,
                has_more=True,
            ),
            requested_limit=1,
            requested_offset=0,
        )


def test_validation_rejects_ordinary_offset_past_the_owning_page():
    with pytest.raises(ValueError, match="offset.*range"):
        validate_library_conversation_page(
            _conversation_response([], offset=40, total=21),
            requested_limit=20,
            requested_offset=40,
        )


@pytest.mark.parametrize(
    ("items", "total", "has_more"),
    [
        ([{"id": f"conv-{index}"} for index in range(20)], 20, True),
        ([{"id": f"conv-{index}"} for index in range(20)], 21, False),
    ],
)
def test_validation_rejects_has_more_disagreement(
    items: list[dict[str, str]], total: int, has_more: bool
):
    with pytest.raises(ValueError, match="has_more"):
        validate_library_conversation_page(
            _conversation_response(items, total=total, has_more=has_more),
            requested_limit=20,
            requested_offset=0,
        )


@pytest.mark.parametrize("has_more", [None, 0, 1, "false"])
def test_validation_requires_boolean_has_more(has_more: object):
    with pytest.raises(ValueError, match="has_more"):
        validate_library_conversation_page(
            _conversation_response([{"id": "one"}], has_more=has_more),
            requested_limit=20,
            requested_offset=0,
        )


def test_validation_accepts_fresh_empty_collection():
    page = validate_library_conversation_page(
        _conversation_response([], total=0),
        requested_limit=20,
        requested_offset=0,
    )

    assert page.items == ()
    assert page.limit == 20
    assert page.offset == 0
    assert page.total == 0
    assert page.has_more is False


def test_validation_preserves_exact_service_item_order():
    items = [
        {"id": "older", "updated_at": "2025-01-01T00:00:00Z"},
        {"id": "newer", "updated_at": "2026-01-01T00:00:00Z"},
    ]

    page = validate_library_conversation_page(
        _conversation_response(items, total=2),
        requested_limit=20,
        requested_offset=0,
    )
    state = build_library_conversations_state(
        page.items,
        page_size=page.limit,
        total_count=page.total,
        now=NOW,
    )

    assert [item["id"] for item in page.items] == ["older", "newer"]
    assert [row.conversation_id for row in state.rows] == ["older", "newer"]


def test_builder_rejects_malformed_rows_before_rendering():
    with pytest.raises(ValueError, match="stable conversation identity"):
        build_library_conversations_state(
            [{"id": "one"}, {}],
            total_count=2,
            now=NOW,
        )


def test_builder_delegates_all_pager_projection_to_pure_display(monkeypatch):
    sentinel = LibraryPagerDisplay(
        title_count=99,
        range_copy="sentinel range",
        page_copy="sentinel page",
        status_copy="sentinel status",
        previous_disabled=False,
        next_disabled=True,
        previous_reason="sentinel previous",
        next_reason="sentinel next",
        retry_visible=True,
    )
    captured: dict[str, object] = {}

    def fake_build_library_pager_display(**kwargs: object) -> LibraryPagerDisplay:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        conversations_state,
        "build_library_pager_display",
        fake_build_library_pager_display,
    )

    state = build_library_conversations_state(
        [{"id": "one"}],
        page=2,
        requested_page=3,
        page_size=20,
        total_count=21,
        loading=True,
        now=NOW,
    )

    assert captured == {
        "applied_page": 2,
        "requested_page": 3,
        "page_size": 20,
        "row_count": 1,
        "total": 21,
        "freshness": "fresh",
        "loading": True,
        "error_copy": "",
        "stale_copy": "",
    }
    assert state.pager is sentinel
    assert state.range_copy == sentinel.range_copy
    assert state.page_copy == sentinel.page_copy
    assert state.previous_disabled is sentinel.previous_disabled
    assert state.next_disabled is sentinel.next_disabled
