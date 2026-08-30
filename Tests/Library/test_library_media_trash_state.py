"""Pure display-state contracts for the Library media Trash view (task-4025).

The Trash view is the third ``_library_media_view`` value of the Browse ▸
Media canvas ("list"/"viewer"/"trash") — these tests pin its pure state
builder the same way ``test_library_media_state.py`` pins the list's.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from tldw_chatbook.Library.library_media_state import (
    LIBRARY_MEDIA_TRASH_EMPTY_COPY,
    LibraryMediaTrashRow,
    LibraryMediaTrashState,
    MediaTrashBrowseState,
    MediaTrashMutationTarget,
    MediaTrashScope,
    apply_media_trash_result,
    begin_media_trash_mutation,
    begin_media_trash_request,
    build_media_trash_result,
    build_library_media_trash_state,
    cancel_media_trash_delete_confirmation,
    commit_media_trash_mutation,
    fail_media_trash_mutation,
    fail_media_trash_request,
    open_media_trash_delete_confirmation,
    select_media_trash_item,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _trash_item(media_id: int) -> dict[str, object]:
    return {
        "id": f"local:media:{media_id}",
        "backing_media_id": media_id,
        "title": f"Trash {media_id}",
        "media_type": "pdf",
        "trash_date": "2026-08-11T11:57:00+00:00",
    }


def _trash_page(
    *,
    scope: MediaTrashScope = MediaTrashScope(),
    total: int = 1,
    items: list[dict[str, object]] | None = None,
    types: list[str] | None = None,
) -> dict[str, object]:
    return {
        "items": [_trash_item(1)] if items is None else items,
        "total": total,
        "limit": 20,
        "offset": scope.offset,
        "types": ["pdf"] if types is None else types,
    }


def test_media_trash_scope_normalizes_and_bounds_coordinates():
    scope = MediaTrashScope(query="  doc  ", media_type=" pdf ", page=2)

    assert scope == MediaTrashScope(query="doc", media_type="pdf", page=2)
    assert scope.page_size == 20
    assert scope.offset == 20


@pytest.mark.parametrize("page", [True, False, 1.0, "1", None, 0, -1])
def test_media_trash_scope_rejects_invalid_pages(page: object):
    with pytest.raises((TypeError, ValueError), match="page"):
        MediaTrashScope(page=page)  # type: ignore[arg-type]


def test_media_trash_scope_rejects_offset_overflow_and_invalid_queries():
    with pytest.raises(ValueError, match="offset"):
        MediaTrashScope(page=(2**63 - 1) // 20 + 2)
    with pytest.raises(ValueError, match="query"):
        MediaTrashScope(query="nul\x00query")
    with pytest.raises(ValueError, match="200"):
        MediaTrashScope(query="x" * 201)


@pytest.mark.parametrize("media_type", ["", "   "])
def test_media_trash_scope_normalizes_blank_type_to_none(media_type: str):
    assert MediaTrashScope(media_type=media_type).media_type is None


def test_media_trash_result_is_exact_and_immutable():
    raw_item = _trash_item(1)
    payload = _trash_page(items=[raw_item])

    result = build_media_trash_result(MediaTrashScope(), payload)
    raw_item["title"] = "mutated"
    payload["types"].append("video")  # type: ignore[union-attr]

    assert isinstance(result.items[0], MappingProxyType)
    assert result.items[0]["title"] == "Trash 1"
    assert result.types == ("pdf",)
    with pytest.raises(TypeError):
        result.items[0]["title"] = "blocked"  # type: ignore[index]


@pytest.mark.parametrize("key", ["items", "total", "limit", "offset", "types"])
def test_media_trash_result_requires_exact_five_key_envelope(key: str):
    missing = _trash_page()
    del missing[key]
    with pytest.raises(ValueError, match="exactly five"):
        build_media_trash_result(MediaTrashScope(), missing)

    extra = _trash_page()
    extra["private"] = "sentinel"
    with pytest.raises(ValueError, match="exactly five"):
        build_media_trash_result(MediaTrashScope(), extra)


@pytest.mark.parametrize(
    "item",
    [
        {"id": "local:media:1"},
        {**_trash_item(1), "private": "sentinel"},
    ],
)
def test_media_trash_result_requires_exact_five_item_keys(item: dict[str, object]):
    with pytest.raises(ValueError, match="exactly five"):
        build_media_trash_result(MediaTrashScope(), _trash_page(items=[item]))


@pytest.mark.parametrize("backing_id", [True, False, 0, -1, 1.0, "1"])
def test_media_trash_result_requires_positive_non_bool_backing_ids(
    backing_id: object,
):
    item = _trash_item(1)
    item["backing_media_id"] = backing_id
    with pytest.raises(ValueError, match="backing_media_id"):
        build_media_trash_result(MediaTrashScope(), _trash_page(items=[item]))


def test_media_trash_result_rejects_duplicate_or_noncanonical_ids():
    items = [_trash_item(1), _trash_item(2)]
    items[1]["id"] = items[0]["id"]
    items[1]["backing_media_id"] = items[0]["backing_media_id"]
    with pytest.raises(ValueError, match="unique"):
        build_media_trash_result(MediaTrashScope(), _trash_page(total=2, items=items))

    item = _trash_item(1)
    item["id"] = "media:1"
    with pytest.raises(ValueError, match="canonical"):
        build_media_trash_result(MediaTrashScope(), _trash_page(items=[item]))


@pytest.mark.parametrize(
    ("field", "value", "copy"),
    [
        ("title", "", "title"),
        ("title", "   ", "title"),
        ("media_type", " pdf ", "media_type"),
        ("media_type", "", "media_type"),
        ("trash_date", "not-a-date", "trash_date"),
        ("trash_date", " 2026-08-11T00:00:00+00:00 ", "trash_date"),
        ("trash_date", 1, "trash_date"),
    ],
)
def test_media_trash_result_rejects_malformed_summary_values(
    field: str, value: object, copy: str
):
    item = _trash_item(1)
    item[field] = value
    with pytest.raises((TypeError, ValueError), match=copy):
        build_media_trash_result(MediaTrashScope(), _trash_page(items=[item]))


def test_media_trash_result_rejects_iso_date_without_time():
    item = _trash_item(1)
    item["trash_date"] = "2026-08-30"

    with pytest.raises(ValueError, match="trash_date"):
        build_media_trash_result(MediaTrashScope(), _trash_page(items=[item]))


@pytest.mark.parametrize(
    "trash_date", [None, "2026-08-11T00:00:00", "2026-08-11T00:00:00Z"]
)
def test_media_trash_result_accepts_iso_or_none_trash_date(
    trash_date: str | None,
):
    item = _trash_item(1)
    item["trash_date"] = trash_date

    result = build_media_trash_result(MediaTrashScope(), _trash_page(items=[item]))

    assert result.items[0]["trash_date"] == trash_date


def test_media_trash_result_requires_exact_page_cardinality_and_coordinates():
    with pytest.raises(ValueError, match="count"):
        build_media_trash_result(MediaTrashScope(), _trash_page(total=2))

    wrong_limit = _trash_page()
    wrong_limit["limit"] = 19
    with pytest.raises(ValueError, match="limit"):
        build_media_trash_result(MediaTrashScope(), wrong_limit)

    wrong_offset = _trash_page()
    wrong_offset["offset"] = 20
    with pytest.raises(ValueError, match="offset"):
        build_media_trash_result(MediaTrashScope(), wrong_offset)


@pytest.mark.parametrize(
    "types",
    [["video", "pdf"], ["pdf", "pdf"], ["pdf", ""], [" pdf"]],
)
def test_media_trash_result_requires_sorted_unique_nonblank_facets(types: list[str]):
    with pytest.raises(ValueError, match="types"):
        build_media_trash_result(MediaTrashScope(), _trash_page(types=types))


def test_media_trash_result_detects_out_of_range_scope():
    scope = MediaTrashScope(page=3)
    result = build_media_trash_result(
        scope,
        _trash_page(scope=scope, total=21, items=[]),
    )

    assert result.last_page == 2
    assert result.out_of_range is True


def _trash_result(
    scope: MediaTrashScope = MediaTrashScope(),
    *,
    total: int = 2,
) -> object:
    offset = scope.offset
    count = min(20, max(total - offset, 0))
    items = [_trash_item(offset + index + 1) for index in range(count)]
    return build_media_trash_result(
        scope,
        _trash_page(scope=scope, total=total, items=items),
    )


def test_entry_result_selects_first_row_only_for_entry_authority():
    state = begin_media_trash_request(
        MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
    )
    applied = apply_media_trash_result(state, _trash_result())

    assert applied.selected_id == "local:media:1"
    assert applied.applied_result is not None
    assert applied.applied_result.scope == MediaTrashScope()
    assert applied.retained_items == applied.applied_result.items
    assert applied.types == ("pdf",)
    assert applied.freshness == "fresh"
    assert applied.loading is False

    superseded = begin_media_trash_request(
        applied, MediaTrashScope(query="new"), origin="search"
    )
    searched = apply_media_trash_result(
        superseded, _trash_result(MediaTrashScope(query="new"))
    )
    assert searched.selected_id == ""


@pytest.mark.parametrize("origin", ["search", "type", "previous", "next", "mutation"])
def test_non_entry_result_leaves_selection_empty(origin: str):
    state = begin_media_trash_request(
        MediaTrashBrowseState(),
        MediaTrashScope(),
        origin=origin,  # type: ignore[arg-type]
    )

    applied = apply_media_trash_result(state, _trash_result())

    assert applied.selected_id == ""


def test_new_scope_clears_selection_before_request_applies():
    entered = apply_media_trash_result(
        begin_media_trash_request(
            MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
        ),
        _trash_result(),
    )
    selected = select_media_trash_item(entered, "local:media:2")

    requested = begin_media_trash_request(
        selected, MediaTrashScope(page=2), origin="next"
    )

    assert requested.selected_id == ""
    assert requested.confirmation_target is None
    assert requested.loading is True
    assert requested.applied_result is entered.applied_result
    assert requested.retained_items is entered.retained_items


@pytest.mark.parametrize(
    ("scope", "origin", "copy"),
    [
        (
            MediaTrashScope(query="failed"),
            "search",
            "Filter not applied — showing All Trash.",
        ),
        (
            MediaTrashScope(page=2),
            "next",
            "Page 2 not loaded — showing page 1.",
        ),
    ],
)
def test_failed_filter_or_page_retains_prior_fresh_page_and_retry_target(
    scope: MediaTrashScope,
    origin: str,
    copy: str,
):
    entered = apply_media_trash_result(
        begin_media_trash_request(
            MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
        ),
        _trash_result(),
    )
    retained = entered.retained_items
    requested = begin_media_trash_request(
        entered,
        scope,
        origin=origin,  # type: ignore[arg-type]
    )

    failed = fail_media_trash_request(requested, scope, copy=copy)

    assert failed.requested_scope == scope
    assert failed.applied_result is entered.applied_result
    assert failed.retained_items is retained
    assert failed.freshness == "fresh"
    assert failed.loading is False
    assert failed.error_copy == copy
    assert failed.failed_scope == scope
    assert failed.failed_origin == origin
    assert failed.selected_id == ""


def test_selection_preserves_failed_request_copy_and_retry_target():
    entered = apply_media_trash_result(
        begin_media_trash_request(
            MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
        ),
        _trash_result(),
    )
    failed_scope = MediaTrashScope(page=2)
    failed = fail_media_trash_request(
        begin_media_trash_request(entered, failed_scope, origin="next"),
        failed_scope,
        copy="Page 2 not loaded — showing page 1.",
    )

    selected = select_media_trash_item(failed, "local:media:2")

    assert selected.selected_id == "local:media:2"
    assert selected.error_copy == "Page 2 not loaded — showing page 1."
    assert selected.failed_scope == failed_scope
    assert selected.failed_origin == "next"


def test_initial_failure_has_no_rows_or_fabricated_freshness():
    scope = MediaTrashScope()
    requested = begin_media_trash_request(
        MediaTrashBrowseState(), scope, origin="entry"
    )

    failed = fail_media_trash_request(requested, scope, copy="Could not load Trash.")

    assert failed.applied_result is None
    assert failed.retained_items == ()
    assert failed.types == ()
    assert failed.freshness == "uninitialized"
    assert failed.error_copy == "Could not load Trash."


def test_confirmation_captures_full_immutable_selected_identity():
    entered = apply_media_trash_result(
        begin_media_trash_request(
            MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
        ),
        _trash_result(),
    )
    selected = select_media_trash_item(entered, "local:media:2")

    confirming = open_media_trash_delete_confirmation(selected)

    assert confirming.confirmation_target == MediaTrashMutationTarget(
        stable_id="local:media:2",
        backing_media_id=2,
        title="Trash 2",
        media_type="pdf",
        trash_date="2026-08-11T11:57:00+00:00",
        page_index=1,
    )
    assert (
        cancel_media_trash_delete_confirmation(confirming).confirmation_target is None
    )


def test_precommit_mutation_failure_keeps_row_selection_and_fresh_boundary():
    entered = apply_media_trash_result(
        begin_media_trash_request(
            MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
        ),
        _trash_result(),
    )
    confirming = open_media_trash_delete_confirmation(entered)
    target = confirming.confirmation_target
    assert target is not None

    pending = begin_media_trash_mutation(confirming)
    failed = fail_media_trash_mutation(
        pending, target, copy="Could not delete this item."
    )

    assert failed.retained_items is entered.retained_items
    assert failed.selected_id == target.stable_id
    assert failed.freshness == "fresh"
    assert failed.mutation_pending is False
    assert failed.error_copy == "Could not delete this item."


def test_committed_mutation_removes_target_and_becomes_stale_loading():
    entered = apply_media_trash_result(
        begin_media_trash_request(
            MediaTrashBrowseState(), MediaTrashScope(), origin="entry"
        ),
        _trash_result(),
    )
    confirming = open_media_trash_delete_confirmation(entered)
    target = confirming.confirmation_target
    assert target is not None
    pending = begin_media_trash_mutation(confirming)

    committed = commit_media_trash_mutation(
        pending, target, notice="Deleted 'Trash 1' permanently."
    )

    assert [item["id"] for item in committed.retained_items] == ["local:media:2"]
    assert committed.applied_result is entered.applied_result
    assert committed.freshness == "stale"
    assert committed.loading is True
    assert committed.stale_copy == "List may be out of date."
    assert committed.selected_id == ""
    assert committed.mutation_pending is False
    assert committed.committed_notice == "Deleted 'Trash 1' permanently."


def test_rows_preserve_seam_order_with_trashed_age_secondary():
    """Rows keep the seam's own trash_date-DESC order (never re-sorted) and
    the secondary reads '{type} · trashed {age}' when a trash_date exists."""
    records = [
        {
            "id": "9",
            "title": "Newest trashed",
            "type": "pdf",
            "trash_date": "2026-08-11T11:57:00+00:00",  # 3m ago
        },
        {
            "id": "4",
            "title": "Older trashed",
            "type": "video",
            "trash_date": "2026-08-11T10:00:00+00:00",  # 2h ago
        },
    ]

    state = build_library_media_trash_state(records, total=2, now=NOW)

    assert isinstance(state, LibraryMediaTrashState)
    assert [row.media_id for row in state.rows] == ["9", "4"]
    assert state.rows[0].secondary == "pdf · trashed 3m"
    assert state.rows[1].secondary == "video · trashed 2h"
    for row in state.rows:
        assert isinstance(row, LibraryMediaTrashRow)


def test_secondary_falls_back_without_trash_date_or_type():
    """No trash_date -> bare type; no type -> 'media' (the list's own
    fallback vocabulary, not a new invention)."""
    records = [
        {"id": "1", "title": "No date", "type": "audio"},
        {"id": "2", "title": "No type", "trash_date": "2026-08-11T11:00:00+00:00"},
        {"id": "3", "title": "Nothing"},
    ]

    state = build_library_media_trash_state(records, total=3, now=NOW)

    assert state.rows[0].secondary == "audio"
    assert state.rows[1].secondary == "media"
    assert state.rows[2].secondary == "media"


def test_selected_id_falls_back_to_first_row():
    records = [
        {"id": "5", "title": "A", "type": "pdf"},
        {"id": "6", "title": "B", "type": "pdf"},
    ]

    state = build_library_media_trash_state(
        records, total=2, selected_id="not-there", now=NOW
    )

    assert state.selected_id == "5"
    assert state.rows[0].selected is True
    assert state.rows[1].selected is False


def test_selected_id_honored_when_present():
    records = [
        {"id": "5", "title": "A", "type": "pdf"},
        {"id": "6", "title": "B", "type": "pdf"},
    ]

    state = build_library_media_trash_state(records, total=2, selected_id="6", now=NOW)

    assert state.selected_id == "6"
    assert [row.selected for row in state.rows] == [False, True]


def test_empty_trash_yields_honest_empty_copy_and_no_selection():
    state = build_library_media_trash_state((), total=0, now=NOW)

    assert state.rows == ()
    assert state.count == 0
    assert state.empty_copy == LIBRARY_MEDIA_TRASH_EMPTY_COPY
    assert state.selected_id == ""
    assert state.loading is False


def test_loading_state_from_none_records_suppresses_empty_copy():
    """records=None means the fetch has not landed yet -- the widget shows a
    loading line, never the 'Trash is empty' copy (which would be a lie)."""
    state = build_library_media_trash_state(None, total=0, now=NOW)

    assert state.loading is True
    assert state.rows == ()
    assert state.empty_copy == ""


def test_error_passthrough_suppresses_empty_copy():
    state = build_library_media_trash_state(
        (), total=0, error="Could not load Trash.", now=NOW
    )

    assert state.error == "Could not load Trash."
    assert state.empty_copy == ""


def test_truncation_status_names_shown_versus_total():
    """When the seam's total exceeds the fetched rows, the status says so
    honestly instead of implying the whole trash is listed."""
    records = [{"id": str(i), "title": f"T{i}", "type": "pdf"} for i in range(3)]

    state = build_library_media_trash_state(records, total=10, now=NOW)

    assert state.count == 10
    assert state.status_copy == "showing 3 of 10"


def test_no_truncation_status_when_all_rows_shown():
    records = [{"id": "1", "title": "T", "type": "pdf"}]

    state = build_library_media_trash_state(records, total=1, now=NOW)

    assert state.status_copy == ""


def test_notice_passthrough_for_restore_feedback():
    state = build_library_media_trash_state(
        (), total=0, notice="Restored 'A title'.", now=NOW
    )

    assert state.notice == "Restored 'A title'."


def test_tolerates_invalid_and_id_key_fallback_records():
    records = [
        "not-a-mapping",
        {"title": "No id at all"},
        {"media_id": "7", "title": "Media-id key", "media_type": "audio"},
        {"uuid": "u-8", "title": "Uuid key"},
    ]

    state = build_library_media_trash_state(records, total=2, now=NOW)

    assert [row.media_id for row in state.rows] == ["7", "u-8"]
    assert state.rows[0].media_type == "audio"


def test_untitled_fallback_for_missing_title():
    records = [{"id": "1"}, {"id": "2", "title": "   "}]

    state = build_library_media_trash_state(records, total=2, now=NOW)

    assert state.rows[0].title == "Untitled media"
    assert state.rows[1].title == "Untitled media"
