"""Pure display-state contracts for the Library Browse ▸ Media canvas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_chatbook.Library.library_media_state import (
    MediaBrowseScope,
    LibraryMediaRow,
    LibraryMediaCanvasState,
    build_media_browse_result,
    build_library_media_browse_state,
    build_library_media_state,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def _summary_item(media_id: int) -> dict[str, object]:
    return {
        "id": f"local:media:{media_id}",
        "backing_media_id": media_id,
        "title": f"Media {media_id}",
        "media_type": "document",
        "updated_at": "2026-08-16T00:00:00+00:00",
    }


def _page(scope: MediaBrowseScope, *, total: int) -> dict[str, object]:
    count = min(20, max(total - scope.offset, 0))
    return {
        "items": [_summary_item(scope.offset + index + 1) for index in range(count)],
        "total": total,
        "limit": 20,
        "offset": scope.offset,
    }


def test_media_browse_scope_normalizes_query_but_preserves_literal_type() -> None:
    scope = MediaBrowseScope(
        query="  needle  ", media_type="  video  ", sort_by="title_asc", page=3
    )

    assert scope.query == "needle"
    assert scope.media_type == "  video  "
    assert scope.page_size == 20
    assert scope.offset == 40
    assert scope.with_page(2).same_except_page(scope)
    assert scope.fingerprint != scope.with_page(2).fingerprint
    with pytest.raises(Exception):
        scope.page = 4  # type: ignore[misc]


@pytest.mark.parametrize("page", [True, 0, -1, 2**63 // 20 + 2])
def test_media_browse_scope_rejects_invalid_or_overflowing_pages(page: object) -> None:
    with pytest.raises(ValueError, match="page"):
        MediaBrowseScope(page=page)  # type: ignore[arg-type]


def test_empty_query_relevance_cannot_misdescribe_database_order() -> None:
    assert MediaBrowseScope(sort_by="relevance").sort_by == "last_modified_desc"
    assert MediaBrowseScope(query="find", sort_by="relevance").sort_by == "relevance"


@pytest.mark.parametrize("media_type", ["All", "all", "ALL"])
def test_media_browse_scope_preserves_literal_all_type_values(media_type: str) -> None:
    assert MediaBrowseScope(media_type=media_type).media_type == media_type
    assert (
        MediaBrowseScope(media_type=f" {media_type} ").media_type == f" {media_type} "
    )


@pytest.mark.parametrize("media_type", [None, "", "   "])
def test_media_browse_scope_uses_none_for_unfiltered_type(
    media_type: str | None,
) -> None:
    assert MediaBrowseScope(media_type=media_type).media_type is None


def test_media_browse_result_preserves_exact_order_and_detaches_items() -> None:
    scope = MediaBrowseScope(page=2)
    payload = _page(scope, total=23)
    result = build_media_browse_result(scope, payload)

    assert [item["backing_media_id"] for item in result.items] == [21, 22, 23]
    assert result.total == 23
    assert result.limit == 20
    assert result.offset == 20
    assert result.last_page == 2
    assert result.out_of_range is False
    payload["items"][0]["title"] = "mutated"  # type: ignore[index]
    assert result.items[0]["title"] == "Media 21"
    with pytest.raises(TypeError):
        result.items[0]["title"] = "mutated"  # type: ignore[index]


def test_media_browse_result_permits_coherent_empty_out_of_range_page() -> None:
    scope = MediaBrowseScope(page=9)
    result = build_media_browse_result(scope, _page(scope, total=45))

    assert result.items == ()
    assert result.out_of_range is True
    assert result.last_page == 3


@pytest.mark.parametrize("field", ["items", "total", "limit", "offset"])
def test_media_browse_result_requires_every_exact_envelope_field(field: str) -> None:
    scope = MediaBrowseScope()
    payload = _page(scope, total=1)
    del payload[field]

    with pytest.raises((TypeError, ValueError), match=field):
        build_media_browse_result(scope, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("limit", 19), ("offset", 1), ("total", True), ("offset", None)],
)
def test_media_browse_result_rejects_repaired_or_mismatched_coordinates(
    field: str, value: object
) -> None:
    scope = MediaBrowseScope()
    payload = _page(scope, total=1)
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        build_media_browse_result(scope, payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.pop("id"),
        lambda item: item.__setitem__("id", None),
        lambda item: item.__setitem__("backing_media_id", True),
        lambda item: item.__setitem__("backing_media_id", "1"),
        lambda item: item.__setitem__("id", "local:media:2"),
        lambda item: item.__setitem__("extra", "forbidden"),
    ],
)
def test_media_browse_result_rejects_malformed_identity_and_shape(mutate) -> None:
    scope = MediaBrowseScope()
    payload = _page(scope, total=1)
    mutate(payload["items"][0])  # type: ignore[index]

    with pytest.raises((TypeError, ValueError)):
        build_media_browse_result(scope, payload)


def test_media_browse_result_rejects_duplicate_page_identity() -> None:
    scope = MediaBrowseScope()
    payload = _page(scope, total=2)
    payload["items"][1]["backing_media_id"] = 1  # type: ignore[index]
    payload["items"][1]["id"] = "local:media:1"  # type: ignore[index]

    with pytest.raises(ValueError, match="unique"):
        build_media_browse_result(scope, payload)


def test_media_browse_result_rejects_wrong_exact_cardinality() -> None:
    scope = MediaBrowseScope(page=2)
    payload = _page(scope, total=22)
    payload["items"] = payload["items"][:1]  # type: ignore[index]

    with pytest.raises(ValueError, match="count"):
        build_media_browse_result(scope, payload)


def test_authoritative_media_page_projection_preserves_order_and_complete_facets() -> (
    None
):
    scope = MediaBrowseScope(media_type="video")
    payload = _page(scope, total=2)
    payload["items"][0]["updated_at"] = "2020-01-01T00:00:00+00:00"  # type: ignore[index]
    payload["items"][1]["updated_at"] = "2026-01-01T00:00:00+00:00"  # type: ignore[index]
    result = build_media_browse_result(scope, payload)

    state = build_library_media_browse_state(
        result,
        type_options=("audio", "document", "video"),
        now=NOW,
    )

    assert [row.media_id for row in state.rows] == ["local:media:1", "local:media:2"]
    assert state.type_options == (None, "audio", "document", "video")
    assert state.active_type == "video"
    assert state.count == 2


def test_authoritative_projection_distinguishes_unfiltered_from_literal_all_types() -> (
    None
):
    scope = MediaBrowseScope()
    result = build_media_browse_result(scope, _page(scope, total=1))

    state = build_library_media_browse_state(
        result,
        type_options=("ALL", "All", "all", " pdf ", "pdf"),
        now=NOW,
    )

    assert state.active_type is None
    assert state.type_options == (None, " pdf ", "ALL", "All", "all", "pdf")


def test_rows_with_type_and_age_secondary_and_missing_last():
    """Rows sorted by recency with secondary showing '{type} · {age}' or fallback."""
    records = [
        {
            "id": "media-b",
            "title": "Beta Video",
            "type": "video",
            "ingestion_date": "2026-07-06T10:00:00+00:00",  # 2h old
        },
        {
            "id": "media-c",
            "title": "Gamma Audio",
            "type": "audio",
            # no updated timestamp
        },
        {
            "id": "media-a",
            "title": "Alpha PDF",
            "type": "pdf",
            "ingestion_date": "2026-07-06T11:57:00+00:00",  # 3m old
        },
    ]

    state = build_library_media_state(records, now=NOW)

    assert isinstance(state, LibraryMediaCanvasState)
    assert [row.media_id for row in state.rows] == ["media-a", "media-b", "media-c"]
    assert state.rows[0].secondary == "pdf · 3m"
    assert state.rows[1].secondary == "video · 2h"
    # No age available -> no " · {age}" suffix
    assert state.rows[2].secondary == "audio"
    for row in state.rows:
        assert isinstance(row, LibraryMediaRow)


def test_type_options_enumerated_and_sorted():
    """type_options = ('All',) + sorted(distinct non-empty types, preserve title-case)."""
    records = [
        {
            "id": "1",
            "title": "One",
            "type": "Video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "2",
            "title": "Two",
            "type": "audio",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
        {
            "id": "3",
            "title": "Three",
            "type": "PDF",
            "ingestion_date": "2026-07-06T09:00:00+00:00",
        },
        {
            "id": "4",
            "title": "Four",
            "type": "Video",
            "ingestion_date": "2026-07-06T08:00:00+00:00",
        },
        {
            "id": "5",
            "title": "Five",
            "type": "",
            "ingestion_date": "2026-07-06T07:00:00+00:00",
        },  # empty type, skip
    ]

    state = build_library_media_state(records, now=NOW)

    # Distinct types: {Video, audio, PDF}; sorted alphabetically; empty excluded
    assert state.type_options == ("All", "PDF", "Video", "audio")


def test_active_type_filter_with_status_copy():
    """When active_type != 'All', filter rows to that type and show status."""
    records = [
        {
            "id": "1",
            "title": "Video One",
            "type": "video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "2",
            "title": "PDF One",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
        {
            "id": "3",
            "title": "Video Two",
            "type": "video",
            "ingestion_date": "2026-07-06T09:00:00+00:00",
        },
        {
            "id": "4",
            "title": "Audio One",
            "type": "audio",
            "ingestion_date": "2026-07-06T08:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, active_type="video", now=NOW)

    assert len(state.rows) == 2
    assert [row.media_id for row in state.rows] == ["1", "3"]
    assert state.status_copy == "2 of 4 · type: video"
    assert state.empty_copy == ""


def test_active_type_filter_no_match_empty_copy():
    """When active_type != 'All' and no matches, empty_copy shows specific message."""
    records = [
        {
            "id": "1",
            "title": "Video One",
            "type": "video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "2",
            "title": "PDF One",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, active_type="audio", now=NOW)

    assert state.rows == ()
    assert state.status_copy == "0 of 2 · type: audio"
    assert state.empty_copy == "No media of type 'audio'."
    assert state.selected_id == ""
    assert state.preview_lines == ()


def test_no_records_yields_default_empty_copy():
    """Empty records list yields default empty_copy and no status_copy."""
    state = build_library_media_state([], now=NOW)

    assert state.rows == ()
    assert state.status_copy == ""
    assert (
        state.empty_copy
        == "No media in your Library yet. Import something to see it here."
    )
    assert state.selected_id == ""
    assert state.preview_lines == ()
    assert state.active_type == "All"


def test_selected_id_not_present_falls_back_to_first_row():
    """When selected_id not in filtered+limited rows, fallback to first row."""
    records = [
        {
            "id": "media-a",
            "title": "Alpha",
            "type": "video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "media-b",
            "title": "Beta",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, selected_id="does-not-exist", now=NOW)

    assert state.selected_id == "media-a"
    assert state.rows[0].selected is True
    assert state.rows[1].selected is False


def test_preview_lines_for_selected_row():
    """Preview lines show: title, Type: {type or 'unknown'}, Updated: {age or 'unknown'}."""
    records = [
        {
            "id": "media-a",
            "title": "Alpha Video",
            "type": "video",
            "ingestion_date": "2026-07-06T11:57:00+00:00",  # 3m old
        },
        {
            "id": "media-b",
            "title": "Beta Audio",
            # No type, no ingestion_date
        },
    ]

    selected_a = build_library_media_state(records, selected_id="media-a", now=NOW)
    assert selected_a.preview_lines == (
        "Alpha Video",
        "Type: video",
        "Updated: 3m",
    )

    selected_b = build_library_media_state(records, selected_id="media-b", now=NOW)
    assert selected_b.preview_lines == (
        "Beta Audio",
        "Type: unknown",
        "Updated: unknown",
    )


def test_limit_truncates_rows_to_max_after_sorting():
    """Limit applied after sorting and filtering to keep only most recent."""
    records = [
        {
            "id": f"media-{i}",
            "title": f"Media {i}",
            "type": "video",
            "ingestion_date": f"2026-07-06T{11 - i:02d}:00:00+00:00",
        }
        for i in range(5)
    ]

    state = build_library_media_state(records, now=NOW, limit=2)

    # Most recent two: media-0 (11:00) and media-1 (10:00)
    assert [row.media_id for row in state.rows] == ["media-0", "media-1"]
    assert len(state.rows) == 2


def test_id_title_type_key_fallbacks():
    """Test key fallbacks: media_id/id/uuid, title, type/media_type."""
    records = [
        {
            "media_id": "mid-99",
            "title": "  Fallback Media  ",
            "media_type": "video",
            "ingestion_date": "2026-07-06T11:57:00+00:00",
        },
        {
            "uuid": "uuid-77",
            "title": "UUID Media",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, now=NOW)

    assert len(state.rows) == 2
    row_a = state.rows[0]  # sorted by recency
    assert row_a.media_id == "mid-99"
    assert row_a.title == "Fallback Media"
    assert row_a.secondary == "video · 3m"

    row_b = state.rows[1]
    assert row_b.media_id == "uuid-77"
    assert row_b.title == "UUID Media"
    assert row_b.secondary == "pdf · 2h"


def test_untitled_fallback_for_missing_title():
    """Missing or empty title defaults to 'Untitled media'."""
    records = [
        {
            "id": "1",
            "title": None,
            "type": "video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "2",
            "title": "  ",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
        {
            "id": "3",
            "type": "audio",
            "ingestion_date": "2026-07-06T09:00:00+00:00",
        },  # No title key
    ]

    state = build_library_media_state(records, now=NOW)

    assert state.rows[0].title == "Untitled media"
    assert state.rows[1].title == "Untitled media"
    assert state.rows[2].title == "Untitled media"


def test_empty_type_secondary_fallback():
    """Record with empty or missing type shows secondary as 'media' regardless of age."""
    records = [
        {
            "id": "media-a",
            "title": "No Type Media",
            "type": "",
            "ingestion_date": "2026-07-06T11:57:00+00:00",
        },
        {
            "id": "media-b",
            "title": "Missing Type Media",
            # No type key
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, now=NOW)

    # When type is empty/missing, secondary must be "media" regardless of age
    assert state.rows[0].secondary == "media"
    assert state.rows[1].secondary == "media"


def test_media_secondary_fallback_when_no_type_no_age():
    """When both type and age are missing, secondary = 'media'."""
    records = [
        {
            "id": "media-a",
            "title": "No Type No Age",
            # No type, no updated timestamp
        },
    ]

    state = build_library_media_state(records, now=NOW)

    assert state.rows[0].secondary == "media"


def test_updated_key_fallbacks():
    """Test timestamp key fallbacks: last_modified, ingestion_date, date, updated_at."""
    records = [
        {"id": "1", "title": "A", "last_modified": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "B", "ingestion_date": "2026-07-06T10:00:00+00:00"},
        {"id": "3", "title": "C", "date": "2026-07-06T09:00:00+00:00"},
        {"id": "4", "title": "D", "updated_at": "2026-07-06T08:00:00+00:00"},
    ]

    state = build_library_media_state(records, now=NOW)

    # Most recent first
    assert [row.media_id for row in state.rows] == ["1", "2", "3", "4"]


def test_count_tracks_total_pre_type_filter():
    """count reflects total records pre-filter, status_copy shows filtered count."""
    records = [
        {
            "id": "1",
            "title": "Video One",
            "type": "video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "2",
            "title": "PDF One",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
        {
            "id": "3",
            "title": "Video Two",
            "type": "video",
            "ingestion_date": "2026-07-06T09:00:00+00:00",
        },
        {
            "id": "4",
            "title": "Audio One",
            "type": "audio",
            "ingestion_date": "2026-07-06T08:00:00+00:00",
        },
        {
            "id": "5",
            "title": "Video Three",
            "type": "video",
            "ingestion_date": "2026-07-06T07:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, active_type="video", now=NOW)

    assert state.count == 5  # Total records
    assert len(state.rows) == 3  # Only video type
    assert state.status_copy == "3 of 5 · type: video"


def test_tolerates_invalid_and_missing_records():
    """Records with None, non-mapping, empty dict, invalid fields are skipped gracefully."""
    records = [
        None,  # Not a mapping
        "not-a-mapping",  # String, not a mapping
        {},  # No id key -> skipped
        {
            "id": "invalid",
            "title": None,  # Invalid title -> uses fallback
            "type": None,  # Invalid type -> uses empty/fallback
            "ingestion_date": "garbage",  # Invalid timestamp -> parsed as None
        },
        {
            "id": "valid",
            "title": "Valid Media",
            "type": "video",
            "ingestion_date": "2026-07-06T11:57:00+00:00",
        },
    ]

    # Should not raise, should return only the valid record
    state = build_library_media_state(records, now=NOW)

    assert len(state.rows) == 2
    valid_ids = [row.media_id for row in state.rows]
    assert "invalid" in valid_ids
    assert "valid" in valid_ids
    # Check that "invalid" record was processed with fallbacks
    invalid_row = next(r for r in state.rows if r.media_id == "invalid")
    assert invalid_row.title == "Untitled media"
    assert invalid_row.secondary == "media"


def test_no_type_status_copy_when_active_type_all():
    """When active_type='All', status_copy is empty."""
    records = [
        {
            "id": "1",
            "title": "Video",
            "type": "video",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
        {
            "id": "2",
            "title": "PDF",
            "type": "pdf",
            "ingestion_date": "2026-07-06T10:00:00+00:00",
        },
    ]

    state = build_library_media_state(records, active_type="All", now=NOW)

    assert state.status_copy == ""
    assert state.empty_copy == ""


def test_status_copy_uses_pre_limit_count():
    """status_copy shows count of filtered entries (pre-limit), not displayed rows (post-limit)."""
    # 100 video records + 10 other-type records = 110 total
    records = [
        {
            "id": f"video-{i}",
            "title": f"Video {i}",
            "type": "video",
            "ingestion_date": f"2026-07-06T{11 - (i % 12):02d}:00:00+00:00",
        }
        for i in range(100)
    ] + [
        {
            "id": f"other-{i}",
            "title": f"Other {i}",
            "type": "audio",
            "ingestion_date": f"2026-07-06T{10 - (i % 10):02d}:00:00+00:00",
        }
        for i in range(10)
    ]

    # Filter to video type with limit=75
    state = build_library_media_state(records, active_type="video", now=NOW, limit=75)

    # Exactly 75 rows displayed (post-limit)
    assert len(state.rows) == 75
    # But status_copy shows all 100 video records (pre-limit)
    assert state.status_copy == "100 of 110 · type: video"


def test_active_type_absent_from_records_stays_in_type_options():
    """When active_type is not in records, it is still included in type_options."""
    records = [
        {
            "id": "1",
            "title": "PDF One",
            "type": "pdf",
            "ingestion_date": "2026-07-06T11:00:00+00:00",
        },
    ]

    # Request active_type="video" even though no records have type="video"
    state = build_library_media_state(records, active_type="video", now=NOW)

    # "video" must be in type_options to avoid InvalidSelectValueError
    assert "video" in state.type_options
    # "video" should come after "All" in sorted order
    assert state.type_options == ("All", "pdf", "video")
    # No rows match the filter
    assert state.rows == ()
    # Empty copy reflects the filtered type
    assert state.empty_copy == "No media of type 'video'."


def test_confirming_bulk_delete_defaults_false_and_passes_through():
    """task-2853 AC3: the bulk-delete confirm flag is a pure passthrough,
    like ``select_mode`` -- no computation, just carried onto the state so
    the canvas can render the confirm row in place of the normal toolbar.
    """
    records = [{"id": "1", "title": "A", "type": "video"}]

    default_state = build_library_media_state(records, select_mode=True)
    assert default_state.confirming_bulk_delete is False

    confirming_state = build_library_media_state(
        records, select_mode=True, confirming_bulk_delete=True
    )
    assert confirming_state.confirming_bulk_delete is True


def test_delete_receipt_count_defaults_zero_and_passes_through():
    """task-4022 AC2: like ``confirming_bulk_delete``, the bulk-delete
    receipt count is a pure passthrough -- no computation, just carried
    onto the state so the canvas can render the "✓ deleted · N items"
    row. Negative input (defensive only -- no real caller passes one)
    floors to 0 rather than rendering a nonsensical receipt."""
    records = [{"id": "1", "title": "A", "type": "video"}]

    default_state = build_library_media_state(records)
    assert default_state.delete_receipt_count == 0

    receipt_state = build_library_media_state(records, delete_receipt_count=3)
    assert receipt_state.delete_receipt_count == 3

    floored_state = build_library_media_state(records, delete_receipt_count=-1)
    assert floored_state.delete_receipt_count == 0
