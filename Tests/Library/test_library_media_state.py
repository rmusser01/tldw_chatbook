"""Pure display-state contracts for the Library Browse ▸ Media canvas."""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Library.library_media_state import (
    LibraryMediaRow,
    LibraryMediaCanvasState,
    build_library_media_state,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


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
        {"id": "1", "title": "One", "type": "Video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "Two", "type": "audio", "ingestion_date": "2026-07-06T10:00:00+00:00"},
        {"id": "3", "title": "Three", "type": "PDF", "ingestion_date": "2026-07-06T09:00:00+00:00"},
        {"id": "4", "title": "Four", "type": "Video", "ingestion_date": "2026-07-06T08:00:00+00:00"},
        {"id": "5", "title": "Five", "type": "", "ingestion_date": "2026-07-06T07:00:00+00:00"},  # empty type, skip
    ]

    state = build_library_media_state(records, now=NOW)

    # Distinct types: {Video, audio, PDF}; sorted alphabetically; empty excluded
    assert state.type_options == ("All", "PDF", "Video", "audio")


def test_active_type_filter_with_status_copy():
    """When active_type != 'All', filter rows to that type and show status."""
    records = [
        {"id": "1", "title": "Video One", "type": "video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "PDF One", "type": "pdf", "ingestion_date": "2026-07-06T10:00:00+00:00"},
        {"id": "3", "title": "Video Two", "type": "video", "ingestion_date": "2026-07-06T09:00:00+00:00"},
        {"id": "4", "title": "Audio One", "type": "audio", "ingestion_date": "2026-07-06T08:00:00+00:00"},
    ]

    state = build_library_media_state(records, active_type="video", now=NOW)

    assert len(state.rows) == 2
    assert [row.media_id for row in state.rows] == ["1", "3"]
    assert state.status_copy == "2 of 4 · type: video"
    assert state.empty_copy == ""


def test_active_type_filter_no_match_empty_copy():
    """When active_type != 'All' and no matches, empty_copy shows specific message."""
    records = [
        {"id": "1", "title": "Video One", "type": "video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "PDF One", "type": "pdf", "ingestion_date": "2026-07-06T10:00:00+00:00"},
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
    assert state.empty_copy == "No media in your Library yet. Ingest something to see it here."
    assert state.selected_id == ""
    assert state.preview_lines == ()
    assert state.active_type == "All"


def test_selected_id_not_present_falls_back_to_first_row():
    """When selected_id not in filtered+limited rows, fallback to first row."""
    records = [
        {"id": "media-a", "title": "Alpha", "type": "video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "media-b", "title": "Beta", "type": "pdf", "ingestion_date": "2026-07-06T10:00:00+00:00"},
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
        {"id": "1", "title": None, "type": "video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "  ", "type": "pdf", "ingestion_date": "2026-07-06T10:00:00+00:00"},
        {"id": "3", "type": "audio", "ingestion_date": "2026-07-06T09:00:00+00:00"},  # No title key
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
        {"id": "1", "title": "Video One", "type": "video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "PDF One", "type": "pdf", "ingestion_date": "2026-07-06T10:00:00+00:00"},
        {"id": "3", "title": "Video Two", "type": "video", "ingestion_date": "2026-07-06T09:00:00+00:00"},
        {"id": "4", "title": "Audio One", "type": "audio", "ingestion_date": "2026-07-06T08:00:00+00:00"},
        {"id": "5", "title": "Video Three", "type": "video", "ingestion_date": "2026-07-06T07:00:00+00:00"},
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
        {"id": "1", "title": "Video", "type": "video", "ingestion_date": "2026-07-06T11:00:00+00:00"},
        {"id": "2", "title": "PDF", "type": "pdf", "ingestion_date": "2026-07-06T10:00:00+00:00"},
    ]

    state = build_library_media_state(records, active_type="All", now=NOW)

    assert state.status_copy == ""
    assert state.empty_copy == ""


def test_status_copy_uses_pre_limit_count():
    """status_copy shows count of filtered entries (pre-limit), not displayed rows (post-limit)."""
    # 100 video records + 10 other-type records = 110 total
    records = (
        [
            {
                "id": f"video-{i}",
                "title": f"Video {i}",
                "type": "video",
                "ingestion_date": f"2026-07-06T{11 - (i % 12):02d}:00:00+00:00",
            }
            for i in range(100)
        ]
        + [
            {
                "id": f"other-{i}",
                "title": f"Other {i}",
                "type": "audio",
                "ingestion_date": f"2026-07-06T{10 - (i % 10):02d}:00:00+00:00",
            }
            for i in range(10)
        ]
    )

    # Filter to video type with limit=75
    state = build_library_media_state(records, active_type="video", now=NOW, limit=75)

    # Exactly 75 rows displayed (post-limit)
    assert len(state.rows) == 75
    # But status_copy shows all 100 video records (pre-limit)
    assert state.status_copy == "100 of 110 · type: video"


def test_active_type_absent_from_records_stays_in_type_options():
    """When active_type is not in records, it is still included in type_options."""
    records = [
        {"id": "1", "title": "PDF One", "type": "pdf", "ingestion_date": "2026-07-06T11:00:00+00:00"},
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
