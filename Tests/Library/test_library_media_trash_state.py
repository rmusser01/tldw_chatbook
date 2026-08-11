"""Pure display-state contracts for the Library media Trash view (task-4025).

The Trash view is the third ``_library_media_view`` value of the Browse ▸
Media canvas ("list"/"viewer"/"trash") — these tests pin its pure state
builder the same way ``test_library_media_state.py`` pins the list's.
"""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Library.library_media_state import (
    LIBRARY_MEDIA_TRASH_EMPTY_COPY,
    LibraryMediaTrashRow,
    LibraryMediaTrashState,
    build_library_media_trash_state,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


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

    state = build_library_media_trash_state(
        records, total=2, selected_id="6", now=NOW
    )

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
