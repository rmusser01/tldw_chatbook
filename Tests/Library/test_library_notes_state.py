"""Pure display-state contracts for the Library notes canvas."""
from datetime import datetime, timezone

from tldw_chatbook.Library.library_notes_state import (
    NOTES_SORT_MODES,
    LibraryNoteEditorState,
    LibraryNotesListRow,
    build_library_note_editor_state,
    build_library_notes_list_state,
    next_notes_sort_mode,
    notes_autosave_status_text,
    sort_notes_records,
)

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

NOTE_A = {"id": "n-1", "title": "Q3 retro", "content": "alpha body",
          "last_modified": "2026-07-07T11:57:00+00:00", "version": 2}
NOTE_B = {"id": "n-2", "title": "Reading list", "content": "bravo body",
          "last_modified": "2026-07-06T12:00:00+00:00", "version": 1}


def test_list_state_builds_rows_with_age_and_header():
    state = build_library_notes_list_state([NOTE_A, NOTE_B], now=NOW)
    assert state.header_copy == "Notes (2)"
    assert state.rows[0] == LibraryNotesListRow(note_id="n-1", title="Q3 retro", age_label="3m")
    assert state.rows[1].age_label == "1d"
    assert state.empty_copy == ""


def test_list_state_empty_uses_quiet_copy():
    state = build_library_notes_list_state([], now=NOW)
    assert state.rows == ()
    assert state.empty_copy == "No notes yet. Create one to see it here."


def test_list_state_filter_note_reflects_active_filter():
    state = build_library_notes_list_state([NOTE_A], filter_note="retro", now=NOW)
    assert state.status_copy == "filter: retro · 1 result"


def test_list_state_tolerates_missing_fields():
    state = build_library_notes_list_state([{"id": "x"}], now=NOW)
    assert state.rows[0].title == "Untitled"
    assert state.rows[0].age_label == ""


def test_sort_mode_cycles_and_wraps():
    assert next_notes_sort_mode("newest") == "oldest"
    assert next_notes_sort_mode("oldest") == "title"
    assert next_notes_sort_mode("title") == "newest"
    assert next_notes_sort_mode("bogus") == "newest"


def test_sort_records_newest_oldest_title():
    newest = sort_notes_records([NOTE_B, NOTE_A], "newest")
    assert [n["id"] for n in newest] == ["n-1", "n-2"]
    oldest = sort_notes_records([NOTE_A, NOTE_B], "oldest")
    assert [n["id"] for n in oldest] == ["n-2", "n-1"]
    by_title = sort_notes_records([NOTE_A, NOTE_B], "title")
    assert [n["id"] for n in by_title] == ["n-1", "n-2"]  # "Q3..." < "Reading..."


def test_editor_state_builds_fields_and_meta_line():
    detail = {"id": "n-1", "title": "Q3 retro", "content": "alpha body",
              "version": 2, "last_modified": "2026-07-07T11:57:00+00:00",
              "created_at": "2026-07-01T10:00:00+00:00",
              "keywords": ["retro", "q3"]}
    state = build_library_note_editor_state(detail, now=NOW)
    assert state.note_id == "n-1"
    assert state.title == "Q3 retro"
    assert state.content == "alpha body"
    assert state.keywords_text == "retro, q3"
    assert state.version == 2
    assert state.has_note is True
    assert "Created 6d" in state.meta_line and "Modified 3m" in state.meta_line
    assert "v2" in state.meta_line


def test_editor_state_none_detail_yields_empty():
    state = build_library_note_editor_state(None, now=NOW)
    assert state.has_note is False
    assert state.note_id == ""


def test_autosave_status_text_variants():
    assert notes_autosave_status_text("idle", word_count=2) == "2 words"
    assert notes_autosave_status_text("saving", word_count=2) == "2 words · saving…"
    assert notes_autosave_status_text("saved", word_count=2) == "2 words · saved"
    assert notes_autosave_status_text("conflict", word_count=2) == "2 words · changed elsewhere"
    assert notes_autosave_status_text("error", word_count=2) == "2 words · save failed"
