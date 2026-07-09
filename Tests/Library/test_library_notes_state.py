"""Pure display-state contracts for the Library notes canvas."""
from datetime import datetime, timezone

from tldw_chatbook.Library.library_notes_state import (
    NOTES_SORT_MODES,
    LibraryNoteEditorState,
    LibraryNotesListRow,
    build_library_note_editor_state,
    build_library_notes_list_state,
    build_note_export_content,
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


EXPORT_NOW = datetime(2026, 7, 7, 9, 30, 15)


def test_export_content_markdown_has_frontmatter_and_heading():
    text = build_note_export_content(
        "Q3 retro", "alpha body", "retro, q3", "n-1", "markdown", now=EXPORT_NOW
    )
    assert text.startswith("---\n")
    assert "title: Q3 retro\n" in text
    assert "date: 2026-07-07 09:30:15\n" in text
    assert "keywords: retro, q3\n" in text
    assert "note_id: n-1\n" in text
    assert "---\n\n# Q3 retro\n\nalpha body" in text


def test_export_content_text_has_header_and_rule():
    text = build_note_export_content(
        "Q3 retro", "alpha body", "retro, q3", "n-1", "text", now=EXPORT_NOW
    )
    assert text.startswith("Title: Q3 retro\n")
    assert "Date: 2026-07-07 09:30:15\n" in text
    assert "Keywords: retro, q3\n" in text
    assert "Note ID: n-1\n" in text
    assert "=" * 50 in text
    assert text.endswith("alpha body")


def test_export_content_blank_title_falls_back_to_untitled():
    markdown_text = build_note_export_content("   ", "body", "", "n-2", "markdown", now=EXPORT_NOW)
    assert "title: Untitled Note\n" in markdown_text
    assert "# Untitled Note" in markdown_text
    text = build_note_export_content("", "body", "", "n-2", "text", now=EXPORT_NOW)
    assert text.startswith("Title: Untitled Note\n")


def test_export_content_now_defaults_when_omitted():
    text = build_note_export_content("Title", "body", "", "n-3", "text")
    assert "Date: " in text
    # No fixed value to assert against, but the stamp must be well-formed.
    date_line = next(line for line in text.splitlines() if line.startswith("Date: "))
    datetime.strptime(date_line.removeprefix("Date: "), "%Y-%m-%d %H:%M:%S")


def test_resolve_note_template_placeholders_substitutes_known_keys():
    from tldw_chatbook.Library.library_notes_state import resolve_note_template_placeholders

    resolved = resolve_note_template_placeholders(
        "T - {date} {time} {datetime}", now=datetime(2026, 7, 8, 9, 30)
    )
    assert resolved == "T - 2026-07-08 09:30 2026-07-08 09:30"


def test_resolve_note_template_placeholders_degrades_on_malformed():
    from tldw_chatbook.Library.library_notes_state import resolve_note_template_placeholders

    assert resolve_note_template_placeholders("{unknown_key}") == "{unknown_key}"
    assert resolve_note_template_placeholders("stray { brace") == "stray { brace"


def test_resolve_note_template_placeholders_resolves_known_keys_leaves_unknown_literal():
    """A per-key resolution: an unknown placeholder sitting alongside a
    known one must not block the known one from being substituted."""
    from tldw_chatbook.Library.library_notes_state import resolve_note_template_placeholders

    resolved = resolve_note_template_placeholders(
        "X {date} {unknown}", now=datetime(2026, 7, 8, 9, 30)
    )
    assert resolved == "X 2026-07-08 {unknown}"


def test_note_template_keywords_parses_comma_string_and_sequences():
    from tldw_chatbook.Library.library_notes_state import note_template_keywords

    assert note_template_keywords({"keywords": "meeting, notes"}) == ("meeting", "notes")
    assert note_template_keywords({"keywords": ["a", " b ", ""]}) == ("a", "b")
    assert note_template_keywords({"keywords": ""}) == ()
    assert note_template_keywords({}) == ()
    assert note_template_keywords(None) == ()


def test_build_note_template_rows_excludes_blank_and_resolves_titles():
    from tldw_chatbook.Library.library_notes_state import build_library_note_template_rows

    templates = {
        "blank": {"title": "New Note", "content": "", "description": "Empty note template"},
        "meeting": {
            "title": "Meeting Notes - {date}",
            "content": "x",
            "description": "Template for meeting notes",
        },
    }
    rows = build_library_note_template_rows(templates, now=datetime(2026, 7, 8, 9, 30))

    assert [row.template_key for row in rows] == ["meeting"]
    assert rows[0].label == "Meeting notes"
    assert rows[0].resolved_title == "Meeting Notes - 2026-07-08"


def test_build_note_template_rows_malformed_value_degrades_to_key_label():
    from tldw_chatbook.Library.library_notes_state import build_library_note_template_rows

    rows = build_library_note_template_rows({"bug_report": "not-a-mapping"})

    assert rows[0].template_key == "bug_report"
    assert rows[0].label == "Bug report"
    assert rows[0].resolved_title == ""


def test_build_note_template_rows_drops_secondary_when_it_repeats_label():
    from tldw_chatbook.Library.library_notes_state import build_library_note_template_rows

    templates = {"todo": {"title": "Todo list", "description": "Todo list template"}}
    rows = build_library_note_template_rows(templates)

    assert rows[0].label == "Todo list"
    assert rows[0].resolved_title == ""


def test_build_note_template_rows_project_and_research_resolve_dateful_titles():
    """L2b.2 task-4 rider: ``project``/``research`` used to end in a
    dangling separator ("Project: ", "Research Notes - "), which rendered
    as an empty/awkward secondary line. Both now carry a real ``{date}``
    placeholder like every other non-blank template, so their resolved
    title is a real secondary rather than a blank or truncated one."""
    from tldw_chatbook.Library.library_notes_state import build_library_note_template_rows

    templates = {
        "project": {"title": "Project Plan - {date}", "description": "Project planning template"},
        "research": {"title": "Research Notes - {date}", "description": "Research notes template"},
    }
    rows = build_library_note_template_rows(templates, now=datetime(2026, 7, 8, 9, 30))
    rows_by_key = {row.template_key: row for row in rows}

    assert rows_by_key["project"].resolved_title == "Project Plan - 2026-07-08"
    assert rows_by_key["research"].resolved_title == "Research Notes - 2026-07-08"
