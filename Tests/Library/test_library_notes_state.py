"""Pure display and lossless-session contracts for Library Database Notes."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest
from rich.cells import cell_len

from tldw_chatbook.Library.library_notes_state import (
    DATABASE_NOTE_BODY_MAX_CHARS,
    DATABASE_NOTE_KEYWORD_MAX_CHARS,
    DATABASE_NOTE_TITLE_MAX_CHARS,
    DatabaseNoteDraft,
    DatabaseNoteSavePayload,
    LibraryNoteSessionSnapshot,
    LibraryNotesFocusIdentity,
    LibraryNotesListRow,
    LibraryNotesOperationState,
    NormalizedDatabaseNote,
    NoteValidationVeto,
    build_library_note_editor_state,
    build_library_notes_list_state,
    build_note_export_content,
    ellipsize_note_title_cells,
    next_notes_sort_mode,
    notes_autosave_status_text,
    patch_note_records_after_save,
    sort_notes_records,
    validate_database_note_draft,
)

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

NOTE_A = {
    "id": "n-1",
    "title": "Q3 retro",
    "content": "alpha body",
    "last_modified": "2026-07-07T11:57:00+00:00",
    "version": 2,
}
NOTE_B = {
    "id": "n-2",
    "title": "Reading list",
    "content": "bravo body",
    "last_modified": "2026-07-06T12:00:00+00:00",
    "version": 1,
}


def test_list_state_builds_rows_with_age_and_header():
    state = build_library_notes_list_state([NOTE_A, NOTE_B], now=NOW)
    assert state.header_copy == "Notes (2)"
    assert state.rows[0] == LibraryNotesListRow(
        note_id="n-1", title="Q3 retro", age_label="3m"
    )
    assert state.rows[1].age_label == "1d"
    assert state.empty_copy == ""


def test_list_state_empty_uses_quiet_copy():
    state = build_library_notes_list_state([], total_count=0, now=NOW)
    assert state.rows == ()
    assert state.empty_copy == "No notes yet. Create your first note."
    assert state.empty_kind == "source-empty"
    assert state.total_count == 0
    assert state.result_count == 0


def test_list_state_filtered_empty_does_not_claim_the_source_is_empty():
    state = build_library_notes_list_state(
        [], filter_note="[draft] <plan>", total_count=2, now=NOW
    )

    assert state.header_copy == "Notes (2)"
    assert state.status_copy == "filter: [draft] <plan> · 0 results"
    assert state.empty_copy == "No notes match “[draft] <plan>”. Clear the filter."
    assert state.empty_kind == "filter-empty"
    assert state.total_count == 2
    assert state.result_count == 0


def test_list_state_filter_note_reflects_active_filter():
    state = build_library_notes_list_state([NOTE_A], filter_note="retro", now=NOW)
    assert state.status_copy == "filter: retro · 1 result"


def test_list_state_filter_status_is_one_row_cell_bounded_and_plain():
    state = build_library_notes_list_state(
        [],
        filter_note="[draft]\n" + "界" * 80,
        total_count=2,
        now=NOW,
    )

    assert "\n" not in state.status_copy
    assert cell_len(state.status_copy) <= 52
    assert state.status_copy.endswith("…")


def test_list_state_exposes_sort_chooser_and_active_operation_status():
    state = build_library_notes_list_state(
        [NOTE_A],
        sort_choices_visible=True,
        operation_status="Import note…",
        now=NOW,
    )

    assert state.sort_choices_visible is True
    assert state.operation_status == "Import note…"
    assert state.status_copy == "Import note…"
    assert state.empty_kind == "populated"


@pytest.mark.parametrize(
    ("phase", "expected"),
    (
        ("running", "Import…"),
        ("complete", "Import complete."),
        ("failed", "Import failed — choose another file."),
    ),
)
def test_note_operation_state_formats_typed_status(phase, expected):
    state = LibraryNotesOperationState(
        kind="import",
        token=7,
        phase=phase,
        region="navigator",
        failure_next_action="choose another file",
    )

    assert state.status_line == expected
    assert state.running is (phase == "running")


def test_note_operation_state_formats_committed_recovery_status():
    state = LibraryNotesOperationState(
        kind="import",
        token=8,
        phase="complete",
        region="navigator",
        completion_next_action="select the new note from Notes to open",
    )

    assert (
        state.status_line == "Import complete — select the new note from Notes to open."
    )


def test_list_state_tolerates_missing_fields():
    state = build_library_notes_list_state([{"id": "x"}], now=NOW)
    assert state.rows[0].title == "Untitled"
    assert state.rows[0].age_label == ""


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("title", 7),
        ("body", {"not": "text"}),
        ("keywords", ["valid", 7]),
    ),
)
def test_database_note_validation_vetoes_non_text_field_shapes(field, value):
    values = {
        "title": "Valid title",
        "body": "Valid body",
        "keywords": "valid, keywords",
    }
    values[field] = value
    draft = DatabaseNoteDraft(
        note_id="",
        title=values["title"],
        body=values["body"],
        keywords_text=values["keywords"],
        revision=1,
    )

    outcome = validate_database_note_draft(draft)

    assert isinstance(outcome, NoteValidationVeto)
    assert outcome.field == field
    assert "must be text" in outcome.message.lower()


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
    detail = {
        "id": "n-1",
        "title": "Q3 retro",
        "content": "alpha body",
        "version": 2,
        "last_modified": "2026-07-07T11:57:00+00:00",
        "created_at": "2026-07-01T10:00:00+00:00",
        "keywords": ["retro", "q3"],
    }
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
    assert (
        notes_autosave_status_text("conflict", word_count=2)
        == "2 words · changed elsewhere"
    )
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
    markdown_text = build_note_export_content(
        "   ", "body", "", "n-2", "markdown", now=EXPORT_NOW
    )
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
    from tldw_chatbook.Library.library_notes_state import (
        resolve_note_template_placeholders,
    )

    resolved = resolve_note_template_placeholders(
        "T - {date} {time} {datetime}", now=datetime(2026, 7, 8, 9, 30)
    )
    assert resolved == "T - 2026-07-08 09:30 2026-07-08 09:30"


def test_resolve_note_template_placeholders_degrades_on_malformed():
    from tldw_chatbook.Library.library_notes_state import (
        resolve_note_template_placeholders,
    )

    assert resolve_note_template_placeholders("{unknown_key}") == "{unknown_key}"
    assert resolve_note_template_placeholders("stray { brace") == "stray { brace"


def test_resolve_note_template_placeholders_resolves_known_keys_leaves_unknown_literal():
    """A per-key resolution: an unknown placeholder sitting alongside a
    known one must not block the known one from being substituted."""
    from tldw_chatbook.Library.library_notes_state import (
        resolve_note_template_placeholders,
    )

    resolved = resolve_note_template_placeholders(
        "X {date} {unknown}", now=datetime(2026, 7, 8, 9, 30)
    )
    assert resolved == "X 2026-07-08 {unknown}"


def test_note_template_keywords_parses_comma_string_and_sequences():
    from tldw_chatbook.Library.library_notes_state import note_template_keywords

    assert note_template_keywords({"keywords": "meeting, notes"}) == (
        "meeting",
        "notes",
    )
    assert note_template_keywords({"keywords": ["a", " b ", ""]}) == ("a", "b")
    assert note_template_keywords({"keywords": ""}) == ()
    assert note_template_keywords({}) == ()
    assert note_template_keywords(None) == ()


def test_build_note_template_rows_excludes_blank_and_resolves_titles():
    from tldw_chatbook.Library.library_notes_state import (
        build_library_note_template_rows,
    )

    templates = {
        "blank": {
            "title": "New Note",
            "content": "",
            "description": "Empty note template",
        },
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
    from tldw_chatbook.Library.library_notes_state import (
        build_library_note_template_rows,
    )

    rows = build_library_note_template_rows({"bug_report": "not-a-mapping"})

    assert rows[0].template_key == "bug_report"
    assert rows[0].label == "Bug report"
    assert rows[0].resolved_title == ""


def test_build_note_template_rows_drops_secondary_when_it_repeats_label():
    from tldw_chatbook.Library.library_notes_state import (
        build_library_note_template_rows,
    )

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
    from tldw_chatbook.Library.library_notes_state import (
        build_library_note_template_rows,
    )

    templates = {
        "project": {
            "title": "Project Plan - {date}",
            "description": "Project planning template",
        },
        "research": {
            "title": "Research Notes - {date}",
            "description": "Research notes template",
        },
    }
    rows = build_library_note_template_rows(templates, now=datetime(2026, 7, 8, 9, 30))
    rows_by_key = {row.template_key: row for row in rows}

    assert rows_by_key["project"].resolved_title == "Project Plan - 2026-07-08"
    assert rows_by_key["research"].resolved_title == "Research Notes - 2026-07-08"


# (task-184) patch_note_records_after_save: the in-memory list refresh
# applied by a successful in-canvas save, so Back-to-list shows the new
# title, a fresh relative age, and correct Newest ordering immediately.


def test_patch_note_records_after_save_updates_title_and_last_modified():
    saved_stamp = "2026-07-07T12:00:00+00:00"
    patched = patch_note_records_after_save(
        [NOTE_A, NOTE_B], "n-2", title="Reading list (edited)", modified_at=saved_stamp
    )

    assert patched[0] == NOTE_A
    assert patched[1]["title"] == "Reading list (edited)"
    assert patched[1]["last_modified"] == saved_stamp
    # Other fields pass through untouched.
    assert patched[1]["content"] == "bravo body"
    assert patched[1]["version"] == 1
    # The original record is never mutated in place.
    assert NOTE_B["title"] == "Reading list"


def test_patch_note_records_after_save_resorts_to_front_with_fresh_age():
    saved_stamp = NOW.isoformat()
    patched = patch_note_records_after_save(
        [NOTE_A, NOTE_B], "n-2", title="Reading list (edited)", modified_at=saved_stamp
    )

    ordered = sort_notes_records(list(patched), "newest")
    state = build_library_notes_list_state(ordered, now=NOW)

    assert state.rows[0].note_id == "n-2"
    assert state.rows[0].title == "Reading list (edited)"
    assert state.rows[0].age_label == "now"


def test_patch_note_records_after_save_leaves_unknown_ids_and_shapes_alone():
    records = [NOTE_A, "not-a-mapping"]
    patched = patch_note_records_after_save(
        records, "missing-id", title="X", modified_at="2026-07-07T12:00:00+00:00"
    )

    assert patched == (NOTE_A, "not-a-mapping")
    assert patch_note_records_after_save(None, "n-1", title="X", modified_at="s") == ()


def test_save_payload_preserves_valid_raw_content_exactly():
    draft = DatabaseNoteDraft(
        note_id="n-1",
        title="[draft] <plan>",
        body="line 1\n<script example>\x3c/script>",
        keywords_text="alpha, βeta",
        revision=7,
    )

    result = validate_database_note_draft(draft)

    assert result == DatabaseNoteSavePayload(
        title="[draft] <plan>",
        body="line 1\n<script example>\x3c/script>",
        keywords=("alpha", "βeta"),
        revision=7,
    )


@pytest.mark.parametrize(
    ("draft", "field"),
    (
        (
            DatabaseNoteDraft(
                "n", "x" * (DATABASE_NOTE_TITLE_MAX_CHARS + 1), "", "", 1
            ),
            "title",
        ),
        (
            DatabaseNoteDraft(
                "n", "ok", "x" * (DATABASE_NOTE_BODY_MAX_CHARS + 1), "", 1
            ),
            "body",
        ),
        (
            DatabaseNoteDraft(
                "n",
                "ok",
                "",
                "x" * (DATABASE_NOTE_KEYWORD_MAX_CHARS + 1),
                1,
            ),
            "keywords",
        ),
        (DatabaseNoteDraft("n", "bad\x00title", "", "", 1), "title"),
        (DatabaseNoteDraft("n", " spaced ", "", "", 1), "title"),
        (DatabaseNoteDraft("n", "ok", "bad\x00body", "", 1), "body"),
    ),
)
def test_invalid_or_transforming_payload_is_a_typed_veto(draft, field):
    result = validate_database_note_draft(draft)

    assert isinstance(result, NoteValidationVeto)
    assert result.field == field
    assert result.revision == draft.revision
    assert result.message


def test_keyword_delimiter_whitespace_is_syntax_not_content():
    draft = DatabaseNoteDraft("n", "ok", "", " alpha , , βeta ", 1)

    result = validate_database_note_draft(draft)

    assert isinstance(result, DatabaseNoteSavePayload)
    assert result.keywords == ("alpha", "βeta")


def test_keyword_limit_is_per_semantic_token_not_aggregate_field():
    keywords = ", ".join(f"topic-{index:02d}" for index in range(20))
    assert len(keywords) > DATABASE_NOTE_KEYWORD_MAX_CHARS

    result = validate_database_note_draft(DatabaseNoteDraft("n", "ok", "", keywords, 1))

    assert isinstance(result, DatabaseNoteSavePayload)
    assert result.keywords == tuple(f"topic-{index:02d}" for index in range(20))


def test_casefold_duplicate_keywords_are_vetoed_not_silently_deduplicated():
    result = validate_database_note_draft(
        DatabaseNoteDraft("n", "ok", "", "Straße, STRASSE", 1)
    )

    assert isinstance(result, NoteValidationVeto)
    assert result.field == "keywords"


def test_cell_ellipsis_honors_wide_unicode_and_keeps_raw_title():
    raw = "研究計画 [draft] roadmap"

    visible = ellipsize_note_title_cells(raw, 10)

    assert cell_len(visible) <= 10
    assert visible.endswith("…")
    assert raw == "研究計画 [draft] roadmap"


def test_cell_ellipsis_keeps_persisted_line_separators_out_of_one_row_header():
    raw = "first line\nsecond\tline"

    visible = ellipsize_note_title_cells(raw, 40)

    assert visible == "first line second line"
    assert "\n" not in visible and "\t" not in visible
    assert raw == "first line\nsecond\tline"


@pytest.mark.parametrize(
    ("budget", "expected"),
    ((0, ""), (1, "…"), (20, "short title")),
)
def test_cell_ellipsis_handles_zero_tiny_and_untruncated_budgets(budget, expected):
    assert ellipsize_note_title_cells("short title", budget) == expected


def test_session_snapshot_is_immutable_and_exposes_canonical_draft_values():
    baseline = NormalizedDatabaseNote(
        note_id="n-1",
        title="Original",
        body="baseline body",
        keywords=("one",),
        version=4,
        created_at="2026-07-01T00:00:00+00:00",
        modified_at="2026-07-02T00:00:00+00:00",
    )
    draft = DatabaseNoteDraft("n-1", "Draft", "new body", "one, two", 3)
    snapshot = LibraryNoteSessionSnapshot(
        baseline=baseline,
        draft=draft,
        session_generation=9,
        saved_revision=2,
        dirty=True,
        saving=False,
        in_conflict=False,
        conflict_generation=0,
        status_message="Unsaved changes",
    )

    assert snapshot.note_id == "n-1"
    assert snapshot.title == "Draft"
    assert snapshot.body == "new body"
    assert snapshot.keywords_text == "one, two"
    assert snapshot.draft_revision == 3
    assert snapshot.version == 4
    with pytest.raises(FrozenInstanceError):
        snapshot.dirty = False


def test_focus_identity_is_portable_immutable_value_state():
    identity = LibraryNotesFocusIdentity(
        stage="notes",
        region="editor",
        note_id="n-1",
        semantic_role="body",
        body_selection_start=(3, 4),
        body_selection_end=(5, 6),
        scroll_offset=(0, 7),
    )

    assert identity == LibraryNotesFocusIdentity(
        stage="notes",
        region="editor",
        note_id="n-1",
        semantic_role="body",
        body_selection_start=(3, 4),
        body_selection_end=(5, 6),
        scroll_offset=(0, 7),
    )
    with pytest.raises(FrozenInstanceError):
        identity.semantic_role = "title"
