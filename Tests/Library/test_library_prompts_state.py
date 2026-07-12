"""Pure display-state contracts for the Library prompts canvas."""
import sqlite3
from datetime import datetime, timezone

from tldw_chatbook.DB.Prompts_DB import ConflictError
from tldw_chatbook.Library.library_prompts_state import (
    PromptEditorState,
    PromptListRow,
    build_prompt_editor_state,
    build_prompts_list_state,
    classify_prompt_save_error,
    prompt_editor_meta_line,
)

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone.utc)

PROMPT_A = {
    "id": 1,
    "name": "Summarize",
    "author": "Alice",
    "details": "Summarizes text",
    "system_prompt": "You are helpful.",
    "user_prompt": "Summarize: {text}",
    "keywords": ["writing", "summary"],
    "last_modified": "2026-07-07T11:57:00+00:00",
    "version": 2,
}
PROMPT_B = {
    "id": 2,
    "name": "brainstorm",
    "author": "",
    "keywords": [],
    "last_modified": "2026-07-06T12:00:00+00:00",
    "version": 1,
}
PROMPT_C = {
    "id": 3,
    "name": "Zeta ideas",
    "author": None,
    "details": "Ideas for the offsite",
    "keywords": ["kw1", "kw2"],
    "last_modified": "2026-07-07T11:00:00+00:00",
}


def test_list_state_newest_sort_orders_by_modified_desc():
    state = build_prompts_list_state([PROMPT_B, PROMPT_A], query="", sort="newest", now=NOW)
    assert [row.prompt_id for row in state.rows] == [1, 2]
    assert state.count == 2
    assert state.sort == "newest"


def test_list_state_name_sort_alpha_ci():
    state = build_prompts_list_state([PROMPT_A, PROMPT_B], query="", sort="name", now=NOW)
    assert [row.name for row in state.rows] == ["brainstorm", "Summarize"]
    assert state.sort == "name"


def test_list_state_query_matches_name_case_insensitively():
    state = build_prompts_list_state([PROMPT_A, PROMPT_B], query="BRAIN", sort="newest", now=NOW)
    assert [row.prompt_id for row in state.rows] == [2]
    assert state.count == 1


def test_list_state_query_matches_details_case_insensitively():
    """D2/U1: the filter matches ``details`` -- a field list-page records
    actually carry (unlike ``keywords``, which real list rows never do --
    see ``_prompts_page_records_or_empty``)."""
    state = build_prompts_list_state([PROMPT_A, PROMPT_B], query="SUMMARIZES", sort="newest", now=NOW)
    assert [row.prompt_id for row in state.rows] == [1]


def test_list_state_query_does_not_silently_match_keywords_absent_from_list_rows():
    """D2/U1 regression: the old behavior matched ``keywords`` -- a field
    real list-page records never carry -- which could never actually match
    anything in production. PROMPT_A's ``keywords`` field only exists here
    because this fixture also doubles for the editor-detail-shaped tests
    below; "WRITING" (one of its keywords) is absent from every record's
    name/details, so the filter must now find nothing."""
    state = build_prompts_list_state([PROMPT_A, PROMPT_B], query="WRITING", sort="newest", now=NOW)
    assert state.rows == ()


def test_list_state_secondary_omits_empty_details():
    state = build_prompts_list_state([PROMPT_B], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(prompt_id=2, name="brainstorm", secondary="1d")


def test_list_state_secondary_shows_details_and_age():
    state = build_prompts_list_state([PROMPT_A], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=1, name="Summarize", secondary="Summarizes text · 3m"
    )


def test_list_state_secondary_ignores_author_and_keywords_even_when_present():
    """D2/U1: author/keywords are dropped from the secondary line entirely
    now, even when a record happens to carry them (PROMPT_C's ``author``/
    ``keywords`` here only exist because this fixture doubles for the
    editor-detail tests below) -- only details + age surface."""
    state = build_prompts_list_state([PROMPT_C], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=3, name="Zeta ideas", secondary="Ideas for the offsite · 1h"
    )


def test_editor_state_maps_fetch_prompt_details_fields():
    state = build_prompt_editor_state(PROMPT_A)
    assert state == PromptEditorState(
        prompt_id=1,
        name="Summarize",
        author="Alice",
        details="Summarizes text",
        system_prompt="You are helpful.",
        user_prompt="Summarize: {text}",
        keywords_csv="writing, summary",
        version=2,
        created="",
        modified="2026-07-07T11:57:00+00:00",
    )


def test_editor_state_tolerates_empty_mapping():
    state = build_prompt_editor_state({})
    assert state == PromptEditorState(
        prompt_id=None,
        name="",
        author="",
        details="",
        system_prompt="",
        user_prompt="",
        keywords_csv="",
        version=None,
        created="",
        modified="",
    )


def test_classify_soft_deleted_name():
    message = "Prompt 'Foo' exists but is soft-deleted. Use overwrite to restore/update."
    assert classify_prompt_save_error(None, message, None) == "soft-deleted-name"


def test_classify_conflict_error():
    assert classify_prompt_save_error(None, "", ConflictError("x")) == "conflict"


def test_classify_name_in_use_from_integrity_error():
    exc = sqlite3.IntegrityError("UNIQUE constraint failed: Prompts.name")
    assert classify_prompt_save_error(None, "", exc) == "name-in-use"


def test_classify_ok():
    assert classify_prompt_save_error(5, "", None) == "ok"


def test_classify_error_fallback():
    assert classify_prompt_save_error(None, "boom", RuntimeError("boom")) == "error"


def test_meta_line_new_prompt_sentinel_overrides_modified_and_version():
    """Task 8b D1: a blank, not-yet-saved editor state (``prompt_id=None``)
    renders "New prompt", never "Modified … · vN" -- even when the caller
    (a malformed record) happens to also carry ``modified``/``version``."""
    state = build_prompt_editor_state({"last_modified": "2026-07-07T11:00:00+00:00", "version": 3})
    assert state.prompt_id is None
    assert prompt_editor_meta_line(state) == "New prompt"


def test_meta_line_existing_prompt_unaffected_by_new_prompt_sentinel():
    state = build_prompt_editor_state(PROMPT_A)
    assert prompt_editor_meta_line(state, now=NOW) == "Modified 3m · v2"


def test_meta_line_appends_unsaved_marker_when_dirty():
    """U6 (Task 8c): a dirty editor's meta line gets a trailing unsaved
    marker -- ``dirty`` is a plain pure-function input, not derived from
    ``PromptEditorState`` itself."""
    state = build_prompt_editor_state(PROMPT_A)
    assert prompt_editor_meta_line(state, now=NOW, dirty=True) == (
        "Modified 3m · v2 · • Unsaved changes"
    )


def test_meta_line_omits_unsaved_marker_when_not_dirty():
    """``dirty`` defaults to ``False`` -- existing callers that never pass
    it keep the exact same rendering as before this change."""
    state = build_prompt_editor_state(PROMPT_A)
    assert prompt_editor_meta_line(state, now=NOW, dirty=False) == "Modified 3m · v2"
    assert prompt_editor_meta_line(state, now=NOW) == "Modified 3m · v2"


def test_meta_line_new_prompt_sentinel_appends_unsaved_marker_when_dirty():
    """The "New prompt" sentinel also gets the unsaved marker once the user
    starts typing into a blank create-flow record (dirty becomes True)."""
    state = build_prompt_editor_state({})
    assert prompt_editor_meta_line(state, dirty=True) == "New prompt · • Unsaved changes"
    assert prompt_editor_meta_line(state) == "New prompt"
