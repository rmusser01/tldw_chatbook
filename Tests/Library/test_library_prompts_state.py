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


def test_list_state_query_matches_keyword_case_insensitively():
    state = build_prompts_list_state([PROMPT_A, PROMPT_B], query="WRITING", sort="newest", now=NOW)
    assert [row.prompt_id for row in state.rows] == [1]


def test_list_state_secondary_omits_empty_author_and_keywords():
    state = build_prompts_list_state([PROMPT_B], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(prompt_id=2, name="brainstorm", secondary="1d")


def test_list_state_secondary_includes_all_parts():
    state = build_prompts_list_state([PROMPT_A], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=1, name="Summarize", secondary="Alice · writing, summary · 3m"
    )


def test_list_state_secondary_omits_only_empty_author():
    state = build_prompts_list_state([PROMPT_C], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=3, name="Zeta ideas", secondary="kw1, kw2 · 1h"
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
