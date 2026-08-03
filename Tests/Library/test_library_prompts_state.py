"""Pure display-state contracts for the Library prompts canvas."""

from dataclasses import replace
import sqlite3
from datetime import datetime, timezone

import pytest

from tldw_chatbook.DB.Prompts_DB import ConflictError
from tldw_chatbook.Library.library_prompts_state import (
    PromptArtifactDraft,
    PromptListRow,
    prepare_prompt_artifact_save,
    build_prompt_editor_state,
    build_prompts_list_state,
    classify_prompt_save_error,
    prompt_editor_meta_line,
    require_artifact_save_supported,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    blank_recipe,
    outcome_first_recipe,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    PromptCapabilityError,
    PromptSourceCapabilities,
    local_prompt_capabilities,
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
    state = build_prompts_list_state(
        [PROMPT_B, PROMPT_A], query="", sort="newest", now=NOW
    )
    assert [row.prompt_id for row in state.rows] == [1, 2]
    assert state.count == 2
    assert state.sort == "newest"


def test_list_state_name_sort_alpha_ci():
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="", sort="name", now=NOW
    )
    assert [row.name for row in state.rows] == ["brainstorm", "Summarize"]
    assert state.sort == "name"


def test_list_state_query_matches_name_case_insensitively():
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="BRAIN", sort="newest", now=NOW
    )
    assert [row.prompt_id for row in state.rows] == [2]
    assert state.count == 1


def test_list_state_query_matches_details_case_insensitively():
    """D2/U1: the filter matches ``details`` -- a field list-page records
    actually carry (unlike ``keywords``, which real list rows never do --
    see ``_prompts_page_records_or_empty``)."""
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="SUMMARIZES", sort="newest", now=NOW
    )
    assert [row.prompt_id for row in state.rows] == [1]


def test_list_state_query_does_not_silently_match_keywords_absent_from_list_rows():
    """D2/U1 regression: the old behavior matched ``keywords`` -- a field
    real list-page records never carry -- which could never actually match
    anything in production. PROMPT_A's ``keywords`` field only exists here
    because this fixture also doubles for the editor-detail-shaped tests
    below; "WRITING" (one of its keywords) is absent from every record's
    name/details, so the filter must now find nothing."""
    state = build_prompts_list_state(
        [PROMPT_A, PROMPT_B], query="WRITING", sort="newest", now=NOW
    )
    assert state.rows == ()


def test_list_state_secondary_omits_empty_details():
    state = build_prompts_list_state([PROMPT_B], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=2, name="brainstorm", secondary="1d"
    )


def test_list_state_secondary_shows_details_and_age():
    state = build_prompts_list_state([PROMPT_A], query="", sort="newest", now=NOW)
    assert state.rows[0] == PromptListRow(
        prompt_id=1,
        name="Summarize",
        secondary="Summarizes text · 3m",
        lane_summary="System + User",
    )


def test_list_rows_label_prompt_recipe_source_and_normalized_lane_summary():
    recipe = {
        **PROMPT_A,
        "id": 9,
        "name": "Outcome first",
        "artifact_type": "recipe",
        "backend": "server",
        "has_system_prompt": True,
        "has_user_prompt": False,
    }
    empty_prompt = {
        **PROMPT_B,
        "id": 10,
        "has_system_prompt": False,
        "has_user_prompt": False,
    }

    state = build_prompts_list_state(
        [recipe, empty_prompt], query="", sort="name", now=NOW
    )

    rows = {row.prompt_id: row for row in state.rows}
    assert rows[9].artifact_type == "recipe"
    assert rows[9].type_label == "Recipe"
    assert rows[9].source_label == "Server"
    assert rows[9].lane_summary == "System only"
    assert rows[10].type_label == "Prompt"
    assert rows[10].lane_summary == "Empty"


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
    assert (
        state.prompt_id,
        state.name,
        state.author,
        state.details,
        state.system_prompt,
        state.user_prompt,
        state.keywords_csv,
        state.version,
        state.created,
        state.modified,
    ) == (
        1,
        "Summarize",
        "Alice",
        "Summarizes text",
        "You are helpful.",
        "Summarize: {text}",
        "writing, summary",
        2,
        "",
        "2026-07-07T11:57:00+00:00",
    )
    assert state.block_editor_state is not None


def _v2_detail(*, artifact_type: str = "prompt") -> dict[str, object]:
    kind = "block_recipe" if artifact_type == "recipe" else "block_prompt"
    return {
        "id": 17,
        "name": "Structured",
        "artifact_type": artifact_type,
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": {
            "schema_version": 2,
            "kind": kind,
            "lanes": [
                {
                    "id": "system",
                    "blocks": [
                        {
                            "id": "role",
                            "title": "Role",
                            "syntax": "markdown",
                            "content": "Be precise.",
                            "mapping_hint": "Define the model's role.",
                        }
                    ],
                },
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "xml",
                            "xml_tag": "goal",
                            "content": "Ship the release.",
                        }
                    ],
                },
            ],
        },
        "system_prompt": "stale compatibility text",
        "user_prompt": "stale compatibility text",
        "version": 4,
        "backend": "local",
    }


def test_editor_state_decodes_supported_v2_into_shared_immutable_block_state():
    state = build_prompt_editor_state(_v2_detail())

    assert state.artifact_type == "prompt"
    assert state.definition_state == "supported_v2"
    assert state.block_editor_state is not None
    assert state.block_editor_state.definition.kind == "block_prompt"
    assert state.compiled_system_preview == "# Role\n\nBe precise."
    assert state.compiled_user_preview == "<goal>Ship the release.</goal>"
    assert state.compatibility_stale is True


def test_editor_state_decomposes_legacy_prompt_without_changing_lane_origins():
    detail = {
        **PROMPT_A,
        "system_prompt": "  exact system\n",
        "user_prompt": "exact user\n\n",
    }

    state = build_prompt_editor_state(detail)

    assert state.definition_state == "legacy"
    assert state.block_editor_state is not None
    assert state.block_editor_state.compiled_system == "  exact system\n"
    assert state.block_editor_state.compiled_user == "exact user\n\n"
    assert state.block_editor_state.system_origin is not None
    assert state.block_editor_state.user_origin is not None


def test_editor_state_keeps_foreign_or_malformed_artifacts_read_only_and_visible():
    detail = _v2_detail(artifact_type="recipe")
    detail["prompt_schema_version"] = 1

    state = build_prompt_editor_state(detail)

    assert state.artifact_type == "recipe"
    assert state.definition_state == "foreign_v1"
    assert state.block_editor_state is None
    assert state.compiled_system_preview == "stale compatibility text"
    assert state.can_convert_as_new is True
    assert "read-only" in state.compatibility_reason.lower()


def test_outcome_first_recipe_has_stable_blank_markdown_blocks_in_both_lanes():
    first = outcome_first_recipe()
    second = outcome_first_recipe()

    assert first == second
    assert first is not second
    assert first.kind == "block_recipe"
    assert tuple(block.id for block in first.lanes[0].blocks) == (
        "role",
        "personality",
        "collaboration-style",
    )
    assert tuple(block.id for block in first.lanes[1].blocks) == (
        "goal",
        "success-criteria",
        "context-evidence",
        "constraints",
        "output",
        "stop-rules",
    )
    assert all(
        block.syntax == "markdown"
        and block.content == ""
        and block.mapping_hint
        and block.xml_tag is None
        for lane in first.lanes
        for block in lane.blocks
    )


def test_blank_recipe_is_a_fresh_immutable_two_lane_recipe():
    first = blank_recipe()
    second = blank_recipe()

    assert first == second
    assert first is not second
    assert first.kind == "block_recipe"
    assert tuple(lane.id for lane in first.lanes) == ("system", "user")
    assert all(lane.blocks == () for lane in first.lanes)


def _draft(*, artifact_type: str = "recipe") -> PromptArtifactDraft:
    definition = outcome_first_recipe()
    if artifact_type == "prompt":
        definition = replace(definition, kind="block_prompt")
    return PromptArtifactDraft(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        definition=definition,
        system_prompt="",
        user_prompt="",
        definition_bytes=b"{}",
        request_bytes=b"{}",
    )


def test_require_artifact_save_supported_accepts_exact_local_recipe_contract():
    require_artifact_save_supported(_draft(), local_prompt_capabilities())


def test_require_artifact_save_supported_rejects_type_kind_mismatch():
    draft = replace(_draft(), artifact_type="prompt")

    with pytest.raises(ValueError, match="artifact_type.*kind.*agree"):
        require_artifact_save_supported(draft, local_prompt_capabilities())


def test_require_artifact_save_supported_names_source_limit_and_recovery():
    capabilities = replace(local_prompt_capabilities(), compiled_lane_limit=3)
    draft = replace(_draft(), user_prompt="four")

    with pytest.raises(ValueError, match="user_prompt.*3 characters.*shorten"):
        require_artifact_save_supported(draft, capabilities)


def test_require_artifact_save_supported_names_definition_and_request_byte_limits():
    definition_limited = replace(local_prompt_capabilities(), definition_limit=1)
    request_limited = replace(local_prompt_capabilities(), request_limit=1)

    with pytest.raises(ValueError, match="prompt_definition.*1 UTF-8 bytes"):
        require_artifact_save_supported(_draft(), definition_limited)
    with pytest.raises(ValueError, match="request.*1 UTF-8 bytes"):
        require_artifact_save_supported(_draft(), request_limited)


def test_require_artifact_save_supported_rejects_missing_kind_capability():
    capabilities = replace(local_prompt_capabilities(), structured_kinds=frozenset())

    with pytest.raises(PromptCapabilityError, match="structured kind"):
        require_artifact_save_supported(_draft(), capabilities)


def test_require_artifact_save_supported_guards_update_version_and_capability():
    capabilities: PromptSourceCapabilities = replace(
        local_prompt_capabilities(), conditional_update=False
    )

    with pytest.raises(ValueError, match="conditional update.*save as new"):
        require_artifact_save_supported(
            _draft(), capabilities, update_original=True, expected_version=3
        )
    with pytest.raises(ValueError, match="current version.*Reload"):
        require_artifact_save_supported(
            _draft(), local_prompt_capabilities(), update_original=True
        )


def test_prepare_recipe_save_defaults_to_empty_content_and_preserves_structure():
    definition = outcome_first_recipe()
    populated = replace(
        definition,
        lanes=(
            replace(
                definition.lanes[0],
                blocks=(replace(definition.lanes[0].blocks[0], content="Architect"),),
            ),
            definition.lanes[1],
        ),
    )
    state = build_prompt_editor_state(
        {
            "artifact_type": "recipe",
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": {
                "kind": populated.kind,
                "schema_version": populated.schema_version,
                "lanes": [
                    {
                        "id": lane.id,
                        "blocks": [
                            {
                                "id": block.id,
                                "title": block.title,
                                "syntax": block.syntax,
                                "content": block.content,
                                "mapping_hint": block.mapping_hint,
                            }
                            for block in lane.blocks
                        ],
                    }
                    for lane in populated.lanes
                ],
            },
        }
    ).block_editor_state
    assert state is not None

    draft, payload, saved_state = prepare_prompt_artifact_save(
        state,
        artifact_type="recipe",
        include_recipe_starter_content=False,
        request_fields={"name": "Outcome first", "keywords": None},
    )

    assert draft.artifact_type == "recipe"
    assert draft.system_prompt == ""
    assert all(
        block.content == ""
        for lane in saved_state.definition.lanes
        for block in lane.blocks
    )
    assert saved_state.definition.lanes[0].blocks[0].title == "Role"
    assert (
        saved_state.definition.lanes[0].blocks[0].mapping_hint
        == "Define the model's function and job."
    )
    assert payload["artifact_type"] == "recipe"
    assert "keywords" not in payload
    assert payload["prompt_definition"]["kind"] == "block_recipe"
    assert draft.definition_bytes
    assert draft.request_bytes


def test_prepare_recipe_save_preserves_content_only_when_explicitly_selected():
    state = build_prompt_editor_state(
        {"system_prompt": "Stay direct.", "user_prompt": "Draft the plan."}
    ).block_editor_state
    assert state is not None

    draft, payload, saved_state = prepare_prompt_artifact_save(
        state,
        artifact_type="recipe",
        include_recipe_starter_content=True,
        request_fields={"name": "Planning recipe"},
    )

    assert draft.system_prompt == "Stay direct."
    assert draft.user_prompt == "Draft the plan."
    assert saved_state.artifact_type == "recipe"
    assert payload["prompt_definition"]["kind"] == "block_recipe"


def test_editor_state_resolves_prompt_id_from_local_id_when_id_is_composite_string():
    """Critical regression: the REAL production seam
    (``PromptScopeService.get_prompt`` -> ``normalize_prompt_record``, see
    ``tldw_chatbook/Prompt_Management/prompt_normalizers.py``) returns
    ``detail["id"]`` as the COMPOSITE STRING ``"<backend>:prompt:<uuid>"``
    -- the raw local numeric id lives under ``detail["local_id"]`` instead.
    ``_to_int`` silently swallows the ``ValueError`` on the composite
    string, so ``build_prompt_editor_state`` used to return
    ``prompt_id=None`` for every EXISTING saved prompt loaded this way,
    which made ``prompt_editor_meta_line`` render "New prompt" instead of
    "Modified ... · vN". ``build_prompt_editor_state`` must prefer
    ``local_id`` when present."""
    detail = {
        "id": "local:prompt:9f4e2f0a-1111-2222-3333-444455556666",
        "backend": "local",
        "source_id": "9f4e2f0a-1111-2222-3333-444455556666",
        "local_id": 7,
        "server_id": None,
        "uuid": "9f4e2f0a-1111-2222-3333-444455556666",
        "name": "Summarize",
        "author": "Alice",
        "details": "Summarizes text",
        "system_prompt": "You are helpful.",
        "user_prompt": "Summarize: {text}",
        "keywords": ["writing", "summary"],
        "version": 2,
        "last_modified": "2026-07-07T11:57:00+00:00",
    }
    state = build_prompt_editor_state(detail)
    assert state.prompt_id == 7
    assert prompt_editor_meta_line(state, now=NOW) == "Modified 3m · v2"


def test_editor_state_prompt_id_none_when_local_id_absent_and_id_is_composite_string():
    """The server-backend shape (``local_id`` present but ``None``, ``id``
    a composite string) must still resolve to ``prompt_id=None`` rather
    than raising -- unchanged from before this fix (server prompts were
    never resolvable via the plain ``id`` field either)."""
    detail = {
        "id": "server:prompt:9f4e2f0a-1111-2222-3333-444455556666",
        "backend": "server",
        "local_id": None,
        "server_id": 7,
        "name": "Summarize",
    }
    state = build_prompt_editor_state(detail)
    assert state.prompt_id is None


def test_editor_state_prompt_id_none_for_blank_create_flow_detail():
    """The D1 blank-create / Duplicate-action detail shapes
    (``_enter_library_prompt_create_editor``,
    ``handle_library_prompt_duplicate``) never carry an ``id`` or
    ``local_id`` key at all -- ``prompt_id`` must stay ``None`` so the
    editor still renders "New prompt", not a false "Modified ... · vN"."""
    detail = {
        "name": "Brand New (copy)",
        "author": "Alice",
        "details": "d",
        "system_prompt": "s",
        "user_prompt": "u",
        "keywords": "kw1, kw2",
    }
    state = build_prompt_editor_state(detail)
    assert state.prompt_id is None
    assert prompt_editor_meta_line(state) == "New prompt"


def test_editor_state_tolerates_empty_mapping():
    state = build_prompt_editor_state({})
    assert (
        state.prompt_id,
        state.name,
        state.author,
        state.details,
        state.system_prompt,
        state.user_prompt,
        state.keywords_csv,
        state.version,
        state.created,
        state.modified,
    ) == (None, "", "", "", "", "", "", None, "", "")
    assert state.block_editor_state is not None


def test_classify_soft_deleted_name():
    message = (
        "Prompt 'Foo' exists but is soft-deleted. Use overwrite to restore/update."
    )
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
    state = build_prompt_editor_state(
        {"last_modified": "2026-07-07T11:00:00+00:00", "version": 3}
    )
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
    assert (
        prompt_editor_meta_line(state, dirty=True) == "New prompt · • Unsaved changes"
    )
    assert prompt_editor_meta_line(state) == "New prompt"
