"""Lossless state contracts for the Library Prompts adaptive reader."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Library.library_prompts_reader_state import (
    PromptReaderRequest,
    PromptReaderState,
    fail_prompt_reader_request,
    select_prompt_for_reader,
    set_prompt_reader_mode,
    settle_prompt_reader_request,
    update_prompt_reader_basic_lane,
    validate_prompt_reader_draft,
)
from tldw_chatbook.Library.library_prompts_state import (
    build_prompt_editor_state,
    prepare_prompt_artifact_save,
)


def _structured_detail(
    *,
    prompt_id: int = 17,
    version: int = 4,
    name: str = "Release assistant",
    user_title: str = "Delivery contract",
) -> dict[str, object]:
    return {
        "id": prompt_id,
        "local_id": prompt_id,
        "name": name,
        "author": "Advanced Author",
        "details": "Keep this description from the Advanced projection.",
        "keywords": ["release", "advanced-only"],
        "artifact_type": "prompt",
        "prompt_format": "structured",
        "prompt_schema_version": 2,
        "prompt_definition": {
            "schema_version": 2,
            "kind": "block_prompt",
            "lanes": [
                {
                    "id": "system",
                    "blocks": [
                        {
                            "id": "role",
                            "title": "Specialized role",
                            "syntax": "markdown",
                            "content": "Be exact.",
                            "mapping_hint": "Advanced-only system mapping hint.",
                        }
                    ],
                },
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "delivery",
                            "title": user_title,
                            "syntax": "xml",
                            "xml_tag": "delivery_contract",
                            "content": "Ship it.",
                            "mapping_hint": "Advanced-only user mapping hint.",
                        }
                    ],
                },
            ],
        },
        "system_prompt": "stale compatibility text",
        "user_prompt": "stale compatibility text",
        "version": version,
        "backend": "local",
    }


def _loaded_reader() -> tuple[PromptReaderState, object]:
    draft = build_prompt_editor_state(_structured_detail())
    selected, request = select_prompt_for_reader(
        PromptReaderState(), draft.prompt_id or 0, version=draft.version or 0
    )
    return settle_prompt_reader_request(selected, request, draft), draft


def test_prompt_reader_defaults_to_basic_and_supports_three_projections() -> None:
    state = PromptReaderState()

    assert state.mode == "basic"
    assert set_prompt_reader_mode(state, "advanced").mode == "advanced"
    assert set_prompt_reader_mode(state, "info").mode == "info"
    assert set_prompt_reader_mode(state, "basic").mode == "basic"

    with pytest.raises(ValueError, match="basic, advanced, or info"):
        set_prompt_reader_mode(state, "future")  # type: ignore[arg-type]


def test_prompt_reader_settlement_requires_every_fence_and_references_one_draft() -> None:
    draft = build_prompt_editor_state(_structured_detail())
    selected, request = select_prompt_for_reader(PromptReaderState(), 17, version=4)

    assert request == PromptReaderRequest(
        destination="prompts",
        prompt_id=17,
        version=4,
        generation=1,
    )
    assert selected.selected_id == 17
    assert selected.loading is True
    assert selected.draft is None

    mismatches = (
        replace(request, destination="skills"),
        replace(request, prompt_id=18),
        replace(request, version=5),
        replace(request, generation=2),
    )
    for stale_request in mismatches:
        assert settle_prompt_reader_request(selected, stale_request, draft) is selected

    assert (
        settle_prompt_reader_request(
            selected,
            request,
            replace(draft, prompt_id=18),
        )
        is selected
    )
    assert (
        settle_prompt_reader_request(
            selected,
            request,
            replace(draft, version=5),
        )
        is selected
    )

    settled = settle_prompt_reader_request(selected, request, draft)

    assert settled.loaded_id == 17
    assert settled.loaded_version == 4
    assert settled.loaded_generation == 1
    assert settled.loading is False
    assert settled.loaded_actions_eligible is True
    assert settled.draft is draft
    assert settled.draft.block_editor_state is draft.block_editor_state


def test_prompt_reader_retains_previous_draft_during_load_and_failure() -> None:
    loaded, draft = _loaded_reader()
    selected, request = select_prompt_for_reader(loaded, 18, version=2)

    assert selected.draft is draft
    assert selected.loaded_id == 17
    assert selected.selected_id == 18
    assert selected.loading is True
    assert selected.loaded_actions_eligible is False

    failed = fail_prompt_reader_request(selected, request, "Prompt unavailable.")

    assert failed.draft is draft
    assert failed.error == "Prompt unavailable."
    assert failed.loading is False
    assert failed.unavailable is False
    assert failed.loaded_actions_eligible is False

    assert (
        fail_prompt_reader_request(
            selected, replace(request, generation=request.generation + 1), "late"
        )
        is selected
    )


def test_basic_edit_preserves_advanced_only_fields_byte_for_byte_in_save_payload() -> None:
    loaded, original_draft = _loaded_reader()
    original_definition = _structured_detail()["prompt_definition"]
    assert isinstance(original_definition, dict)
    expected_definition = json.loads(json.dumps(original_definition))
    expected_definition["lanes"][1]["blocks"][0]["content"] = (
        "Ship the safer release."
    )

    updated = update_prompt_reader_basic_lane(
        loaded,
        lane="user",
        content="Ship the safer release.",
    )
    updated_draft = updated.draft
    assert updated_draft is not None
    assert updated_draft is not original_draft
    assert updated_draft.author == "Advanced Author"
    assert updated_draft.details == (
        "Keep this description from the Advanced projection."
    )
    assert updated_draft.keywords_csv == "release, advanced-only"
    assert updated_draft.block_editor_state is not None

    artifact, payload, _prepared = prepare_prompt_artifact_save(
        updated_draft.block_editor_state,
        artifact_type=updated_draft.artifact_type,
        include_recipe_starter_content=True,
        request_fields={
            "name": updated_draft.name,
            "author": updated_draft.author,
            "details": updated_draft.details,
            "keywords": ["release", "advanced-only"],
            "expected_version": updated_draft.version,
        },
    )

    assert payload["prompt_definition"] == expected_definition
    assert payload["author"] == "Advanced Author"
    assert payload["details"] == (
        "Keep this description from the Advanced projection."
    )
    assert payload["keywords"] == ["release", "advanced-only"]
    assert artifact.definition_bytes == json.dumps(
        expected_definition,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_validation_returns_owning_mode_and_target_without_mutating_draft() -> None:
    invalid_draft = build_prompt_editor_state(
        _structured_detail(user_title="")
    )
    selected, request = select_prompt_for_reader(PromptReaderState(), 17, version=4)
    state = settle_prompt_reader_request(selected, request, invalid_draft)

    target = validate_prompt_reader_draft(state)

    assert target is not None
    assert target.mode == "advanced"
    assert target.control_id == "library-prompt-block-editor"
    assert target.block_id == "delivery"
    assert target.block_field == "title"
    assert "title" in target.message.lower()
    assert state.mode == "basic"
    assert state.draft is invalid_draft

    unnamed = replace(state, draft=replace(invalid_draft, name=""))
    name_target = validate_prompt_reader_draft(unnamed)

    assert name_target is not None
    assert name_target.mode == "basic"
    assert name_target.control_id == "library-prompt-name"
    assert name_target.block_id is None
    assert name_target.block_field is None
    assert unnamed.draft is not None
    assert unnamed.draft.name == ""
