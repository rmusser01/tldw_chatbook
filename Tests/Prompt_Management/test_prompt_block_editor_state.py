"""Pure-state contract for the shared Prompt/Recipe block editor."""

from __future__ import annotations

from hashlib import sha256

import pytest

from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    LegacyLaneOrigin,
    PromptBlock,
    PromptLane,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor_state import (
    ADDITIONAL_CONTEXT_RESERVED_PREFIX,
    PromptBlockEditorState,
    add_block,
    delete_block,
    duplicate_block,
    move_block,
    set_artifact_type,
    update_block,
)


def _definition() -> BlockArtifactDefinition:
    return BlockArtifactDefinition(
        kind="block_prompt",
        schema_version=2,
        lanes=(
            PromptLane(
                id="system",
                blocks=(
                    PromptBlock(
                        id="role",
                        title="Role",
                        syntax="markdown",
                        content="Be exact.",
                    ),
                ),
            ),
            PromptLane(
                id="user",
                blocks=(
                    PromptBlock(
                        id="goal",
                        title="Goal",
                        syntax="freeform",
                        content="Explain the result.",
                    ),
                    PromptBlock(
                        id="context",
                        title="Context",
                        syntax="freeform",
                        content="Use the supplied evidence.",
                    ),
                ),
            ),
        ),
    )


def _definition_with_additional_context(
    *,
    kind: str = "block_prompt",
    lane_id: str = "user",
    block_id: str = ADDITIONAL_CONTEXT_RESERVED_PREFIX,
) -> BlockArtifactDefinition:
    definition = _definition()
    lanes = list(definition.lanes)
    lane_index = 0 if lane_id == "system" else 1
    lane = lanes[lane_index]
    lanes[lane_index] = PromptLane(
        id=lane.id,
        blocks=lane.blocks
        + (
            PromptBlock(
                id=block_id,
                title="Additional context",
                syntax="markdown",
                content="Unmatched evidence.",
            ),
        ),
    )
    return BlockArtifactDefinition(
        kind=kind,  # type: ignore[arg-type]
        schema_version=2,
        lanes=tuple(lanes),  # type: ignore[arg-type]
    )


def _definition_with_duplicate_additional_context() -> BlockArtifactDefinition:
    definition = _definition_with_additional_context()
    user_lane = definition.lanes[1]
    duplicate_lane = PromptLane(
        id="user",
        blocks=user_lane.blocks
        + (
            PromptBlock(
                id=ADDITIONAL_CONTEXT_RESERVED_PREFIX,
                title="Second mapped context",
                syntax="freeform",
                content="Duplicate evidence.",
            ),
        ),
    )
    malformed = object.__new__(BlockArtifactDefinition)
    object.__setattr__(malformed, "kind", "block_prompt")
    object.__setattr__(malformed, "schema_version", 2)
    object.__setattr__(
        malformed,
        "lanes",
        (definition.lanes[0], duplicate_lane),
    )
    return malformed


def _origin(text: str) -> LegacyLaneOrigin:
    return LegacyLaneOrigin(
        text=text,
        fingerprint=sha256(text.encode("utf-8")).hexdigest(),
    )


def _state(*, legacy: bool = False) -> PromptBlockEditorState:
    return PromptBlockEditorState.from_definition(
        artifact_type="prompt",
        definition=_definition(),
        system_origin=_origin("SYSTEM BYTES\n") if legacy else None,
        user_origin=_origin("USER BYTES\n\n") if legacy else None,
    )


def _ids(state: PromptBlockEditorState, lane_index: int) -> list[str]:
    return [block.id for block in state.definition.lanes[lane_index].blocks]


def test_state_compiles_preview_and_preserves_unchanged_legacy_bytes() -> None:
    structured = _state()
    legacy = _state(legacy=True)

    assert structured.compiled_system == "# Role\n\nBe exact."
    assert structured.compiled_user == (
        "Explain the result.\n\nUse the supplied evidence."
    )
    assert legacy.compiled_system == "SYSTEM BYTES\n"
    assert legacy.compiled_user == "USER BYTES\n\n"
    assert legacy.issues == ()


def test_add_block_uses_stable_collision_safe_ids_and_exact_lane_order() -> None:
    state = add_block(_state(), "user", title="Evidence")
    state = add_block(state, "user", title="Constraints")

    assert _ids(state, 1) == ["goal", "context", "block", "block-2"]
    assert state.dirty_block_ids == frozenset({"block", "block-2"})
    assert state.definition.lanes[1].blocks[-1].title == "Constraints"


def test_add_and_rename_reject_reserved_additional_context_ids() -> None:
    with pytest.raises(ValueError, match="reserved for mapped Additional context"):
        add_block(
            _state(),
            "user",
            block_id=ADDITIONAL_CONTEXT_RESERVED_PREFIX,
        )

    with pytest.raises(ValueError, match="reserved for mapped Additional context"):
        update_block(
            _state(),
            "goal",
            id=f"{ADDITIONAL_CONTEXT_RESERVED_PREFIX}-2",
        )


def test_prompt_mount_accepts_one_canonical_user_additional_context_block() -> None:
    state = PromptBlockEditorState.from_definition(
        artifact_type="prompt",
        definition=_definition_with_additional_context(),
    )

    mapped = state.definition.lanes[1].blocks[-1]
    assert mapped.id == ADDITIONAL_CONTEXT_RESERVED_PREFIX
    assert mapped.content == "Unmatched evidence."
    assert state.compiled_user.endswith("# Additional context\n\nUnmatched evidence.")


def test_mapped_additional_context_edits_and_reorder_survive_rebuild() -> None:
    state = PromptBlockEditorState.from_definition(
        artifact_type="prompt",
        definition=_definition_with_additional_context(),
    )
    state = update_block(
        state,
        ADDITIONAL_CONTEXT_RESERVED_PREFIX,
        title="Evidence appendix",
        syntax="xml",
        xml_tag="evidence_appendix",
        content="Edited unmatched evidence.",
    )
    state = move_block(state, ADDITIONAL_CONTEXT_RESERVED_PREFIX, -1)
    state = move_block(state, ADDITIONAL_CONTEXT_RESERVED_PREFIX, -1)

    rebuilt = PromptBlockEditorState.from_definition(
        artifact_type="prompt",
        definition=state.definition,
    )

    mapped = rebuilt.definition.lanes[1].blocks[0]
    assert (
        mapped.id,
        mapped.title,
        mapped.syntax,
        mapped.xml_tag,
        mapped.content,
    ) == (
        ADDITIONAL_CONTEXT_RESERVED_PREFIX,
        "Evidence appendix",
        "xml",
        "evidence_appendix",
        "Edited unmatched evidence.",
    )
    assert rebuilt.compiled_user.startswith(
        "<evidence_appendix>Edited unmatched evidence.</evidence_appendix>"
    )


def test_recipe_mount_rejects_exact_additional_context_id() -> None:
    with pytest.raises(ValueError, match="reserved for mapped Additional context"):
        PromptBlockEditorState.from_definition(
            artifact_type="recipe",
            definition=_definition_with_additional_context(kind="block_recipe"),
        )


def test_prompt_system_lane_rejects_exact_additional_context_id() -> None:
    with pytest.raises(ValueError, match="reserved for mapped Additional context"):
        PromptBlockEditorState.from_definition(
            artifact_type="prompt",
            definition=_definition_with_additional_context(lane_id="system"),
        )


def test_prompt_mount_rejects_duplicate_exact_additional_context_ids() -> None:
    with pytest.raises(ValueError, match="reserved for mapped Additional context"):
        PromptBlockEditorState.from_definition(
            artifact_type="prompt",
            definition=_definition_with_duplicate_additional_context(),
        )


@pytest.mark.parametrize(
    "block_id",
    [
        "additional_context",
        "additional-context-2",
        "Additional-Context",
    ],
)
def test_prompt_user_lane_rejects_other_reserved_namespace_spellings(
    block_id: str,
) -> None:
    with pytest.raises(ValueError, match="reserved for mapped Additional context"):
        PromptBlockEditorState.from_definition(
            artifact_type="prompt",
            definition=_definition_with_additional_context(block_id=block_id),
        )


def test_move_block_honors_boundaries_and_preserves_ids() -> None:
    state = _state()

    assert move_block(state, "goal", -1) is state
    moved = move_block(state, "goal", 1)
    assert _ids(moved, 1) == ["context", "goal"]
    assert moved.dirty_block_ids == frozenset({"goal"})
    assert move_block(moved, "goal", 1) is moved


def test_duplicate_inserts_after_source_with_a_new_stable_id() -> None:
    state = duplicate_block(_state(), "goal")

    assert _ids(state, 1) == ["goal", "goal-copy", "context"]
    duplicate = state.definition.lanes[1].blocks[1]
    assert duplicate.title == "Goal copy"
    assert duplicate.content == "Explain the result."
    assert duplicate.id != "goal"
    assert state.dirty_block_ids == frozenset({"goal-copy"})


def test_duplicate_avoids_existing_copy_id_collisions() -> None:
    once = duplicate_block(_state(), "goal")
    twice = duplicate_block(once, "goal")

    assert _ids(twice, 1) == ["goal", "goal-copy-2", "goal-copy", "context"]


def test_duplicate_keeps_an_empty_xml_tag_draft_repairable() -> None:
    state = update_block(_state(), "goal", syntax="xml", xml_tag="goal")
    state = update_block(state, "goal", xml_tag="")

    duplicated = duplicate_block(state, "goal")

    source, copy, _context = duplicated.definition.lanes[1].blocks
    assert (source.xml_tag, copy.xml_tag) == ("", "")
    assert copy.content == source.content == "Explain the result."
    assert copy.title == "Goal copy"
    assert copy.id == "goal-copy"
    assert [(issue.block_id, issue.code) for issue in duplicated.issues] == [
        ("goal", "invalid_xml_name"),
        ("goal-copy", "invalid_xml_name"),
    ]


def test_delete_removes_only_the_target_and_records_the_dirty_id() -> None:
    state = delete_block(_state(), "goal")

    assert _ids(state, 1) == ["context"]
    assert state.dirty_block_ids == frozenset({"goal"})
    assert state.compiled_user == "Use the supplied evidence."


def test_syntax_change_defaults_xml_tag_then_surfaces_invalid_xml_beside_block() -> (
    None
):
    xml_state = update_block(_state(), "goal", syntax="xml")

    goal = xml_state.definition.lanes[1].blocks[0]
    assert goal.syntax == "xml"
    assert goal.xml_tag == "goal"
    assert xml_state.compiled_user.startswith("<goal>Explain the result.</goal>")

    invalid = update_block(xml_state, "goal", xml_tag="bad tag")
    assert invalid.definition.lanes[1].blocks[0].content == "Explain the result."
    assert invalid.issues == (
        invalid.issues[0].__class__(
            block_id="goal",
            field="xml_tag",
            code="invalid_xml_name",
            message="XML tag must start with a letter or underscore and contain only XML name characters.",
        ),
    )


def test_xml_wrapper_collision_is_a_content_issue_without_losing_content() -> None:
    state = update_block(_state(), "goal", syntax="xml", xml_tag="goal")
    state = update_block(state, "goal", content="Keep <goal>nested</goal> exactly")

    assert state.definition.lanes[1].blocks[0].content == (
        "Keep <goal>nested</goal> exactly"
    )
    [issue] = state.issues
    assert (issue.block_id, issue.field, issue.code) == (
        "goal",
        "content",
        "xml_wrapper_collision",
    )


def test_clearing_xml_tag_keeps_transient_input_and_content_for_recovery() -> None:
    state = update_block(_state(), "goal", syntax="xml", xml_tag="goal")

    cleared = update_block(state, "goal", xml_tag="")

    goal = cleared.definition.lanes[1].blocks[0]
    assert goal.xml_tag == ""
    assert goal.content == "Explain the result."
    assert [(issue.field, issue.code) for issue in cleared.issues] == [
        ("xml_tag", "invalid_xml_name")
    ]


def test_editing_one_legacy_lane_keeps_the_other_origin_byte_exact() -> None:
    state = update_block(_state(legacy=True), "goal", content="Rewritten")

    assert state.system_origin is not None
    assert state.user_origin is None
    assert state.compiled_system == "SYSTEM BYTES\n"
    assert state.compiled_user == "Rewritten\n\nUse the supplied evidence."
    assert state.dirty_block_ids == frozenset({"goal"})


def test_identical_update_returns_same_state_without_invalidating_origins() -> None:
    state = PromptBlockEditorState.from_definition(
        artifact_type="prompt",
        definition=_definition(),
        dirty_block_ids=frozenset({"context"}),
        system_origin=_origin("SYSTEM BYTES\n"),
        user_origin=_origin("USER BYTES\n\n"),
    )

    unchanged = update_block(state, "goal", content="Explain the result.")

    assert unchanged is state
    assert unchanged.system_origin is state.system_origin
    assert unchanged.user_origin is state.user_origin
    assert unchanged.compiled_system == "SYSTEM BYTES\n"
    assert unchanged.compiled_user == "USER BYTES\n\n"
    assert unchanged.dirty_block_ids == frozenset({"context"})


def test_artifact_type_change_keeps_definition_and_kind_in_lockstep() -> None:
    recipe = set_artifact_type(_state(), "recipe")

    assert recipe.artifact_type == "recipe"
    assert recipe.definition.kind == "block_recipe"
    assert recipe.definition.lanes == _state().definition.lanes
    assert set_artifact_type(recipe, "recipe") is recipe


def test_unknown_block_operations_fail_explicitly() -> None:
    with pytest.raises(KeyError, match="missing"):
        update_block(_state(), "missing", content="nope")
    with pytest.raises(KeyError, match="missing"):
        delete_block(_state(), "missing")
