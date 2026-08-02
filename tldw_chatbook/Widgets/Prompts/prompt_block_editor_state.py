"""Immutable state transitions for the shared Prompt/Recipe block editor."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Literal

from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    ArtifactType,
    BlockArtifactDefinition,
    BlockSyntax,
    LegacyLaneOrigin,
    PromptBlock,
    PromptLane,
)
from tldw_chatbook.Prompt_Management.prompt_block_compiler import compile_block


LaneId = Literal["system", "user"]
ValidationField = Literal["id", "title", "syntax", "xml_tag", "content"]

ADDITIONAL_CONTEXT_RESERVED_PREFIX = "additional-context"

_XML_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


@dataclass(frozen=True)
class PromptBlockValidationIssue:
    """One actionable validation issue attached to a stable block identity."""

    block_id: str
    field: ValidationField
    code: str
    message: str


@dataclass(frozen=True)
class PromptBlockEditorState:
    """Complete immutable state for a System/User block editor."""

    artifact_type: ArtifactType
    definition: BlockArtifactDefinition
    compiled_system: str
    compiled_user: str
    issues: tuple[PromptBlockValidationIssue, ...]
    dirty_block_ids: frozenset[str]
    system_origin: LegacyLaneOrigin | None = None
    user_origin: LegacyLaneOrigin | None = None

    @classmethod
    def from_definition(
        cls,
        *,
        artifact_type: ArtifactType,
        definition: BlockArtifactDefinition,
        dirty_block_ids: frozenset[str] = frozenset(),
        system_origin: LegacyLaneOrigin | None = None,
        user_origin: LegacyLaneOrigin | None = None,
    ) -> PromptBlockEditorState:
        """Build validated editor state and deterministic compiled previews."""
        _validate_artifact_pair(artifact_type, definition)
        for lane in definition.lanes:
            for block in lane.blocks:
                _reject_reserved_id(block.id)
        issues = validate_block_artifact(definition)
        compiled_system = _compile_editor_lane(
            definition.lanes[0], origin=system_origin
        )
        compiled_user = _compile_editor_lane(definition.lanes[1], origin=user_origin)
        return cls(
            artifact_type=artifact_type,
            definition=definition,
            compiled_system=compiled_system,
            compiled_user=compiled_user,
            issues=issues,
            dirty_block_ids=dirty_block_ids,
            system_origin=system_origin,
            user_origin=user_origin,
        )


def validate_block_artifact(
    definition: BlockArtifactDefinition,
) -> tuple[PromptBlockValidationIssue, ...]:
    """Return deterministic, field-addressable validation issues."""
    issues: list[PromptBlockValidationIssue] = []
    for lane in definition.lanes:
        for block in lane.blocks:
            if not block.title.strip():
                issues.append(
                    PromptBlockValidationIssue(
                        block_id=block.id,
                        field="title",
                        code="empty_title",
                        message="Title is required; enter a block title.",
                    )
                )
            if block.syntax != "xml":
                continue
            xml_tag = block.xml_tag or ""
            if not _XML_NAME.fullmatch(xml_tag):
                issues.append(
                    PromptBlockValidationIssue(
                        block_id=block.id,
                        field="xml_tag",
                        code="invalid_xml_name",
                        message=(
                            "XML tag must start with a letter or underscore and "
                            "contain only XML name characters."
                        ),
                    )
                )
                continue
            collision = re.compile(
                rf"<\s*/?\s*{re.escape(xml_tag)}(?:\s+[^>]*)?\s*/?\s*>"
            )
            if collision.search(block.content):
                issues.append(
                    PromptBlockValidationIssue(
                        block_id=block.id,
                        field="content",
                        code="xml_wrapper_collision",
                        message=(
                            f"Content already contains the <{xml_tag}> wrapper; "
                            "remove or rename that wrapper."
                        ),
                    )
                )
    return tuple(issues)


def update_block(
    state: PromptBlockEditorState,
    block_id: str,
    **changes: str | None,
) -> PromptBlockEditorState:
    """Replace one block by ID and invalidate only its edited legacy lane."""
    lane_index, block_index, block = _locate_block(state.definition, block_id)
    normalized = dict(changes)
    if "id" in normalized:
        new_id = normalized["id"]
        if not isinstance(new_id, str) or not new_id:
            raise ValueError("Block id must be a non-empty string.")
        _reject_reserved_id(new_id)

    syntax = normalized.get("syntax", block.syntax)
    if syntax == "xml":
        if "xml_tag" not in normalized and block.xml_tag is None:
            normalized["xml_tag"] = _default_xml_tag(block.id)
    elif syntax in {"freeform", "markdown"}:
        normalized["xml_tag"] = None

    updated = _replace_editor_block(block, normalized)
    lanes = list(state.definition.lanes)
    blocks = list(lanes[lane_index].blocks)
    blocks[block_index] = updated
    lanes[lane_index] = replace(lanes[lane_index], blocks=tuple(blocks))
    definition = replace(state.definition, lanes=tuple(lanes))
    dirty_ids = state.dirty_block_ids | {block_id, updated.id}
    return _replace_definition(
        state,
        definition,
        dirty_block_ids=dirty_ids,
        edited_lane=lanes[lane_index].id,
    )


def add_block(
    state: PromptBlockEditorState,
    lane_id: LaneId,
    *,
    title: str = "Untitled block",
    syntax: BlockSyntax = "freeform",
    content: str = "",
    xml_tag: str | None = None,
    mapping_hint: str | None = None,
    block_id: str | None = None,
) -> PromptBlockEditorState:
    """Append a new block using a deterministic collision-safe identity."""
    lane_index = _lane_index(state.definition, lane_id)
    existing_ids = _block_ids(state.definition)
    new_id = block_id or _next_id("block", existing_ids)
    _reject_reserved_id(new_id)
    if new_id in existing_ids:
        raise ValueError(f"Block ID already exists: {new_id!r}")
    if syntax == "xml" and xml_tag is None:
        xml_tag = _default_xml_tag(new_id)
    block = PromptBlock(
        id=new_id,
        title=title,
        syntax=syntax,
        content=content,
        xml_tag=xml_tag,
        mapping_hint=mapping_hint,
    )
    lanes = list(state.definition.lanes)
    lanes[lane_index] = replace(
        lanes[lane_index], blocks=(*lanes[lane_index].blocks, block)
    )
    definition = replace(state.definition, lanes=tuple(lanes))
    return _replace_definition(
        state,
        definition,
        dirty_block_ids=state.dirty_block_ids | {new_id},
        edited_lane=lane_id,
    )


def move_block(
    state: PromptBlockEditorState,
    block_id: str,
    direction: Literal[-1, 1],
) -> PromptBlockEditorState:
    """Move a block one position within its lane; boundaries are no-ops."""
    if direction not in {-1, 1}:
        raise ValueError("Block direction must be -1 or 1.")
    lane_index, block_index, _block = _locate_block(state.definition, block_id)
    lane = state.definition.lanes[lane_index]
    target_index = block_index + direction
    if target_index < 0 or target_index >= len(lane.blocks):
        return state
    blocks = list(lane.blocks)
    blocks[block_index], blocks[target_index] = (
        blocks[target_index],
        blocks[block_index],
    )
    lanes = list(state.definition.lanes)
    lanes[lane_index] = replace(lane, blocks=tuple(blocks))
    definition = replace(state.definition, lanes=tuple(lanes))
    return _replace_definition(
        state,
        definition,
        dirty_block_ids=state.dirty_block_ids | {block_id},
        edited_lane=lane.id,
    )


def duplicate_block(
    state: PromptBlockEditorState, block_id: str
) -> PromptBlockEditorState:
    """Insert a content-preserving copy after its source with a fresh ID."""
    lane_index, block_index, block = _locate_block(state.definition, block_id)
    new_id = _next_id(f"{block.id}-copy", _block_ids(state.definition))
    duplicate = replace(block, id=new_id, title=f"{block.title} copy")
    lane = state.definition.lanes[lane_index]
    blocks = list(lane.blocks)
    blocks.insert(block_index + 1, duplicate)
    lanes = list(state.definition.lanes)
    lanes[lane_index] = replace(lane, blocks=tuple(blocks))
    definition = replace(state.definition, lanes=tuple(lanes))
    return _replace_definition(
        state,
        definition,
        dirty_block_ids=state.dirty_block_ids | {new_id},
        edited_lane=lane.id,
    )


def delete_block(
    state: PromptBlockEditorState, block_id: str
) -> PromptBlockEditorState:
    """Delete exactly one block while retaining its ID in dirty tracking."""
    lane_index, block_index, _block = _locate_block(state.definition, block_id)
    lane = state.definition.lanes[lane_index]
    blocks = list(lane.blocks)
    del blocks[block_index]
    lanes = list(state.definition.lanes)
    lanes[lane_index] = replace(lane, blocks=tuple(blocks))
    definition = replace(state.definition, lanes=tuple(lanes))
    return _replace_definition(
        state,
        definition,
        dirty_block_ids=state.dirty_block_ids | {block_id},
        edited_lane=lane.id,
    )


def set_artifact_type(
    state: PromptBlockEditorState, artifact_type: ArtifactType
) -> PromptBlockEditorState:
    """Change Prompt/Recipe type while keeping the canonical kind aligned."""
    if artifact_type not in {"prompt", "recipe"}:
        raise ValueError(f"Unsupported artifact type: {artifact_type!r}")
    if artifact_type == state.artifact_type:
        return state
    definition = replace(
        state.definition,
        kind="block_prompt" if artifact_type == "prompt" else "block_recipe",
    )
    return PromptBlockEditorState.from_definition(
        artifact_type=artifact_type,
        definition=definition,
        dirty_block_ids=state.dirty_block_ids,
        system_origin=state.system_origin,
        user_origin=state.user_origin,
    )


def _replace_definition(
    state: PromptBlockEditorState,
    definition: BlockArtifactDefinition,
    *,
    dirty_block_ids: frozenset[str],
    edited_lane: LaneId,
) -> PromptBlockEditorState:
    return PromptBlockEditorState.from_definition(
        artifact_type=state.artifact_type,
        definition=definition,
        dirty_block_ids=dirty_block_ids,
        system_origin=None if edited_lane == "system" else state.system_origin,
        user_origin=None if edited_lane == "user" else state.user_origin,
    )


def _compile_editor_lane(lane: PromptLane, *, origin: LegacyLaneOrigin | None) -> str:
    if origin is not None:
        return origin.text
    rendered: list[str] = []
    for block in lane.blocks:
        if block.content == "":
            continue
        try:
            value = compile_block(block)
        except ValueError:
            # Invalid transient editor input remains visible in preview while
            # validation blocks Apply/Save. This value is never persisted.
            value = block.content
        if value:
            rendered.append(value)
    return "\n\n".join(rendered)


def _replace_editor_block(
    block: PromptBlock, changes: dict[str, str | None]
) -> PromptBlock:
    """Preserve a repairable empty XML tag without weakening persisted models."""
    syntax = changes.get("syntax", block.syntax)
    xml_tag = changes.get("xml_tag", block.xml_tag)
    if syntax != "xml" or xml_tag != "":
        return replace(block, **changes)

    # PromptBlock intentionally rejects empty XML tags at persistence boundaries.
    # Validate every other draft value through that strict model first, then retain
    # only the empty tag as transient editor input. Validation keeps Apply/Save off.
    validated_changes = dict(changes)
    validated_changes["xml_tag"] = _default_xml_tag(
        str(validated_changes.get("id", block.id))
    )
    validated = replace(block, **validated_changes)
    transient = object.__new__(PromptBlock)
    for field_name in ("id", "title", "syntax", "content", "xml_tag", "mapping_hint"):
        value = "" if field_name == "xml_tag" else getattr(validated, field_name)
        object.__setattr__(transient, field_name, value)
    return transient


def _validate_artifact_pair(
    artifact_type: ArtifactType, definition: BlockArtifactDefinition
) -> None:
    expected_kind = "block_prompt" if artifact_type == "prompt" else "block_recipe"
    if artifact_type not in {"prompt", "recipe"} or definition.kind != expected_kind:
        raise ValueError("Artifact type and block definition kind must agree.")


def _lane_index(definition: BlockArtifactDefinition, lane_id: LaneId) -> int:
    if lane_id not in {"system", "user"}:
        raise ValueError(f"Unsupported lane id: {lane_id!r}")
    return 0 if lane_id == "system" else 1


def _locate_block(
    definition: BlockArtifactDefinition, block_id: str
) -> tuple[int, int, PromptBlock]:
    for lane_index, lane in enumerate(definition.lanes):
        for block_index, block in enumerate(lane.blocks):
            if block.id == block_id:
                return lane_index, block_index, block
    raise KeyError(f"Unknown block id: {block_id}")


def _block_ids(definition: BlockArtifactDefinition) -> set[str]:
    return {block.id for lane in definition.lanes for block in lane.blocks}


def _next_id(base: str, existing_ids: set[str]) -> str:
    if base not in existing_ids:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing_ids:
        suffix += 1
    return f"{base}-{suffix}"


def _reject_reserved_id(block_id: str) -> None:
    normalized = block_id.casefold().replace("_", "-")
    if normalized == ADDITIONAL_CONTEXT_RESERVED_PREFIX or normalized.startswith(
        f"{ADDITIONAL_CONTEXT_RESERVED_PREFIX}-"
    ):
        raise ValueError(
            f"Block ID {block_id!r} is reserved for mapped Additional context."
        )


def _default_xml_tag(block_id: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.:-]", "_", block_id)
    if not tag or not re.match(r"[A-Za-z_]", tag):
        tag = f"block_{tag}"
    return tag
