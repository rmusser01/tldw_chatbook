"""Explicit record dispatch for legacy, server-v1, and Console-v2 prompts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, cast

from .prompt_artifact_models import (
    ArtifactType,
    BlockArtifactDefinition,
    DecodedPromptArtifact,
    PromptBlock,
    PromptLane,
)
from .prompt_block_compiler import compile_block_artifact


def deserialize_definition(value: Any) -> Mapping[str, Any] | None:
    """Return a JSON-object definition, never guessing at non-object input."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return dict(parsed) if isinstance(parsed, Mapping) else None


def _artifact_type(record: Mapping[str, Any]) -> ArtifactType:
    raw_artifact_type = str(record.get("artifact_type") or "prompt")
    if raw_artifact_type not in {"prompt", "recipe"}:
        raise ValueError(f"Unsupported artifact_type: {raw_artifact_type!r}")
    return cast(ArtifactType, raw_artifact_type)


def _compatibility_text(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("system_prompt") or ""), str(record.get("user_prompt") or "")


def _decoded(
    *,
    state: Literal[
        "foreign_v1", "unsupported", "malformed", "mismatched"
    ],
    record: Mapping[str, Any],
    artifact_type: ArtifactType,
    raw: Mapping[str, Any] | None,
) -> DecodedPromptArtifact:
    system, user = _compatibility_text(record)
    return DecodedPromptArtifact(
        state=state,
        artifact_type=artifact_type,
        definition=None,
        raw_definition=raw,
        compiled_system=system,
        compiled_user=user,
        compatibility_stale=False,
    )


def foreign_definition(
    record: Mapping[str, Any],
    artifact_type: ArtifactType,
    raw: Mapping[str, Any] | None,
    *,
    state: Literal["foreign_v1", "unsupported"],
) -> DecodedPromptArtifact:
    """Represent a known-foreign or future definition without parsing it as v2."""
    return _decoded(state=state, record=record, artifact_type=artifact_type, raw=raw)


def malformed_definition(
    record: Mapping[str, Any], artifact_type: ArtifactType
) -> DecodedPromptArtifact:
    """Represent malformed structured input as data, rather than an exception."""
    return _decoded(
        state="malformed",
        record=record,
        artifact_type=artifact_type,
        raw=deserialize_definition(record.get("prompt_definition")),
    )


def _parse_block(raw: Mapping[str, Any]) -> PromptBlock:
    if not isinstance(raw, Mapping):
        raise ValueError("Every block must be an object.")
    required = ("id", "title", "syntax", "content")
    if any(key not in raw for key in required):
        raise ValueError("Every block requires id, title, syntax, and content.")
    return PromptBlock(
        id=raw["id"],
        title=raw["title"],
        syntax=raw["syntax"],
        content=raw["content"],
        xml_tag=raw.get("xml_tag"),
        mapping_hint=raw.get("mapping_hint"),
    )


def _parse_lane(raw: Mapping[str, Any]) -> PromptLane:
    if not isinstance(raw, Mapping):
        raise ValueError("Every lane must be an object.")
    blocks = raw.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("Every lane requires a blocks array.")
    if "id" not in raw:
        raise ValueError("Every lane requires an id.")
    return PromptLane(id=raw["id"], blocks=tuple(_parse_block(block) for block in blocks))


def decode_console_v2(
    record: Mapping[str, Any], *, artifact_type: ArtifactType, raw: Mapping[str, Any]
) -> DecodedPromptArtifact:
    """Decode only the closed Console v2 shape and expose corrupt states safely."""
    if raw.get("schema_version") != 2:
        return _decoded(
            state="mismatched", record=record, artifact_type=artifact_type, raw=raw
        )
    kind = raw.get("kind")
    if kind not in {"block_prompt", "block_recipe"}:
        return _decoded(
            state="malformed", record=record, artifact_type=artifact_type, raw=raw
        )
    try:
        lanes_raw = raw["lanes"]
        if not isinstance(lanes_raw, list):
            raise ValueError("Block artifact lanes must be an array.")
        definition = BlockArtifactDefinition(
            kind=kind,
            schema_version=2,
            lanes=tuple(_parse_lane(lane) for lane in lanes_raw),
        )
    except (KeyError, TypeError, ValueError):
        return _decoded(
            state="malformed", record=record, artifact_type=artifact_type, raw=raw
        )

    expected_kind = "block_prompt" if artifact_type == "prompt" else "block_recipe"
    if definition.kind != expected_kind:
        return _decoded(
            state="mismatched", record=record, artifact_type=artifact_type, raw=raw
        )

    try:
        compiled_system, compiled_user = compile_block_artifact(definition)
    except ValueError:
        return _decoded(
            state="malformed", record=record, artifact_type=artifact_type, raw=raw
        )
    stored_system, stored_user = _compatibility_text(record)
    return DecodedPromptArtifact(
        state="supported_v2",
        artifact_type=artifact_type,
        definition=definition,
        raw_definition=raw,
        compiled_system=compiled_system,
        compiled_user=compiled_user,
        compatibility_stale=(stored_system, stored_user) != (compiled_system, compiled_user),
    )


def decode_prompt_artifact(record: Mapping[str, Any]) -> DecodedPromptArtifact:
    """Classify a prompt record without allowing one schema to impersonate another."""
    artifact_type = _artifact_type(record)
    if str(record.get("prompt_format") or "legacy") == "legacy":
        system, user = _compatibility_text(record)
        return DecodedPromptArtifact(
            state="legacy",
            artifact_type=artifact_type,
            definition=None,
            raw_definition=None,
            compiled_system=system,
            compiled_user=user,
            compatibility_stale=False,
        )

    raw = deserialize_definition(record.get("prompt_definition"))
    version = record.get("prompt_schema_version")
    if version == 1:
        return foreign_definition(record, artifact_type, raw, state="foreign_v1")
    if version != 2:
        return foreign_definition(record, artifact_type, raw, state="unsupported")
    if raw is None:
        return malformed_definition(record, artifact_type)
    if raw.get("definition_kind") == "single_text_recipe":
        return foreign_definition(record, artifact_type, raw, state="unsupported")
    return decode_console_v2(record, artifact_type=artifact_type, raw=raw)
