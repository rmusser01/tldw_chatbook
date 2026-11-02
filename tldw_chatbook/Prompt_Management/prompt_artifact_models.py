"""Immutable artifact values and pure record decoding for Console prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

LOCAL_COMPILED_LANE_LIMIT = 20_000
LOCAL_DEFINITION_LIMIT = 256_000
LOCAL_REQUEST_LIMIT = 512_000


ArtifactType = Literal["prompt", "recipe", "mixin"]
ArtifactDefinitionState = Literal[
    "legacy",
    "supported_v2",
    "supported_v3",
    "foreign_v1",
    "unsupported",
    "malformed",
    "mismatched",
]
BlockSyntax = Literal["freeform", "markdown", "xml"]


@dataclass(frozen=True)
class PromptBlock:
    """One immutable block in a System or User lane."""

    id: str
    title: str
    syntax: BlockSyntax
    content: str
    xml_tag: str | None = None
    mapping_hint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Block id must be a non-empty string.")
        if not isinstance(self.title, str):
            raise ValueError("Block title must be a string.")
        if self.syntax not in {"freeform", "markdown", "xml"}:
            raise ValueError(f"Unsupported block syntax: {self.syntax!r}")
        if not isinstance(self.content, str):
            raise ValueError("Block content must be a string.")
        if self.xml_tag is not None and not isinstance(self.xml_tag, str):
            raise ValueError("Block XML tag must be a string or None.")
        if self.syntax == "xml" and not self.xml_tag:
            raise ValueError("XML blocks require an XML tag.")
        if self.syntax != "xml" and self.xml_tag is not None:
            raise ValueError("Only XML blocks may define an XML tag.")
        if self.mapping_hint is not None and not isinstance(self.mapping_hint, str):
            raise ValueError("Block mapping hint must be a string or None.")


@dataclass(frozen=True)
class PromptLane:
    """An ordered lane of blocks."""

    id: Literal["system", "user"]
    blocks: tuple[PromptBlock, ...]

    def __post_init__(self) -> None:
        if self.id not in {"system", "user"}:
            raise ValueError(f"Unsupported lane id: {self.id!r}")
        if not isinstance(self.blocks, tuple) or not all(
            isinstance(block, PromptBlock) for block in self.blocks
        ):
            raise ValueError("Lane blocks must be a tuple of PromptBlock values.")


@dataclass(frozen=True)
class BlockArtifactDefinition:
    """The Console-specific schema-v2 document shape."""

    kind: Literal["block_prompt", "block_recipe"]
    schema_version: Literal[2]
    lanes: tuple[PromptLane, PromptLane]

    def __post_init__(self) -> None:
        if self.kind not in {"block_prompt", "block_recipe"}:
            raise ValueError(f"Unsupported block artifact kind: {self.kind!r}")
        if self.schema_version != 2:
            raise ValueError("Console block artifacts require schema version 2.")
        if not isinstance(self.lanes, tuple) or len(self.lanes) != 2:
            raise ValueError("A block artifact must have exactly two lanes.")
        if not all(isinstance(lane, PromptLane) for lane in self.lanes):
            raise ValueError("Block artifact lanes must be PromptLane values.")
        if tuple(lane.id for lane in self.lanes) != ("system", "user"):
            raise ValueError("Block artifact lanes must be ordered system then user.")
        seen_ids: set[str] = set()
        for lane in self.lanes:
            for block in lane.blocks:
                if block.id in seen_ids:
                    raise ValueError(f"Block IDs must be globally unique: {block.id!r}")
                seen_ids.add(block.id)


@dataclass(frozen=True)
class LiteralMixinDefinition:
    """A literal instruction with an explicit placement."""

    body: str = field(repr=False)
    placement: Literal["before", "after"]
    kind: Literal["text_mixin"] = "text_mixin"
    schema_version: Literal[3] = 3

    def __post_init__(self) -> None:
        if type(self.body) is not str or not self.body.strip():
            raise ValueError("Mix-in body must be a non-empty string.")
        if type(self.placement) is not str or self.placement not in ("before", "after"):
            raise ValueError("Invalid mix-in placement.")
        if (
            type(self.kind) is not str
            or self.kind != "text_mixin"
            or type(self.schema_version) is not int
            or self.schema_version != 3
        ):
            raise ValueError("Invalid mix-in kind or schema version.")


@dataclass(frozen=True)
class LegacyLaneOrigin:
    """Original legacy bytes and a stable fingerprint for one lane."""

    text: str
    fingerprint: str


@dataclass(frozen=True)
class LegacyDecomposition:
    """A conservative editable view of a legacy prompt record."""

    definition: BlockArtifactDefinition
    system_origin: LegacyLaneOrigin
    user_origin: LegacyLaneOrigin


@dataclass(frozen=True)
class DecodedPromptArtifact:
    """A fully classified prompt record, safe for callers to branch on."""

    state: ArtifactDefinitionState
    artifact_type: ArtifactType
    definition: BlockArtifactDefinition | LiteralMixinDefinition | None
    raw_definition: Mapping[str, Any] | None
    compiled_system: str
    compiled_user: str
    compatibility_stale: bool


def _is_schema_version(value: Any, expected: int) -> bool:
    """Require JSON integer schema versions without Python bool/float coercion."""
    return type(value) is int and value == expected


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


def _compatibility_text(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("system_prompt") or ""), str(record.get("user_prompt") or "")


def _decoded(
    *,
    state: Literal["foreign_v1", "unsupported", "malformed", "mismatched"],
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


def decode_literal_mixin_v3(
    record: Mapping[str, Any], *, raw: Mapping[str, Any] | None
) -> DecodedPromptArtifact:
    """Decode the closed literal mix-in shape without interpreting its body."""
    artifact_type: ArtifactType = "mixin"
    if record.get("prompt_format") != "structured":
        return _decoded(
            state="mismatched", record=record, artifact_type=artifact_type, raw=raw
        )
    if not _is_schema_version(record.get("prompt_schema_version"), 3):
        return _decoded(
            state="unsupported", record=record, artifact_type=artifact_type, raw=raw
        )
    if raw is None:
        return _decoded(
            state="malformed", record=record, artifact_type=artifact_type, raw=raw
        )
    if (
        type(raw.get("kind")) is not str
        or raw.get("kind") != "text_mixin"
        or not _is_schema_version(raw.get("schema_version"), 3)
    ):
        return _decoded(
            state="mismatched", record=record, artifact_type=artifact_type, raw=raw
        )
    try:
        if set(raw) != {"kind", "schema_version", "body", "placement"}:
            raise ValueError("Invalid mix-in fields.")
        definition = LiteralMixinDefinition(
            body=raw["body"], placement=raw["placement"]
        )
    except (KeyError, TypeError, ValueError):
        return _decoded(
            state="malformed", record=record, artifact_type=artifact_type, raw=raw
        )
    system, user = _compatibility_text(record)
    return DecodedPromptArtifact(
        state="supported_v3",
        artifact_type=artifact_type,
        definition=definition,
        raw_definition=raw,
        compiled_system="",
        compiled_user=definition.body,
        compatibility_stale=(system, user) != ("", definition.body),
    )


def outcome_first_recipe() -> BlockArtifactDefinition:
    """Return a fresh built-in outcome-first Recipe definition.

    The factory supplies structure and concise mapping help only. User facts,
    constraints, evidence, and desired outputs intentionally remain blank.
    """
    system_specs = (
        ("role", "Role", "Define the model's function and job."),
        ("personality", "Personality", "Describe the desired tone and demeanor."),
        (
            "collaboration-style",
            "Collaboration style",
            "Describe how the model should work with the user.",
        ),
    )
    user_specs = (
        ("goal", "Goal", "State the user-visible outcome."),
        (
            "context-evidence",
            "Context and evidence",
            "Add available facts, sources, examples, or inputs.",
        ),
        (
            "constraints",
            "Constraints",
            "Name policy, safety, business, evidence, or side-effect limits.",
        ),
        (
            "output",
            "Output",
            "Describe the required answer shape, length, and tone.",
        ),
        (
            "success-criteria",
            "Success criteria",
            "List what must be true before the answer is complete.",
        ),
        (
            "stop-rules",
            "Stop rules",
            "Define when to retry, ask, fall back, abstain, or stop.",
        ),
    )

    def blocks(
        specs: tuple[tuple[str, str, str], ...],
    ) -> tuple[PromptBlock, ...]:
        return tuple(
            PromptBlock(
                id=block_id,
                title=title,
                syntax="markdown",
                content="",
                mapping_hint=mapping_hint,
            )
            for block_id, title, mapping_hint in specs
        )

    return BlockArtifactDefinition(
        kind="block_recipe",
        schema_version=2,
        lanes=(
            PromptLane(id="system", blocks=blocks(system_specs)),
            PromptLane(id="user", blocks=blocks(user_specs)),
        ),
    )


def blank_recipe() -> BlockArtifactDefinition:
    """Return a fresh empty two-lane Recipe working definition."""
    return BlockArtifactDefinition(
        kind="block_recipe",
        schema_version=2,
        lanes=(
            PromptLane(id="system", blocks=()),
            PromptLane(id="user", blocks=()),
        ),
    )
