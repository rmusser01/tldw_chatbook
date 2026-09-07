"""Immutable models for Console schema-v2 prompt artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


ArtifactType = Literal["prompt", "recipe"]
ArtifactDefinitionState = Literal[
    "legacy", "supported_v2", "foreign_v1", "unsupported", "malformed", "mismatched"
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
    definition: BlockArtifactDefinition | None
    raw_definition: Mapping[str, Any] | None
    compiled_system: str
    compiled_user: str
    compatibility_stale: bool


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
