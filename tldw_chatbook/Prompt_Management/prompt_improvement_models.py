"""Immutable contracts for one captured prompt-improvement request."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from typing import Any, Literal

from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    BlockArtifactDefinition,
    PromptBlock,
    PromptLane,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import (
    ComposerDraftSegmentSnapshot,
    ComposerDraftSnapshot,
    ComposerModelProjection,
)


ImprovementMode = Literal["auto", "review", "recipe"]
ImprovementOutcomeKind = Literal[
    "success",
    "no_change",
    "empty",
    "unsupported",
    "cancelled",
    "provider_error",
    "malformed",
    "preservation_veto",
    "context_limit",
    "stale",
]


def fingerprint_text(text: str) -> str:
    """Return the canonical fingerprint for captured optional text."""
    if not isinstance(text, str):
        raise TypeError("Fingerprint input must be text.")
    return f"sha256:{sha256(text.encode('utf-8')).hexdigest()}"


def block_definition_payload(definition: BlockArtifactDefinition) -> dict[str, Any]:
    """Return the closed canonical JSON shape for one block definition."""
    if not isinstance(definition, BlockArtifactDefinition):
        raise TypeError("definition must be a BlockArtifactDefinition")
    return {
        "kind": definition.kind,
        "schema_version": definition.schema_version,
        "lanes": [
            {
                "id": lane.id,
                "blocks": [
                    {
                        "id": block.id,
                        "title": block.title,
                        "syntax": block.syntax,
                        "content": block.content,
                        "xml_tag": block.xml_tag,
                        "mapping_hint": block.mapping_hint,
                    }
                    for block in lane.blocks
                ],
            }
            for lane in definition.lanes
        ],
    }


def fingerprint_block_definition(definition: BlockArtifactDefinition) -> str:
    """Fingerprint the exact canonical captured block definition."""
    encoded = json.dumps(
        block_definition_payload(definition),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _copy_definition(definition: BlockArtifactDefinition) -> BlockArtifactDefinition:
    return BlockArtifactDefinition(
        kind=definition.kind,
        schema_version=definition.schema_version,
        lanes=tuple(
            PromptLane(
                id=lane.id,
                blocks=tuple(
                    PromptBlock(
                        id=block.id,
                        title=block.title,
                        syntax=block.syntax,
                        content=block.content,
                        xml_tag=block.xml_tag,
                        mapping_hint=block.mapping_hint,
                    )
                    for block in lane.blocks
                ),
            )
            for lane in definition.lanes
        ),
    )


def _copy_composer_snapshot(snapshot: ComposerDraftSnapshot) -> ComposerDraftSnapshot:
    return ComposerDraftSnapshot(
        segments=tuple(
            ComposerDraftSegmentSnapshot(
                text=segment.text,
                origin=segment.origin,
                collapse_state=segment.collapse_state,
                label=segment.label,
                generated_boundary=segment.generated_boundary,
                paste_block=segment.paste_block,
            )
            for segment in snapshot.segments
        ),
        cursor_index=snapshot.cursor_index,
        selection=snapshot.selection,
        edit_serial=snapshot.edit_serial,
        generation=snapshot.generation,
        fingerprint=snapshot.fingerprint,
    )


@dataclass(frozen=True)
class PromptImprovementRequestSnapshot:
    """All immutable state captured before one auxiliary completion."""

    request_id: str
    mode: ImprovementMode
    session_id: str
    composer_snapshot: ComposerDraftSnapshot = field(repr=False)
    projection: ComposerModelProjection = field(repr=False)
    system_prompt: str | None = field(repr=False)
    system_fingerprint: str | None = field(repr=False)
    resolution: ConsoleProviderResolution = field(repr=False)
    provider_label: str
    model_label: str
    recipe_source: Literal["local", "server"] | None
    recipe_source_id: str | None
    recipe_version: int | None
    recipe_definition: BlockArtifactDefinition | None = field(repr=False)
    recipe_fingerprint: str | None = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be non-empty text")
        if self.mode not in {"auto", "review", "recipe"}:
            raise ValueError("Unsupported prompt improvement mode")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be non-empty text")
        if not isinstance(self.composer_snapshot, ComposerDraftSnapshot):
            raise TypeError("composer_snapshot must be a ComposerDraftSnapshot")
        if not isinstance(self.projection, ComposerModelProjection):
            raise TypeError("projection must be a ComposerModelProjection")
        if not isinstance(self.resolution, ConsoleProviderResolution):
            raise TypeError("resolution must be a ConsoleProviderResolution")
        if not self.resolution.ready:
            raise ValueError("Pinned provider resolution is not ready")
        if (
            not isinstance(self.resolution.model, str)
            or not self.resolution.model.strip()
        ):
            raise ValueError("Pinned provider model is required")
        for field_name, value in (
            ("provider_label", self.provider_label),
            ("model_label", self.model_label),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")

        has_system = self.system_prompt is not None
        has_system_fingerprint = self.system_fingerprint is not None
        if has_system != has_system_fingerprint:
            raise ValueError("System prompt and fingerprint must be captured together")
        if has_system and (
            not isinstance(self.system_prompt, str)
            or not isinstance(self.system_fingerprint, str)
            or not self.system_fingerprint
        ):
            raise TypeError("Captured system prompt fields must be text")

        recipe_values = (
            self.recipe_source,
            self.recipe_source_id,
            self.recipe_version,
            self.recipe_definition,
            self.recipe_fingerprint,
        )
        if self.mode == "recipe":
            required_recipe_values = recipe_values[1:]
            if any(value is None for value in required_recipe_values):
                raise ValueError("Recipe mode requires complete captured Recipe fields")
            if (
                not isinstance(self.recipe_source_id, str)
                or not self.recipe_source_id.strip()
            ):
                raise ValueError("recipe_source_id must be non-empty text")
            if self.recipe_source_id.startswith("builtin:"):
                if self.recipe_source is not None:
                    raise ValueError("Built-in Recipes cannot capture a saved source")
            elif self.recipe_source not in {"local", "server"}:
                raise ValueError("Saved Recipes require a Local or Server source")
            if (
                isinstance(self.recipe_version, bool)
                or not isinstance(self.recipe_version, int)
                or self.recipe_version < 0
            ):
                raise ValueError("recipe_version must be a non-negative integer")
            if not isinstance(self.recipe_definition, BlockArtifactDefinition):
                raise TypeError("recipe_definition must be a BlockArtifactDefinition")
            if (
                not isinstance(self.recipe_fingerprint, str)
                or not self.recipe_fingerprint
            ):
                raise ValueError("recipe_fingerprint must be non-empty text")
        elif any(value is not None for value in recipe_values):
            raise ValueError("Only Recipe mode may capture Recipe fields")

        object.__setattr__(
            self, "composer_snapshot", _copy_composer_snapshot(self.composer_snapshot)
        )
        object.__setattr__(
            self,
            "projection",
            ComposerModelProjection(
                text=self.projection.text,
                placeholder_nonce=self.projection.placeholder_nonce,
                placeholder_ids=tuple(self.projection.placeholder_ids),
                fingerprint=self.projection.fingerprint,
            ),
        )
        object.__setattr__(self, "resolution", replace(self.resolution))
        if self.recipe_definition is not None:
            object.__setattr__(
                self, "recipe_definition", _copy_definition(self.recipe_definition)
            )


@dataclass(frozen=True)
class PromptImprovementOutcome:
    """Typed result of one headless prompt-improvement attempt."""

    request_id: str
    kind: ImprovementOutcomeKind
    rewritten_prompt: str | None = field(default=None, repr=False)
    filled_definition: BlockArtifactDefinition | None = field(default=None, repr=False)
    provider: str = ""
    model: str = ""
    user_message: str = ""
