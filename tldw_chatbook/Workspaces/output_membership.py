"""Bounded, payload-free provenance stored on a workspace output link."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_OUTPUT_PROVENANCE_BYTES = 32 * 1024
_IDENTIFIER = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
_FINGERPRINT = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
_REVISION = Annotated[int, Field(strict=True, ge=0)]
_LABEL = Annotated[str, StringConstraints(strict=True, max_length=256)]
_PRIVATE_LABEL = re.compile(
    r"(?i)(://|\b(?:bearer\s+\S+|sk-[A-Za-z0-9_-]+|(?:api[_-]?key|password|secret|token)\s*[:=]))"
)


class _OutputSourceVersion(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    source_id: _IDENTIFIER
    catalog_item_id: _IDENTIFIER
    source_type: _IDENTIFIER
    association_version: _REVISION | None = None
    catalog_item_version: _REVISION | None = None
    ready_modes: tuple[Literal["fts", "semantic", "hybrid"], ...] = Field(
        default=(), max_length=3
    )


class _ResearchRunRef(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    backend: Literal["local", "server"]
    workspace_id: _IDENTIFIER
    launch_id: _IDENTIFIER
    run_id: _IDENTIFIER


class OutputGenerationProvenance(BaseModel):
    """Fields the workspace can retain alongside the canonical owner ID."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    operation_id: _IDENTIFIER
    output_kind: Literal[
        "summary",
        "flashcards",
        "quiz",
        "report",
        "compare_sources",
        "mind_map",
        "timeline",
        "data_table",
        "slides",
        "literature_matrix",
        "corpus_gap_finder",
        "executive_brief",
        "evidence_bound_hypotheses",
        "research_proposal_pack",
    ]
    source_snapshot: tuple[_OutputSourceVersion, ...] = Field(
        min_length=1, max_length=100
    )
    settings_fingerprint: _FINGERPRINT
    processing_route_fingerprint: _FINGERPRINT
    provider: _LABEL
    model: _LABEL
    status: Literal["complete", "cancelled", "failed"]
    owner_version: _REVISION | None = None
    supersedes_owner_id: _IDENTIFIER | None = None
    research_run_ref: _ResearchRunRef | None = None

    @model_validator(mode="after")
    def validate_research_run(self):
        if self.research_run_ref is not None and self.output_kind != "report":
            raise ValueError("Only Reports carry Research run provenance")
        return self

    @field_validator("provider", "model")
    @classmethod
    def validate_processing_label(cls, value: str) -> str:
        if _PRIVATE_LABEL.search(value) or any(ord(char) < 32 for char in value):
            raise ValueError(
                "Processing labels must be free of secrets and control characters"
            )
        return value

    @field_validator("source_snapshot")
    @classmethod
    def validate_unique_sources(
        cls, rows: tuple[_OutputSourceVersion, ...]
    ) -> tuple[_OutputSourceVersion, ...]:
        if len({row.source_id for row in rows}) != len(rows) or len(
            {row.catalog_item_id for row in rows}
        ) != len(rows):
            raise ValueError("Output provenance contains duplicate sources")
        return rows


def output_membership_metadata(
    metadata: Mapping[str, object], role: str
) -> dict[str, object]:
    """Validate a copy of the exact fields accepted on an output membership."""
    if not isinstance(metadata, Mapping):
        raise ValueError("Output provenance must be an object")
    if not metadata:
        return {}
    if role != "output":
        raise ValueError("Only output memberships accept generation provenance")
    if set(metadata) != {"research_generation"}:
        raise ValueError("Output provenance contains unsupported fields")
    try:
        generation = OutputGenerationProvenance.model_validate(
            metadata["research_generation"]
        )
    except ValidationError:
        raise ValueError(
            "Output provenance contains unsupported or invalid fields"
        ) from None
    result = {
        "research_generation": generation.model_dump(mode="json", exclude_none=True)
    }
    if (
        len(encode_output_membership_metadata(result).encode("utf-8"))
        > MAX_OUTPUT_PROVENANCE_BYTES
    ):
        raise ValueError("Output provenance exceeds 32 KiB")
    return result


def encode_output_membership_metadata(metadata: dict[str, object]) -> str:
    """Serialize using the same representation used for the size bound."""
    return json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
