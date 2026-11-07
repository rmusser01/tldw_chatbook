"""Shared fail-closed runtime policy for canonical citation provenance."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class CitationProvenanceRuntimePolicy(BaseModel):
    """Immutable recovery switch shared by citation persistence services."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    canonical_writes_enabled: bool = False

    @classmethod
    def from_config(cls) -> "CitationProvenanceRuntimePolicy":
        """Load the typed switch without acquiring identity or key material.

        TASK-33620.8.3: an install still carrying the old shipped
        ``canonical_writes_enabled = false`` may have answered "Keep a record
        of which sources replies cite?"; a saved Keep outranks that ``false``
        (the answer file is read only then; config.toml is never rewritten).
        """

        from tldw_chatbook.config import (
            get_rag_citation_canonical_writes_enabled,
        )

        if get_rag_citation_canonical_writes_enabled():
            return cls(canonical_writes_enabled=True)
        from tldw_chatbook.Chat.citation_records_choice import (
            canonical_writes_enabled_with_answer,
        )

        return cls(canonical_writes_enabled=canonical_writes_enabled_with_answer())


__all__ = ["CitationProvenanceRuntimePolicy"]
