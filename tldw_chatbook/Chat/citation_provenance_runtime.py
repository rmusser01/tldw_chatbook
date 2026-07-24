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
        """Load the typed switch without acquiring identity or key material."""

        from tldw_chatbook.config import (
            get_rag_citation_canonical_writes_enabled,
        )

        return cls(canonical_writes_enabled=get_rag_citation_canonical_writes_enabled())


__all__ = ["CitationProvenanceRuntimePolicy"]
