"""Immutable inspection records, distinct from runtime authorization."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class Diagnostic(FrozenModel):
    code: str
    path: str | None = None
    component_id: str | None = None


class InspectionEvidence(FrozenModel):
    level: Literal["parsed"] = "parsed"
    revision: str | None = None
    adapter: str = "chatbook-portable/1"
    platform: str
    configuration_identity: str | None = None
    observed_at: str


class ComponentRecord(FrozenModel):
    component_id: str
    kind: Literal["skill", "mcp", "command", "rule", "agent", "hook"]
    local_id: str
    path: str
    definition_json: str = "{}"
    support: Literal["supported", "adapted", "unsupported", "invalid"] = "supported"
    selection: Literal["selected", "excluded"] = "selected"
    availability: Literal[
        "disabled",
        "needs_trust",
        "needs_configuration",
        "ready",
        "failed",
        "recovery_required",
    ] = "disabled"
    evidence: tuple[InspectionEvidence, ...] = ()
    dependencies: tuple[str, ...] = ()
    activation_blockers: tuple[str, ...] = ()


class DialectCandidate(FrozenModel):
    dialect: str
    format_version: str | None = None
    adapter_version: str | None = None
    root_manifest: str
    overlays: tuple[str, ...] = ()
    support: Literal["supported", "unsupported"] = "supported"
    qualified: bool = False


class PackageInspection(FrozenModel):
    source_identity: str
    dialect: str | None = None
    format_version: str | None = None
    adapter_version: str | None = None
    root_manifest: str | None = None
    overlay_identities: tuple[str, ...] = ()
    candidates: tuple[DialectCandidate, ...] = ()
    content_digest: str | None = None
    source_digest: str | None = None
    materialized_identity: str | None = None
    link_targets: Mapping[str, str] = Field(default_factory=dict)
    effective_digest: str | None = None
    inventory: Mapping[str, ComponentRecord] = Field(default_factory=dict)
    dependency_edges: Mapping[str, tuple[str, ...]] = Field(default_factory=dict)
    variables_json: str = "{}"
    activation_blockers: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    rejected: bool = False

    @field_validator("inventory", "dependency_edges", "link_targets", mode="after")
    @classmethod
    def freeze_mapping(cls, value):
        return MappingProxyType(dict(value))

    @field_serializer("inventory", "dependency_edges", "link_targets")
    def serialize_mapping(self, value):
        return dict(value)
