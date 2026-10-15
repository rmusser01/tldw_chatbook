"""Immutable inspection records, distinct from runtime authorization."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)


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


class AcquisitionReference(FrozenModel):
    """Compact producer-authenticated provenance, with no recursive inspection."""

    operation_id: str = Field(min_length=1, max_length=200)
    acquisition_nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    transport: Literal["https", "ssh", "local"]
    origin: str = Field(max_length=4096)
    repository: str = Field(max_length=4096)
    ref: str = Field(min_length=1, max_length=1024)
    subdirectory: str | None = Field(default=None, max_length=4096)
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    tree: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    materialized_identity: str = Field(min_length=1, max_length=4096)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class CatalogBinding(FrozenModel):
    source_id: str = Field(min_length=1, max_length=100)
    catalog: AcquisitionReference
    package: AcquisitionReference
    catalog_path: Literal[
        ".agents/plugins/marketplace.json",
        ".cursor-plugin/marketplace.json",
        ".claude-plugin/marketplace.json",
    ]
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_identity: str = Field(min_length=1, max_length=256)
    entry_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_locator_json: str = Field(min_length=1, max_length=8192)

    @model_validator(mode="after")
    def bounded_binding(self):
        if len(self.model_dump_json().encode()) > 16384:
            raise ValueError("catalog_binding_limit")
        from .git_source import SourceLocator

        SourceLocator.model_validate_json(self.update_locator_json)
        return self


class PackageInspection(FrozenModel):
    source_identity: str
    catalog_binding: CatalogBinding | None = None
    acquisition_reference: AcquisitionReference | None = None
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

    @model_validator(mode="after")
    def exclusive_source_binding(self):
        if self.catalog_binding is not None and self.acquisition_reference is not None:
            raise ValueError("ambiguous_acquisition_binding")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_catalog(self, handler):
        result = handler(self)
        if self.catalog_binding is None:
            result.pop("catalog_binding", None)
        if self.acquisition_reference is None:
            result.pop("acquisition_reference", None)
        return result

    @field_validator("inventory", "dependency_edges", "link_targets", mode="after")
    @classmethod
    def freeze_mapping(cls, value):
        return MappingProxyType(dict(value))

    @field_serializer("inventory", "dependency_edges", "link_targets")
    def serialize_mapping(self, value):
        return dict(value)


class AcquiredSnapshot(FrozenModel):
    """Frozen acquired bytes/provenance; inspection is never activation."""

    operation_id: str
    acquisition_nonce: str
    transport: Literal["https", "ssh", "local"]
    origin: str
    repository: str
    ref: str
    subdirectory: str | None = None
    commit: str | None = None
    tree: str | None = None
    materialized_identity: str
    inspection: PackageInspection
    diagnostics: tuple[Diagnostic, ...] = ()
