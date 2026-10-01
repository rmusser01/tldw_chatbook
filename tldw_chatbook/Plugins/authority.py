"""Closed, domain-separated logical plugin authority."""

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from tldw_chatbook.Skills_Interop.skill_trust_crypto import canonical_json, sha256_hex

PURPOSES = frozenset(
    {
        "snapshot",
        "prepared",
        "committed",
        "issued_operation",
        "legacy_cutover",
        "archive-pin-v1",
    }
)


def authority_message(purpose: str, payload: dict) -> bytes:
    """Encode an authenticated message with a non-interchangeable purpose."""
    if purpose not in PURPOSES:
        raise ValueError("plugin_authority_purpose_invalid")
    _validate_json(payload)
    return (
        b"chatbook.plugins.v1\0"
        + purpose.encode("ascii")
        + b"\0"
        + canonical_json(payload)
    )


def _validate_json(value: object) -> None:
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("authority requires string keys")
        for child in value.values():
            _validate_json(child)
    elif type(value) is list:
        for child in value:
            _validate_json(child)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("nonfinite authority")
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("non-JSON authority")


Identifier = Annotated[
    str, Field(min_length=1, max_length=256, pattern=r"^[^\s\x00-\x1f]+$")
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Generation = Annotated[int, Field(ge=0, le=2**63 - 1)]
Text = Annotated[str, Field(max_length=16384)]


class AuthorityModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class PluginMarker(AuthorityModel):
    """Exact rollback-protected authority identity."""

    generation: Generation
    operation_id: Identifier
    recovery_snapshot_digest: Digest

    @model_validator(mode="after")
    def bootstrap_identity(self):
        if (self.generation == 0) != (self.operation_id == "bootstrap"):
            raise ValueError("invalid bootstrap marker")
        return self


class IssuedOperationIdentity(AuthorityModel):
    """Authenticated issuance only; never execution permission or commit proof."""

    generation: Annotated[int, Field(ge=1, le=2**63 - 1)]
    binding_digest: Digest
    nonce: Digest


class LegacyCutoverEntry(AuthorityModel):
    operation_id: Identifier
    result_digest: Digest


class LegacyCutover(AuthorityModel):
    schema_version: Literal[1]
    marker: PluginMarker
    entries: Annotated[list[LegacyCutoverEntry], Field(max_length=1001)]

    @model_validator(mode="after")
    def ordered_unique(self):
        ids = [entry.operation_id for entry in self.entries]
        if ids != sorted(set(ids)):
            raise ValueError("cutover entries must be sorted and unique")
        return self


class Installation(AuthorityModel):
    installation_id: Identifier
    revision_digest: Digest | None
    activation_default: bool
    alias: (
        Annotated[
            str, Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9.-]*$")
        ]
        | None
    ) = None


class Revision(AuthorityModel):
    installation_id: Identifier
    revision_digest: Digest
    source_identity: Text
    dialect: Identifier | None
    format_version: Identifier | None
    adapter_version: Identifier | None
    root_manifest: Text | None
    overlay_identities: list[Text]
    content_digest: Digest | None
    source_digest: Digest | None
    materialized_identity: Text | None
    link_targets: dict[Text, Text]
    activation_blockers: list[Identifier]
    variables_digest: Digest
    rejected: bool


class Component(AuthorityModel):
    installation_id: Identifier
    revision_digest: Digest
    component_id: Identifier
    kind: Literal["skill", "mcp", "command", "rule", "agent", "hook"]
    local_id: Identifier
    path: Text
    definition_digest: Digest
    support: Literal["supported", "adapted", "unsupported", "invalid"]
    selection: Literal["selected", "excluded"]
    dependencies: list[Identifier]
    activation_blockers: list[Identifier]

    @model_validator(mode="after")
    def validate_identity(self):
        if self.component_id != f"{self.kind}:{self.local_id}":
            raise ValueError("component identity mismatch")
        if self.support in ("invalid", "unsupported") and not self.activation_blockers:
            raise ValueError("unsupported component requires blocker")
        return self


class Selection(AuthorityModel):
    installation_id: Identifier
    revision_digest: Digest
    component_id: Identifier
    selected: bool


class Activation(AuthorityModel):
    installation_id: Identifier
    workspace_id: Identifier
    intent: Literal["inherit", "enabled", "disabled"]


class CredentialBinding(AuthorityModel):
    reference_id: Identifier
    authority_generation: Generation
    authentication_method: Identifier
    issuer: Text | None
    audience: Text | None
    endpoint_origin: Text | None
    principal: Text | None
    scopes: list[Text] | None
    identity_state: Literal["verified", "opaque_reviewed"]


class Mapping(AuthorityModel):
    installation_id: Identifier
    mapping_id: Identifier
    component_id: Identifier
    revision_digest: Digest
    kind: Literal["connection", "tool", "model", "configuration"]
    target_reference: Identifier
    definition_digest: Digest
    configuration_digest: Digest
    credential_bindings: list[CredentialBinding]


class AuthorityGeneration(AuthorityModel):
    installation_id: Identifier
    scope_kind: Literal["installation", "global_default", "workspace"]
    workspace_id: Text
    generation: Generation
    revoked: bool

    @model_validator(mode="after")
    def validate_scope(self):
        if (self.scope_kind == "workspace") != bool(self.workspace_id):
            raise ValueError("invalid generation scope")
        return self


class RevisionTrust(AuthorityModel):
    installation_id: Identifier
    revision_digest: Digest
    reviewed: bool


class Tombstone(AuthorityModel):
    installation_id: Identifier
    generation: Generation
    operation_id: Identifier


class RuntimeCheckpoint(AuthorityModel):
    version: Literal[1]
    session_nonce: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    namespace_id: Digest
    marker: PluginMarker
    phase: Literal["dirty", "clean"]


class PhysicalIdentity(AuthorityModel):
    device: Generation
    inode: Annotated[int, Field(ge=1, le=2**64 - 1)]
    birth_seconds: Annotated[int, Field(ge=1, le=2**63 - 1)]
    birth_nanoseconds: Annotated[int, Field(ge=0, lt=10**9)]


class RootBinding(AuthorityModel):
    platform: Literal["darwin-apfs-boot-v1"]
    boot_id: Annotated[
        str, Field(pattern=r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
    ]
    anchors: Annotated[list[PhysicalIdentity], Field(min_length=2, max_length=2)]
    leaf: PhysicalIdentity


class RootCustody(AuthorityModel):
    attached_installation_id: Identifier | None
    state: Literal["creating", "present", "cleaned_absent"]
    binding: RootBinding | None

    @model_validator(mode="after")
    def present_binding(self):
        if self.state != "creating" and self.binding is None:
            raise ValueError("root custody requires binding")
        return self


class RootCleanup(AuthorityModel):
    action: Literal["create", "delete", "attach"]
    attachment: Identifier | None
    group_id: Identifier
    phase: Literal["creating", "waiting", "deleting"]
    target_count: Annotated[int, Field(ge=1, le=256)]
    target_digest: Digest
    reviewed_generation: Generation
    binding_digest: Digest


class RootResult(AuthorityModel):
    group_id: Identifier
    phase: Literal[
        "creating",
        "created",
        "waiting",
        "deleting",
        "complete",
        "cancelled",
        "attached",
        "reconciled",
    ]
    target_digest: Digest
    root_ids: Annotated[list[Identifier], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def sorted_unique(self):
        if self.root_ids != sorted(set(self.root_ids)):
            raise ValueError("root result requires sorted exact members")
        return self


class DataRoot(AuthorityModel):
    root_id: Identifier
    installation_id: Identifier
    workspace_id: Identifier | None
    path: Text
    generation: Generation
    deletion_fenced: bool
    custody: RootCustody | None = None
    cleanup: RootCleanup | None = None

    @model_serializer(mode="wrap")
    def preserve_absent_custody(self, handler):
        value = handler(self)
        for field in ("custody", "cleanup"):
            if field not in self.model_fields_set:
                value.pop(field, None)
        return value

    @model_validator(mode="after")
    def validate_path(self):
        from tldw_chatbook.Utils.path_validation import validate_path_simple

        path = validate_path_simple(self.path, probe_existing=False)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("root reference must be absolute without traversal")
        return self


class RetiredRevision(AuthorityModel):
    revision_digest: Digest
    materialized_identity: Text
    device: Generation
    inode: Generation
    root_device: Generation
    root_inode: Generation
    anchor_device: Generation
    anchor_inode: Generation


class OperationResult(AuthorityModel):
    operation_id: Identifier
    installation_id: Identifier
    kind: Literal[
        "install",
        "update",
        "activate",
        "select",
        "configure",
        "trust",
        "revoke",
        "uninstall",
        "fence_data",
        "root_data",
        "retain",
        "recover",
    ]
    revision_digest: Digest | None
    result: Literal["committed"]

    root_result: RootResult | None = None

    retired_revisions: (
        Annotated[list[RetiredRevision], Field(min_length=1, max_length=1000)] | None
    ) = None

    @model_validator(mode="after")
    def retention_payload(self):
        if (self.kind == "root_data") != (self.root_result is not None):
            raise ValueError("root result belongs only to root_data")
        if self.kind != "root_data" and "root_result" in self.model_fields_set:
            raise ValueError("unexpected root result")
        if self.kind != "retain" and "retired_revisions" in self.model_fields_set:
            raise ValueError("cleanup payload belongs only to retain")
        if (self.kind == "retain") != (self.retired_revisions is not None):
            raise ValueError("retain requires its closed cleanup payload")
        if self.retired_revisions is not None:
            rows = [row.model_dump(mode="json") for row in self.retired_revisions]
            identities = [row["revision_digest"] for row in rows]
            if (
                identities != sorted(set(identities))
                or len(canonical_json(rows)) > 1024 * 1024
            ):
                raise ValueError("retention payload bounds or duplicate identity")
        return self

    @model_serializer(mode="wrap")
    def preserve_absent_retention(self, handler):
        value = handler(self)
        if self.root_result is None:
            value.pop("root_result", None)
        if self.retired_revisions is None:
            value.pop("retired_revisions", None)
        return value


class CompleteAuthority(AuthorityModel):
    schema_version: Literal[1]
    installations: list[Installation]
    revisions: list[Revision]
    components: list[Component]
    selections: list[Selection]
    activation: list[Activation]
    mappings: list[Mapping]
    authority_generations: list[AuthorityGeneration]
    revision_trust: list[RevisionTrust]
    tombstones: list[Tombstone]
    data_roots: list[DataRoot]
    operation_result: OperationResult | None


_ROW_KEYS = {
    "installations": ("installation_id",),
    "revisions": ("installation_id", "revision_digest"),
    "components": ("installation_id", "revision_digest", "component_id"),
    "selections": ("installation_id", "revision_digest", "component_id"),
    "activation": ("installation_id", "workspace_id"),
    "mappings": ("installation_id", "mapping_id"),
    "authority_generations": ("installation_id", "scope_kind", "workspace_id"),
    "revision_trust": ("installation_id", "revision_digest"),
    "tombstones": ("installation_id",),
    "data_roots": ("root_id",),
}


def empty_snapshot() -> dict:
    """Return the only snapshot allowed by explicit namespace bootstrap."""
    return {
        "schema_version": 1,
        **{key: [] for key in _ROW_KEYS},
        "operation_result": None,
    }


def canonical_snapshot(snapshot: dict) -> dict:
    """Validate the closed complete schema, references and deterministic row order."""
    _validate_json(snapshot)
    if type(snapshot.get("schema_version")) is not int:
        raise ValueError("invalid authority version")
    result = CompleteAuthority.model_validate(snapshot).model_dump(mode="json")
    # Preserve exact legacy logical bytes: an absent alias is not a new grant.
    for source, row in zip(snapshot["installations"], result["installations"]):
        if "alias" not in source:
            row.pop("alias", None)
    aliases = [row["alias"] for row in result["installations"] if row.get("alias")]
    if len(set(aliases)) != len(aliases):
        raise ValueError("duplicate installation alias")
    for name, keys in _ROW_KEYS.items():
        rows = result[name]
        if len(rows) > 100000:
            raise ValueError("authority row limit")
        ids = [tuple(row[key] for key in keys) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate authority identity")
        rows.sort(key=lambda row: tuple(row[key] for key in keys))
    installations = {row["installation_id"] for row in result["installations"]}
    revisions = {
        (r["installation_id"], r["revision_digest"]) for r in result["revisions"]
    }
    components = {
        (r["installation_id"], r["revision_digest"], r["component_id"])
        for r in result["components"]
    }
    for row in result["installations"]:
        if (
            row["revision_digest"] is not None
            and (row["installation_id"], row["revision_digest"]) not in revisions
        ):
            raise ValueError("missing selected revision")
    for name in ("revisions", "activation", "mappings", "authority_generations"):
        if any(r["installation_id"] not in installations for r in result[name]):
            raise ValueError("missing installation")
    for name in ("components", "revision_trust"):
        if any(
            (r["installation_id"], r["revision_digest"]) not in revisions
            for r in result[name]
        ):
            raise ValueError("missing revision")
    for name in ("selections", "mappings"):
        if any(
            (r["installation_id"], r["revision_digest"], r["component_id"])
            not in components
            for r in result[name]
        ):
            raise ValueError("missing component")
    for row in result["components"]:
        for field in ("dependencies", "activation_blockers"):
            if len(row[field]) != len(set(row[field])):
                raise ValueError("duplicate component constraint")
            row[field].sort()
        # Missing dependencies stay authenticated blockers, never vanish.
        for dependency in row["dependencies"]:
            if (
                row["installation_id"],
                row["revision_digest"],
                dependency,
            ) not in components and not row["activation_blockers"]:
                raise ValueError("missing unblocked dependency")
    for row in result["mappings"]:
        bindings = row["credential_bindings"]
        if len({r["reference_id"] for r in bindings}) != len(bindings):
            raise ValueError("duplicate credential binding")
        bindings.sort(key=lambda binding: binding["reference_id"])
        for binding in bindings:
            if binding["scopes"] is not None:
                if len(binding["scopes"]) != len(set(binding["scopes"])):
                    raise ValueError("duplicate scope")
                binding["scopes"].sort()
    return result


def snapshot_digest(snapshot: dict) -> str:
    """Digest canonical plaintext authority, independent of encryption randomness."""
    return sha256_hex(authority_message("snapshot", canonical_snapshot(snapshot)))
