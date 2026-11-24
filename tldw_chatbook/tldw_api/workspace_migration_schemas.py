"""Bounded request and receipt types for Server workspace migrations."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


MAX_WORKSPACE_MIGRATION_CHUNKS = 512
_Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
_Key = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
_Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[a-f0-9]{64}$")]
_Name = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=256),
]
_ChunkKind = Annotated[
    str,
    StringConstraints(
        strict=True, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$"
    ),
]
_ByteCount = Annotated[int, Field(strict=True, ge=0, le=2 * 1024 * 1024)]
_OwnerVersion = Annotated[int, Field(strict=True, ge=0, lt=2**63)]
_ChunkCount = Annotated[
    int, Field(strict=True, ge=0, le=MAX_WORKSPACE_MIGRATION_CHUNKS)
]


class _MigrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class WorkspaceMigrationChunkDeclaration(_MigrationRequest):
    id: _Identifier
    sha256: _Digest
    byte_count: _ByteCount
    chunk_kind: _ChunkKind = "research_copy_receipt"


class WorkspaceMigrationAuditManifest(_MigrationRequest):
    transfer_id: _Identifier
    source_workspace_id: _Identifier
    destination_workspace_id: _Identifier
    source_retained: Literal[True] = True


class WorkspaceMigrationCreateRequest(_MigrationRequest):
    id: _Identifier
    idempotency_key: _Key
    target_workspace_id: _Identifier
    target_workspace_name: _Name
    source_product: Literal["tldw-chatbook"] = "tldw-chatbook"
    manifest_hash: _Digest
    declared_chunks: tuple[WorkspaceMigrationChunkDeclaration, ...] = Field(
        default=(), max_length=MAX_WORKSPACE_MIGRATION_CHUNKS
    )
    manifest: WorkspaceMigrationAuditManifest | None = None

    @model_validator(mode="after")
    def _check_declared_items(self):
        if len({chunk.id for chunk in self.declared_chunks}) != len(
            self.declared_chunks
        ):
            raise ValueError("Server migration requires unique declared chunk IDs")
        if self.manifest is not None and (
            self.manifest.transfer_id != self.id
            or self.manifest.destination_workspace_id != self.target_workspace_id
        ):
            raise ValueError("The audit manifest belongs to a different migration")
        return self


class WorkspaceMigrationChunkMetadata(_MigrationRequest):
    manifest_hash: _Digest
    item_key: _Digest
    item_type: Literal["media", "note", "artifact", "study_deck", "quiz"]
    destination_owner_id: _Identifier
    destination_owner_version: _OwnerVersion | None = None
    destination_snapshot_hash: _Digest | None = None

    @model_validator(mode="after")
    def _check_owner_revision(self):
        if (
            self.destination_owner_version is None
            and self.destination_snapshot_hash is None
        ):
            raise ValueError(
                "A migration chunk requires the destination owner revision"
            )
        return self


class WorkspaceMigrationChunkUploadRequest(_MigrationRequest):
    sha256: _Digest
    byte_count: _ByteCount
    chunk_kind: _ChunkKind = "research_copy_receipt"
    metadata: WorkspaceMigrationChunkMetadata


class WorkspaceMigrationFinalizeRequest(_MigrationRequest):
    manifest_hash: _Digest


class WorkspaceMigrationClientDeleteAckRequest(_MigrationRequest):
    acknowledged_manifest_hash: _Digest


class WorkspaceMigrationChunkReceiptResponse(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    id: _Identifier
    migration_id: _Identifier
    sha256: _Digest
    byte_count: _ByteCount
    chunk_kind: _ChunkKind
    metadata: dict[str, Any] = Field(default_factory=dict, repr=False)
    status: Literal["accepted"]


class WorkspaceMigrationResponse(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    id: _Identifier
    idempotency_key: _Key
    target_workspace_id: _Identifier
    target_workspace_name: _Name
    source_product: _Identifier
    manifest_hash: _Digest
    status: Literal["created", "finalized", "failed"]
    declared_chunk_count: _ChunkCount
    accepted_chunk_count: _ChunkCount
    missing_chunk_ids: tuple[_Identifier, ...] = Field(
        max_length=MAX_WORKSPACE_MIGRATION_CHUNKS
    )
    chunks: tuple[WorkspaceMigrationChunkReceiptResponse, ...] = Field(
        max_length=MAX_WORKSPACE_MIGRATION_CHUNKS
    )

    @model_validator(mode="after")
    def _check_chunk_counts(self):
        accepted = {row.id for row in self.chunks}
        missing = set(self.missing_chunk_ids)
        if (
            len(accepted) != len(self.chunks)
            or len(missing) != len(self.missing_chunk_ids)
            or accepted & missing
            or any(row.migration_id != self.id for row in self.chunks)
            or len(accepted) != self.accepted_chunk_count
            or len(accepted | missing) != self.declared_chunk_count
            or self.status == "finalized"
            and missing
        ):
            raise ValueError(
                "The Server returned inconsistent migration chunk identities"
            )
        return self
