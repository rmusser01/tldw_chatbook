"""Audit completed Copy steps on the exact destination Server."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
import json
from uuid import UUID

from tldw_chatbook.tldw_api.exceptions import APIResponseError
from tldw_chatbook.tldw_api.workspace_migration_client import WorkspaceMigrationClient
from tldw_chatbook.tldw_api.workspace_migration_schemas import (
    MAX_WORKSPACE_MIGRATION_CHUNKS,
    WorkspaceMigrationAuditManifest,
    WorkspaceMigrationChunkDeclaration,
    WorkspaceMigrationChunkMetadata,
    WorkspaceMigrationChunkReceiptResponse,
    WorkspaceMigrationChunkUploadRequest,
    WorkspaceMigrationCreateRequest,
    WorkspaceMigrationResponse,
)

from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource, _safe_identity
from .copy_models import ResearchCopyItem, ResearchCopyManifest
from .copy_receipts import CopyItemReceipt, CopyItemStage, CopyReceiptFailure


_PRODUCT = "tldw-chatbook"


def _idempotency_key(manifest: ResearchCopyManifest) -> str:
    return f"{_PRODUCT}:{manifest.manifest_hash}"


def _declare(
    manifest: ResearchCopyManifest, item: ResearchCopyItem
) -> WorkspaceMigrationChunkDeclaration:
    payload = json.dumps(
        asdict(item), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return WorkspaceMigrationChunkDeclaration(
        id=manifest.item_idempotency_key(item),
        sha256=sha256(payload).hexdigest(),
        byte_count=len(payload),
    )


def _declarations(
    manifest: ResearchCopyManifest,
) -> dict[str, WorkspaceMigrationChunkDeclaration]:
    if manifest.destination.data_source is not WorkspaceDataSource.SERVER:
        raise ValueError("The migration requires a Server destination")
    declarations = (
        _declare(manifest, item) for item in manifest.items if item.actionable
    )
    result = {row.id: row for row in declarations}
    if not 1 <= len(result) <= MAX_WORKSPACE_MIGRATION_CHUNKS:
        raise ValueError(
            f"Select 1 to {MAX_WORKSPACE_MIGRATION_CHUNKS} supported items per Server Copy"
        )
    return result


class ServerMigrationReceiptService:
    """Use the Server's audit API after verifying the real owner write."""

    def __init__(self, client_for: Callable[[QualifiedWorkspaceRef], object]) -> None:
        self.client_for = client_for

    def _assert_current(self, ref: QualifiedWorkspaceRef, client) -> None:
        if self.client_for(ref) is not client:
            raise ValueError("The selected destination server changed")

    @staticmethod
    def _validate_state(
        manifest: ResearchCopyManifest,
        state: WorkspaceMigrationResponse,
        declarations: dict[str, WorkspaceMigrationChunkDeclaration],
    ) -> WorkspaceMigrationResponse:
        if (
            state.id,
            state.idempotency_key,
            state.target_workspace_id,
            state.source_product,
            state.manifest_hash,
        ) != (
            manifest.transfer_id,
            _idempotency_key(manifest),
            manifest.destination.workspace_id,
            _PRODUCT,
            manifest.manifest_hash,
        ):
            raise ValueError(
                "The Server migration belongs to a different Copy or workspace"
            )
        if {chunk.id for chunk in state.chunks} | set(
            state.missing_chunk_ids
        ) != declarations.keys():
            raise ValueError(
                "The Server migration does not match the declared Copy items"
            )
        items = {
            manifest.item_idempotency_key(item): item
            for item in manifest.items
            if item.actionable
        }
        for chunk in state.chunks:
            declaration = declarations[chunk.id]
            metadata = WorkspaceMigrationChunkMetadata.model_validate(chunk.metadata)
            if (chunk.sha256, chunk.byte_count, chunk.chunk_kind) != (
                declaration.sha256,
                declaration.byte_count,
                declaration.chunk_kind,
            ) or (metadata.manifest_hash, metadata.item_key, metadata.item_type) != (
                manifest.manifest_hash,
                chunk.id,
                items[chunk.id].source.kind,
            ):
                raise ValueError(
                    "The Server migration contains a different item receipt"
                )
        return state

    async def _read(
        self, manifest, client, api, declarations
    ) -> WorkspaceMigrationResponse:
        state = await api.get(manifest.transfer_id)
        self._assert_current(manifest.destination, client)
        return self._validate_state(manifest, state, declarations)

    async def begin(
        self, manifest: ResearchCopyManifest, *, target_name: str
    ) -> WorkspaceMigrationResponse:
        """Create the audit session for a newly minted workspace or recover the same session."""
        declarations = _declarations(manifest)
        ref = manifest.destination
        client = self.client_for(ref)
        api = WorkspaceMigrationClient(client)
        try:
            return await self._read(manifest, client, api, declarations)
        except APIResponseError as exc:
            if exc.status_code != 404:
                raise
        self._assert_current(ref, client)
        try:
            minted = UUID(ref.workspace_id)
        except ValueError as exc:
            raise ValueError("Choose a new Server workspace for this Copy") from exc
        if minted.version != 4 or str(minted) != ref.workspace_id:
            raise ValueError("Choose a new Server workspace for this Copy")
        try:
            await client.get_workspace(ref.workspace_id)
        except APIResponseError as exc:
            if exc.status_code != 404:
                raise
        else:
            raise ValueError(
                "Choose a new Server workspace. The migration initializes its Study policy"
            )
        self._assert_current(ref, client)
        request = WorkspaceMigrationCreateRequest(
            id=manifest.transfer_id,
            idempotency_key=_idempotency_key(manifest),
            target_workspace_id=ref.workspace_id,
            target_workspace_name=_safe_identity(target_name, "workspace name"),
            manifest_hash=manifest.manifest_hash,
            declared_chunks=tuple(declarations.values()),
            manifest=WorkspaceMigrationAuditManifest(
                transfer_id=manifest.transfer_id,
                source_workspace_id=manifest.source.workspace_id,
                destination_workspace_id=ref.workspace_id,
            ),
        )
        state = await api.begin(request)
        self._assert_current(ref, client)
        return self._validate_state(manifest, state, declarations)

    async def status(
        self, manifest: ResearchCopyManifest
    ) -> WorkspaceMigrationResponse:
        declarations = _declarations(manifest)
        client = self.client_for(manifest.destination)
        return await self._read(
            manifest, client, WorkspaceMigrationClient(client), declarations
        )

    async def ack_item(
        self,
        manifest: ResearchCopyManifest,
        receipt: CopyItemReceipt,
    ) -> WorkspaceMigrationChunkReceiptResponse:
        declarations = _declarations(manifest)
        destination = receipt.destination
        if (
            receipt.stage is not CopyItemStage.VERIFIED
            or receipt.rolled_back
            or receipt.failure is not CopyReceiptFailure.NONE
            or destination is None
            or destination.workspace != manifest.destination
            or destination.kind is not receipt.item.source.kind
            or not destination.has_revision
            or manifest.item_idempotency_key(receipt.item) != receipt.item_key
            or receipt.item_key not in declarations
        ):
            raise ValueError(
                "A Server chunk requires a verified destination owner receipt"
            )
        declaration = declarations[receipt.item_key]
        metadata = WorkspaceMigrationChunkMetadata(
            manifest_hash=manifest.manifest_hash,
            item_key=receipt.item_key,
            item_type=receipt.item.source.kind,
            destination_owner_id=destination.owner_id,
            destination_owner_version=destination.version,
            destination_snapshot_hash=destination.snapshot_hash or None,
        )
        client = self.client_for(manifest.destination)
        api = WorkspaceMigrationClient(client)
        state = await self._read(manifest, client, api, declarations)
        previous = next(
            (row for row in state.chunks if row.id == receipt.item_key), None
        )
        if previous is not None:
            if previous.metadata != metadata.model_dump(mode="json", exclude_none=True):
                raise ValueError(
                    "The Server receipt points to a different destination owner"
                )
            return previous
        if state.status != "created":
            raise ValueError("The Server migration cannot accept another receipt")
        saved = await api.put_chunk(
            manifest.transfer_id,
            receipt.item_key,
            WorkspaceMigrationChunkUploadRequest(
                sha256=declaration.sha256,
                byte_count=declaration.byte_count,
                chunk_kind=declaration.chunk_kind,
                metadata=metadata,
            ),
        )
        self._assert_current(manifest.destination, client)
        return saved

    async def finalize(
        self, manifest: ResearchCopyManifest
    ) -> WorkspaceMigrationResponse:
        declarations = _declarations(manifest)
        client = self.client_for(manifest.destination)
        api = WorkspaceMigrationClient(client)
        state = await self._read(manifest, client, api, declarations)
        if state.status == "finalized":
            return state
        if state.missing_chunk_ids:
            raise ValueError("The Server migration has missing item receipts")
        if state.status != "created":
            raise ValueError("The Server migration cannot be finalized")
        await api.finalize(manifest.transfer_id, manifest.manifest_hash)
        state = await self._read(manifest, client, api, declarations)
        if state.status != "finalized":
            raise ValueError("The Server did not confirm finalization on readback")
        return state
