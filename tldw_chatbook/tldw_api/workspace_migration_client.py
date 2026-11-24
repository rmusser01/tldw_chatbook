"""Use the authenticated root transport for audited workspace migrations."""

import re
from typing import Annotated

from pydantic import Field, TypeAdapter

from .workspace_migration_schemas import (
    WorkspaceMigrationChunkReceiptResponse,
    WorkspaceMigrationChunkUploadRequest,
    WorkspaceMigrationClientDeleteAckRequest,
    WorkspaceMigrationCreateRequest,
    WorkspaceMigrationFinalizeRequest,
    WorkspaceMigrationResponse,
)


_PATH = "/api/v1/workspaces/migrations"
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SESSION_LIST = TypeAdapter(
    Annotated[tuple[WorkspaceMigrationResponse, ...], Field(max_length=100)]
)


def _segment(value: str) -> str:
    if not isinstance(value, str) or not _SEGMENT.fullmatch(value):
        raise ValueError("Choose a bounded Server migration identifier")
    return value


class WorkspaceMigrationClient:
    """Bind the migration API to an already authenticated TLDWAPIClient."""

    def __init__(self, root_client) -> None:
        self.root_client = root_client

    @staticmethod
    def _session(response, expected_id: str) -> WorkspaceMigrationResponse:
        result = WorkspaceMigrationResponse.model_validate(response)
        if result.id != expected_id:
            raise ValueError("The Server returned a different migration")
        return result

    async def begin(
        self, request: WorkspaceMigrationCreateRequest
    ) -> WorkspaceMigrationResponse:
        response = await self.root_client._request(
            "POST", _PATH, json_data=request.model_dump(mode="json", exclude_none=True)
        )
        result = self._session(response, request.id)
        if (
            result.idempotency_key,
            result.target_workspace_id,
            result.manifest_hash,
            result.source_product,
        ) != (
            request.idempotency_key,
            request.target_workspace_id,
            request.manifest_hash,
            request.source_product,
        ):
            raise ValueError("The Server returned a different migration manifest")
        return result

    async def list_sessions(self) -> tuple[WorkspaceMigrationResponse, ...]:
        return _SESSION_LIST.validate_python(
            await self.root_client._request("GET", _PATH)
        )

    async def get(self, migration_id: str) -> WorkspaceMigrationResponse:
        response = await self.root_client._request(
            "GET", f"{_PATH}/{_segment(migration_id)}"
        )
        return self._session(response, migration_id)

    async def put_chunk(
        self,
        migration_id: str,
        chunk_id: str,
        request: WorkspaceMigrationChunkUploadRequest,
    ) -> WorkspaceMigrationChunkReceiptResponse:
        response = await self.root_client._request(
            "PUT",
            f"{_PATH}/{_segment(migration_id)}/chunks/{_segment(chunk_id)}",
            json_data=request.model_dump(mode="json", exclude_none=True),
        )
        result = WorkspaceMigrationChunkReceiptResponse.model_validate(response)
        if (
            result.id,
            result.migration_id,
            result.sha256,
            result.byte_count,
            result.chunk_kind,
        ) != (
            chunk_id,
            migration_id,
            request.sha256,
            request.byte_count,
            request.chunk_kind,
        ) or result.metadata != request.metadata.model_dump(
            mode="json", exclude_none=True
        ):
            raise ValueError("The Server returned a different migration chunk receipt")
        return result

    async def finalize(
        self, migration_id: str, manifest_hash: str
    ) -> WorkspaceMigrationResponse:
        request = WorkspaceMigrationFinalizeRequest(manifest_hash=manifest_hash)
        response = await self.root_client._request(
            "POST",
            f"{_PATH}/{_segment(migration_id)}/finalize",
            json_data=request.model_dump(mode="json"),
        )
        result = self._session(response, migration_id)
        if result.status != "finalized" or result.manifest_hash != manifest_hash:
            raise ValueError("The Server did not finalize this migration")
        return result

    async def acknowledge_client_deletion(
        self, migration_id: str, manifest_hash: str
    ) -> bool:
        request = WorkspaceMigrationClientDeleteAckRequest(
            acknowledged_manifest_hash=manifest_hash
        )
        response = await self.root_client._request(
            "POST",
            f"{_PATH}/{_segment(migration_id)}/client-delete-ack",
            json_data=request.model_dump(mode="json"),
        )
        return isinstance(response, dict) and response.get("ok") is True
