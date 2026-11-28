"""Checked workspace migration routes reuse the authenticated root client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.workspace_migration_client import WorkspaceMigrationClient
from tldw_chatbook.tldw_api.workspace_migration_schemas import (
    WorkspaceMigrationCreateRequest,
)


MIGRATION_PATH = "/api/v1/workspaces/migrations"
HASH = "a" * 64


def session(**overrides):
    return {
        "id": "copy-id",
        "idempotency_key": "record-copy",
        "target_workspace_id": "target",
        "target_workspace_name": "Target",
        "source_product": "tldw-chatbook",
        "manifest_hash": HASH,
        "status": "created",
        "declared_chunk_count": 0,
        "accepted_chunk_count": 0,
        "missing_chunk_ids": [],
        "chunks": [],
        **overrides,
    }


@pytest.mark.asyncio
async def test_workspace_migration_client_uses_exact_authenticated_paths():
    root = AsyncMock()
    root._request.side_effect = (
        session(),
        [session()],
        session(),
        session(status="finalized"),
    )
    api = WorkspaceMigrationClient(root)
    request = WorkspaceMigrationCreateRequest(
        id="copy-id",
        idempotency_key="record-copy",
        target_workspace_id="target",
        target_workspace_name="Target",
        manifest_hash=HASH,
    )
    assert (await api.begin(request)).target_workspace_id == "target"
    assert len(await api.list_sessions()) == 1
    assert (await api.get("copy-id")).manifest_hash == HASH
    assert (await api.finalize("copy-id", HASH)).status == "finalized"
    assert [call.args for call in root._request.await_args_list] == [
        ("POST", MIGRATION_PATH),
        ("GET", MIGRATION_PATH),
        ("GET", MIGRATION_PATH + "/copy-id"),
        ("POST", MIGRATION_PATH + "/copy-id/finalize"),
    ]
    assert (
        root._request.await_args_list[0].kwargs["json_data"]["source_product"]
        == "tldw-chatbook"
    )


@pytest.mark.asyncio
async def test_workspace_migration_rejects_other_receipts_and_unsafe_route_segments():
    root = AsyncMock()
    root._request.return_value = session(id="different")
    api = WorkspaceMigrationClient(root)
    with pytest.raises(ValueError, match="different migration"):
        await api.get("copy-id")
    with pytest.raises(ValueError, match="identifier"):
        await api.get("copy-id/../workspaces")
    assert root._request.await_count == 1


@pytest.mark.asyncio
async def test_legacy_deletion_ack_route_has_an_exact_typed_wire_contract():
    root = AsyncMock()
    root._request.return_value = {"ok": True}
    api = WorkspaceMigrationClient(root)
    assert await api.acknowledge_client_deletion("copy-id", HASH)
    root._request.assert_awaited_once_with(
        "POST",
        MIGRATION_PATH + "/copy-id/client-delete-ack",
        json_data={"acknowledged_manifest_hash": HASH},
    )
