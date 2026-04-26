from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ClientChangesPayload,
    ServerChangesResponse,
    SyncLogEntry,
    SyncOperation,
    SyncSendEntity,
    SyncSendLogEntry,
    TLDWAPIClient,
)


def _change() -> SyncSendLogEntry:
    return SyncSendLogEntry(
        change_id=12,
        entity=SyncSendEntity.MEDIA,
        entity_uuid="media-uuid-1",
        operation=SyncOperation.UPDATE,
        timestamp="2026-04-25T12:00:00Z",
        client_id="chatbook-client-1",
        version=3,
        payload='{"uuid":"media-uuid-1","title":"Updated"}',
    )


@pytest.mark.asyncio
async def test_sync_client_routes_send_and_get_changes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"status": "success"},
            {
                "changes": [
                    {
                        "change_id": 33,
                        "entity": "Keywords",
                        "entity_uuid": "keyword-uuid-1",
                        "operation": "create",
                        "timestamp": "2026-04-25T12:01:00Z",
                        "server_timestamp": "2026-04-25T12:01:01Z",
                        "client_id": "server-client",
                        "version": 1,
                        "payload": '{"uuid":"keyword-uuid-1","keyword":"paper"}',
                    }
                ],
                "latest_change_id": 33,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    sent = await client.send_sync_changes(
        ClientChangesPayload(
            client_id="chatbook-client-1",
            changes=[_change()],
            last_processed_server_id=30,
        )
    )
    pulled = await client.get_sync_changes(client_id="chatbook-client-1", since_change_id=30)

    assert sent == {"status": "success"}
    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/sync/send")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "client_id": "chatbook-client-1",
        "changes": [
            {
                "change_id": 12,
                "entity": "Media",
                "entity_uuid": "media-uuid-1",
                "operation": "update",
                "timestamp": "2026-04-25T12:00:00Z",
                "client_id": "chatbook-client-1",
                "version": 3,
                "payload": '{"uuid":"media-uuid-1","title":"Updated"}',
            }
        ],
        "last_processed_server_id": 30,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/sync/get")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "client_id": "chatbook-client-1",
        "since_change_id": 30,
    }
    assert isinstance(pulled, ServerChangesResponse)
    assert isinstance(pulled.changes[0], SyncLogEntry)
    assert pulled.changes[0].entity == "Keywords"
