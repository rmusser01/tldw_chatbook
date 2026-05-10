from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ClientChangesPayload,
    ServerChangesResponse,
    SyncLogEntry,
    SyncOperation,
    SyncSendEntity,
    SyncSendLogEntry,
    SyncV2CapabilitiesResponse,
    SyncV2DatasetEnrollRequest,
    SyncV2DeviceRegisterRequest,
    SyncV2Envelope,
    SyncV2PullResponse,
    SyncV2PushRequest,
    SyncV2PushResponse,
    SyncV2RestoreManifestResponse,
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


def test_sync_v2_envelope_rejects_plaintext_private_payload():
    with pytest.raises(ValueError, match="payload_clear.body"):
        SyncV2Envelope(
            client_envelope_id="env-1",
            dataset_id="dataset-1",
            domain="notes",
            entity_id="note-1",
            operation="upsert",
            adapter_version=1,
            payload_clear={"body": "clear text should stay client-side"},
            payload_hash="sha256:payload",
        )


@pytest.mark.asyncio
async def test_sync_v2_client_routes_protocol_endpoints(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "protocol_version": 2,
                "min_supported_protocol_version": 2,
                "supported_domains": ["notes", "chat", "workspaces", "source_cache", "media"],
                "supported_operations": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
                "encryption_policies": ["client_private_v1", "server_trusted", "shared_workspace_v1"],
                "max_batch_size": 100,
                "max_envelope_payload_bytes": 262144,
                "max_attachment_bytes": 1048576,
            },
            {
                "device_id": "device-1",
                "server_capabilities": {
                    "protocol_version": 2,
                    "min_supported_protocol_version": 2,
                    "supported_domains": ["notes"],
                    "supported_operations": ["upsert"],
                    "encryption_policies": ["client_private_v1"],
                    "max_batch_size": 100,
                    "max_envelope_payload_bytes": 262144,
                    "max_attachment_bytes": 1048576,
                },
                "required_actions": [],
            },
            {
                "dataset_id": "dataset-1",
                "scope_type": "personal",
                "encryption_policy": "client_private_v1",
                "domains": ["notes"],
                "cursors": {"notes": "4"},
                "key_setup_required": False,
            },
            {"dataset_id": "dataset-1", "accepted": [], "rejected": [], "conflicts": [], "next_cursor": "5"},
            {"dataset_id": "dataset-1", "envelopes": [], "next_cursor": "6", "has_more": False},
            {"datasets": [], "devices": [], "generated_at": "2026-05-10T00:00:00Z"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    capabilities = await client.get_sync_v2_capabilities()
    device = await client.register_sync_v2_device(
        SyncV2DeviceRegisterRequest(display_name="Laptop", supported_domains=["notes"])
    )
    dataset = await client.enroll_sync_v2_dataset(
        SyncV2DatasetEnrollRequest(device_id=device.device_id, domains=["notes"])
    )
    pushed = await client.push_sync_v2_envelopes(
        SyncV2PushRequest(dataset_id=dataset.dataset_id, device_id=device.device_id, envelopes=[])
    )
    pulled = await client.pull_sync_v2_envelopes(
        dataset_id=dataset.dataset_id,
        device_id=device.device_id,
        cursor=pushed.next_cursor,
        domains=["notes"],
        page_size=10,
    )
    manifest = await client.get_sync_v2_restore_manifest(dataset_ids=[dataset.dataset_id], domains=["notes"])

    assert isinstance(capabilities, SyncV2CapabilitiesResponse)
    assert device.device_id == "device-1"
    assert dataset.cursors == {"notes": "4"}
    assert isinstance(pushed, SyncV2PushResponse)
    assert isinstance(pulled, SyncV2PullResponse)
    assert isinstance(manifest, SyncV2RestoreManifestResponse)
    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("GET", "/api/v1/sync/capabilities"),
        ("POST", "/api/v1/sync/devices/register"),
        ("POST", "/api/v1/sync/datasets/enroll"),
        ("POST", "/api/v1/sync/push"),
        ("GET", "/api/v1/sync/pull"),
        ("GET", "/api/v1/sync/restore-manifest"),
    ]
    assert mocked.await_args_list[4].kwargs["params"] == {
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "cursor": "5",
        "domain": ["notes"],
        "page_size": 10,
        "include_own_changes": False,
    }
    assert mocked.await_args_list[5].kwargs["params"] == {
        "dataset_id": ["dataset-1"],
        "domain": ["notes"],
    }
