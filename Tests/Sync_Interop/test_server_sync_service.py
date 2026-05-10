from unittest.mock import Mock

import pytest

from tldw_chatbook.Sync_Interop import ServerSyncService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.tldw_api import ClientChangesPayload, SyncOperation, SyncSendEntity, SyncSendLogEntry


def _payload() -> ClientChangesPayload:
    return ClientChangesPayload(
        client_id="chatbook-client-1",
        changes=[
            SyncSendLogEntry(
                change_id=12,
                entity=SyncSendEntity.MEDIA,
                entity_uuid="media-uuid-1",
                operation=SyncOperation.UPDATE,
                timestamp="2026-04-25T12:00:00Z",
                client_id="chatbook-client-1",
                version=3,
                payload='{"uuid":"media-uuid-1","title":"Updated"}',
            )
        ],
        last_processed_server_id=30,
    )


class FakeSyncClient:
    def __init__(self):
        self.calls = []

    async def send_sync_changes(self, request_data):
        self.calls.append(("send_sync_changes", request_data.model_dump(mode="json")))
        return {"status": "success"}

    async def get_sync_changes(self, *, client_id, since_change_id=0):
        self.calls.append(("get_sync_changes", client_id, since_change_id))
        return {
            "changes": [
                {
                    "change_id": 33,
                    "entity": "Keywords",
                    "entity_uuid": "keyword-uuid-1",
                    "operation": "create",
                    "timestamp": "2026-04-25T12:01:00Z",
                    "client_id": "server-client",
                    "version": 1,
                    "payload": '{"uuid":"keyword-uuid-1","keyword":"paper"}',
                }
            ],
            "latest_change_id": 33,
        }

    async def get_sync_v2_capabilities(self):
        self.calls.append(("get_sync_v2_capabilities",))
        return {
            "protocol_version": 2,
            "min_supported_protocol_version": 2,
            "supported_domains": ["notes", "chat", "workspaces", "source_cache", "media"],
            "supported_operations": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
            "encryption_policies": ["client_private_v1"],
            "max_batch_size": 100,
            "max_envelope_payload_bytes": 262144,
            "max_attachment_bytes": 1048576,
        }

    async def register_sync_v2_device(self, request_data):
        self.calls.append(("register_sync_v2_device", request_data.model_dump(mode="json")))
        return {
            "device_id": request_data.device_id or "device-1",
            "server_capabilities": await self.get_sync_v2_capabilities(),
            "required_actions": [],
        }

    async def enroll_sync_v2_dataset(self, request_data):
        self.calls.append(("enroll_sync_v2_dataset", request_data.model_dump(mode="json")))
        return {
            "dataset_id": request_data.dataset_id or "dataset-1",
            "scope_type": request_data.scope_type,
            "encryption_policy": request_data.encryption_policy,
            "domains": request_data.domains,
            "workspace_id": request_data.workspace_id,
            "cursors": {"notes": "4"},
            "key_setup_required": False,
        }

    async def push_sync_v2_envelopes(self, request_data):
        self.calls.append(("push_sync_v2_envelopes", request_data.model_dump(mode="json")))
        return {"dataset_id": request_data.dataset_id, "accepted": [], "rejected": [], "conflicts": [], "next_cursor": "5"}

    async def pull_sync_v2_envelopes(
        self,
        *,
        dataset_id,
        device_id,
        cursor=None,
        domains=None,
        page_size=None,
        include_own_changes=False,
    ):
        self.calls.append(
            (
                "pull_sync_v2_envelopes",
                dataset_id,
                device_id,
                cursor,
                domains,
                page_size,
                include_own_changes,
            )
        )
        return {"dataset_id": dataset_id, "envelopes": [], "next_cursor": "6", "has_more": False}

    async def store_sync_v2_key_recovery_bundle(self, request_data):
        self.calls.append(("store_sync_v2_key_recovery_bundle", request_data.model_dump(mode="json")))
        return {
            "key_record_id": "key-record-1",
            "dataset_id": request_data.dataset_id,
            "device_id": request_data.device_id,
            "key_purpose": request_data.key_purpose,
            "recovery_hint": request_data.recovery_hint,
            "created_at": "2026-05-10T00:00:00Z",
        }


@pytest.mark.asyncio
async def test_server_sync_service_routes_transport_with_policy_actions():
    client = FakeSyncClient()
    policy = Mock()
    service = ServerSyncService(client=client, policy_enforcer=policy)

    sent = await service.send_changes(_payload())
    pulled = await service.get_changes(client_id="chatbook-client-1", since_change_id=30)

    assert sent == {"status": "success"}
    assert pulled["latest_change_id"] == 33
    assert client.calls == [
        (
            "send_sync_changes",
            {
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
            },
        ),
        ("get_sync_changes", "chatbook-client-1", 30),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "sync.changes.launch.server",
        "sync.changes.observe.server",
    ]


@pytest.mark.asyncio
async def test_server_sync_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeSyncClient()
    service = ServerSyncService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.send_changes(_payload())

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


class FreshClientProvider:
    def __init__(self):
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = object()
        self.clients.append(client)
        return client


def test_server_sync_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerSyncService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_sync_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerSyncService.from_server_context_provider(provider)

    assert isinstance(service, ServerSyncService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_sync_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerSyncService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_sync_service_from_config_returns_provider_backed_service():
    service = ServerSyncService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerSyncService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


@pytest.mark.asyncio
async def test_server_sync_service_runs_sync_v2_no_content_dry_run_and_persists_state(tmp_path):
    client = FakeSyncClient()
    policy = Mock()
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    service = ServerSyncService(client=client, policy_enforcer=policy, state_repository=repo)

    result = await service.run_v2_dry_run(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        display_name="Laptop",
        domains=["notes"],
    )

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )
    cursor = repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    )

    assert result["dry_run"] is True
    assert result["device_id"] == "device-1"
    assert result["dataset_id"] == "dataset-1"
    assert result["pushed_envelopes"] == 0
    assert result["pulled_envelopes"] == 0
    assert result["next_cursor"] == "6"
    assert profile["profile_mode"] == "local_first"
    assert profile["dataset_cursors"] == {"notes": "4", "sync_v2": "6"}
    assert cursor.cursor == "6"
    assert client.calls[0] == ("get_sync_v2_capabilities",)
    assert client.calls[-1] == (
        "pull_sync_v2_envelopes",
        "dataset-1",
        "device-1",
        None,
        ["notes"],
        1,
        False,
    )
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "sync.v2.dry_run.server"
    ]


@pytest.mark.asyncio
async def test_server_sync_service_stores_v2_recovery_bundle_with_policy_gate():
    client = FakeSyncClient()
    policy = Mock()
    service = ServerSyncService(client=client, policy_enforcer=policy)

    response = await service.store_v2_recovery_bundle(
        dataset_id="dataset-1",
        device_id="device-1",
        wrapped_key_blob="wrapped",
        kdf_metadata={"algorithm": "scrypt"},
        recovery_hint="personal laptop",
    )

    assert response["key_record_id"] == "key-record-1"
    assert client.calls[-1] == (
        "store_sync_v2_key_recovery_bundle",
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped",
            "kdf_metadata": {"algorithm": "scrypt"},
            "recovery_hint": "personal laptop",
            "rotation_of_key_record_id": None,
        },
    )
    assert policy.require_allowed.call_args.kwargs["action_id"] == "sync.v2.keys.store.server"
