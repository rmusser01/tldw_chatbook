from unittest.mock import Mock

import pytest

from tldw_chatbook.Sync_Interop import ServerSyncService
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
