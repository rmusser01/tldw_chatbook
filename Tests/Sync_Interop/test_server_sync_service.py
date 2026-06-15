from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from tldw_chatbook.Sync_Interop import ServerSyncService
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError, RuntimeSourceState
from tldw_chatbook.tldw_api import (
    ClientChangesPayload,
    SyncOperation,
    SyncSendEntity,
    SyncSendLogEntry,
    SyncV2Envelope,
)


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


def _server_runtime_state() -> RuntimeSourceState:
    return RuntimeSourceState(
        active_source="server",
        server_configured=True,
        server_reachability="reachable",
        server_reachability_checked_at=datetime.now(timezone.utc),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime.now(timezone.utc),
    )


def _local_runtime_state() -> RuntimeSourceState:
    return RuntimeSourceState(active_source="local")


def _sync_v2_envelope(
    *,
    dataset_id: str = "dataset-1",
    device_id: str | None = "device-1",
    domain: str = "notes",
    entity_id: str = "note-1",
) -> SyncV2Envelope:
    return SyncV2Envelope(
        client_envelope_id=f"{device_id or 'unknown'}:{domain}:{entity_id}:1",
        dataset_id=dataset_id,
        device_id=device_id,
        domain=domain,
        entity_id=entity_id,
        operation="upsert",
        adapter_version=1,
        stable_key=entity_id,
        payload_hash=f"sha256:{entity_id}",
    )


class FakeSyncClient:
    def __init__(self, *, push_response=None, pull_response=None):
        self.calls = []
        self.push_response = push_response
        self.pull_response = pull_response

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
            "protocol_version": "sync-v2-m1",
            "min_supported_protocol_version": "sync-v2-m1",
            "domains": ["notes", "chat", "workspaces", "source_cache", "media"],
            "operations": {
                "notes": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
                "chat": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
                "workspaces": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
                "source_cache": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
                "media": ["upsert", "delete", "link", "unlink", "resolve_conflict"],
            },
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
        if self.push_response is not None:
            return self.push_response
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
        if self.pull_response is not None:
            return self.pull_response
        return {"dataset_id": dataset_id, "envelopes": [], "next_cursor": "6", "has_more": False}

    async def get_sync_v2_restore_manifest(self, *, dataset_ids=None, domains=None):
        self.calls.append(("get_sync_v2_restore_manifest", dataset_ids, domains))
        return {
            "datasets": [
                {
                    "dataset_id": "dataset-1",
                    "scope_type": "personal",
                    "encryption_policy": "client_private_v1",
                    "domains": domains or ["notes"],
                    "approximate_counts": {"notes": 1},
                    "unresolved_conflicts": 1,
                    "key_recovery_available": True,
                }
            ],
            "devices": [{"device_id": "device-1", "display_name": "Laptop"}],
            "generated_at": "2026-05-10T00:00:00Z",
            "filters_applied": {"dataset_id": dataset_ids, "domain": domains},
        }

    async def list_sync_v2_conflicts(self, *, dataset_id, status=None):
        self.calls.append(("list_sync_v2_conflicts", dataset_id, status))
        return [
            {
                "conflict_id": "conflict-1",
                "dataset_id": dataset_id,
                "domain": "notes",
                "entity_id": "note-1",
                "conflict_type": "encrypted_content_edit",
                "status": status or "unresolved",
            }
        ]

    async def resolve_sync_v2_conflict(self, conflict_id, request_data):
        self.calls.append(("resolve_sync_v2_conflict", conflict_id, request_data.model_dump(mode="json")))
        return {
            "conflict_id": conflict_id,
            "dataset_id": "dataset-1",
            "domain": "notes",
            "entity_id": "note-1",
            "conflict_type": "encrypted_content_edit",
            "status": "resolved",
            "resolved_by_envelope_id": None,
        }

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

    async def list_sync_v2_key_recovery_bundles(
        self,
        *,
        dataset_id,
        device_id=None,
        key_purpose="dataset_recovery",
    ):
        self.calls.append(("list_sync_v2_key_recovery_bundles", dataset_id, device_id, key_purpose))
        return {
            "dataset_id": dataset_id,
            "key_records": [
                {
                    "key_record_id": "key-record-1",
                    "dataset_id": dataset_id,
                    "device_id": device_id,
                    "key_purpose": key_purpose,
                    "wrapped_key_blob": "wrapped:opaque-key",
                    "kdf_metadata": {"algorithm": "scrypt"},
                    "recovery_hint": "personal laptop",
                    "rotation_of_key_record_id": None,
                    "created_at": "2026-05-10T00:00:00Z",
                    "revoked_at": None,
                }
            ],
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
    assert profile["profile_mode"] == "local_first_sync"
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
async def test_server_sync_service_preserves_requested_sync_v2_profile_mode(tmp_path):
    client = FakeSyncClient()
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    service = ServerSyncService(client=client, state_repository=repo)

    await service.run_v2_dry_run(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        display_name="Laptop",
        domains=["notes"],
        profile_mode="local_first",
    )

    profile = repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )
    assert profile["profile_mode"] == "local_first"


@pytest.mark.asyncio
async def test_server_sync_service_rejects_mismatched_dry_run_push_response_before_persisting(
    tmp_path,
):
    client = FakeSyncClient(
        push_response={
            "dataset_id": "other-dataset",
            "accepted": [],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "5",
        }
    )
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    service = ServerSyncService(client=client, state_repository=repo)

    with pytest.raises(ValueError, match="push response dataset_id"):
        await service.run_v2_dry_run(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            display_name="Laptop",
            domains=["notes"],
        )

    assert repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    ) is None
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor is None
    assert [call[0] for call in client.calls] == [
        "get_sync_v2_capabilities",
        "register_sync_v2_device",
        "get_sync_v2_capabilities",
        "enroll_sync_v2_dataset",
        "push_sync_v2_envelopes",
    ]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_uncheckpointed_dry_run_pull_response_before_persisting(
    tmp_path,
):
    envelope = _sync_v2_envelope(device_id="remote-device")
    client = FakeSyncClient(
        pull_response={
            "dataset_id": "dataset-1",
            "envelopes": [envelope.model_dump(mode="json")],
            "next_cursor": None,
            "has_more": False,
        }
    )
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    service = ServerSyncService(client=client, state_repository=repo)

    with pytest.raises(ValueError, match="envelopes.*next_cursor"):
        await service.run_v2_dry_run(
            server_profile_id="server-a",
            authenticated_principal_id="user-a",
            workspace_scope="workspace-1",
            display_name="Laptop",
            domains=["notes"],
        )

    assert repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    ) is None
    assert repo.get_remote_pull_cursor(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="sync_v2",
        remote_collection="dataset-1",
    ).cursor is None
    assert client.calls[-1] == (
        "pull_sync_v2_envelopes",
        "dataset-1",
        "device-1",
        None,
        ["notes"],
        1,
        False,
    )


@pytest.mark.asyncio
async def test_server_sync_service_rejects_mismatched_v2_push_dataset_before_dispatch():
    client = FakeSyncClient()
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="dataset_id"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[_sync_v2_envelope(dataset_id="other-dataset")],
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_sync_service_rejects_mismatched_v2_push_device_before_dispatch():
    client = FakeSyncClient()
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="device_id"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[_sync_v2_envelope(device_id="other-device")],
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_sync_service_rejects_v2_push_domain_outside_allowlist_before_dispatch():
    client = FakeSyncClient()
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="domain"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            domains=["notes"],
            envelopes=[_sync_v2_envelope(domain="chat")],
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_sync_service_rejects_duplicate_v2_push_envelope_ids_before_dispatch():
    client = FakeSyncClient()
    service = ServerSyncService(client=client)
    envelope = _sync_v2_envelope()

    with pytest.raises(ValueError, match="duplicate client_envelope_id"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[envelope, envelope],
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_sync_service_rejects_mismatched_v2_push_response_dataset_after_dispatch():
    envelope = _sync_v2_envelope()
    client = FakeSyncClient(
        push_response={
            "dataset_id": "other-dataset",
            "accepted": [{"client_envelope_id": envelope.client_envelope_id}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "5",
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="push response dataset_id"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[envelope],
        )

    assert [call[0] for call in client.calls] == ["push_sync_v2_envelopes"]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_missing_v2_push_response_dataset_after_dispatch():
    envelope = _sync_v2_envelope()
    client = FakeSyncClient(
        push_response={
            "accepted": [{"client_envelope_id": envelope.client_envelope_id}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "5",
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="push response dataset_id"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[envelope],
        )

    assert [call[0] for call in client.calls] == ["push_sync_v2_envelopes"]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_unknown_v2_push_response_id_after_dispatch():
    envelope = _sync_v2_envelope()
    client = FakeSyncClient(
        push_response={
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": "unknown-envelope"}],
            "rejected": [],
            "conflicts": [],
            "next_cursor": "5",
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="unknown client_envelope_id"):
        await service.push_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[envelope],
        )

    assert [call[0] for call in client.calls] == ["push_sync_v2_envelopes"]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_mismatched_v2_pull_response_dataset_after_dispatch():
    client = FakeSyncClient(
        pull_response={
            "dataset_id": "other-dataset",
            "envelopes": [],
            "next_cursor": "cursor-2",
            "has_more": False,
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="pull response dataset_id"):
        await service.pull_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            domains=["notes"],
        )

    assert [call[0] for call in client.calls] == ["pull_sync_v2_envelopes"]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_missing_v2_pull_response_dataset_after_dispatch():
    client = FakeSyncClient(
        pull_response={
            "envelopes": [],
            "next_cursor": "cursor-2",
            "has_more": False,
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="pull response dataset_id"):
        await service.pull_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            domains=["notes"],
        )

    assert [call[0] for call in client.calls] == ["pull_sync_v2_envelopes"]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_own_device_v2_pull_response_after_dispatch():
    envelope = _sync_v2_envelope(device_id="device-1")
    client = FakeSyncClient(
        pull_response={
            "dataset_id": "dataset-1",
            "envelopes": [envelope.model_dump(mode="json")],
            "next_cursor": "cursor-2",
            "has_more": False,
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="own device"):
        await service.pull_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            domains=["notes"],
        )

    assert [call[0] for call in client.calls] == ["pull_sync_v2_envelopes"]


@pytest.mark.asyncio
async def test_server_sync_service_rejects_nonempty_v2_pull_response_without_next_cursor_after_dispatch():
    envelope = _sync_v2_envelope(device_id="remote-device")
    client = FakeSyncClient(
        pull_response={
            "dataset_id": "dataset-1",
            "envelopes": [envelope.model_dump(mode="json")],
            "next_cursor": None,
            "has_more": False,
        }
    )
    service = ServerSyncService(client=client)

    with pytest.raises(ValueError, match="envelopes.*next_cursor"):
        await service.pull_v2_envelopes(
            dataset_id="dataset-1",
            device_id="device-1",
            domains=["notes"],
        )

    assert [call[0] for call in client.calls] == ["pull_sync_v2_envelopes"]


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


@pytest.mark.asyncio
async def test_server_sync_service_lists_v2_recovery_bundles_with_policy_gate():
    client = FakeSyncClient()
    policy = Mock()
    service = ServerSyncService(client=client, policy_enforcer=policy)

    response = await service.list_v2_recovery_bundles(
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
    )

    assert response["key_records"][0]["wrapped_key_blob"] == "wrapped:opaque-key"
    assert client.calls[-1] == (
        "list_sync_v2_key_recovery_bundles",
        "dataset-1",
        "device-1",
        "dataset_recovery",
    )
    assert policy.require_allowed.call_args.kwargs["action_id"] == "sync.v2.keys.retrieve.server"


@pytest.mark.asyncio
async def test_server_sync_service_sync_v2_methods_are_allowed_by_real_runtime_policy(tmp_path):
    client = FakeSyncClient()
    service = ServerSyncService(
        client=client,
        policy_enforcer=ServicePolicyEnforcer(state_provider=_server_runtime_state),
        state_repository=SyncStateRepository(tmp_path / "sync_state.db"),
    )

    dry_run = await service.run_v2_dry_run(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        display_name="Laptop",
        domains=["notes"],
    )
    await service.store_v2_recovery_bundle(
        dataset_id="dataset-1",
        device_id="device-1",
        wrapped_key_blob="wrapped",
        kdf_metadata={"algorithm": "scrypt"},
    )
    await service.list_v2_recovery_bundles(dataset_id="dataset-1", device_id="device-1")
    await service.get_v2_restore_manifest(dataset_ids=["dataset-1"], domains=["notes"])
    await service.push_v2_envelopes(dataset_id="dataset-1", device_id="device-1", envelopes=[])
    await service.pull_v2_envelopes(dataset_id="dataset-1", device_id="device-1", domains=["notes"])
    await service.list_v2_conflicts(dataset_id="dataset-1")
    await service.resolve_v2_conflict(
        conflict_id="conflict-1",
        action="accept_remote",
        resolved_by_device_id="device-1",
    )

    assert dry_run["dataset_id"] == "dataset-1"
    assert [call[0] for call in client.calls] == [
        "get_sync_v2_capabilities",
        "register_sync_v2_device",
        "get_sync_v2_capabilities",
        "enroll_sync_v2_dataset",
        "push_sync_v2_envelopes",
        "pull_sync_v2_envelopes",
        "store_sync_v2_key_recovery_bundle",
        "list_sync_v2_key_recovery_bundles",
        "get_sync_v2_restore_manifest",
        "push_sync_v2_envelopes",
        "pull_sync_v2_envelopes",
        "list_sync_v2_conflicts",
        "resolve_sync_v2_conflict",
    ]


@pytest.mark.asyncio
async def test_server_sync_service_sync_v2_policy_denial_stops_before_dispatch():
    client = FakeSyncClient()
    service = ServerSyncService(
        client=client,
        policy_enforcer=ServicePolicyEnforcer(state_provider=_local_runtime_state),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_v2_recovery_bundles(dataset_id="dataset-1")

    assert exc.value.reason_code == "wrong_source"
    assert client.calls == []


@pytest.mark.asyncio
async def test_server_sync_service_routes_restore_manifest_pull_and_conflicts_with_policy_gates():
    client = FakeSyncClient()
    policy = Mock()
    service = ServerSyncService(client=client, policy_enforcer=policy)

    manifest = await service.get_v2_restore_manifest(dataset_ids=["dataset-1"], domains=["notes"])
    pushed = await service.push_v2_envelopes(
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[],
        idempotency_key="idem-1",
        last_known_cursor="cursor-0",
    )
    pulled = await service.pull_v2_envelopes(
        dataset_id="dataset-1",
        device_id="device-1",
        cursor="cursor-1",
        domains=["notes"],
        page_size=25,
    )
    conflicts = await service.list_v2_conflicts(dataset_id="dataset-1", status="unresolved")
    resolved = await service.resolve_v2_conflict(
        conflict_id="conflict-1",
        action="accept_remote",
        resolved_by_device_id="device-1",
        notes="restored remote copy",
    )

    assert manifest["datasets"][0]["key_recovery_available"] is True
    assert pushed["next_cursor"] == "5"
    assert pulled["next_cursor"] == "6"
    assert conflicts[0]["status"] == "unresolved"
    assert resolved["status"] == "resolved"
    assert client.calls[-5:] == [
        ("get_sync_v2_restore_manifest", ["dataset-1"], ["notes"]),
        (
            "push_sync_v2_envelopes",
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "envelopes": [],
                "idempotency_key": "idem-1",
                "last_known_cursor": "cursor-0",
            },
        ),
        ("pull_sync_v2_envelopes", "dataset-1", "device-1", "cursor-1", ["notes"], 25, False),
        ("list_sync_v2_conflicts", "dataset-1", "unresolved"),
        (
            "resolve_sync_v2_conflict",
            "conflict-1",
            {
                "conflict_id": "conflict-1",
                "action": "accept_remote",
                "resolution_envelope": None,
                "resolved_by_device_id": "device-1",
                "notes": "restored remote copy",
            },
        ),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "sync.v2.restore_manifest.observe.server",
        "sync.v2.push.server",
        "sync.v2.restore.pull.server",
        "sync.v2.conflicts.observe.server",
        "sync.v2.conflicts.resolve.server",
    ]
