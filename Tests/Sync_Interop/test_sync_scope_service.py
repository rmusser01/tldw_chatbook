import pytest

from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
from tldw_chatbook.Sync_Interop.sync_scope_service import SyncScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeSyncService:
    def __init__(self):
        self.calls = []

    async def send_changes(self, request_data):
        self.calls.append(("send_changes", request_data))
        return {"status": "success"}

    async def get_changes(self, *, client_id, since_change_id=0):
        self.calls.append(("get_changes", client_id, since_change_id))
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

    async def run_v2_dry_run(self, **kwargs):
        self.calls.append(("run_v2_dry_run", kwargs))
        return {
            "dry_run": True,
            "server_profile_id": kwargs["server_profile_id"],
            "workspace_scope": kwargs.get("workspace_scope"),
            "device_id": "device-1",
            "dataset_id": "dataset-1",
            "domains": kwargs.get("domains") or ["notes"],
            "profile_mode": "local_first",
        }


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_sync_scope_service_routes_server_transport_and_normalizes_records():
    server = FakeSyncService()
    policy = FakePolicyEnforcer()
    scope = SyncScopeService(server_service=server, policy_enforcer=policy)

    sent = await scope.send_changes(mode="server", request_data={"client_id": "chatbook-client-1", "changes": []})
    pulled = await scope.get_changes(mode="server", client_id="chatbook-client-1", since_change_id=30)

    assert sent["backend"] == "server"
    assert sent["record_id"] == "server:sync_batch:chatbook-client-1"
    assert pulled["backend"] == "server"
    assert pulled["record_id"] == "server:sync_delta:chatbook-client-1:33"
    assert pulled["changes"][0]["record_id"] == "server:sync_change:33"
    assert server.calls == [
        ("send_changes", {"client_id": "chatbook-client-1", "changes": []}),
        ("get_changes", "chatbook-client-1", 30),
    ]
    assert policy.calls == ["sync.changes.launch.server", "sync.changes.observe.server"]


@pytest.mark.asyncio
async def test_sync_scope_service_honestly_rejects_local_mode_as_transport_only():
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Server sync transport is unavailable in local mode"):
        await scope.get_changes(mode="local", client_id="chatbook-client-1")

    assert server.calls == []


@pytest.mark.asyncio
async def test_sync_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("server_auth_required"))

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.get_changes(mode="server", client_id="chatbook-client-1")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_sync_scope_service_reports_known_unsupported_capabilities():
    scope = SyncScopeService(server_service=FakeSyncService())

    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "sync.transport_only.server",
            "source": "server",
            "supported": False,
            "reason_code": "sync_engine_missing",
            "user_message": "Server sync transport wrappers are available, but Chatbook has not enabled automatic local/server mirroring yet.",
            "affected_action_ids": [],
        }
    ]
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "sync.transport.remote.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server sync send/get transport is unavailable in local/offline mode; local file-note sync remains separate.",
            "affected_action_ids": ["sync.changes.launch.server", "sync.changes.observe.server"],
        }
    ]


def test_sync_scope_service_reports_unsupported_unsyncable_domain():
    scope = SyncScopeService(server_service=FakeSyncService())

    report = scope.list_unsupported_sync_domains(
        domains=["notes", "workspace_notes", "media", "research", "chat_metadata", "unknown"],
        server_profile_id="server-a",
        workspace_id="workspace-1",
    )

    assert report == [
        {
            "operation_id": "sync.domain.unsupported.unknown",
            "source": "server",
            "supported": False,
            "reason_code": "not_registered",
            "user_message": "Domain 'unknown' is not registered for sync dry-run readiness.",
            "affected_action_ids": [],
            "domain": "unknown",
            "server_profile_id": "server-a",
            "workspace_id": "workspace-1",
            "write_enabled": False,
        }
    ]


def test_sync_scope_service_records_dry_run_mirror_report_from_repository_identity_map(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_identity_mapping(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        entity_type="note",
        local_entity_id="local-note-1",
        remote_entity_id="remote-note-1",
        mapping_status="confirmed",
    )
    scope = SyncScopeService(server_service=FakeSyncService(), state_repository=repo)

    report = scope.record_dry_run_mirror_report(
        mode="server",
        domain="notes",
        entity_type="note",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        local_records=[{"id": "local-note-1", "version": "local-v2"}],
        remote_records=[{"id": "remote-note-1", "version": "remote-v1"}],
    )

    stored = repo.list_mirror_reports(domain="notes")
    profile_state = repo.get_sync_profile_state(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert report["backend"] == "server"
    assert report["record_id"] == f"server:sync_mirror_report:{report['report_id']}"
    assert report["report"]["dry_run"] is True
    assert report["report"]["write_enabled"] is False
    assert report["report"]["mapped_count"] == 1
    assert report["report"]["actions"][0]["identity"]["local_entity_id"] == "local-note-1"
    assert stored[0]["report_id"] == report["report_id"]
    assert profile_state["last_mirror_report_id"] == report["report_id"]


def test_sync_scope_service_requires_repository_for_dry_run_mirror_reports():
    scope = SyncScopeService(server_service=FakeSyncService())

    with pytest.raises(ValueError, match="Sync state repository is unavailable"):
        scope.record_dry_run_mirror_report(
            mode="server",
            domain="notes",
            entity_type="note",
            server_profile_id="server-a",
            workspace_scope="workspace-1",
        )


def test_sync_scope_service_lists_write_sync_promotion_states_without_dispatch(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.record_mirror_report(
        source_authority="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domain="notes",
        report={
            "dry_run": True,
            "write_enabled": False,
            "mapped_count": 2,
            "actions": [],
        },
    )
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server, state_repository=repo)

    states = scope.list_write_sync_promotion_states(
        domains=["notes", "unknown"],
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert [state.domain for state in states] == ["notes", "unknown"]
    assert states[0].status == "dry-run"
    assert states[0].sync_label == "Sync: dry-run only"
    assert states[0].mirror_label == "Mirror: 2 mapped records"
    assert states[0].mutation_allowed is False
    assert states[1].status == "unavailable"
    assert states[1].sync_label == "Sync: unavailable"
    assert states[1].mutation_allowed is False
    assert server.calls == []


def test_sync_scope_service_write_sync_promotion_state_reports_profile_rollback_without_dispatch(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    repo.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        profile_mode="local_first_sync",
        device_id="device-1",
        dataset_id="dataset-1",
        last_error="rollback required before promotion can continue",
    )
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server, state_repository=repo)

    states = scope.list_write_sync_promotion_states(
        domains=["notes"],
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert len(states) == 1
    assert states[0].status == "rollback-required"
    assert states[0].sync_label == "Sync: rollback required"
    assert states[0].rollback_label == "Rollback: required before writes"
    assert states[0].mutation_allowed is False
    assert server.calls == []


@pytest.mark.asyncio
async def test_sync_scope_service_prepares_local_only_mode_without_sync_side_effects(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server, state_repository=repo)

    result = await scope.prepare_sync_v2_profile_mode(
        profile_mode="local_only",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert result == {
        "dry_run": True,
        "profile_mode": "local_only",
        "backend": "local",
        "server_profile_id": "server-a",
        "workspace_scope": "workspace-1",
        "sync_dataset_created": False,
        "local_sync_enabled": False,
        "server_frontend": False,
    }
    assert server.calls == []
    assert repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    ) is None


@pytest.mark.asyncio
async def test_sync_scope_service_prepares_server_frontend_mode_without_local_sync_state(tmp_path):
    repo = SyncStateRepository(tmp_path / "sync_state.db")
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server, state_repository=repo)

    result = await scope.prepare_sync_v2_profile_mode(
        profile_mode="server_frontend",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    )

    assert result == {
        "dry_run": True,
        "profile_mode": "server_frontend",
        "backend": "server",
        "server_profile_id": "server-a",
        "workspace_scope": "workspace-1",
        "sync_dataset_created": False,
        "local_sync_enabled": False,
        "server_frontend": True,
    }
    assert server.calls == []
    assert repo.get_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
    ) is None


@pytest.mark.asyncio
async def test_sync_scope_service_delegates_local_first_mode_to_sync_v2_dry_run():
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server)

    result = await scope.prepare_sync_v2_profile_mode(
        profile_mode="local_first",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        display_name="Laptop",
        domains=["notes", "chat"],
        client_version="0.1.0",
    )

    assert result["profile_mode"] == "local_first"
    assert result["device_id"] == "device-1"
    assert server.calls == [
        (
            "run_v2_dry_run",
            {
                "server_profile_id": "server-a",
                "authenticated_principal_id": "user-a",
                "workspace_scope": "workspace-1",
                "display_name": "Laptop",
                "domains": ["notes", "chat"],
                "client_version": "0.1.0",
                "scope_type": "personal",
                "encryption_policy": "client_private_v1",
                "profile_mode": "local_first",
            },
        )
    ]


@pytest.mark.asyncio
async def test_sync_scope_service_delegates_canonical_local_first_sync_mode_to_dry_run():
    server = FakeSyncService()
    scope = SyncScopeService(server_service=server)

    result = await scope.prepare_sync_v2_profile_mode(
        profile_mode="local_first_sync",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        display_name="Laptop",
        domains=["notes"],
    )

    assert result["profile_mode"] == "local_first_sync"
    assert result["device_id"] == "device-1"
    assert server.calls[0][0] == "run_v2_dry_run"
    assert server.calls[0][1]["profile_mode"] == "local_first_sync"
