import pytest

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
        domains=["notes", "unknown"],
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
