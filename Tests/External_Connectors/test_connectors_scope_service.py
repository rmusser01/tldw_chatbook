import pytest

from tldw_chatbook.External_Connectors_Interop.connectors_scope_service import ConnectorsScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeConnectorsService:
    def __init__(self):
        self.calls = []

    async def list_providers(self):
        self.calls.append(("list_providers",))
        return [{"name": "drive", "auth_type": "oauth2"}]

    async def authorize_provider(self, provider, **kwargs):
        self.calls.append(("authorize_provider", provider, kwargs))
        return {"auth_url": "https://accounts.example.test/oauth", "state": kwargs.get("state")}

    async def list_accounts(self):
        self.calls.append(("list_accounts",))
        return [{"id": 7, "provider": "drive", "display_name": "Drive"}]

    async def browse_sources(self, provider, **kwargs):
        self.calls.append(("browse_sources", provider, kwargs))
        return {"items": [{"id": "root", "name": "Root"}], "next_cursor": None}

    async def list_sources(self):
        self.calls.append(("list_sources",))
        return [{"id": 11, "provider": "drive", "remote_id": "root", "type": "folder"}]

    async def import_source(self, source_id):
        self.calls.append(("import_source", source_id))
        return {"id": "import-1", "source_id": source_id, "status": "queued"}

    async def get_source_sync_status(self, source_id):
        self.calls.append(("get_source_sync_status", source_id))
        return {"source_id": source_id, "provider": "drive", "state": "idle"}

    async def get_job_status(self, job_id):
        self.calls.append(("get_job_status", job_id))
        return {"id": job_id, "status": "queued"}


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
async def test_connectors_scope_service_routes_server_operations_and_normalizes_records():
    server = FakeConnectorsService()
    policy = FakePolicyEnforcer()
    scope = ConnectorsScopeService(server_service=server, policy_enforcer=policy)

    providers = await scope.list_providers(mode="server")
    authorize = await scope.authorize_provider("drive", mode="server", state="state-1")
    accounts = await scope.list_accounts(mode="server")
    browse = await scope.browse_sources("drive", mode="server", account_id=7)
    sources = await scope.list_sources(mode="server")
    imported = await scope.import_source(11, mode="server")
    sync_status = await scope.get_source_sync_status(11, mode="server")
    job_status = await scope.get_job_status(99, mode="server")

    assert providers[0]["record_id"] == "server:connector_provider:drive"
    assert authorize["backend"] == "server"
    assert accounts[0]["record_id"] == "server:connector_account:7"
    assert browse["items"][0]["record_id"] == "server:connector_remote_source:drive:root"
    assert sources[0]["record_id"] == "server:connector_source:11"
    assert imported["record_id"] == "server:connector_job:import-1"
    assert sync_status["record_id"] == "server:connector_source_sync:11"
    assert job_status["record_id"] == "server:connector_job:99"
    assert server.calls == [
        ("list_providers",),
        ("authorize_provider", "drive", {"state": "state-1"}),
        ("list_accounts",),
        ("browse_sources", "drive", {"account_id": 7}),
        ("list_sources",),
        ("import_source", 11),
        ("get_source_sync_status", 11),
        ("get_job_status", 99),
    ]
    assert policy.calls == [
        "connectors.providers.list.server",
        "connectors.providers.launch.server",
        "connectors.accounts.list.server",
        "connectors.sources.list.server",
        "connectors.sources.list.server",
        "connectors.sources.launch.server",
        "connectors.sources.observe.server",
        "connectors.jobs.observe.server",
    ]


@pytest.mark.asyncio
async def test_connectors_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeConnectorsService()
    scope = ConnectorsScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="External connectors are server-only"):
        await scope.list_accounts(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_connectors_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeConnectorsService()
    scope = ConnectorsScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_accounts(mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_connectors_scope_service_reports_known_unsupported_capabilities():
    scope = ConnectorsScopeService(server_service=FakeConnectorsService())

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "connectors.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "External connectors are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
