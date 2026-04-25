from unittest.mock import Mock

import pytest

from tldw_chatbook.External_Connectors_Interop import ServerConnectorsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


def _account_payload():
    return {"id": 7, "provider": "drive", "display_name": "Drive", "connected": True}


def _source_payload(**overrides):
    payload = {
        "id": 11,
        "account_id": 7,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
        "path": "/Research",
        "options": {"recursive": True},
        "enabled": True,
    }
    payload.update(overrides)
    return payload


class FakeConnectorsClient:
    def __init__(self):
        self.calls = []

    async def list_connector_providers(self):
        self.calls.append(("list_connector_providers",))
        return [{"name": "drive", "auth_type": "oauth2"}]

    async def authorize_connector_provider(self, provider, **kwargs):
        self.calls.append(("authorize_connector_provider", provider, kwargs))
        return {"auth_url": "https://accounts.example.test/oauth", "state": kwargs.get("state")}

    async def complete_connector_oauth_callback(self, provider, **kwargs):
        self.calls.append(("complete_connector_oauth_callback", provider, kwargs))
        return _account_payload()

    async def list_connector_accounts(self):
        self.calls.append(("list_connector_accounts",))
        return [_account_payload()]

    async def delete_connector_account(self, account_id):
        self.calls.append(("delete_connector_account", account_id))
        return True

    async def browse_connector_sources(self, provider, **kwargs):
        self.calls.append(("browse_connector_sources", provider, kwargs))
        return {"items": [{"id": "root", "name": "Root"}], "next_cursor": None}

    async def create_connector_source(self, request_data):
        self.calls.append(("create_connector_source", request_data.model_dump(mode="json")))
        return _source_payload()

    async def list_connector_sources(self):
        self.calls.append(("list_connector_sources",))
        return [_source_payload()]

    async def update_connector_source(self, source_id, request_data):
        self.calls.append(("update_connector_source", source_id, request_data.model_dump(exclude_none=True, mode="json")))
        return _source_payload(id=source_id, enabled=False)

    async def import_connector_source(self, source_id):
        self.calls.append(("import_connector_source", source_id))
        return {"id": "import-1", "source_id": source_id, "type": "import", "status": "queued"}

    async def get_connector_source_sync_status(self, source_id):
        self.calls.append(("get_connector_source_sync_status", source_id))
        return {"source_id": source_id, "provider": "drive", "enabled": True, "state": "idle"}

    async def trigger_connector_source_sync(self, source_id):
        self.calls.append(("trigger_connector_source_sync", source_id))
        return {
            "source_id": source_id,
            "provider": "drive",
            "status": "queued",
            "job": {"id": "sync-1", "source_id": source_id, "type": "incremental_sync", "status": "queued"},
        }

    async def get_connector_job_status(self, job_id):
        self.calls.append(("get_connector_job_status", job_id))
        return {"id": job_id, "status": "queued"}


@pytest.mark.asyncio
async def test_server_connectors_service_routes_user_connector_surface_with_policy_actions():
    client = FakeConnectorsClient()
    policy = Mock()
    service = ServerConnectorsService(client=client, policy_enforcer=policy)

    providers = await service.list_providers()
    authorize = await service.authorize_provider("drive", state="state-1", scopes=["drive.readonly"])
    account = await service.complete_oauth_callback("drive", code="code-1", state="state-1")
    accounts = await service.list_accounts()
    deleted = await service.delete_account(7)
    browsed = await service.browse_sources("drive", account_id=7, parent_remote_id="root")
    created = await service.create_source(
        account_id=7,
        provider="drive",
        remote_id="root",
        type="folder",
        path="/Research",
        options={"recursive": True},
    )
    sources = await service.list_sources()
    patched = await service.update_source(11, enabled=False)
    imported = await service.import_source(11)
    sync_status = await service.get_source_sync_status(11)
    sync_trigger = await service.trigger_source_sync(11)
    job_status = await service.get_job_status(99)

    assert providers[0]["name"] == "drive"
    assert authorize["state"] == "state-1"
    assert account["id"] == 7
    assert accounts[0]["id"] == 7
    assert deleted is True
    assert browsed["items"][0]["id"] == "root"
    assert created["id"] == 11
    assert sources[0]["id"] == 11
    assert patched["enabled"] is False
    assert imported["id"] == "import-1"
    assert sync_status["source_id"] == 11
    assert sync_trigger["job"]["id"] == "sync-1"
    assert job_status["id"] == 99
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "connectors.providers.list.server",
        "connectors.providers.launch.server",
        "connectors.providers.launch.server",
        "connectors.accounts.list.server",
        "connectors.accounts.delete.server",
        "connectors.sources.list.server",
        "connectors.sources.create.server",
        "connectors.sources.list.server",
        "connectors.sources.update.server",
        "connectors.sources.launch.server",
        "connectors.sources.observe.server",
        "connectors.sources.launch.server",
        "connectors.jobs.observe.server",
    ]


@pytest.mark.asyncio
async def test_server_connectors_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeConnectorsClient()
    service = ServerConnectorsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_accounts()

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
