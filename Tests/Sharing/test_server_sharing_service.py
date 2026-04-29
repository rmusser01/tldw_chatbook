import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Sharing.server_sharing_service as sharing_module
import tldw_chatbook.Sharing_Interop.server_sharing_service as sharing_interop_module
from tldw_chatbook.Sharing import ServerSharingService as PublicServerSharingService
from tldw_chatbook.Sharing_Interop import ServerSharingService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeSharingClient:
    def __init__(self):
        self.calls = []

    async def create_share_token(self, request_data):
        self.calls.append(("create_share_token", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 9, "raw_token": "secret-token"}

    async def list_share_tokens(self):
        self.calls.append(("list_share_tokens",))
        return {"tokens": [], "total": 0}

    async def revoke_share_token(self, token_id):
        self.calls.append(("revoke_share_token", token_id))
        return {"detail": "Token revoked"}

    async def share_workspace(self, workspace_id, request_data):
        self.calls.append(("share_workspace", workspace_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 7, "workspace_id": workspace_id}

    async def list_shared_with_me(self):
        self.calls.append(("list_shared_with_me",))
        return {"items": [], "total": 0}

    async def preview_public_share(self, token):
        self.calls.append(("preview_public_share", token))
        return {"resource_type": "workspace", "access_level": "view_chat"}

    async def list_sharing_audit_events(self, **kwargs):
        self.calls.append(("list_sharing_audit_events", kwargs))
        return {
            "events": [
                {
                    "id": 12,
                    "event_type": "share.created",
                    "resource_type": kwargs.get("resource_type") or "workspace",
                    "resource_id": kwargs.get("resource_id") or "ws-1",
                    "owner_user_id": 1,
                }
            ],
            "total": 1,
        }


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class FreshClientProvider:
    def __init__(self, factory):
        self.factory = factory
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = self.factory()
        self.clients.append(client)
        return client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


async def _exercise_public_sharing(service):
    return await service.list_share_tokens()


async def _exercise_interop_sharing(service):
    return await service.list_links()


SHARING_IMPORT_PATHS = [
    (PublicServerSharingService, sharing_module, _exercise_public_sharing),
    (ServerSharingService, sharing_interop_module, _exercise_interop_sharing),
]


def test_server_sharing_service_modules_do_not_reference_legacy_config_client_builders():
    for _, module, _ in SHARING_IMPORT_PATHS:
        source = inspect.getsource(module)

        assert "build_runtime_api_client_from_config" not in source
        assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), SHARING_IMPORT_PATHS)
async def test_server_sharing_service_direct_client_takes_precedence_over_provider(
    service_cls,
    _module,
    exercise,
):
    client = FakeSharingClient()
    provider = ExplodingClientProvider()
    service = service_cls(client=client, client_provider=provider)

    result = await exercise(service)

    assert result["total"] == 0
    assert provider.build_calls == 0
    assert client.calls == [("list_share_tokens",)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), SHARING_IMPORT_PATHS)
async def test_server_sharing_service_from_server_context_provider_is_lazy(
    service_cls,
    _module,
    exercise,
):
    client = FakeSharingClient()
    provider = FakeClientProvider(client)
    service = service_cls.from_server_context_provider(provider)

    assert isinstance(service, service_cls)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await exercise(service)

    assert result["total"] == 0
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [("list_share_tokens",)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), SHARING_IMPORT_PATHS)
async def test_server_sharing_service_re_resolves_provider_without_service_local_client_cache(
    service_cls,
    _module,
    exercise,
):
    provider = FreshClientProvider(FakeSharingClient)
    service = service_cls.from_server_context_provider(provider)

    await exercise(service)
    await exercise(service)

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [("list_share_tokens",)]
    assert provider.clients[1].calls == [("list_share_tokens",)]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


@pytest.mark.parametrize(("service_cls", "_module", "_exercise"), SHARING_IMPORT_PATHS)
def test_server_sharing_service_from_config_returns_provider_backed_service(
    service_cls,
    _module,
    _exercise,
):
    service = service_cls.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, service_cls)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


@pytest.mark.asyncio
async def test_server_sharing_service_routes_links_and_permissions_with_policy_actions():
    client = FakeSharingClient()
    policy = Mock()
    service = ServerSharingService(client=client, policy_enforcer=policy)

    created = await service.create_link(resource_type="workspace", resource_id="ws-1")
    links = await service.list_links()
    revoked = await service.revoke_link(9)
    share = await service.share_workspace(workspace_id="ws-1", share_scope_type="team", share_scope_id=3)
    shared = await service.list_shared_with_me()
    preview = await service.inspect_public_link("secret-token")
    events = await service.observe_link_events(resource_type="workspace", resource_id="ws-1")

    assert created["raw_token"] == "secret-token"
    assert links["total"] == 0
    assert revoked["detail"] == "Token revoked"
    assert share["workspace_id"] == "ws-1"
    assert shared["total"] == 0
    assert preview["resource_type"] == "workspace"
    assert events["events"][0]["event_type"] == "share.created"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "sharing.links.create.server",
        "sharing.links.list.server",
        "sharing.links.revoke.server",
        "sharing.permissions.configure.server",
        "sharing.links.list.server",
        "sharing.links.inspect.server",
        "sharing.links.observe.server",
    ]


@pytest.mark.asyncio
async def test_server_sharing_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeSharingClient()
    service = ServerSharingService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_links()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
