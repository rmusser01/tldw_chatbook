import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Outputs.server_outputs_service as outputs_module
import tldw_chatbook.Outputs_Interop.server_outputs_service as outputs_interop_module
from tldw_chatbook.Outputs import ServerOutputsService as PublicServerOutputsService
from tldw_chatbook.Outputs_Interop import ServerOutputsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeOutputsClient:
    def __init__(self):
        self.calls = []

    async def list_output_templates(self, **kwargs):
        self.calls.append(("list_output_templates", kwargs))
        return type("Response", (), {"model_dump": lambda self, **kwargs: {"items": [], "total": 0}})()

    async def create_output_template(self, request_data):
        self.calls.append(("create_output_template", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 1, "name": "Newsletter"}

    async def preview_output_template(self, template_id, request_data):
        self.calls.append(("preview_output_template", template_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"rendered": "# Digest", "format": "md"}

    async def list_outputs(self, **kwargs):
        self.calls.append(("list_outputs", kwargs))
        return {"items": [], "total": 0, "page": 1, "size": 50}

    async def create_output(self, request_data):
        self.calls.append(("create_output", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 9, "title": "Digest"}

    async def delete_output(self, output_id, **kwargs):
        self.calls.append(("delete_output", output_id, kwargs))
        return {"success": True, "file_deleted": False}


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


async def _exercise_public_outputs(service, *, limit=50):
    return await service.list_output_templates(limit=limit)


async def _exercise_interop_outputs(service, *, limit=50):
    return await service.list_templates(limit=limit)


OUTPUTS_IMPORT_PATHS = [
    (PublicServerOutputsService, outputs_module, _exercise_public_outputs),
    (ServerOutputsService, outputs_interop_module, _exercise_interop_outputs),
]


def test_server_outputs_service_modules_do_not_reference_legacy_config_client_builders():
    for _, module, _ in OUTPUTS_IMPORT_PATHS:
        source = inspect.getsource(module)

        assert "build_runtime_api_client_from_config" not in source
        assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), OUTPUTS_IMPORT_PATHS)
async def test_server_outputs_service_direct_client_takes_precedence_over_provider(
    service_cls,
    _module,
    exercise,
):
    client = FakeOutputsClient()
    provider = ExplodingClientProvider()
    service = service_cls(client=client, client_provider=provider)

    result = await exercise(service, limit=25)

    assert result["total"] == 0
    assert provider.build_calls == 0
    assert client.calls == [("list_output_templates", {"q": None, "limit": 25, "offset": 0})]


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), OUTPUTS_IMPORT_PATHS)
async def test_server_outputs_service_from_server_context_provider_is_lazy(
    service_cls,
    _module,
    exercise,
):
    client = FakeOutputsClient()
    provider = FakeClientProvider(client)
    service = service_cls.from_server_context_provider(provider)

    assert isinstance(service, service_cls)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await exercise(service, limit=25)

    assert result["total"] == 0
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [("list_output_templates", {"q": None, "limit": 25, "offset": 0})]


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), OUTPUTS_IMPORT_PATHS)
async def test_server_outputs_service_re_resolves_provider_without_service_local_client_cache(
    service_cls,
    _module,
    exercise,
):
    provider = FreshClientProvider(FakeOutputsClient)
    service = service_cls.from_server_context_provider(provider)

    await exercise(service, limit=25)
    await exercise(service, limit=10)

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [("list_output_templates", {"q": None, "limit": 25, "offset": 0})]
    assert provider.clients[1].calls == [("list_output_templates", {"q": None, "limit": 10, "offset": 0})]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


@pytest.mark.parametrize(("service_cls", "_module", "_exercise"), OUTPUTS_IMPORT_PATHS)
def test_server_outputs_service_from_config_returns_provider_backed_service(
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
async def test_server_outputs_service_routes_templates_artifacts_and_render_with_policy_actions():
    client = FakeOutputsClient()
    policy = Mock()
    service = ServerOutputsService(client=client, policy_enforcer=policy)

    templates = await service.list_templates(q="news")
    created_template = await service.create_template(
        name="Newsletter",
        type="newsletter_markdown",
        format="md",
        body="# {{ title }}",
    )
    preview = await service.preview_template(1, data={"title": "Digest"})
    artifacts = await service.list_artifacts(type="newsletter_markdown")
    created_artifact = await service.create_artifact(template_id=1, data={"title": "Digest"})
    deleted_artifact = await service.delete_artifact(9, hard=True)

    assert templates["total"] == 0
    assert created_template["id"] == 1
    assert preview["rendered"] == "# Digest"
    assert artifacts["total"] == 0
    assert created_artifact["id"] == 9
    assert deleted_artifact["success"] is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "outputs.templates.list.server",
        "outputs.templates.create.server",
        "outputs.render_jobs.launch.server",
        "outputs.artifacts.list.server",
        "outputs.artifacts.create.server",
        "outputs.artifacts.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_outputs_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeOutputsClient()
    service = ServerOutputsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_templates()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
