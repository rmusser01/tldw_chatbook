import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.WebClipper.server_web_clipper_service as web_clipper_module
import tldw_chatbook.Web_Clipper_Interop.server_web_clipper_service as web_clipper_interop_module
from tldw_chatbook.WebClipper import ServerWebClipperService as PublicServerWebClipperService
from tldw_chatbook.Web_Clipper_Interop import ServerWebClipperService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeWebClipperClient:
    def __init__(self):
        self.calls = []

    async def save_web_clip(self, request_data):
        self.calls.append(("save_web_clip", request_data.model_dump(exclude_none=True, mode="json")))
        return {"clip_id": "clip-1", "note_id": "note-1", "status": "saved"}

    async def get_web_clip_status(self, clip_id):
        self.calls.append(("get_web_clip_status", clip_id))
        return {"clip_id": clip_id, "status": "saved"}

    async def persist_web_clip_enrichment(self, clip_id, request_data):
        self.calls.append(("persist_web_clip_enrichment", clip_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"clip_id": clip_id, "status": "complete"}


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


async def _exercise_public_web_clipper(service, clip_id="clip-1"):
    return await service.get_clip_status(clip_id)


async def _exercise_interop_web_clipper(service, clip_id="clip-1"):
    return await service.get_status(clip_id)


WEB_CLIPPER_IMPORT_PATHS = [
    (PublicServerWebClipperService, web_clipper_module, _exercise_public_web_clipper),
    (ServerWebClipperService, web_clipper_interop_module, _exercise_interop_web_clipper),
]


def test_server_web_clipper_service_modules_do_not_reference_legacy_config_client_builders():
    for _, module, _ in WEB_CLIPPER_IMPORT_PATHS:
        source = inspect.getsource(module)

        assert "build_runtime_api_client_from_config" not in source
        assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), WEB_CLIPPER_IMPORT_PATHS)
async def test_server_web_clipper_service_direct_client_takes_precedence_over_provider(
    service_cls,
    _module,
    exercise,
):
    client = FakeWebClipperClient()
    provider = ExplodingClientProvider()
    service = service_cls(client=client, client_provider=provider)

    result = await exercise(service)

    assert result["status"] == "saved"
    assert provider.build_calls == 0
    assert client.calls == [("get_web_clip_status", "clip-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), WEB_CLIPPER_IMPORT_PATHS)
async def test_server_web_clipper_service_from_server_context_provider_is_lazy(
    service_cls,
    _module,
    exercise,
):
    client = FakeWebClipperClient()
    provider = FakeClientProvider(client)
    service = service_cls.from_server_context_provider(provider)

    assert isinstance(service, service_cls)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await exercise(service)

    assert result["status"] == "saved"
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [("get_web_clip_status", "clip-1")]


@pytest.mark.asyncio
@pytest.mark.parametrize(("service_cls", "_module", "exercise"), WEB_CLIPPER_IMPORT_PATHS)
async def test_server_web_clipper_service_re_resolves_provider_without_service_local_client_cache(
    service_cls,
    _module,
    exercise,
):
    provider = FreshClientProvider(FakeWebClipperClient)
    service = service_cls.from_server_context_provider(provider)

    await exercise(service, "clip-1")
    await exercise(service, "clip-2")

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [("get_web_clip_status", "clip-1")]
    assert provider.clients[1].calls == [("get_web_clip_status", "clip-2")]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


@pytest.mark.parametrize(("service_cls", "_module", "_exercise"), WEB_CLIPPER_IMPORT_PATHS)
def test_server_web_clipper_service_from_config_returns_provider_backed_service(
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
async def test_server_web_clipper_service_routes_with_policy_actions():
    client = FakeWebClipperClient()
    policy = Mock()
    service = ServerWebClipperService(client=client, policy_enforcer=policy)

    saved = await service.save_clip(
        clip_id="clip-1",
        clip_type="article",
        source_url="https://example.com",
        source_title="Example",
        content={"visible_body": "Body"},
    )
    status = await service.get_status("clip-1")
    enrichment = await service.persist_enrichment(
        clip_id="clip-1",
        enrichment_type="ocr",
        status="complete",
        source_note_version=1,
    )

    assert saved["note_id"] == "note-1"
    assert status["clip_id"] == "clip-1"
    assert enrichment["status"] == "complete"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "web_clipper.capture.server",
        "web_clipper.status.server",
        "web_clipper.capture.server",
    ]


@pytest.mark.asyncio
async def test_server_web_clipper_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeWebClipperClient()
    service = ServerWebClipperService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_status("clip-1")

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
