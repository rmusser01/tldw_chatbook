from unittest.mock import Mock

import pytest

from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.Web_Scraping_Interop import ServerWebScrapingService


class FakeWebScrapingClient:
    def __init__(self):
        self.calls = []

    async def get_web_scraping_status(self):
        self.calls.append(("get_web_scraping_status",))
        return {"initialized": True}

    async def get_web_scraping_job_status(self, job_id):
        self.calls.append(("get_web_scraping_job_status", job_id))
        return {"job_id": job_id, "status": "running"}

    async def cancel_web_scraping_job(self, job_id):
        self.calls.append(("cancel_web_scraping_job", job_id))
        return {"job_id": job_id, "status": "cancelled"}

    async def get_web_scraping_progress(self, task_id):
        self.calls.append(("get_web_scraping_progress", task_id))
        return {"task_id": task_id, "status": "in_progress"}

    async def get_web_scraping_cookies(self, domain):
        self.calls.append(("get_web_scraping_cookies", domain))
        return {"domain": domain, "cookies": []}

    async def set_web_scraping_cookies(self, domain, cookies):
        self.calls.append(("set_web_scraping_cookies", domain, cookies))
        return {"domain": domain, "status": "success"}

    async def check_web_scraping_duplicate(self, url):
        self.calls.append(("check_web_scraping_duplicate", url))
        return {"url": url, "is_duplicate": False}


@pytest.mark.asyncio
async def test_server_web_scraping_service_routes_with_policy_actions():
    client = FakeWebScrapingClient()
    policy = Mock()
    service = ServerWebScrapingService(client=client, policy_enforcer=policy)

    status = await service.get_status()
    job = await service.get_job_status("job-1")
    cancelled = await service.cancel_job("job-1")
    progress = await service.get_progress("task-1")
    cookies = await service.get_cookies("example.com")
    set_cookies = await service.set_cookies("example.com", [{"name": "session"}])
    duplicate = await service.check_duplicate("https://example.com")

    assert status["record_id"] == "server:web_scraping:status"
    assert job["record_id"] == "server:web_scraping_job:job-1"
    assert cancelled["record_id"] == "server:web_scraping_job:job-1"
    assert progress["record_id"] == "server:web_scraping_progress:task-1"
    assert cookies["record_id"] == "server:web_scraping_cookies:example.com"
    assert set_cookies["record_id"] == "server:web_scraping_cookies:example.com"
    assert duplicate["record_id"] == "server:web_scraping_duplicate:https://example.com"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "media.web_scraping.status.server",
        "media.web_scraping.detail.server",
        "media.web_scraping.cancel.server",
        "media.web_scraping.observe.server",
        "media.web_scraping.cookies.detail.server",
        "media.web_scraping.cookies.update.server",
        "media.web_scraping.inspect.server",
    ]


@pytest.mark.asyncio
async def test_server_web_scraping_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeWebScrapingClient()
    service = ServerWebScrapingService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.get_status()

    assert exc.value.reason_code == "server_unreachable"
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


def test_server_web_scraping_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerWebScrapingService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_web_scraping_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerWebScrapingService.from_server_context_provider(provider)

    assert isinstance(service, ServerWebScrapingService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_web_scraping_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerWebScrapingService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_web_scraping_service_from_config_returns_provider_backed_service():
    service = ServerWebScrapingService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerWebScrapingService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
