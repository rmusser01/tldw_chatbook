import pytest

from tldw_chatbook.Translation_Interop.server_translation_service import ServerTranslationService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeTranslationClient:
    def __init__(self):
        self.calls = []

    async def translate_text(self, request_data):
        self.calls.append(("translate_text", request_data))
        return {
            "translated_text": "Bonjour",
            "detected_source_language": request_data.source_language,
            "target_language": request_data.target_language,
            "model_used": request_data.model or "server-default",
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
async def test_server_translation_service_launches_text_translation_with_policy_and_normalization():
    client = FakeTranslationClient()
    policy = FakePolicyEnforcer()
    service = ServerTranslationService(client, policy_enforcer=policy)

    result = await service.translate_text(
        {
            "text": "Hello",
            "target_language": "French",
            "source_language": "English",
            "model": "fast-model",
        }
    )

    assert result == {
        "translated_text": "Bonjour",
        "detected_source_language": "English",
        "target_language": "French",
        "model_used": "fast-model",
        "backend": "server",
        "record_id": "server:translation:text",
    }
    assert client.calls[0][0] == "translate_text"
    assert client.calls[0][1].text == "Hello"
    assert client.calls[0][1].target_language == "French"
    assert policy.calls == ["translation.text.launch.server"]


@pytest.mark.asyncio
async def test_server_translation_service_denies_before_dispatch():
    client = FakeTranslationClient()
    service = ServerTranslationService(client, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await service.translate_text({"text": "Hello", "target_language": "French"})

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


def test_server_translation_service_direct_client_takes_precedence_over_provider():
    client = object()
    provider = ExplodingClientProvider()
    service = ServerTranslationService(client=client, client_provider=provider)

    assert service._require_client() is client
    assert provider.build_calls == 0


def test_server_translation_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    service = ServerTranslationService.from_server_context_provider(provider)

    assert isinstance(service, ServerTranslationService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1


def test_server_translation_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider()
    service = ServerTranslationService.from_server_context_provider(provider)

    first_client = service._require_client()
    second_client = service._require_client()

    assert service.client is None
    assert provider.build_calls == 2
    assert first_client is not second_client
    assert provider.clients == [first_client, second_client]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_translation_service_from_config_returns_provider_backed_service():
    service = ServerTranslationService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerTranslationService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client
