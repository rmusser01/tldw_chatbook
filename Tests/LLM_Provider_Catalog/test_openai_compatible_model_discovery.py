from __future__ import annotations

import httpx
import pytest

from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    build_models_url,
    discover_openai_compatible_models,
    fingerprint_endpoint,
    normalize_models_response,
    supports_openai_compatible_model_discovery,
)
from tldw_chatbook.LLM_Provider_Catalog import openai_compatible_model_discovery


def test_chat_completions_url_maps_to_models_url():
    assert (
        build_models_url("https://api.example.test/v1/chat/completions", "custom")
        == "https://api.example.test/v1/models"
    )


def test_llamacpp_completion_url_maps_to_v1_models():
    assert (
        build_models_url("http://127.0.0.1:9099/completion", "llama_cpp")
        == "http://127.0.0.1:9099/v1/models"
    )


def test_llamacpp_completion_url_drops_userinfo():
    assert (
        build_models_url("https://user:secret@example.test/completion", "llama_cpp")
        == "https://example.test/v1/models"
    )


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("localhost:8000", "http://localhost:8000/v1/models"),
        ("https://api.example.test", "https://api.example.test/v1/models"),
        (
            "https://api.example.test/api/v1/chat/completions",
            "https://api.example.test/api/v1/models",
        ),
        (
            "https://api.example.test/openai/v1/chat/completions",
            "https://api.example.test/openai/v1/models",
        ),
    ],
)
def test_common_base_and_prefixed_openai_paths_map_to_models(endpoint, expected):
    assert supports_openai_compatible_model_discovery("custom", endpoint) is True
    assert build_models_url(endpoint, "custom") == expected


def test_space_separated_provider_label_can_infer_openai_compatible_base_url():
    assert (
        supports_openai_compatible_model_discovery("Local LLM", "http://127.0.0.1:8000")
        is True
    )


@pytest.mark.parametrize(
    "provider", ["anthropic", "google", "cohere", "huggingface", "ollama"]
)
def test_native_provider_base_urls_do_not_infer_openai_compatibility(provider):
    assert (
        supports_openai_compatible_model_discovery(provider, "https://api.example.test")
        is False
    )


@pytest.mark.parametrize(
    "provider", ["anthropic", "google", "cohere", "huggingface", "ollama"]
)
def test_native_provider_explicit_openai_compatible_paths_are_eligible(provider):
    assert (
        supports_openai_compatible_model_discovery(
            provider,
            "https://api.example.test/v1",
        )
        is True
    )


def test_malformed_endpoint_port_is_not_eligible_and_does_not_crash():
    endpoint = "https://user:secret@example.test:bad/v1/models"

    assert supports_openai_compatible_model_discovery("custom", endpoint) is False
    assert fingerprint_endpoint(endpoint) == "https://example.test/v1/models"


def test_native_kobold_generate_is_not_eligible():
    assert (
        supports_openai_compatible_model_discovery(
            "koboldcpp",
            "http://127.0.0.1:5001/api/v1/generate",
        )
        is False
    )


def test_kobold_with_explicit_v1_endpoint_is_eligible():
    assert (
        supports_openai_compatible_model_discovery(
            "koboldcpp",
            "http://127.0.0.1:5001/v1",
        )
        is True
    )


def test_ollama_native_tags_is_not_eligible_in_v1():
    assert (
        supports_openai_compatible_model_discovery(
            "ollama",
            "http://127.0.0.1:11434/api/tags",
        )
        is False
    )


def test_normalizes_openai_models_response_to_raw_model_ids():
    payload = {"data": [{"id": "gpt-4.1"}, {"id": "gpt-4.1-mini"}]}

    models = normalize_models_response(
        payload,
        provider="OpenAI",
        provider_list_key="OpenAI",
        endpoint_fingerprint="abc",
        now_iso="2026-06-04T12:00:00Z",
    )

    assert [model.model_id for model in models] == ["gpt-4.1", "gpt-4.1-mini"]
    assert models[0].source == "runtime_discovered"


def test_response_metadata_does_not_include_sensitive_headers():
    payload = {
        "data": [
            {
                "id": "model-a",
                "owned_by": "org",
                "api_key": "secret",
                "sessionToken": "secret",
                "metadata": {
                    "access_token": "secret",
                    "accessToken": "secret",
                    "auth_token": "secret",
                    "bearerToken": "secret",
                    "client_secret": "secret",
                    "credential": "secret",
                    "id_token": "secret",
                    "privateKey": "secret",
                    "private_key": "secret",
                    "refreshToken": "secret",
                    "refresh_token": "secret",
                    "session_token": "secret",
                    "safe": "visible",
                },
                "variants": [
                    {"token": "secret"},
                    {"bearer_token": "secret"},
                    {"safe": "visible"},
                ],
            }
        ]
    }

    models = normalize_models_response(
        payload,
        provider="Custom",
        provider_list_key="Custom",
        endpoint_fingerprint="abc",
        now_iso="2026-06-04T12:00:00Z",
    )

    assert "authorization" not in {key.lower() for key in models[0].metadata_raw_safe}
    assert "api_key" not in models[0].metadata_raw_safe
    assert "sessionToken" not in models[0].metadata_raw_safe
    assert "access_token" not in models[0].metadata_raw_safe["metadata"]
    assert "accessToken" not in models[0].metadata_raw_safe["metadata"]
    assert "auth_token" not in models[0].metadata_raw_safe["metadata"]
    assert "bearerToken" not in models[0].metadata_raw_safe["metadata"]
    assert "client_secret" not in models[0].metadata_raw_safe["metadata"]
    assert "credential" not in models[0].metadata_raw_safe["metadata"]
    assert "id_token" not in models[0].metadata_raw_safe["metadata"]
    assert "privateKey" not in models[0].metadata_raw_safe["metadata"]
    assert "private_key" not in models[0].metadata_raw_safe["metadata"]
    assert "refreshToken" not in models[0].metadata_raw_safe["metadata"]
    assert "refresh_token" not in models[0].metadata_raw_safe["metadata"]
    assert "session_token" not in models[0].metadata_raw_safe["metadata"]
    assert models[0].metadata_raw_safe["metadata"]["safe"] == "visible"
    assert "token" not in models[0].metadata_raw_safe["variants"][0]
    assert "bearer_token" not in models[0].metadata_raw_safe["variants"][1]
    assert models[0].metadata_raw_safe["variants"][2]["safe"] == "visible"


def test_duplicate_model_ids_preserve_first_occurrence():
    payload = {
        "data": [
            {"id": "model-a", "owned_by": "first"},
            {"id": "model-b", "owned_by": "second"},
            {"id": "model-a", "owned_by": "duplicate"},
        ]
    }

    models = normalize_models_response(
        payload,
        provider="Custom",
        provider_list_key="Custom",
        endpoint_fingerprint="abc",
        now_iso="2026-06-04T12:00:00Z",
    )

    assert [model.model_id for model in models] == ["model-a", "model-b"]
    assert models[0].metadata_raw_safe["owned_by"] == "first"


@pytest.mark.parametrize(
    "payload",
    [
        {"data": "not-a-list"},
        {"data": [{"object": "model"}]},
        {"data": [{"id": ""}]},
    ],
)
def test_invalid_models_response_raises_value_error(payload):
    with pytest.raises(ValueError, match="Invalid models response"):
        normalize_models_response(
            payload,
            provider="Custom",
            provider_list_key="Custom",
            endpoint_fingerprint="abc",
            now_iso="2026-06-04T12:00:00Z",
        )


def test_endpoint_fingerprint_redacts_credentials_and_query_params():
    fingerprint = fingerprint_endpoint(
        "https://user:secret@example.test:8443/v1/models?api_key=secret&debug=1"
    )

    assert fingerprint == "https://example.test:8443/v1/models"
    assert "secret" not in fingerprint
    assert "api_key" not in fingerprint


def test_endpoint_fingerprint_redacts_credentials_for_unsupported_scheme():
    fingerprint = fingerprint_endpoint(
        "ftp://user:secret@example.test/v1/models?api_key=secret"
    )

    assert fingerprint == "ftp://example.test/v1/models"
    assert "secret" not in fingerprint
    assert "api_key" not in fingerprint


def test_endpoint_fingerprint_redacts_credentials_for_host_only_malformed_port():
    fingerprint = fingerprint_endpoint(
        "user:secret@example.test:bad/v1/models?api_key=secret"
    )

    assert fingerprint == "http://example.test/v1/models"
    assert "secret" not in fingerprint
    assert "api_key" not in fingerprint


def test_endpoint_fingerprint_redacts_credentials_for_unparseable_authority():
    fingerprint = fingerprint_endpoint(
        "https://user:secret@[::1/v1/models?api_key=secret"
    )

    assert fingerprint == "https://[invalid-endpoint]"
    assert "user" not in fingerprint
    assert "secret" not in fingerprint
    assert "api_key" not in fingerprint


def test_endpoint_with_userinfo_but_no_host_is_not_eligible_or_leaky():
    endpoint = "https://user:secret@/v1"

    assert supports_openai_compatible_model_discovery("custom", endpoint) is False
    assert fingerprint_endpoint(endpoint) == "https://[invalid-endpoint]"


@pytest.mark.asyncio
async def test_malformed_endpoint_reports_distinct_error_kind():
    """TASK-367: a malformed URL (e.g. a dropped 'h' → invalid scheme) must be
    reported as a distinct 'malformed_endpoint' error, not misdiagnosed as a
    missing-/v1-path 'unsupported_endpoint' problem."""
    malformed = await discover_openai_compatible_models(
        provider="custom",
        provider_list_key="custom",
        endpoint="ttp://127.0.0.1:9099/v1",
        api_key=None,
    )
    assert malformed.status == "unsupported"
    assert malformed.error.kind == "malformed_endpoint"

    # A valid URL whose PATH is not OpenAI-compatible stays unsupported_endpoint.
    unsupported = await discover_openai_compatible_models(
        provider="koboldcpp",
        provider_list_key="koboldcpp",
        endpoint="http://127.0.0.1:5001/api/v1/generate",
        api_key=None,
    )
    assert unsupported.status == "unsupported"
    assert unsupported.error.kind == "unsupported_endpoint"


@pytest.mark.asyncio
async def test_discovers_models_with_authorization_header_when_api_key_present():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"id": "runtime-a"}, {"id": "runtime-b"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_openai_compatible_models(
            provider="Custom",
            provider_list_key="Custom",
            endpoint="https://api.example.test/v1/chat/completions",
            api_key="test-key",
            client=client,
        )

    assert result.status == "success"
    assert [model.model_id for model in result.models] == ["runtime-a", "runtime-b"]
    assert str(requests[0].url) == "https://api.example.test/v1/models"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert "test-key" not in (result.endpoint_fingerprint or "")


@pytest.mark.asyncio
async def test_discovery_owned_async_client_uses_context_manager(monkeypatch):
    events: list[str] = []

    class FakeAsyncClient:
        def __init__(self, *, timeout):
            events.append(f"init:{timeout}")

        async def __aenter__(self):
            events.append("enter")
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            events.append("exit")

        async def get(self, url, headers=None, params=None):
            events.append(f"get:{url}")
            return httpx.Response(
                200,
                json={"data": [{"id": "runtime-a"}]},
                request=httpx.Request("GET", url),
            )

        async def aclose(self):
            events.append("manual-close")

    monkeypatch.setattr(
        openai_compatible_model_discovery.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    result = await discover_openai_compatible_models(
        provider="Custom",
        provider_list_key="Custom",
        endpoint="https://api.example.test/v1",
        api_key=None,
    )

    assert result.status == "success"
    assert events == [
        "init:10.0",
        "enter",
        "get:https://api.example.test/v1/models",
        "exit",
    ]


@pytest.mark.asyncio
async def test_discovery_returns_typed_error_for_invalid_response():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": "invalid"})
        )
    ) as client:
        result = await discover_openai_compatible_models(
            provider="Custom",
            provider_list_key="Custom",
            endpoint="https://api.example.test/v1",
            api_key=None,
            client=client,
        )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "invalid_response"


@pytest.mark.asyncio
async def test_discovery_returns_typed_error_for_non_json_response():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text="not-json")
        )
    ) as client:
        result = await discover_openai_compatible_models(
            provider="Custom",
            provider_list_key="Custom",
            endpoint="https://api.example.test/v1",
            api_key=None,
            client=client,
        )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "invalid_response"


@pytest.mark.asyncio
async def test_discovery_returns_typed_error_for_request_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await discover_openai_compatible_models(
            provider="Custom",
            provider_list_key="Custom",
            endpoint="https://api.example.test/v1",
            api_key=None,
            client=client,
        )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "request_failed"


@pytest.mark.asyncio
async def test_discovery_returns_typed_error_for_unsupported_endpoint():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        result = await discover_openai_compatible_models(
            provider="ollama",
            provider_list_key="Ollama",
            endpoint="http://127.0.0.1:11434/api/tags",
            api_key=None,
            client=client,
        )

    assert result.status == "unsupported"
    assert result.error is not None
    assert result.error.kind == "unsupported_endpoint"


def test_openrouter_api_v1_path_maps_to_models():
    assert supports_openai_compatible_model_discovery(
        "openrouter", "https://openrouter.ai/api/v1"
    ) is True
    assert (
        build_models_url("https://openrouter.ai/api/v1", "openrouter")
        == "https://openrouter.ai/api/v1/models"
    )


def test_zai_paas_v4_path_maps_to_models():
    assert supports_openai_compatible_model_discovery(
        "zai", "https://api.z.ai/api/paas/v4"
    ) is True
    assert (
        build_models_url("https://api.z.ai/api/paas/v4", "zai")
        == "https://api.z.ai/api/paas/v4/models"
    )


def test_anthropic_uses_x_api_key_headers():
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        build_discovery_auth_headers,
    )

    headers = build_discovery_auth_headers("anthropic", "sk-ant-test")
    assert headers == {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    assert build_discovery_auth_headers("openai", "sk-test") == {
        "Authorization": "Bearer sk-test"
    }
    assert build_discovery_auth_headers("anthropic", None) == {}


@pytest.mark.asyncio
async def test_anthropic_paginates_with_after_id():
    requests: list[dict] = []
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(dict(request.url.params))
        seen_headers.update({k.lower(): v for k, v in request.headers.items()})
        page = len(requests)
        payload = (
            {"data": [{"id": f"claude-{page}"}], "has_more": True, "last_id": f"claude-{page}"}
            if page == 1
            else {"data": [{"id": "claude-2"}], "has_more": False, "last_id": "claude-2"}
        )
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await discover_openai_compatible_models(
            provider="anthropic",
            provider_list_key="Anthropic",
            endpoint="https://api.anthropic.com/v1",
            api_key="sk-ant-test",
            client=client,
        )
    assert result.status == "success"
    assert [m.model_id for m in result.models] == ["claude-1", "claude-2"]
    assert requests[0] == {"limit": "1000"}
    assert requests[1] == {"limit": "1000", "after_id": "claude-1"}
    # Anthropic auth headers, not Bearer:
    assert seen_headers["x-api-key"] == "sk-ant-test"
    assert seen_headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in seen_headers


@pytest.mark.asyncio
async def test_openai_does_not_paginate():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert "limit" not in dict(request.url.params)
        return httpx.Response(200, json={"data": [{"id": "gpt-x"}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await discover_openai_compatible_models(
            provider="openai",
            provider_list_key="OpenAI",
            endpoint="https://api.openai.com/v1",
            api_key="sk-test",
            client=client,
        )
    assert result.status == "success"
    assert calls == 1


@pytest.mark.asyncio
async def test_discovery_maps_401_to_missing_credentials():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(401))
    ) as client:
        result = await discover_openai_compatible_models(
            provider="openai",
            provider_list_key="OpenAI",
            endpoint="https://api.openai.com/v1",
            api_key="sk-bad",
            client=client,
        )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "missing_credentials"


@pytest.mark.asyncio
async def test_discovery_maps_500_to_request_failed():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500))
    ) as client:
        result = await discover_openai_compatible_models(
            provider="openai",
            provider_list_key="OpenAI",
            endpoint="https://api.openai.com/v1",
            api_key="sk-test",
            client=client,
        )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.kind == "request_failed"
