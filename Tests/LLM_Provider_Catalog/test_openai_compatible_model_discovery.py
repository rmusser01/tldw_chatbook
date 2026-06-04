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


@pytest.mark.parametrize("provider", ["anthropic", "google", "cohere", "huggingface", "ollama"])
def test_native_provider_base_urls_do_not_infer_openai_compatibility(provider):
    assert (
        supports_openai_compatible_model_discovery(provider, "https://api.example.test")
        is False
    )


@pytest.mark.parametrize("provider", ["anthropic", "google", "cohere", "huggingface", "ollama"])
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
                "metadata": {
                    "access_token": "secret",
                    "accessToken": "secret",
                    "auth_token": "secret",
                    "client_secret": "secret",
                    "credential": "secret",
                    "id_token": "secret",
                    "privateKey": "secret",
                    "private_key": "secret",
                    "refreshToken": "secret",
                    "refresh_token": "secret",
                    "safe": "visible",
                },
                "variants": [{"token": "secret"}, {"safe": "visible"}],
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

    assert "authorization" not in {
        key.lower() for key in models[0].metadata_raw_safe
    }
    assert "api_key" not in models[0].metadata_raw_safe
    assert "access_token" not in models[0].metadata_raw_safe["metadata"]
    assert "accessToken" not in models[0].metadata_raw_safe["metadata"]
    assert "auth_token" not in models[0].metadata_raw_safe["metadata"]
    assert "client_secret" not in models[0].metadata_raw_safe["metadata"]
    assert "credential" not in models[0].metadata_raw_safe["metadata"]
    assert "id_token" not in models[0].metadata_raw_safe["metadata"]
    assert "privateKey" not in models[0].metadata_raw_safe["metadata"]
    assert "private_key" not in models[0].metadata_raw_safe["metadata"]
    assert "refreshToken" not in models[0].metadata_raw_safe["metadata"]
    assert "refresh_token" not in models[0].metadata_raw_safe["metadata"]
    assert models[0].metadata_raw_safe["metadata"]["safe"] == "visible"
    assert "token" not in models[0].metadata_raw_safe["variants"][0]
    assert models[0].metadata_raw_safe["variants"][1]["safe"] == "visible"


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
