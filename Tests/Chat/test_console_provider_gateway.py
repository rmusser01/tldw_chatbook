import asyncio
import builtins
import http.server
import json
import threading

import httpx
import pytest

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import (
    GENERATION_READ_TIMEOUT_SECONDS,
    PROBE_TIMEOUT_SECONDS,
    ConsoleProviderGateway,
    ConsoleProviderResolution,
    LlamaCppProviderConfig,
    build_llamacpp_chat_payload,
    safe_provider_error_copy,
)
from tldw_chatbook.Chat.console_provider_support import resolve_console_provider_identity


def test_llamacpp_payload_includes_supported_sampling_params() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        temperature=0.4,
        top_p=0.7,
        min_p=0.03,
        top_k=20,
        max_tokens=300,
    )

    assert payload == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "temperature": 0.4,
        "top_p": 0.7,
        "min_p": 0.03,
        "top_k": 20,
        "max_tokens": 300,
    }


def test_llamacpp_payload_omits_blank_provider_defaults() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[],
        stream=False,
        temperature=None,
        top_p=None,
        min_p=None,
        top_k=None,
        max_tokens=None,
    )

    assert payload == {"model": "m", "messages": [], "stream": False}


def test_llamacpp_payload_includes_explicit_top_k_zero() -> None:
    payload = build_llamacpp_chat_payload(
        model="m",
        messages=[],
        stream=False,
        top_k=0,
    )

    assert payload == {"model": "m", "messages": [], "stream": False, "top_k": 0}


@pytest.mark.asyncio
async def test_llamacpp_prefers_explicit_model_but_still_probes_reachability():
    seen_paths = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", explicit_model="explicit-model")
    )

    assert resolved.ready is True
    assert resolved.model == "explicit-model"
    assert seen_paths == ["/health"]


@pytest.mark.asyncio
async def test_llamacpp_prefers_configured_model_but_still_probes_reachability():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(404, text="no health route, but server is reachable")

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", configured_model="configured-model")
    )

    assert resolved.ready is True
    assert resolved.model == "configured-model"


@pytest.mark.asyncio
async def test_llamacpp_explicit_model_blocks_when_reachability_probe_cannot_connect():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        raise httpx.ConnectError("connection refused", request=request)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(
        LlamaCppProviderConfig(base_url="http://127.0.0.1:9099", explicit_model="explicit-model")
    )

    assert resolved.ready is False
    assert resolved.model == "explicit-model"
    assert "not reachable" in resolved.visible_copy


@pytest.mark.asyncio
async def test_llamacpp_uses_first_models_endpoint_result_when_no_configured_model():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is True
    assert resolved.model == "server-model"


@pytest.mark.asyncio
async def test_llamacpp_unreachable_server_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is False
    assert resolved.model is None
    assert "Provider blocked" in resolved.visible_copy
    assert "127.0.0.1:9099" in resolved.visible_copy


@pytest.mark.asyncio
async def test_llamacpp_empty_models_without_configured_model_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is False
    assert resolved.model is None
    assert resolved.visible_copy == "Provider blocked: select or configure a llama.cpp model."


@pytest.mark.asyncio
async def test_llamacpp_non_object_models_payload_returns_blocked_recovery_copy():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_llamacpp(LlamaCppProviderConfig(base_url="http://127.0.0.1:9099"))

    assert resolved.ready is False
    assert resolved.model is None
    assert resolved.visible_copy == "Provider blocked: select or configure a llama.cpp model."


@pytest.mark.asyncio
async def test_resolve_for_send_dispatches_llamacpp_selection():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="http://127.0.0.1:9099")
    )

    assert resolved.ready is True
    assert resolved.provider == "llama_cpp"
    assert resolved.model == "server-model"


@pytest.mark.asyncio
async def test_resolve_for_send_copies_sampling_fields_to_llamacpp_resolution():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            temperature=0.4,
            top_p=0.8,
            min_p=0.05,
            top_k=30,
            max_tokens=500,
            seed=11,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            reasoning_effort="high",
            reasoning_summary="auto",
            verbosity="medium",
            thinking_effort="low",
            thinking_budget_tokens=2048,
            streaming=False,
        )
    )

    assert resolved.ready is True
    assert resolved.temperature == 0.4
    assert resolved.top_p == 0.8
    assert resolved.min_p == 0.05
    assert resolved.top_k == 30
    assert resolved.max_tokens == 500
    assert resolved.seed == 11
    assert resolved.presence_penalty == 0.2
    assert resolved.frequency_penalty == 0.3
    assert resolved.reasoning_effort == "high"
    assert resolved.reasoning_summary == "auto"
    assert resolved.verbosity == "medium"
    assert resolved.thinking_effort == "low"
    assert resolved.thinking_budget_tokens == 2048
    assert resolved.streaming is False


@pytest.mark.asyncio
async def test_resolve_for_send_normalizes_scheme_less_llamacpp_base_url_before_http():
    seen_urls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="127.0.0.1:9099/v1")
    )

    assert resolved.ready is True
    assert resolved.base_url == "http://127.0.0.1:9099"
    assert seen_urls == ["http://127.0.0.1:9099/v1/models"]


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_invalid_llamacpp_base_url_before_http():
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": "server-model"}]})

    gateway = ConsoleProviderGateway(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="file:///etc/passwd")
    )

    assert resolved.ready is False
    assert resolved.base_url == "file:///etc/passwd"
    assert "invalid llama.cpp base URL" in resolved.visible_copy
    assert requests == []


@pytest.mark.asyncio
async def test_gateway_resolves_direct_llamacpp_without_importing_chat_functions(monkeypatch):
    real_import = builtins.__import__

    def fail_chat_functions_import(name, *args, **kwargs):
        if name == "tldw_chatbook.Chat.Chat_Functions":
            raise AssertionError("direct llama resolution should not import Chat_Functions")
        return real_import(name, *args, **kwargs)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(builtins, "__import__", fail_chat_functions_import)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = ConsoleProviderGateway(http_client=client)

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="llama_cpp", base_url="http://127.0.0.1:9099", explicit_model="m")
    )

    assert resolved.ready is True
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_unsupported_provider_with_recovery_copy():
    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="future_provider",
            temperature=0.3,
            top_p=0.9,
            min_p=0.02,
            top_k=40,
            max_tokens=600,
            streaming=False,
        )
    )

    assert resolved.ready is False
    assert resolved.provider == "future_provider"
    assert resolved.temperature == 0.3
    assert resolved.top_p == 0.9
    assert resolved.min_p == 0.02
    assert resolved.top_k == 40
    assert resolved.max_tokens == 600
    assert resolved.streaming is False
    assert resolved.visible_copy == (
        "Provider blocked: 'future_provider' is not available in Console yet. Choose a supported provider."
    )


@pytest.mark.asyncio
async def test_resolve_for_send_openai_uses_env_key_and_execution_key() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={"OPENAI_API_KEY": "sk-test-secret"},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1", streaming=False)
    )

    assert resolved.ready is True
    assert resolved.provider == "openai"
    assert resolved.readiness_key == "openai"
    assert resolved.execution_key == "openai"
    assert resolved.api_key == "sk-test-secret"
    assert resolved.api_key_source == "env:OPENAI_API_KEY"
    assert "sk-test-secret" not in resolved.visible_copy
    assert "sk-test-secret" not in repr(resolved)


@pytest.mark.asyncio
async def test_resolve_for_send_all_chat_api_handlers_are_console_supported() -> None:
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.Chat.provider_readiness import PROVIDERS_REQUIRING_API_KEY_KEYS

    handler_keys = frozenset(API_CALL_HANDLERS)
    api_settings: dict[str, dict[str, str]] = {}
    for provider in handler_keys:
        identity = resolve_console_provider_identity(
            provider,
            handler_keys=handler_keys,
        )
        settings = api_settings.setdefault(
            identity.readiness_key,
            {"model": f"{identity.readiness_key}-model"},
        )
        if identity.readiness_key in PROVIDERS_REQUIRING_API_KEY_KEYS:
            settings["api_key"] = f"test-key-for-{identity.readiness_key}"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"}))
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            config_provider=lambda: {"api_settings": api_settings},
            environ={},
        )

        for provider in sorted(handler_keys):
            identity = resolve_console_provider_identity(
                provider,
                handler_keys=handler_keys,
            )
            resolved = await gateway.resolve_for_send(
                ConsoleProviderSelection(provider=provider, explicit_model="console-sweep-model")
            )

            assert resolved.ready is True, provider
            assert resolved.readiness_key == identity.readiness_key, provider
            assert resolved.execution_key == identity.execution_key, provider
            assert "not available in Console yet" not in resolved.visible_copy
            assert "WIP" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_supported_provider_missing_key_blocks_without_wip() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"anthropic": {"api_key_env_var": "ANTHROPIC_API_KEY"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="anthropic", explicit_model="claude-sonnet")
    )

    assert resolved.ready is False
    assert "Missing API key" in resolved.visible_copy
    assert "not wired" not in resolved.visible_copy
    assert "WIP" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_custom_alias_uses_custom_openai_execution_key() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"custom": {"model": "m"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(ConsoleProviderSelection(provider="Custom", configured_model="m"))

    assert resolved.ready is True
    assert resolved.readiness_key == "custom"
    assert resolved.execution_key == "custom-openai-api"


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_generic_base_url_override_that_differs_from_config() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="ollama",
            explicit_model="llama3",
            base_url="http://user:secret@127.0.0.1:9999/v1",
        )
    )

    assert resolved.ready is False
    assert "save the endpoint in Settings" in resolved.visible_copy
    assert "Selected endpoint: http://127.0.0.1:9999/v1" in resolved.visible_copy
    assert "Saved endpoint: http://127.0.0.1:11434" in resolved.visible_copy
    assert "user" not in resolved.visible_copy
    assert "secret" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_ignores_cloud_session_base_url_without_configured_endpoint() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {"openai": {"api_key": "unit-test-key", "model": "gpt-4.1"}},
        },
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-4.1",
            base_url="http://127.0.0.1:9999/v1",
        )
    )

    assert resolved.ready is True
    assert resolved.readiness_key == "openai"
    assert resolved.execution_key == "openai"
    assert "save the endpoint in Settings" not in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_accepts_generic_base_url_matching_config_with_trailing_slash() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="ollama", explicit_model="llama3", base_url="http://127.0.0.1:11434/")
    )

    assert resolved.ready is True
    assert resolved.base_url == "http://127.0.0.1:11434/"
    assert resolved.model == "llama3"


@pytest.mark.asyncio
async def test_resolve_for_send_accepts_generic_base_url_matching_default_port() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"ollama": {"api_url": "http://example.test"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="ollama", explicit_model="llama3", base_url="http://example.test:80/")
    )

    assert resolved.ready is True


@pytest.mark.asyncio
async def test_resolve_for_send_blocks_malformed_generic_base_url_without_crashing() -> None:
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    resolved = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="ollama", explicit_model="llama3", base_url="http://[::1")
    )

    assert resolved.ready is False
    assert "save the endpoint in Settings" in resolved.visible_copy


@pytest.mark.asyncio
async def test_resolve_for_send_reads_config_provider_at_resolution_time() -> None:
    configs = [
        {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "old-model"}}},
        {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY", "model": "new-model"}}},
    ]

    def config_provider() -> dict[str, object]:
        return configs.pop(0)

    gateway = ConsoleProviderGateway(
        config_provider=config_provider,
        environ={"OPENAI_API_KEY": "sk-test-secret"},
    )

    first = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai"))
    second = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai"))

    assert first.ready is True
    assert first.model == "old-model"
    assert second.ready is True
    assert second.model == "new-model"
    assert configs == []


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_yields_content_chunks():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.method == "POST"
        body = (
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_falls_back_to_non_streaming_when_stream_rejected():
    request_payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(request.read())
        if len(request_payloads) == 1:
            return httpx.Response(400, json={"error": "streaming disabled"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback completion"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["fallback completion"]
    assert b'"stream":true' in request_payloads[0]
    assert b'"stream":false' in request_payloads[1]


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_falls_back_when_sse_has_no_content_chunks():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, content=b"data: {not-json}\n\ndata: [DONE]\n\n")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback after bad sse"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["fallback after bad sse"]
    assert calls == 2


@pytest.mark.asyncio
async def test_llamacpp_stream_chat_ignores_non_object_json_sse_lines():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"data: []\n\n"
            b"data: null\n\n"
            b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body)

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )

    chunks = [
        chunk
        async for chunk in gateway.stream_llamacpp_chat(
            base_url="http://127.0.0.1:9099",
            model="test-model",
            messages=[{"role": "user", "content": "say hello"}],
        )
    ]

    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_stream_chat_dispatches_llamacpp_resolution():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            content=b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="http://127.0.0.1:9099",
        )
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="llama_cpp",
            base_url="http://127.0.0.1:9099",
            explicit_model="test-model",
        )
    )

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hello"}])]

    assert chunks == ["ok"]


@pytest.mark.asyncio
async def test_stream_chat_non_streaming_resolution_yields_completion_once() -> None:
    seen_payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    gateway = ConsoleProviderGateway(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="m",
        ready=True,
        streaming=False,
        temperature=0.2,
    )

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["done"]
    assert seen_payloads == [
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "temperature": 0.2,
        }
    ]


@pytest.mark.asyncio
async def test_stream_chat_generic_non_streaming_yields_completion_once() -> None:
    calls = []

    def fake_chat_api_call(**kwargs):
        calls.append(kwargs)
        return "generic done"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-4.1",
            streaming=False,
            temperature=0.2,
            top_p=0.9,
            min_p=0.05,
            top_k=40,
            max_tokens=256,
            seed=123,
            presence_penalty=0.4,
            frequency_penalty=0.5,
            reasoning_effort="high",
            reasoning_summary="auto",
            verbosity="medium",
        )
    )

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["generic done"]
    assert calls == [
        {
            "api_endpoint": "openai",
            "messages_payload": [{"role": "user", "content": "hi"}],
            "api_key": "sk-test",
            "model": "gpt-4.1",
            "streaming": False,
            "temp": 0.2,
            "topp": 0.9,
            "maxp": 0.9,
            "minp": 0.05,
            "topk": 40,
            "max_tokens": 256,
            "seed": 123,
            "presence_penalty": 0.4,
            "frequency_penalty": 0.5,
            "reasoning_effort": "high",
            "reasoning_summary": "auto",
            "verbosity": "medium",
        }
    ]


@pytest.mark.asyncio
async def test_stream_chat_generic_sync_generator_yields_ordered_chunks() -> None:
    def fake_chat_api_call(**_kwargs):
        yield "hel"
        yield {"choices": [{"delta": {"content": "lo"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_stream_chat_generic_completion_dict_yields_message_content() -> None:
    def fake_chat_api_call(**_kwargs):
        return {"choices": [{"message": {"content": "complete dict"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["complete dict"]


def test_normalize_generic_provider_response_shapes() -> None:
    unsupported = "Provider returned an unsupported response shape."
    no_content = "Provider returned no assistant content."

    class IterableSdkResponse:
        def __iter__(self):
            yield {"content": "do not dump"}

    assert list(ConsoleProviderGateway.normalize_provider_response({"content": "body"})) == ["body"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response({"choices": [{"message": {"content": "choice"}}]})
    ) == ["choice"]
    assert list(ConsoleProviderGateway.normalize_provider_response({"generated_text": "generated"})) == ["generated"]
    assert list(ConsoleProviderGateway.normalize_provider_response([{"content": "do not dump"}])) == [unsupported]
    assert list(ConsoleProviderGateway.normalize_provider_response(({"content": "do not dump"},))) == [unsupported]
    assert list(ConsoleProviderGateway.normalize_provider_response(b"hello \xff")) == ["hello \ufffd"]
    assert list(ConsoleProviderGateway.normalize_provider_response(iter(()))) == [no_content]
    assert list(ConsoleProviderGateway.normalize_provider_response({"unexpected": {"secret": "do not dump"}})) == [
        unsupported
    ]
    assert list(ConsoleProviderGateway.normalize_provider_response(IterableSdkResponse())) == [unsupported]


def test_normalize_generic_provider_response_dict_precedence() -> None:
    payload = {
        "choices": [
            {
                "delta": {"content": "delta"},
                "message": {"content": "message"},
                "text": "choice text",
            }
        ],
        "message": {"content": "top message"},
        "content": "content",
        "text": "text",
        "response": "response",
        "generated_text": "generated",
    }

    assert list(ConsoleProviderGateway.normalize_provider_response(payload)) == ["delta"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "choices": [{"message": {"content": "message"}, "text": "choice text"}],
                "message": {"content": "top message"},
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["message"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "choices": [{"text": "choice text"}],
                "message": {"content": "top message"},
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["choice text"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {
                "message": {"content": "top message"},
                "content": "content",
                "text": "text",
                "response": "response",
                "generated_text": "generated",
            }
        )
    ) == ["top message"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"content": "content", "text": "text", "response": "response", "generated_text": "generated"}
        )
    ) == ["content"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response(
            {"text": "text", "response": "response", "generated_text": "generated"}
        )
    ) == ["text"]
    assert list(
        ConsoleProviderGateway.normalize_provider_response({"response": "response", "generated_text": "generated"})
    ) == ["response"]
    assert list(ConsoleProviderGateway.normalize_provider_response({"generated_text": "generated"})) == ["generated"]


def test_normalize_google_gemini_candidates_response() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "OK"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 1,
            "totalTokenCount": 6,
        },
    }

    assert list(ConsoleProviderGateway.normalize_provider_response(payload)) == ["OK"]


def test_safe_provider_error_copy_redacts_secret_like_values() -> None:
    copy = safe_provider_error_copy(
        "openai",
        RuntimeError(
            "Authorization: Bearer sk-1234567890abcdef "
            "https://user:secret@example.test/v1 password=hunter2 token=abc123"
        ),
    )

    assert "sk-1234567890abcdef" not in copy
    assert "Bearer" not in copy
    assert "user:secret@" not in copy
    assert "hunter2" not in copy
    assert "abc123" not in copy
    assert "openai" in copy


def test_safe_provider_error_copy_classifies_provider_exceptions() -> None:
    cases = [
        (ChatAuthenticationError(), "authentication failed"),
        (ChatRateLimitError(), "rate limit exceeded"),
        (ChatBadRequestError(), "bad request"),
        (ChatConfigurationError(), "configuration error"),
        (ChatProviderError(), "provider unavailable"),
        (RuntimeError("boom"), "unexpected provider error"),
    ]

    for exc, category in cases:
        copy = safe_provider_error_copy("openai", exc)
        assert f"Provider error from openai: {category}." in copy
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            assert f"Status: {status_code}." in copy
        else:
            assert "Status:" not in copy


def test_safe_provider_error_copy_includes_status_code_when_available() -> None:
    copy = safe_provider_error_copy("openai", ChatProviderError(status_code=503))

    assert copy == "Provider error from openai: provider unavailable. Status: 503."


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_string_chunks_yield_content_only() -> None:
    def fake_chat_api_call(**_kwargs):
        yield 'data: {"choices":[{"delta":{"content":"hel"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"lo"}}]}'
        yield "data: [DONE]"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["hel", "lo"]


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_byte_chunks_yield_content_only() -> None:
    def fake_chat_api_call(**_kwargs):
        yield b'data: {"choices":[{"delta":{"content":"bytes"}}]}'
        yield b"data: [DONE]"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    chunks = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    assert chunks == ["bytes"]


@pytest.mark.asyncio
async def test_stream_chat_generic_cancel_ignores_late_chunks() -> None:
    gate = threading.Event()

    def fake_chat_api_call(**_kwargs):
        yield "first"
        gate.wait(timeout=1)
        yield "late"

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="m"))
    stream = gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])

    assert await anext(stream) == "first"
    await stream.aclose()
    gate.set()


@pytest.mark.asyncio
async def test_stream_chat_generic_provider_error_raises_sanitized_exception() -> None:
    def fake_chat_api_call(**_kwargs):
        raise RuntimeError("Authorization: Bearer sk-1234567890abcdef")

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    message = str(exc_info.value)
    assert message == "Provider error from openai: unexpected provider error."
    assert "sk-1234567890abcdef" not in message
    assert "Bearer" not in message


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_error_raises_sanitized_exception() -> None:
    def fake_chat_api_call(**_kwargs):
        yield 'data: {"error":{"message":"Authorization: Bearer sk-1234567890abcdef"}}'

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    message = str(exc_info.value)
    assert message == "Provider error from openai: unexpected provider error."
    assert "sk-1234567890abcdef" not in message
    assert "Bearer" not in message


@pytest.mark.asyncio
async def test_stream_chat_generic_sse_byte_error_raises_sanitized_exception() -> None:
    def fake_chat_api_call(**_kwargs):
        yield b'data: {"error":{"message":"Authorization: Bearer sk-1234567890abcdef"}}'

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=fake_chat_api_call,
    )
    resolution = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"))

    with pytest.raises(ChatProviderError) as exc_info:
        _ = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]

    message = str(exc_info.value)
    assert message == "Provider error from openai: unexpected provider error."
    assert "sk-1234567890abcdef" not in message
    assert "Bearer" not in message


@pytest.mark.asyncio
async def test_gateway_closes_owned_http_client():
    gateway = ConsoleProviderGateway()

    assert gateway.http_client.is_closed is False

    await gateway.aclose()

    assert gateway.http_client.is_closed is True


@pytest.mark.asyncio
async def test_gateway_does_not_close_injected_http_client():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    gateway = ConsoleProviderGateway(http_client=client)

    await gateway.aclose()

    assert client.is_closed is False
    await client.aclose()


@pytest.mark.asyncio
async def test_owned_http_client_uses_generous_generation_read_timeout():
    """The owned client must not cap slow local generations at the old 30s."""
    gateway = ConsoleProviderGateway()
    try:
        timeout = gateway.http_client.timeout
        assert timeout.read == GENERATION_READ_TIMEOUT_SECONDS
        assert timeout.read >= 120
        assert timeout.write >= 120
        assert timeout.pool >= 120
        assert timeout.connect is not None and timeout.connect <= 30
    finally:
        await gateway.aclose()


@pytest.mark.asyncio
async def test_llamacpp_probes_use_short_per_request_timeout():
    """Readiness probes stay snappy even though generation reads are long."""
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.extensions.get("timeout", {}))))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=GENERATION_READ_TIMEOUT_SECONDS,
        )
    )

    explicit = await gateway.resolve_llamacpp(LlamaCppProviderConfig(explicit_model="m"))
    discovered = await gateway.resolve_llamacpp(LlamaCppProviderConfig())

    assert explicit.ready is True
    assert discovered.model == "model-a"
    assert [path for path, _ in seen] == ["/health", "/v1/models"]
    for path, timeout in seen:
        assert timeout.get("connect") == PROBE_TIMEOUT_SECONDS, path
        assert timeout.get("read") == PROBE_TIMEOUT_SECONDS, path


@pytest.mark.asyncio
async def test_llamacpp_generation_calls_keep_client_level_timeout():
    """Generation requests inherit the client timeout, not the probe override."""
    client_timeout = 222.0
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.extensions.get("timeout", {}))))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "slow answer"}}]},
        )

    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            timeout=client_timeout,
        )
    )

    completion = await gateway.complete_llamacpp_chat(
        base_url="http://127.0.0.1:9099",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert completion == "slow answer"
    assert [path for path, _ in seen] == ["/v1/chat/completions"]
    assert seen[0][1].get("read") == client_timeout
    assert seen[0][1].get("read") != PROBE_TIMEOUT_SECONDS


class _JSONOKHandler(http.server.BaseHTTPRequestHandler):
    """Minimal local HTTP server: real sockets, real httpcore connection pool.

    A ``httpx.MockTransport`` does not reproduce the loop-binding bug below
    (it never touches httpcore's real ``AsyncConnectionPool``, so no
    loop-bound lock/event is ever created) -- only genuine socket traffic
    does, which is why this fixture spins up a real (if tiny) HTTP server
    instead. ``protocol_version`` must be HTTP/1.1 (``BaseHTTPRequestHandler``
    defaults to 1.0, which closes the connection after every response and
    happens to sidestep the pool-level lock reuse this test targets) -- real
    llama.cpp servers speak keep-alive HTTP/1.1, which is what actually
    reproduced the live crash this regression test is pinned to.
    """

    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 -- BaseHTTPRequestHandler naming
        body = b'{"data": [{"id": "model-a"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D102 -- silence default stderr logging
        pass


@pytest.fixture
def local_http_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _JSONOKHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_owned_http_client_survives_agent_bridge_style_loop_swap(local_http_server):
    """Regression (Task 8 live gate): every agent turn crashed against a real
    llama.cpp server with ``RuntimeError: <asyncio.locks.Event ...> is bound
    to a different event loop``. Root cause: the gateway's OWNED httpx
    client was reused verbatim across the app's main event loop (readiness
    probes, awaited in-place) and the agent bridge's per-turn
    ``asyncio.run()`` worker-thread loop (``console_agent_bridge.
    _StreamingModelAdapter.chat_call``) -- httpx/httpcore bind their
    internal connection-pool lock/event objects to whichever loop first
    touches them, so a second, concurrently-running loop reusing the same
    client always raised. This drives the exact same two-loop shape: a
    background thread keeps a loop alive indefinitely (like the Textual app
    loop) while a fresh ``asyncio.run()`` (like the agent bridge) reuses the
    same gateway afterward.
    """
    gateway = ConsoleProviderGateway()

    async def probe() -> bool:
        return await gateway._is_reachable(local_http_server)

    main_loop = asyncio.new_event_loop()
    main_loop_ready = threading.Event()

    def run_main_loop() -> None:
        asyncio.set_event_loop(main_loop)
        main_loop_ready.set()
        main_loop.run_forever()

    main_thread = threading.Thread(target=run_main_loop, daemon=True)
    main_thread.start()
    main_loop_ready.wait(timeout=2)
    try:
        # First use: a readiness probe awaited on the (still-running) main
        # loop -- binds the owned client's internal locks to `main_loop`.
        first = asyncio.run_coroutine_threadsafe(probe(), main_loop).result(timeout=5)
        assert first is True

        # Second use: the agent bridge's worker thread bridges via a BRAND
        # NEW asyncio.run() loop while `main_loop` is still alive elsewhere.
        # Before the fix this raised RuntimeError("... is bound to a
        # different event loop") on every single agent turn.
        second = asyncio.run(probe())
        assert second is True
    finally:
        main_loop.call_soon_threadsafe(main_loop.stop)
        main_thread.join(timeout=2)


def test_injected_http_client_is_never_swapped_across_loops():
    """Injected clients (test doubles / callers that own their own client)
    must never be silently replaced -- only the gateway's OWNED client is
    loop-swapped."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    gateway = ConsoleProviderGateway(http_client=client)

    async def active_client_identity() -> int:
        return id(gateway._active_http_client())

    first = asyncio.run(active_client_identity())
    second = asyncio.run(active_client_identity())

    assert first == second == id(client)
    asyncio.run(client.aclose())


def test_active_http_client_swap_is_mutually_exclusive_across_threads():
    """PR #629 Fix 1(a) (Gemini HIGH x2 + Qodo-8): the check-and-swap of
    ``http_client``/``_client_loop`` must be a single atomic critical
    section guarded by one lock, not two independently-racy reads/writes --
    otherwise a concurrent caller from another thread/loop can interleave
    with an in-flight swap and desync the client/loop pair (see the
    interleaving hammer test below for the crash this produces in
    practice). Proven deterministically here (no reliance on GIL
    scheduling luck): thread A is parked *inside* the swap via a
    monkeypatched, blocking ``_new_owned_http_client``, and thread B's
    concurrent call must provably fail to complete while A is still in
    flight -- only completing once A releases and the lock is free."""
    gateway = ConsoleProviderGateway()
    original_new_client = ConsoleProviderGateway._new_owned_http_client
    entered = threading.Event()
    release = threading.Event()

    def blocking_new_client():
        # Only the FIRST call (thread A's) blocks -- a concurrent second
        # call (thread B's) that is *not* actually serialized by a lock
        # would sail straight through this on its own turn and finish its
        # swap well before thread A ever releases, which is exactly the
        # unlocked-race behavior this test must catch.
        if not entered.is_set():
            entered.set()
            release.wait(timeout=5)
        return original_new_client()

    ConsoleProviderGateway._new_owned_http_client = staticmethod(blocking_new_client)
    loop_a = asyncio.new_event_loop()
    loop_b = asyncio.new_event_loop()
    thread_a: threading.Thread | None = None
    thread_b: threading.Thread | None = None
    second_done = threading.Event()
    try:
        def call_a() -> None:
            async def go() -> None:
                gateway._active_http_client()
            loop_a.run_until_complete(go())

        thread_a = threading.Thread(target=call_a)
        thread_a.start()
        assert entered.wait(timeout=5), "thread A must have entered the swap"

        def call_b() -> None:
            async def go() -> None:
                gateway._active_http_client()
            loop_b.run_until_complete(go())
            second_done.set()

        thread_b = threading.Thread(target=call_b)
        thread_b.start()

        # Thread A is still parked inside its swap -- give thread B ample
        # opportunity to race ahead if the swap were not actually
        # serialized by a lock.
        premature = second_done.wait(timeout=0.5)
        assert premature is False, (
            "a concurrent swap completed while another thread's swap was "
            "still in flight -- the check-and-swap is not atomic"
        )
        assert second_done.wait(timeout=5)
    finally:
        # Always unblock thread A and drain both threads before touching
        # the loops, whether or not the assertions above passed -- an
        # early failure must not leave a loop "running" (from the other
        # thread's still-in-flight run_until_complete) when we try to
        # close it.
        release.set()
        if thread_a is not None:
            thread_a.join(timeout=5)
        if thread_b is not None:
            thread_b.join(timeout=5)
        # Re-wrap in `staticmethod(...)`: plain-function reassignment onto
        # the class would otherwise bind `self` as an implicit first
        # argument on the next instance access, breaking every other test
        # in this module that constructs a gateway afterward.
        ConsoleProviderGateway._new_owned_http_client = staticmethod(original_new_client)
        loop_a.close()
        loop_b.close()


def test_first_swap_still_schedules_close_of_the_original_owned_client(monkeypatch):
    """PR #629 Fix 1(b) (Gemini HIGH + Qodo-8): the very first swap has
    ``_client_loop`` still ``None`` (there is no previous loop to close the
    replaced client on), and the old code's ``if stale_loop is not None:``
    guard skipped scheduling a close entirely in that case -- silently
    leaking the client created in ``__init__``. The replaced client must
    always be closed best-effort, even on the first swap."""
    gateway = ConsoleProviderGateway()
    original_client = gateway.http_client
    scheduled: list[tuple[int, object]] = []

    def fake_schedule(client, loop):
        scheduled.append((id(client), loop))

    monkeypatch.setattr(
        ConsoleProviderGateway, "_schedule_stale_client_close", staticmethod(fake_schedule)
    )

    async def touch() -> None:
        gateway._active_http_client()

    asyncio.run(touch())

    assert scheduled, "the original owned client must be scheduled for close on the first swap too"
    assert scheduled[0][0] == id(original_client)


def test_active_http_client_concurrent_swap_never_leaves_client_bound_to_wrong_loop(
    local_http_server,
):
    """PR #629 Fix 1(a) (Gemini HIGH x2 + Qodo-8): the check-and-swap of
    ``http_client``/``_client_loop`` was not atomic, so concurrent callers
    from different threads/loops (e.g. the app loop's readiness probe
    racing the agent worker thread's per-turn loop) could interleave the
    read-then-write and leave the client bound to one loop while
    ``_client_loop`` records a different one -- the next probe on the
    recorded loop then reuses a client bound elsewhere and crashes with
    "bound to a different event loop". This hammers many persistent loops
    (each in its own OS thread, lined up on a barrier every round so they
    all race the swap concurrently) against the gateway's single owned
    client and asserts every single real request against a local server
    succeeds -- a mismatch manifests as a genuine RuntimeError out of
    httpx/httpcore, not just a stale internal-state assertion."""
    gateway = ConsoleProviderGateway()
    thread_count = 6
    rounds = 20
    barrier = threading.Barrier(thread_count)
    loops: list[asyncio.AbstractEventLoop] = []
    ready_events: list[threading.Event] = []
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def run_loop_thread(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    loop_threads = []
    for _ in range(thread_count):
        loop = asyncio.new_event_loop()
        ready = threading.Event()
        loops.append(loop)
        ready_events.append(ready)
        thread = threading.Thread(target=run_loop_thread, args=(loop, ready), daemon=True)
        loop_threads.append(thread)
        thread.start()
    for ready in ready_events:
        assert ready.wait(timeout=2)

    def hammer(loop: asyncio.AbstractEventLoop) -> None:
        for _ in range(rounds):
            try:
                barrier.wait(timeout=10)
            except threading.BrokenBarrierError:
                return
            try:
                future = asyncio.run_coroutine_threadsafe(
                    gateway._is_reachable(local_http_server), loop
                )
                assert future.result(timeout=10) is True
            except BaseException as exc:  # noqa: BLE001 -- collected, asserted below
                with errors_lock:
                    errors.append(exc)

    workers = [
        threading.Thread(target=hammer, args=(loop,), daemon=True) for loop in loops
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)

    for loop in loops:
        loop.call_soon_threadsafe(loop.stop)
    for thread in loop_threads:
        thread.join(timeout=2)

    assert errors == []
