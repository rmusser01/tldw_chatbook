import builtins
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
