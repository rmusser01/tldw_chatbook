"""Server-first context limits, bounded discovery, and shared consumers."""

import asyncio
from dataclasses import replace

import httpx
import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_context_window import (
    ContextWindowCache,
    ContextWindowTarget,
    resolve_context_window,
)
from tldw_chatbook.Chat.console_session_settings import build_console_context_estimate
from tldw_chatbook.model_capabilities import ModelCapabilities


@pytest.fixture(autouse=True)
def known_capabilities(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.model_capabilities._global_capabilities", ModelCapabilities({})
    )


@pytest.fixture(autouse=True)
def enabled_egress_policy(monkeypatch):
    """Exercise real URL policy with isolated settings and no host exemptions."""
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(egress, "_config_enabled", lambda: True)
    monkeypatch.setattr(egress, "_config_allowed_hosts", frozenset)


@pytest.mark.parametrize("value", [None, 0, -1, True, "128000", 1.5, 10**400])
def test_invalid_server_window_uses_exact_model_before_family(value):
    result = resolve_context_window("openai", "gpt-4o", server_tokens=value)
    assert result.tokens == 128000
    assert result.source == "model catalog"


def test_priority_and_unknown_system_fallback():
    assert resolve_context_window("openai", "gpt-4o", server_tokens=4096).tokens == 4096
    assert resolve_context_window("anthropic", "unknown").tokens == 200000
    result = resolve_context_window("llama_cpp", "unknown.gguf")
    assert (result.tokens, result.verified) == (32000, False)
    estimate = build_console_context_estimate(
        [], "llama_cpp", "unknown.gguf", token_counter=lambda *args: 0
    )
    assert estimate.token_limit == result.tokens
    assert estimate.token_limit_verified is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "family,path,payload",
    [
        ("llama_cpp", "/props", {"default_generation_settings": {"n_ctx": 8192}}),
        ("vllm", "/v1/models", {"data": [{"id": "selected", "max_model_len": 8192}]}),
        (
            "ollama",
            "/api/ps",
            {"models": [{"name": "selected", "context_length": 8192}]},
        ),
    ],
)
async def test_server_capacity_and_authorization_are_cached(family, path, payload):
    calls = []

    async def handler(request):
        calls.append(request)
        assert request.url.path == path
        if path == "/props":
            assert dict(request.url.params) == {
                "model": "selected",
                "autoload": "false",
            }
        assert request.headers["authorization"] == "Bearer fake-test-key"
        return httpx.Response(200, json=payload)

    cache = ContextWindowCache()
    target = ContextWindowTarget(
        "custom-ep:a", family, "http://localhost:9000", "selected", "fake-test-key"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await cache.resolve(target, client)
        second = await cache.resolve(target, client)
    assert first == second
    assert (first.tokens, first.source, first.verified) == (
        8192,
        "server metadata",
        True,
    )
    assert len(calls) == 1
    assert "fake-test-key" not in repr(target)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "http://169.254.169.254",
        "http://[fd00:ec2::254]",
        "http://100.100.100.200",
        "http://metadata.google.internal",
    ],
)
async def test_metadata_egress_is_denied_before_request(endpoint):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200, json={"default_generation_settings": {"n_ctx": 8192}}
        )

    cache = ContextWindowCache()
    target = ContextWindowTarget("llama_cpp", "llama_cpp", endpoint, "selected")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await cache.resolve(target, client)
        assert await cache.resolve(target, client) == result
    assert requests == []
    assert result.tokens == 32000
    assert result.verified is False


@pytest.mark.asyncio
async def test_single_flight_and_identity_isolation():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def handler(request):
        calls.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(
            200, json={"data": [{"id": "selected", "max_model_len": 64000}]}
        )

    cache = ContextWindowCache()
    target = ContextWindowTarget(
        "custom-ep:a", "custom", "http://localhost:9000", "selected"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        one = asyncio.create_task(cache.resolve(target, client))
        await entered.wait()
        two = asyncio.create_task(cache.resolve(target, client))
        await asyncio.sleep(0)
        release.set()
        assert (await one).tokens == (await two).tokens == 64000
        assert len(calls) == 1
        for changes in (
            {"owner": "custom-ep:b"},
            {"api_key": "different"},
            {"model": "other"},
            {"endpoint": "http://localhost:9001"},
        ):
            await cache.resolve(replace(target, **changes), client)
        assert len(calls) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": [{"id": "other", "max_model_len": 999999}]},
        {"data": [{"id": "selected", "max_model_len": True}]},
        {"data": []},
    ],
)
async def test_invalid_or_unrelated_metadata_falls_back(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        result = await ContextWindowCache().resolve(
            ContextWindowTarget("vllm", "vllm", "http://localhost:9000", "selected"),
            client,
        )
    assert result.tokens == 32000
    assert not result.verified


@pytest.mark.asyncio
async def test_timeout_failure_is_cached_from_completion_and_cancellation_settles_waiters():
    entered = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()

    target = ContextWindowTarget(
        "custom", "custom", "http://localhost:9000", "selected"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = ContextWindowCache(timeout=0.02)
        assert (await cache.resolve(target, client)).tokens == 32000
        assert (await cache.resolve(target, client)).tokens == 32000
        assert calls == 1
        cache = ContextWindowCache()
        entered.clear()
        leader = asyncio.create_task(cache.resolve(target, client))
        await entered.wait()
        follower = asyncio.create_task(cache.resolve(target, client))
        await asyncio.sleep(0)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        assert (await asyncio.wait_for(follower, 0.2)).tokens == 32000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(302, headers={"Location": "http://other.invalid"}),
        httpx.Response(200, content=b"x" * 262145),
        httpx.Response(200, content=b"not json"),
    ],
)
async def test_failed_oversized_or_redirected_metadata_is_optional(response):
    calls = []

    def handler(request):
        calls.append(request)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ContextWindowCache().resolve(
            ContextWindowTarget(
                "custom", "custom", "http://localhost:9000", "selected"
            ),
            client,
        )
    assert result.tokens == 32000
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gateway_display_preparation_and_compaction_share_remote_capacity():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        build_console_context_control_state,
    )

    async def handler(request):
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": 12000}}
            )
        return httpx.Response(200, json={"status": "ok"})

    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="selected",
        base_url="http://localhost:9000",
        max_tokens=1024,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ConsoleProviderGateway(http_client=client)
        remote = await gateway.resolve_context_window(settings)
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=settings.provider,
                base_url=settings.base_url,
                explicit_model=settings.model,
                max_tokens=settings.max_tokens,
            )
        )
        prepared = gateway.prepare_chat_request(
            resolution, [{"role": "user", "content": "hello"}]
        )
        estimate = build_console_context_estimate(
            [],
            settings.provider,
            settings.model,
            context_window=gateway.cached_context_window(settings),
        )
        state = build_console_context_control_state(
            settings=settings, estimate=estimate
        )
    assert (
        remote.tokens
        == estimate.token_limit
        == prepared.capacity.context_window_tokens
        == state.model_window_tokens
        == 12000
    )
    assert (
        state.safe_input_ceiling_tokens
        == prepared.capacity.effective_input_ceiling_tokens
        == 10464
    )
    assert state.resolved_policy.can_compact
    assert state.resolved_policy.safety_verified


def test_estimated_capacity_keeps_automatic_budget_without_claiming_verification():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        build_console_context_control_state,
    )

    gateway = ConsoleProviderGateway()
    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url="http://localhost:9000",
        model="unlisted",
        ready=True,
        max_tokens=1024,
    )
    prepared = gateway.prepare_chat_request(
        resolution, [{"role": "user", "content": "hello"}]
    )
    estimate = build_console_context_estimate([], "llama_cpp", "unlisted")
    state = build_console_context_control_state(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="unlisted", max_tokens=1024
        ),
        estimate=estimate,
    )
    assert prepared.capacity.context_window_tokens == estimate.token_limit == 32000
    assert not prepared.capacity.safety_verified
    assert prepared.capacity.limit_source == "estimated"
    assert state.resolved_policy.can_compact
    assert not state.resolved_policy.safety_verified


@pytest.mark.asyncio
async def test_custom_registry_metadata_uses_family_and_entry_credential():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderSelection,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/v1/models"
        assert request.headers["Authorization"] == "Bearer entry-key"
        return httpx.Response(
            200, json={"data": [{"id": "selected", "context_length": 48000}]}
        )

    config = {
        "custom_endpoints": {
            "gpu-node": {
                "display_name": "GPU",
                "family": "openai_compatible",
                "base_url": "http://localhost:9000/v1",
                "models": ["selected"],
                "api_key_env": "ENTRY_KEY",
            }
        },
        "api_settings": {"custom": {"api_key": "wrong-family-key"}},
    }
    settings = ConsoleSessionSettings(provider="custom-ep:gpu-node", model="selected")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ConsoleProviderGateway(
            http_client=client,
            config_provider=lambda: config,
            environ={"ENTRY_KEY": "entry-key"},
        )
        window = await gateway.resolve_context_window(settings)
        resolution = await gateway.resolve_for_send(
            ConsoleProviderSelection(
                provider=settings.provider, explicit_model=settings.model
            )
        )
        assert resolution.ready
        assert (
            window
            == resolution.context_window
            == gateway.cached_context_window(settings)
        )
        assert window.tokens == 48000
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.loopback_network
@private_profile_test
async def test_real_http_transport_reads_serving_capacity(
    request,
):
    """Exercise the real client over loopback, with an isolated metadata server."""
    import json

    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    requests = []

    async def serve(reader, writer):
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            body = json.dumps(
                {"default_generation_settings": {"n_ctx": 48000}}
            ).encode()
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        gateway = ConsoleProviderGateway()
        try:
            result = await gateway.resolve_context_window(
                ConsoleSessionSettings(
                    provider="llama_cpp",
                    model="local/model",
                    base_url=f"http://127.0.0.1:{port}",
                )
            )
            assert result.tokens == 48000
            assert b"autoload=false" in requests[0]
            assert b"model=local%2Fmodel" in requests[0]
        finally:
            await gateway.aclose()


@pytest.mark.asyncio
async def test_failed_probe_is_not_retried_on_every_send(monkeypatch):
    """TASK-32923: failures were cached for 5 s, so an unreachable or slow
    metadata endpoint added up to the 1 s probe timeout to nearly every send.
    """
    from tldw_chatbook.Chat import console_context_window

    clock = [1000.0]
    monkeypatch.setattr(console_context_window, "monotonic", lambda: clock[0])
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    target = ContextWindowTarget("custom", "custom", "http://localhost:9000", "m")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = ContextWindowCache()
        await cache.resolve(target, client)
        clock[0] += 30
        await cache.resolve(target, client)
        assert len(calls) == 1
        clock[0] += 31
        await cache.resolve(target, client)
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_openrouter_is_not_probed():
    """OpenRouter's model list (~750 KB) always exceeds the 256 KB cap, so the
    probe could only ever download 256 KB and fail; the catalog answers."""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=b"x" * 262145)

    target = ContextWindowTarget(
        "openrouter", "openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-x"
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ContextWindowCache().resolve(target, client)
    assert calls == []
    assert result.tokens == 200000  # upstream provider fallback, as before
