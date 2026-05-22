import httpx
import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    LlamaCppProviderConfig,
)


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
async def test_resolve_for_send_blocks_unsupported_provider_with_wip_copy():
    gateway = ConsoleProviderGateway(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    )

    resolved = await gateway.resolve_for_send(ConsoleProviderSelection(provider="openai"))

    assert resolved.ready is False
    assert resolved.provider == "openai"
    assert resolved.visible_copy == "WIP: Console native provider 'openai' is not wired yet. Select llama.cpp for this slice."


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
