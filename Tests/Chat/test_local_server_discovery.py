"""Pure local-server discovery tests (task-188) using httpx.MockTransport."""

import httpx
import pytest

from tldw_chatbook.Chat.local_server_discovery import (
    DEFAULT_LLAMACPP_DISCOVERY_URL,
    DEFAULT_OLLAMA_DISCOVERY_URL,
    DiscoveredLocalServer,
    LocalModelProbeResult,
    build_local_server_candidates,
    discover_local_servers,
    is_localhost_url,
    normalize_probe_base_url,
    probe_models_endpoint,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _openai_models_payload(*model_ids: str) -> dict:
    return {"object": "list", "data": [{"id": model_id} for model_id in model_ids]}


# --- candidate building -----------------------------------------------------


def test_candidates_include_wellknown_defaults_first() -> None:
    candidates = build_local_server_candidates({})

    assert [candidate.base_url for candidate in candidates] == [
        DEFAULT_LLAMACPP_DISCOVERY_URL,
        DEFAULT_OLLAMA_DISCOVERY_URL,
    ]
    assert [candidate.provider_key for candidate in candidates] == ["llama_cpp", "ollama"]


def test_candidates_add_configured_local_endpoints_and_strip_api_paths() -> None:
    candidates = build_local_server_candidates(
        {
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099/v1"},
                "vllm": {"api_url": "http://localhost:8000/v1/models"},
                "openai": {"api_url": "http://127.0.0.1:5000"},
            }
        }
    )

    urls = {candidate.base_url for candidate in candidates}
    assert "http://127.0.0.1:9099" in urls
    assert "http://localhost:8000" in urls
    # openai is not a local-provider section; its endpoint is never a candidate.
    assert "http://127.0.0.1:5000" not in urls


def test_candidates_never_include_non_localhost_hosts() -> None:
    candidates = build_local_server_candidates(
        {
            "api_settings": {
                "vllm": {"api_url": "http://192.168.1.5:8000"},
                "ollama": {"api_url": "https://ollama.example.com"},
                "koboldcpp": {"api_url": "http://127.0.0.2:5001"},
            }
        }
    )

    hosts = {httpx.URL(candidate.base_url).host for candidate in candidates}
    assert hosts <= {"127.0.0.1", "localhost"}


def test_candidates_dedupe_repeated_urls() -> None:
    candidates = build_local_server_candidates(
        {
            "api_settings": {
                "llama_cpp": {
                    "api_url": DEFAULT_LLAMACPP_DISCOVERY_URL,
                    "base_url": f"{DEFAULT_LLAMACPP_DISCOVERY_URL}/",
                },
            }
        }
    )

    urls = [candidate.base_url for candidate in candidates]
    assert urls.count(DEFAULT_LLAMACPP_DISCOVERY_URL) == 1


def test_normalize_and_localhost_helpers() -> None:
    assert normalize_probe_base_url("127.0.0.1:8080") == "http://127.0.0.1:8080"
    assert normalize_probe_base_url("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    assert normalize_probe_base_url("ftp://127.0.0.1:8080") is None
    assert normalize_probe_base_url("") is None
    assert is_localhost_url("http://localhost:1234") is True
    assert is_localhost_url("http://127.0.0.1:1234") is True
    assert is_localhost_url("http://127.0.0.2:1234") is False
    assert is_localhost_url("http://example.com") is False


# --- discover_local_servers -------------------------------------------------


@pytest.mark.asyncio
async def test_discovers_llamacpp_server_at_default_url() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1" and request.url.port == 8080:
            assert request.url.path == "/v1/models"
            return httpx.Response(200, json=_openai_models_payload("qwen-3", "phi-4"))
        raise httpx.ConnectError("refused", request=request)

    servers = await discover_local_servers({}, http_client=_client(handler))

    assert servers == (
        DiscoveredLocalServer(
            provider_key="llama_cpp",
            base_url=DEFAULT_LLAMACPP_DISCOVERY_URL,
            model_ids=("qwen-3", "phi-4"),
        ),
    )


@pytest.mark.asyncio
async def test_no_servers_found_returns_empty_tuple() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await discover_local_servers({}, http_client=_client(handler)) == ()


@pytest.mark.asyncio
async def test_timeouts_are_not_found_and_never_raise() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    assert await discover_local_servers({}, http_client=_client(handler)) == ()


@pytest.mark.asyncio
async def test_non_localhost_config_endpoints_are_never_probed() -> None:
    probed_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        probed_hosts.append(request.url.host)
        raise httpx.ConnectError("refused", request=request)

    await discover_local_servers(
        {"api_settings": {"vllm": {"api_url": "http://192.168.1.5:8000"}}},
        http_client=_client(handler),
    )

    assert probed_hosts
    assert set(probed_hosts) <= {"127.0.0.1", "localhost"}
    assert "192.168.1.5" not in probed_hosts


@pytest.mark.asyncio
async def test_ollama_candidate_falls_back_to_api_tags() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port != 11434:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(404, text="not found")
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})

    servers = await discover_local_servers({}, http_client=_client(handler))

    assert servers == (
        DiscoveredLocalServer(
            provider_key="ollama",
            base_url=DEFAULT_OLLAMA_DISCOVERY_URL,
            model_ids=("llama3:latest",),
        ),
    )


@pytest.mark.asyncio
async def test_llamacpp_candidate_gets_no_ollama_fallback() -> None:
    seen_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8080:
            seen_paths.append(request.url.path)
            return httpx.Response(404, text="not found")
        raise httpx.ConnectError("refused", request=request)

    servers = await discover_local_servers({}, http_client=_client(handler))

    assert servers == ()
    assert seen_paths == ["/v1/models"]


# --- probe_models_endpoint (settings-modal Discover button) ------------------


@pytest.mark.asyncio
async def test_probe_success_returns_model_ids() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json=_openai_models_payload("m-a", "m-b"))

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        provider_key="llama_cpp",
        http_client=_client(handler),
    )

    assert result == LocalModelProbeResult(
        ok=True,
        base_url="http://127.0.0.1:9099",
        model_ids=("m-a", "m-b"),
    )


@pytest.mark.asyncio
async def test_probe_success_with_empty_model_list() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": []})

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(handler),
    )

    assert result.ok is True
    assert result.model_ids == ()


@pytest.mark.asyncio
async def test_probe_connect_error_reports_honest_copy() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(handler),
    )

    assert result.ok is False
    assert result.detail == "No models endpoint at http://127.0.0.1:9099."


@pytest.mark.asyncio
async def test_probe_timeout_reports_honest_copy() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(handler),
    )

    assert result.ok is False
    assert result.detail == "Timed out contacting http://127.0.0.1:9099."


@pytest.mark.asyncio
async def test_probe_http_error_status_reports_honest_copy() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(handler),
    )

    assert result.ok is False
    assert result.detail == "No models endpoint at http://127.0.0.1:9099 (HTTP 500)."


@pytest.mark.asyncio
async def test_probe_rejects_unusable_url_without_network() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request expected for an unusable URL")

    result = await probe_models_endpoint(
        "ftp://127.0.0.1:9099",
        http_client=_client(handler),
    )

    assert result.ok is False
    assert result.detail


@pytest.mark.asyncio
async def test_probe_non_json_response_is_not_a_detected_server() -> None:
    """PR #608 review: an HTML page on a default port must not probe ok."""
    client = _client(
        lambda request: httpx.Response(200, text="<html><body>hello</body></html>")
    )
    result = await probe_models_endpoint(
        "http://127.0.0.1:8080", provider_key="llama_cpp", http_client=client
    )
    assert result.ok is False
    assert "No models endpoint" in result.detail


@pytest.mark.asyncio
async def test_probe_unrecognized_json_payload_is_not_a_detected_server() -> None:
    """A JSON API without a models container must not register as an LLM server."""
    client = _client(
        lambda request: httpx.Response(200, json={"status": "ok", "service": "printer"})
    )
    result = await probe_models_endpoint(
        "http://127.0.0.1:8080", provider_key="llama_cpp", http_client=client
    )
    assert result.ok is False
    assert "No models endpoint" in result.detail


@pytest.mark.asyncio
async def test_probe_sanitizes_hostile_model_ids() -> None:
    """Control characters are stripped and ids are bounded at the boundary."""
    hostile = "bad\x1b[31mid\x00" + "x" * 500
    client = _client(
        lambda request: httpx.Response(
            200, json={"data": [{"id": hostile}, {"id": "good-model"}]}
        )
    )
    result = await probe_models_endpoint(
        "http://127.0.0.1:8080", provider_key="llama_cpp", http_client=client
    )
    assert result.ok is True
    assert "good-model" in result.model_ids
    for model_id in result.model_ids:
        assert len(model_id) <= 120
        assert all(ch.isprintable() for ch in model_id)
        assert "\x1b" not in model_id
