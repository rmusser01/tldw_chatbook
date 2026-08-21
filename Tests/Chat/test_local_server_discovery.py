"""Pure local-server discovery tests (task-188) using httpx.MockTransport."""

import httpx
import pytest

import tldw_chatbook.Chat.local_server_discovery as local_discovery_module
from tldw_chatbook.Chat.local_server_discovery import (
    DEFAULT_LLAMACPP_DISCOVERY_URL,
    DEFAULT_OLLAMA_DISCOVERY_URL,
    DiscoveredLocalServer,
    LocalModelProbeResult,
    build_local_server_candidates,
    discover_local_servers,
    is_localhost_url,
    model_ids_from_payload,
    normalize_probe_base_url,
    probe_models_endpoint,
)

# These ARE the tests of the probe implementation, so they opt out of the
# autouse `_no_local_server_probes` guard (Tests/conftest.py, task-15111).
# They never touch the network: every client below is an injected
# httpx.MockTransport, which the socket guard independently confirms.
pytestmark = pytest.mark.local_server_probe
_EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES = 1024 * 1024


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _TrackingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.iterated_chunks = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.iterated_chunks += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _openai_models_payload(*model_ids: str) -> dict:
    return {"object": "list", "data": [{"id": model_id} for model_id in model_ids]}


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ({"data": [{"id": ""}]}, None),
        ({"data": [{"id": " \t "}]}, None),
        ({"data": [{"id": "\x00\x1f"}]}, None),
        ([" \t\x00"], None),
        ({"data": [{"id": "", "name": "fallback-name"}]}, ("fallback-name",)),
        (
            {
                "data": [
                    {"id": "", "name": "\t", "model": "fallback-model"}
                ]
            },
            ("fallback-model",),
        ),
    ),
)
def test_model_ids_require_a_usable_sanitized_identifier(
    payload: object,
    expected: tuple[str, ...] | None,
) -> None:
    assert model_ids_from_payload(payload) == expected


# --- candidate building -----------------------------------------------------


def test_candidates_include_wellknown_defaults_first() -> None:
    candidates = build_local_server_candidates({})

    assert [candidate.base_url for candidate in candidates] == [
        DEFAULT_LLAMACPP_DISCOVERY_URL,
        DEFAULT_OLLAMA_DISCOVERY_URL,
    ]
    assert [candidate.provider_key for candidate in candidates] == [
        "llama_cpp",
        "ollama",
    ]


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
    assert (
        normalize_probe_base_url("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080"
    )
    assert normalize_probe_base_url("ftp://127.0.0.1:8080") is None
    assert normalize_probe_base_url("") is None
    assert is_localhost_url("http://localhost:1234") is True
    assert is_localhost_url("http://127.0.0.1:1234") is True
    assert is_localhost_url("http://127.0.0.2:1234") is False
    assert is_localhost_url("http://example.com") is False


def test_normalize_probe_base_url_uses_contract_persistence_shape() -> None:
    assert (
        normalize_probe_base_url(
            "http://127.0.0.1:8080/proxy/v1/chat/completions"
        )
        == "http://127.0.0.1:8080/proxy"
    )
    assert (
        normalize_probe_base_url("http://127.0.0.1:8080/models")
        == "http://127.0.0.1:8080"
    )


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
async def test_probe_canonicalizes_dotted_local_llamacpp_display_name() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=_openai_models_payload("m-a"))

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        provider_key="local llama.cpp",
        http_client=_client(handler),
    )

    assert result.ok is True
    assert result.model_ids == ("m-a",)
    assert seen == ["http://127.0.0.1:9099/v1/models"]


@pytest.mark.asyncio
async def test_probe_full_chat_url_uses_contract_derived_models_sibling() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=_openai_models_payload("m-a"))

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099/proxy/v1/chat/completions",
        provider_key="llama_cpp",
        http_client=_client(handler),
    )

    assert result.ok is True
    assert seen == ["http://127.0.0.1:9099/proxy/v1/models"]


@pytest.mark.asyncio
async def test_probe_models_url_never_doubles_models_suffix() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=_openai_models_payload("m-a"))

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099/proxy/v1/models",
        provider_key="llama_cpp",
        http_client=_client(handler),
    )

    assert result.ok is True
    assert seen == ["http://127.0.0.1:9099/proxy/v1/models"]


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
async def test_probe_accepts_bare_list_string_model_entry() -> None:
    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(
            lambda request: httpx.Response(200, json=["model-a"])
        ),
    )

    assert result.ok is True
    assert result.model_ids == ("model-a",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_id",
    ("", " \t ", "\x00\x1f"),
)
async def test_local_probe_rejects_listing_with_only_unusable_identifiers(
    model_id: str,
) -> None:
    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(
            lambda request: httpx.Response(
                200,
                json={"data": [{"id": model_id}]},
            )
        ),
    )

    assert result.ok is False
    assert result.model_ids == ()
    assert result.detail == (
        "No models endpoint at http://127.0.0.1:9099 "
        "(unrecognized API payload)."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "expected"),
    (
        ({"id": "", "name": "fallback-name"}, "fallback-name"),
        (
            {"id": "", "name": "\t", "model": "fallback-model"},
            "fallback-model",
        ),
    ),
)
async def test_local_probe_falls_back_to_next_usable_identifier_field(
    entry: dict[str, str],
    expected: str,
) -> None:
    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(
            lambda request: httpx.Response(200, json={"data": [entry]})
        ),
    )

    assert result.ok is True
    assert result.model_ids == (expected,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        {"data": [{"foo": "bar"}]},
        [[]],
        [123],
        {"models": [{"foo": "bar"}]},
    ),
)
async def test_probe_rejects_nonempty_listing_without_recognized_entries(
    payload: object,
) -> None:
    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(
            lambda request: httpx.Response(200, json=payload)
        ),
    )

    assert result.ok is False
    assert result.model_ids == ()
    assert result.detail == (
        "No models endpoint at http://127.0.0.1:9099 "
        "(unrecognized API payload)."
    )


@pytest.mark.asyncio
async def test_probe_handles_recursive_json_as_bounded_failure(monkeypatch) -> None:
    def recursive_loads(body: bytes) -> object:
        raise RecursionError("secret recursive decoder detail")

    monkeypatch.setattr(local_discovery_module.json, "loads", recursive_loads)

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(
            lambda request: httpx.Response(200, content=b'{"data": []}')
        ),
    )

    assert result.ok is False
    assert result.model_ids == ()
    assert result.detail == (
        "No models endpoint at http://127.0.0.1:9099 (not a JSON API)."
    )
    assert "secret" not in repr(result)


@pytest.mark.asyncio
async def test_local_probe_accepts_json_body_at_exact_byte_limit() -> None:
    prefix = b'{"data":[]}'
    body = prefix + b" " * (
        _EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES - len(prefix)
    )

    result = await probe_models_endpoint(
        "http://127.0.0.1:9099",
        http_client=_client(lambda request: httpx.Response(200, content=body)),
    )

    assert result.ok is True
    assert result.model_ids == ()


@pytest.mark.asyncio
async def test_local_probe_rejects_oversized_chunked_body_and_closes_response() -> None:
    stream = _TrackingAsyncStream(
        b"x" * _EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES,
        b"secret-over-limit",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with _client(handler) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:9099/secret-endpoint",
            http_client=client,
        )

    assert result.ok is False
    assert result.detail == "Models response is too large."
    assert "secret" not in result.detail
    assert stream.iterated_chunks == 2
    assert stream.closed is True


@pytest.mark.parametrize("content_encoding", ("gzip", "deflate", "br"))
@pytest.mark.asyncio
async def test_local_probe_rejects_encoded_body_before_decompression(
    content_encoding: str,
) -> None:
    stream = _TrackingAsyncStream(b"compressed-expansion-bomb")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            headers={"content-encoding": content_encoding, "content-length": "24"},
            stream=stream,
        )

    async with _client(handler) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:9099",
            http_client=client,
        )

    assert result.ok is False
    assert result.detail == "Compressed models responses are not supported."
    assert stream.iterated_chunks == 0
    assert stream.closed is True


@pytest.mark.asyncio
async def test_local_probe_bounds_one_raw_chunk_despite_misleading_length() -> None:
    stream = _TrackingAsyncStream(b"x" * (_EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES + 1))

    async with _client(
        lambda request: httpx.Response(
            200, headers={"content-length": "1"}, stream=stream
        )
    ) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:9099", http_client=client
        )

    assert result.ok is False
    assert result.detail == "Models response is too large."
    assert stream.closed is True


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
@pytest.mark.parametrize(
    ("entered", "canaries"),
    (
        (
            "https://user:password@example.test/v1?token=unit-test-token",
            ("user", "password", "token", "unit-test-token"),
        ),
        (
            "https://example.test/path\u202esecret",
            ("\u202e", "secret"),
        ),
        (
            "https://example.test/path\ud800secret",
            ("\ud800", "secret"),
        ),
    ),
    ids=("credential-bearing-url", "bidi-control-url", "surrogate-url"),
)
async def test_invalid_probe_input_never_retains_raw_endpoint_or_secrets(
    entered: str,
    canaries: tuple[str, ...],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("invalid input must not be requested")

    result = await probe_models_endpoint(entered, http_client=_client(handler))
    rendered = repr(result)

    assert result.ok is False
    assert result.base_url == ""
    assert all(canary not in result.base_url for canary in canaries)
    assert all(canary not in rendered for canary in canaries)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_key",
    ("ollama", "Ollama", "local_ollama", "Local Ollama"),
)
async def test_local_probe_normalizes_ollama_provider_before_fallback(
    provider_key: str,
) -> None:
    seen_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"models": [{"name": "llama3"}]})

    result = await probe_models_endpoint(
        "http://127.0.0.1:11434",
        provider_key=provider_key,
        http_client=_client(handler),
    )

    assert result.ok is True
    assert result.model_ids == ("llama3",)
    assert seen_paths == ["/v1/models", "/api/tags"]


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


# --- Roleplay UAT regression: non-chat engines must not register as chat servers ---
# Live repro (origin/dev @ f384a2807): a TTS-only engine was listening on the
# hardcoded llama.cpp discovery port (127.0.0.1:8080) and answered /v1/models
# with a 2xx payload. Discovery treated any 2xx models listing as a chat-capable
# llama.cpp, so first-run onboarding offered "Use detected llama.cpp
# (127.0.0.1:8080)", declared the app Ready, and the user's first message failed
# with HTTP 404 (unknown endpoint: /v1/chat/completions). The payload itself
# declared `"task": "tts"` -- the data needed to reject it was already present.


def _tts_models_payload() -> dict:
    """The real payload shape served by the TTS engine seen in the UAT."""
    return {
        "object": "list",
        "data": [
            {
                "id": "supertonic-3",
                "object": "model",
                "owned_by": "engine",
                "family": "supertonic",
                "task": "tts",
                "mode": "offline",
            }
        ],
    }


@pytest.mark.asyncio
async def test_probe_rejects_tts_only_server() -> None:
    """A models listing whose every entry declares a non-chat task is not a hit."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tts_models_payload())

    async with _client(handler) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:8080", provider_key="llama_cpp", http_client=client
        )

    assert result.ok is False
    assert result.model_ids == ()


@pytest.mark.asyncio
async def test_probe_keeps_server_whose_models_declare_no_task() -> None:
    """llama.cpp does not declare `task`; absence must never mean 'not chat'."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_models_payload("gemma-4-26B.gguf"))

    async with _client(handler) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:9099", provider_key="llama_cpp", http_client=client
        )

    assert result.ok is True
    assert result.model_ids == ("gemma-4-26B.gguf",)


@pytest.mark.asyncio
async def test_probe_keeps_only_chat_capable_models_from_mixed_server() -> None:
    """A server serving both chat and non-chat models offers only the chat ones."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "voice-1", "task": "tts"},
                    {"id": "embed-1", "task": "embedding"},
                    {"id": "chat-1"},
                ],
            },
        )

    async with _client(handler) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:9099", provider_key="llama_cpp", http_client=client
        )

    assert result.ok is True
    assert result.model_ids == ("chat-1",)


@pytest.mark.asyncio
async def test_probe_rejects_non_chat_and_unrecognized_only_listing() -> None:
    """Filtering non-chat entries must leave one recognized model entry."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "voice-1", "task": "tts"},
                    {"id": 12345},  # non-string id: unrecognized, not non-chat
                ],
            },
        )

    async with _client(handler) as client:
        result = await probe_models_endpoint(
            "http://127.0.0.1:9099", provider_key="llama_cpp", http_client=client
        )

    assert result.ok is False
    assert result.model_ids == ()
    assert result.detail == (
        "No models endpoint at http://127.0.0.1:9099 "
        "(unrecognized API payload)."
    )
