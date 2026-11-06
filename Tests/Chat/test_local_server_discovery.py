"""Pure local-server discovery tests (task-188) using httpx.MockTransport."""

import errno
import tomllib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import toml

from Tests.Backup_Recovery.config_test_support import select_config_source

import tldw_chatbook.Chat.local_server_discovery as local_discovery_module
from tldw_chatbook.Chat import console_settings_defaults as defaults_module
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_endpoints import (
    DEFAULT_LLAMACPP_BASE_URL,
    UNSAVED_ENDPOINT_COPY,
    generic_endpoint_differs,
    normalize_generic_endpoint_for_compare,
    safe_endpoint_display,
    unsaved_endpoint_copy,
)
from tldw_chatbook.Chat.local_server_discovery import (
    DEFAULT_LLAMACPP_DISCOVERY_URL,
    DEFAULT_OLLAMA_DISCOVERY_URL,
    DiscoveredLocalServer,
    LocalModelProbeResult,
    build_local_server_candidates,
    connect_error_is_refused,
    discover_local_servers,
    is_localhost_url,
    model_ids_from_payload,
    normalize_probe_base_url,
    probe_models_endpoint,
)
from tldw_chatbook.Chat.console_project_instructions import (
    canonical_provider_endpoint_identity,
    fingerprint_canonical_locator,
    project_instruction_notice_key,
    sanitized_destination_label,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.provider_setup_persistence import (
    ProviderSetupDraft,
    build_provider_setup_mutation,
)
from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
    fingerprint_endpoint,
    supports_openai_compatible_model_discovery,
)
from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
    KOBOLDCPP_NATIVE_ROUTE_REFUSAL,
    URL_BASED_PROVIDER_KEYS,
    koboldcpp_native_endpoint,
    koboldcpp_route_refusal,
    resolve_provider_endpoint,
)
from tldw_chatbook.UI.Console_Modules import model_switcher as model_switcher_module
from tldw_chatbook.UI.Console_Modules.model_switcher import adoption_endpoints_differ

# These ARE the tests of the probe implementation, so they opt out of the
# autouse `_no_local_server_probes` guard (Tests/conftest.py, task-15111).
# They never touch the network: every client below is an injected
# httpx.MockTransport, which the socket guard independently confirms.
pytestmark = pytest.mark.local_server_probe
_EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES = 8 * 1024 * 1024  # TASK-32925: sized for full provider catalogs


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
            {"data": [{"id": "", "name": "\t", "model": "fallback-model"}]},
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


def _connect_error(*causes: BaseException) -> httpx.ConnectError:
    """httpx's chain for one connect: ConnectError <- OSError <- per-address."""
    try:
        try:
            try:
                raise ExceptionGroup("multiple connection attempts failed", list(causes))
            except ExceptionGroup as group:
                raise OSError("All connection attempts failed") from group
        except OSError as os_error:
            raise httpx.ConnectError("All connection attempts failed") from os_error
    except httpx.ConnectError as error:
        return error


def test_a_host_name_refused_on_every_address_reads_as_refused() -> None:
    """TASK-33005.5: ``localhost`` is ::1 and 127.0.0.1, so a stopped server
    there fails as an ExceptionGroup of per-address refusals (seen live at
    localhost:2242). It must read "refused :PORT", not "unreachable"."""
    refused = errno.ECONNREFUSED

    assert connect_error_is_refused(
        _connect_error(
            ConnectionRefusedError(refused, "::1"),
            ConnectionRefusedError(refused, "127.0.0.1"),
        )
    )
    assert not connect_error_is_refused(
        _connect_error(TimeoutError("::1"), OSError(errno.EHOSTUNREACH, "v4"))
    )


def test_candidates_include_wellknown_defaults_first() -> None:
    candidates = build_local_server_candidates({})

    # Each default probes the IPv4 address and hostname, in that order.
    # Keep llama.cpp's two conventions before Ollama and the Custom engines.
    assert [candidate.base_url for candidate in candidates] == [
        DEFAULT_LLAMACPP_DISCOVERY_URL,
        "http://localhost:8080",
        DEFAULT_LLAMACPP_BASE_URL,
        "http://localhost:9099",
        DEFAULT_OLLAMA_DISCOVERY_URL,
        "http://localhost:11434",
        "http://127.0.0.1:1234",
        "http://localhost:1234",
        "http://127.0.0.1:1337",
        "http://localhost:1337",
        "http://127.0.0.1:5001",
        "http://localhost:5001",
    ]
    assert [candidate.provider_key for candidate in candidates] == [
        "llama_cpp", "llama_cpp", "llama_cpp", "llama_cpp",
        "ollama", "ollama",
        "custom", "custom", "custom", "custom", "custom", "custom",
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
        normalize_probe_base_url("http://127.0.0.1:8080/proxy/v1/chat/completions")
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
    seen_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 8080:
            seen_urls.append(str(request.url))
            return httpx.Response(404, text="not found")
        raise httpx.ConnectError("refused", request=request)

    servers = await discover_local_servers({}, http_client=_client(handler))

    assert servers == ()
    assert seen_urls == [
        "http://127.0.0.1:8080/v1/models",
        "http://localhost:8080/v1/models",
    ]


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
        http_client=_client(lambda request: httpx.Response(200, json=["model-a"])),
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
        "No models endpoint at http://127.0.0.1:9099 (unrecognized API payload)."
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
        http_client=_client(lambda request: httpx.Response(200, json=payload)),
    )

    assert result.ok is False
    assert result.model_ids == ()
    assert result.detail == (
        "No models endpoint at http://127.0.0.1:9099 (unrecognized API payload)."
    )


@pytest.mark.asyncio
async def test_probe_handles_recursive_json_as_bounded_failure(monkeypatch) -> None:
    def recursive_loads(body: bytes) -> object:
        raise RecursionError("secret recursive decoder detail")

    # Replace only this module's json: patching json.loads itself also broke
    # the storage admission's control-record read (object_pairs_hook).
    monkeypatch.setattr(
        local_discovery_module, "json", SimpleNamespace(loads=recursive_loads)
    )

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
    body = prefix + b" " * (_EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES - len(prefix))

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
        "No models endpoint at http://127.0.0.1:9099 (unrecognized API payload)."
    )


#: TASK-36517: every provider the endpoint contract serves, as the task's
#: dev probe listed them; koboldcpp on the native route its send needs.
_CONTRACT_PROVIDERS = (
    *sorted(URL_BASED_PROVIDER_KEYS),
    "openai",
    "deepinfra",
    "anthropic",
)


def _endpoint_for(provider: str, host: str, scheme: str = "http://") -> str:
    path = "/api/v1/generate" if provider == "koboldcpp" else "/v1"
    return f"{scheme}{host}:5001{path}"


def _saved_endpoint(provider: str, endpoint: str) -> str | None:
    """The endpoint the provider save writes, or None when it refuses."""
    try:
        mutation = build_provider_setup_mutation(
            ProviderSetupDraft(
                provider=provider,
                model="m",
                endpoint=endpoint,
                credential_source="none",
                credential_revision=0,
                draft_generation=1,
            ),
            {"api_settings": {}},
        )
    except ValueError:
        return None
    (table,) = (
        values
        for section, values in mutation.section_values.items()
        if section.startswith("api_settings.")
    )
    return table.get("api_url") or table.get("api_base_url")


@pytest.mark.parametrize("provider", _CONTRACT_PROVIDERS)
def test_an_underscore_host_saves_and_discovers_like_its_hyphen_spelling(provider):
    """TASK-36517 AC#1/AC#2: a Docker Compose name such as ``kobold_cpp`` was
    refused by the save ("valid host and port") while Discover models listed
    from it. Both now accept it exactly as they accept ``kobold-cpp``."""
    hyphen = _endpoint_for(provider, "kobold-cpp")
    underscore = _endpoint_for(provider, "kobold_cpp")

    assert resolve_provider_endpoint(provider, underscore).errors == ()
    saved = _saved_endpoint(provider, underscore)
    assert saved is not None
    assert saved == _saved_endpoint(provider, hyphen).replace("-", "_")
    assert supports_openai_compatible_model_discovery(provider, underscore) is True
    assert supports_openai_compatible_model_discovery(provider, hyphen) is True


#: TASK-36517 AC#3's hosts: each breaks a host rule other than "_".
_HOSTS_REFUSED_FOR_ANOTHER_REASON = (
    "-kobold",
    "kobold-",
    "-kobold_cpp",
    "kobold_cpp-",
    "kob..old",
    "kob_..old",
    "kob%41ld",
    "kob_%41ld",
    "kob*old",
    "kob_*old",
    "1.2.3",
    "a" * 64,
    "a_" * 32,
    ".".join(["a" * 60] * 5),
)


@pytest.mark.parametrize("host", _HOSTS_REFUSED_FOR_ANOTHER_REASON)
def test_hosts_refused_for_another_reason_are_still_refused(host):
    """TASK-36517 AC#3: allowing "_" relaxes no other host rule, with or
    without an underscore in the host."""
    for provider in _CONTRACT_PROVIDERS:
        endpoint = _endpoint_for(provider, host)
        assert resolve_provider_endpoint(provider, endpoint).errors == (
            "Endpoint URL must include a valid host and port.",
        ), provider
        assert _saved_endpoint(provider, endpoint) is None, provider


def test_an_underscore_host_displays_and_compares_as_itself():
    """TASK-36517 review: Console's display and compare helpers kept the old
    label rule, so every underscore host read "invalid endpoint" and two
    different ones compared equal, as one sentinel."""
    first, second = "http://my_proxy.corp/v1", "http://other_proxy.corp/v1"

    assert safe_endpoint_display(first) == first
    assert normalize_generic_endpoint_for_compare(first) == (
        "http://my_proxy.corp/v1/chat/completions"
    )
    assert normalize_generic_endpoint_for_compare(
        second
    ) != normalize_generic_endpoint_for_compare(first)
    assert generic_endpoint_differs(first, {"api_url": second}) is True
    assert generic_endpoint_differs(first, {"api_url": first}) is False
    # The other label rules still refuse.
    for refused in ("http://-my_proxy.corp/v1", "http://my_proxy-.corp/v1"):
        assert safe_endpoint_display(refused) == "invalid endpoint"


#: Consent keys for hosts the project-instruction parser accepted before
#: TASK-36517, measured at 3795dee3bd: allowing "_" must not move any of them.
_NOTICE_KEYS_BEFORE_UNDERSCORES = {
    "http://kobold-cpp:5001/v1": (
        "735cc3db6de8cffdec48e7e1d3941c7c34b03d812c1a8e68dce0a8f01817180e"
    ),
    "https://My-Proxy.corp/v1/": (
        "aa7ba74582109afee14981af841a0ee435ebc65e722a60584838664840bc6a90"
    ),
    "http://127.0.0.1:5001/api/v1/generate": (
        "230a4eb202e98922dd8500ab912ba358797508c63878267e0edb3f8a0f2c7295"
    ),
    "https://bücher.example/v1": (
        "9c2eb1086a90887408a48e6c5e276b214e1e40d3f0fcd41d8f7b99ef551623a3"
    ),
    "http://[::1]:8080/v1": (
        "904ad8976b81be75c1b119602a56909d35d01517b07eee0aaa2128ca7ffc98b6"
    ),
}


def test_an_underscore_host_gets_a_project_instruction_key_and_its_origin_label():
    """TASK-36517 review r8e-rev2: project instructions kept their own label
    rule, so a Console send to ``kobold_cpp`` raised "invalid provider
    endpoint" while keying its consent (the foreground run was cancelled,
    traces read "[invalid endpoint]")."""
    locator = fingerprint_canonical_locator("workspace:/r8e")
    for endpoint, origin in (
        ("http://kobold_cpp:5001/v1", "http://kobold_cpp:5001"),
        ("http://My_Proxy.corp/v1", "http://my_proxy.corp"),
    ):
        key = project_instruction_notice_key(locator, "koboldcpp", endpoint)
        assert len(key) == 64 and int(key, 16) >= 0
        assert canonical_provider_endpoint_identity(endpoint) == f"{origin}/v1"
        assert sanitized_destination_label("KoboldCpp", endpoint) == (
            f"KoboldCpp ({origin})"
        )
    for endpoint, key in _NOTICE_KEYS_BEFORE_UNDERSCORES.items():
        assert project_instruction_notice_key(locator, "koboldcpp", endpoint) == key
    for refused in (
        "http://-bad.corp/v1",
        "http://bad-.corp/v1",
        "http://-bad_x.corp/v1",
    ):
        with pytest.raises(ValueError, match="invalid provider endpoint"):
            project_instruction_notice_key(locator, "koboldcpp", refused)
        assert sanitized_destination_label("KoboldCpp", refused) == (
            "KoboldCpp (invalid endpoint)"
        )



#: TASK-38963: single-label hosts, the Docker Compose service-name shape.
_SINGLE_LABEL_HOSTS = ("kobold", "kobold_cpp")
#: TASK-39041: the spellings the save reads as ``http://`` without a scheme.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
#: TASK-39041: a link-local IPv6 address with a zone id (refused everywhere).
_ZONE_ID_HOST = "[fe80::1%25en0]"


@pytest.mark.parametrize("scheme", ["http://", ""], ids=["scheme", "no-scheme"])
@pytest.mark.parametrize(
    "host",
    [
        "kobold-cpp",
        "kobold_cpp",
        *_HOSTS_REFUSED_FOR_ANOTHER_REASON,
        *_SINGLE_LABEL_HOSTS,
        *_LOOPBACK_HOSTS,
        _ZONE_ID_HOST,
    ],
)
def test_discover_models_accepts_exactly_the_hosts_the_save_accepts(host, scheme):
    """TASK-38963 AC#1: Discover models parsed the URL itself, so it listed
    from hosts such as ``-kobold`` or ``1.2.3`` and from a schemeless remote
    host, all of which the save refuses. Both now decide by the contract.
    TASK-39041: QwenCloud refused a loopback endpoint typed without a scheme,
    and the others listed from a zone id; both followed the save since."""
    for provider in _CONTRACT_PROVIDERS:
        endpoint = _endpoint_for(provider, host, scheme)
        saves = _saved_endpoint(provider, endpoint) is not None
        assert supports_openai_compatible_model_discovery(provider, endpoint) is (
            saves
        ), (provider, endpoint, saves)
        if host in _LOOPBACK_HOSTS:
            assert saves, (provider, endpoint)


def test_an_ipv6_zone_id_is_refused_by_discover_the_save_and_console():
    """TASK-39041 AC#4. Owner decision (2026-11-06): refuse everywhere. Zone
    ids are link-local only, the save and TTS refused them, and Discover
    listed from them while Console compared one as itself."""
    for provider in _CONTRACT_PROVIDERS:
        endpoint = _endpoint_for(provider, _ZONE_ID_HOST)
        assert _saved_endpoint(provider, endpoint) is None, provider
        assert supports_openai_compatible_model_discovery(provider, endpoint) is (
            False
        ), provider
        assert safe_endpoint_display(endpoint) == "invalid endpoint", provider
        assert normalize_generic_endpoint_for_compare(endpoint) == (
            "invalid endpoint"
        ), provider


@pytest.mark.asyncio
@pytest.mark.parametrize("host", _LOOPBACK_HOSTS)
async def test_qwencloud_discover_reads_a_schemeless_loopback_endpoint_as_http(host):
    """TASK-39041 AC#1: Settings hands the typed Endpoint field to Discover.
    For QwenCloud, ``localhost:5001/v1`` read as the scheme ``localhost``, so
    Discover reported "unsupported" and never asked the server, although **s**
    saves ``http://localhost:5001/v1``."""
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        discover_openai_compatible_models,
    )

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "qwen-local"}]})

    async def discovery_client(**kwargs):
        async with _client(handler) as http_client:
            return await discover_openai_compatible_models(**kwargs, client=http_client)

    statuses = []
    for endpoint in (f"{host}:5001/v1", f"http://{host}:5001/v1"):
        service = LocalLLMProviderCatalogService(
            provider_catalog_loader=lambda: {"QwenCloud": []},
            settings_loader=lambda: {"providers": {"QwenCloud": []}},
            discovery_client=discovery_client,
            environ={},
        )
        result = await service.discover_models(
            provider="QwenCloud",
            staged_settings={
                "api_settings": {
                    "qwencloud": {"api_base_url": endpoint, "api_key": "k"}
                }
            },
            use_shared_cache=False,
        )
        statuses.append(result.status)

    assert statuses == ["success", "success"]
    assert requested == [f"http://{host}:5001/v1/models"] * 2


def test_a_single_label_host_displays_and_compares_as_itself():
    """TASK-38963 AC#2: Console's display and compare took a single-label
    host with a scheme for a pasted token, so ``http://kobold:5001`` read
    "invalid endpoint" and two different such servers compared equal."""
    for endpoint in ("http://kobold:5001", "http://kobold_cpp:5001"):
        assert safe_endpoint_display(endpoint) == endpoint
    assert generic_endpoint_differs(
        "http://other:5001", {"api_url": "http://kobold:5001"}
    ) is True
    assert generic_endpoint_differs(
        "http://kobold:5001/", {"api_url": "http://kobold:5001"}
    ) is False
    assert adoption_endpoints_differ(
        "koboldcpp", "http://kobold:5001", "http://other:5001"
    ) is True
    assert adoption_endpoints_differ(
        "koboldcpp", "http://kobold_cpp:5001", "http://kobold_cpp:5001/"
    ) is False
    # A single label with no port can still be a pasted key: never echoed.
    assert safe_endpoint_display("http://unit-test-key") == "invalid endpoint"
    # Review r10e-rev1: without a port, a path still says it is an endpoint,
    # and two different servers never compare as the one sentinel.
    for endpoint in ("http://kobold/v1", "http://kobold_cpp/api/v1/generate"):
        assert safe_endpoint_display(endpoint) == endpoint
    for saved, other in (
        ("http://kobold/v1", "http://other/v1"),
        ("http://kobold/api/v1/generate", "http://other/api/v1/generate"),
        ("http://kobold", "http://other"),
    ):
        assert generic_endpoint_differs(other, {"api_url": saved}) is True, saved
        assert generic_endpoint_differs(f"{saved}/", {"api_url": saved}) is False
        assert adoption_endpoints_differ("koboldcpp", saved, other) is True, saved
    assert safe_endpoint_display("http://kobold") == "invalid endpoint"
    # Review r10e-rev2: the path counts only after a typed scheme.
    assert safe_endpoint_display("unit-test-key/v1") == "invalid endpoint"


#: TASK-38963 review r10e-rev2: keys holding "/", pasted without a scheme
#: (moved here in rev3 so a PR lane runs it).
_SCHEMELESS_KEYS_WITH_A_SLASH = (
    "abcDEF123/xyz+789=",
    "sk-abc123def456/extra",
    "unit-test-key/v1",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)


@pytest.mark.parametrize("pasted", _SCHEMELESS_KEYS_WITH_A_SLASH)
def test_a_schemeless_token_with_a_slash_is_never_echoed(pasted: str) -> None:
    """A single label's path counts only after a typed scheme: rev1 let any
    path through, so these keys displayed (and compared) as themselves."""
    key = pasted.split("/", 1)[0]

    assert safe_endpoint_display(pasted) == "invalid endpoint"
    assert normalize_generic_endpoint_for_compare(pasted) == "invalid endpoint"
    copy = unsaved_endpoint_copy(pasted, {"api_url": "http://127.0.0.1:9099/v1"})
    assert key not in copy and key.lower() not in copy
    assert "Selected endpoint: invalid endpoint." in copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "saved", "session"),
    [
        (
            "koboldcpp",
            "http://kobold_cpp:5001/api/v1/generate",
            "http://other_kobold:5001/api/v1/generate",
        ),
        # Review r10e-rev1: the same servers without a port.
        ("koboldcpp", "http://kobold/api/v1/generate", "http://other/api/v1/generate"),
        ("vllm", "http://vllm/v1", "http://other_vllm/v1"),
    ],
)
async def test_a_different_single_label_session_endpoint_is_blocked_as_unsaved(
    provider, saved, session
):
    """TASK-38963 AC#3: both single-label endpoints compared as the one
    "invalid endpoint" sentinel, so the unsaved-endpoint guard let a session
    on another server send as though it were the saved one."""

    def no_send(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("a blocked endpoint never sends")

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {provider: {"api_url": saved, "model": "m"}}
        },
        environ={},
        chat_api_call_fn=no_send,
    )

    blocked = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider=provider, base_url=session, explicit_model="m", streaming=False
        )
    )
    same = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider=provider, base_url=saved, explicit_model="m", streaming=False
        )
    )

    assert blocked.ready is False
    assert blocked.visible_copy == (
        f"{UNSAVED_ENDPOINT_COPY} Selected endpoint: {session}. "
        f"Saved endpoint: {saved}."
    )
    assert same.ready is True, same.visible_copy


class _AdoptionStore:
    active_session_id = "session-1"

    def __init__(self) -> None:
        self.ephemeral: list[object] = []

    def adopt_session_ephemeral_endpoint(self, _session_id, *, settings, policy):
        del settings
        self.ephemeral.append(policy)


class _AdoptionScreen:
    """The ChatScreen surface ``adopt_detected_local_server`` reads."""

    def __init__(self, app_config: dict) -> None:
        self.notes: list[str] = []
        self.session_settings: list[object] = []
        self.store = _AdoptionStore()
        self.app_instance = SimpleNamespace(
            notify=lambda message, **_kwargs: self.notes.append(message)
        )
        self._session = SimpleNamespace(
            _replace_active_console_session_settings=self.session_settings.append
        )
        self._app_config = app_config

    @staticmethod
    def _config_section(config, key):
        value = config.get(key, {})
        return value if isinstance(value, dict) else {}

    def _provider_readiness_app_config(self):
        return self._app_config

    def _ensure_console_chat_store(self):
        return self.store

    def _sync_console_chat_core_state(self) -> None:
        pass

    def _sync_console_settings_summary(self) -> None:
        pass

    def _sync_console_transcript_guidance(self) -> None:
        pass


def _adopt(monkeypatch, provider: str, saved: str, detected: str):
    """Adopt ``detected`` over a saved endpoint; return the screen and the save."""
    writes: list[dict] = []

    def fake_save(section_values):
        writes.append(section_values)
        return True

    monkeypatch.setattr(model_switcher_module, "save_settings_to_cli_config", fake_save)
    screen = _AdoptionScreen({"api_settings": {provider: {"api_url": saved}}})
    model_switcher_module.adopt_detected_local_server(
        screen,
        DiscoveredLocalServer(
            provider_key=provider, base_url=detected, model_ids=("detected-model",)
        ),
        ConsoleSessionSettings(provider=provider, model="detected-model"),
    )
    return screen, writes


@pytest.mark.parametrize(
    ("provider", "name", "saved", "detected"),
    [
        (
            "koboldcpp",
            "KoboldCpp",
            "http://kobold_cpp:5001/api/v1/generate",
            "http://127.0.0.1:5001",
        ),
        (
            "koboldcpp",
            "KoboldCpp",
            "http://kobold_cpp/api/v1/generate",
            "http://127.0.0.1:5001",
        ),
        ("llama_cpp", "llama.cpp", "http://llama_cpp:8080", "http://127.0.0.1:9099"),
    ],
)
def test_the_adoption_toast_names_a_saved_single_label_endpoint(
    monkeypatch, provider, name, saved, detected
):
    """TASK-38963 AC#3: the toast read "Keeping the saved endpoint invalid
    endpoint for KoboldCpp"; it names the endpoint it keeps."""
    screen, writes = _adopt(monkeypatch, provider, saved, detected)

    assert screen.notes == [
        (
            f"Keeping the saved endpoint {saved} for {name}; using the detected "
            "server for this session only."
        )
    ]
    assert "api_url" not in writes[0][f"api_settings.{provider}"]


#: TASK-38964: koboldcpp endpoints its native send cannot use, and its route.
_KOBOLD_OFF_ROUTE = (
    "http://127.0.0.1:5001",
    "http://127.0.0.1:5001/v1",
    "http://127.0.0.1:5001/v1/chat/completions",
)
_KOBOLD_NATIVE = "http://127.0.0.1:5001/api/v1/generate"


@pytest.fixture
def _reset_console_default_intents(monkeypatch):
    """Fresh intent bookkeeping, as ``test_console_settings_defaults`` keeps."""
    for name, value in (
        ("_LATEST_INTENT_GENERATION", None),
        ("_LATEST_INTENT_FINGERPRINT", None),
        ("_LATEST_INTENT_ACTION", None),
        ("_LATEST_INTENT_LIFECYCLE", None),
        ("_ACTIVE_INTENT_CALLS", set()),
        ("_PENDING_RETRY_STATE", None),
        ("_ACTIVE_RUNTIME_PUBLICATION_CLAIM", None),
        ("_NEXT_RUNTIME_PUBLICATION_CLAIM_TOKEN", 0),
    ):
        monkeypatch.setattr(defaults_module, name, value, raising=False)


@pytest.mark.parametrize(
    ("endpoint", "written"),
    [*((value, False) for value in _KOBOLD_OFF_ROUTE), (_KOBOLD_NATIVE, True)],
)
def test_make_default_writes_a_koboldcpp_endpoint_only_on_its_native_route(
    tmp_path, monkeypatch, _reset_console_default_intents, endpoint, written
):
    """TASK-38964 AC#2/AC#5: Chat settings' "Make default for new chats"
    wrote a koboldcpp root as is, and chat_with_kobold then posted the
    native payload to it. Only the native route is written."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "api_settings": {
                    "koboldcpp": {"api_url": "http://localhost:5001/api/v1/generate"}
                },
                "chat_defaults": {"provider": "openai", "model": "old-model"},
            }
        ),
        encoding="utf-8",
    )
    select_config_source(monkeypatch, config_path, vars(defaults_module))
    before = config_path.read_bytes()

    outcome = defaults_module.apply_console_default_intent(
        defaults_module.ConsoleDefaultMutationIntent(
            generation=1,
            action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            provider_config_key="koboldcpp",
            literal_model_id="kobold-model",
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
            values={"temperature": 0.25},
            endpoint_patch=defaults_module.ConsoleEndpointPatch(
                endpoint, "koboldcpp", True, True
            ),
        )
    )

    if written:
        assert outcome.runtime_published is True, outcome
        saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert saved["api_settings"]["koboldcpp"]["api_url"] == endpoint
        assert saved["chat_defaults"]["provider"] == "koboldcpp"
    else:
        assert outcome.file_replaced is False
        assert outcome.failure_phase is defaults_module.ConsoleDefaultSavePhase.BEFORE_REPLACE
        assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    ("configured", "native"),
    [
        *((value, _KOBOLD_NATIVE) for value in _KOBOLD_OFF_ROUTE),
        ("http://localhost:5001/kobold", "http://localhost:5001/kobold/api/v1/generate"),
        # Review r10e-rev1: a root at KoboldAI's /api base gets one /api.
        *(
            (f"http://127.0.0.1:5001{path}", _KOBOLD_NATIVE)
            for path in ("/api", "/api/", "/api/v1")
        ),
    ],
)
def test_server_adoption_writes_koboldcpp_s_native_route(
    monkeypatch, configured, native
):
    """TASK-38964 AC#3: adopting a detected koboldcpp wrote its root (the only
    shape discovery finds) as ``api_url``. It writes the root's native route,
    and this chat sends there too."""
    root = local_discovery_module.normalize_probe_base_url(configured)
    assert root is not None and not root.endswith("/generate")

    screen, writes = _adopt(monkeypatch, "koboldcpp", configured, root)

    assert writes[0]["api_settings.koboldcpp"]["api_url"] == native
    assert [settings.base_url for settings in screen.session_settings] == [native]
    assert screen.notes == []


def test_server_adoption_writes_other_providers_detected_endpoint_unchanged(
    monkeypatch,
):
    """TASK-38964 AC#5: only koboldcpp's write moves to its native route."""
    screen, writes = _adopt(
        monkeypatch, "llama_cpp", "http://127.0.0.1:9099", "http://127.0.0.1:9099"
    )

    assert writes[0]["api_settings.llama_cpp"]["api_url"] == "http://127.0.0.1:9099"
    assert [settings.base_url for settings in screen.session_settings] == [
        "http://127.0.0.1:9099"
    ]


def test_koboldcpp_route_rule_lives_in_the_contract_alone():
    """TASK-38964 AC#1: the wizard's /api/v1/generate check moved to the
    contract; every write path reads it there."""
    for endpoint in _KOBOLD_OFF_ROUTE:
        assert koboldcpp_route_refusal("koboldcpp", endpoint) == (
            KOBOLDCPP_NATIVE_ROUTE_REFUSAL
        )
        # A chat URL the contract resolved decides the same way.
        chat_url = resolve_provider_endpoint("koboldcpp", endpoint).chat_url
        assert koboldcpp_route_refusal("koboldcpp", chat_url) == (
            KOBOLDCPP_NATIVE_ROUTE_REFUSAL
        )
        assert koboldcpp_native_endpoint(endpoint) == _KOBOLD_NATIVE
    assert koboldcpp_route_refusal("koboldcpp", _KOBOLD_NATIVE) == ""
    assert koboldcpp_native_endpoint(_KOBOLD_NATIVE) == _KOBOLD_NATIVE
    assert koboldcpp_route_refusal("custom", "http://127.0.0.1:5001") == ""
    assert koboldcpp_route_refusal("koboldcpp", "") == ""
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in (Path(__file__).parents[2] / "tldw_chatbook").rglob("*.py")
    }
    checks = [
        str(path)
        for path, text in sources.items()
        if 'endswith("/api/v1/generate")' in text
        or "first_run_endpoint_refusal" in text
    ]
    assert checks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "custom"])
async def test_discover_names_the_contracts_reason_for_a_refused_host(provider):
    """Review r10e-rev1 of TASK-38963: Discover refused ``-kobold`` with
    "Configure an explicit http:// or https:// /v1 models endpoint", which
    that URL already is. It names the contract's reason, as **s** would."""
    from unittest.mock import AsyncMock

    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )

    client = AsyncMock()
    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {provider: []},
        settings_loader=lambda: {
            "providers": {provider: []},
            "api_settings": {provider: {"api_url": "http://-kobold:8000/v1"}},
        },
        discovery_client=client,
    )

    result = await service.discover_models(provider=provider)

    assert result.status == "unsupported"
    assert result.error.message == "Endpoint URL must include a valid host and port."
    assert result.error.recovery_hint == ""
    assert (
        _saved_endpoint(provider, "http://-kobold:8000/v1") is None
    )  # the save refuses it for that reason
    client.assert_not_awaited()


#: TASK-38963 review r10e-rev3: provider key shapes pasted into an endpoint
#: field without a scheme (the reviewer's probes 8 and 9).
_KEY_SHAPED_ENDPOINTS = (
    "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC1dE",
    "0123456789abcdef" * 4,
    "dapi" + "0123456789abcdef" * 2,
    "csk-" + "ab1cd2ef3gh4" * 4,
    "fw_" + "Ab1Cd2Ef3Gh4" * 2,
    "AIzaSyD-1234567890abcdefghijklmnopqrstu",
    "gsk_" + "a1B2c3D4" * 7,
    "xai-" + "Q" * 80,
    "sk-abc123def456/extra",
    "0123456789abcdef0123456789abcdef.AbCdEfGhIjKlMnOp",
)


@pytest.mark.parametrize("key", _KEY_SHAPED_ENDPOINTS)
def test_a_refused_endpoints_fingerprint_never_carries_a_pasted_key(key):
    """The fingerprint of an endpoint the contract refuses read
    ``http://<key>/...``; it is the sentinel."""
    assert resolve_provider_endpoint("custom", key).errors
    fingerprint = fingerprint_endpoint(key)

    assert fingerprint == "[invalid-endpoint]"
    assert key.lower()[:12] not in fingerprint.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", _KEY_SHAPED_ENDPOINTS[:2])
async def test_a_failed_catalog_refresh_logs_no_part_of_a_pasted_key(tmp_path, key):
    """Auto-refresh logged "failed for MistralAI (fingerprint http://<key>/)"
    at INFO, past the file sink's redaction, on every launch."""
    from loguru import logger

    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
        ModelCatalogSettings,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
        ModelCatalogDiskStore,
    )

    service = LocalLLMProviderCatalogService(
        provider_catalog_loader=lambda: {"MistralAI": ["saved-1"]},
        settings_loader=lambda: {
            "providers": {"MistralAI": ["saved-1"]},
            "api_settings": {"mistralai": {"api_base_url": key, "api_key": "k"}},
        },
        save_discovered_models_callback=lambda _values: True,
        environ={},
    )
    logged: list[str] = []
    sink_id = logger.add(lambda message: logged.append(str(message)), level="DEBUG")
    try:
        report = await service.refresh_stale_configured_providers(
            catalog_settings=ModelCatalogSettings(),
            disk_store=ModelCatalogDiskStore(tmp_path / "cache.json"),
            provider_list_keys=("MistralAI",),
        )
    finally:
        logger.remove(sink_id)

    assert report.outcomes[0].status == "failed", report
    text = "".join(logged).lower()
    assert "auto-refresh failed for mistralai" in text
    for start in range(0, len(key) - 7, 4):
        assert key.lower()[start : start + 8] not in text, key[start : start + 8]


#: Providers whose Console default write takes an endpoint patch, each with
#: a saved table that makes the pair ready.
_DEFAULT_WRITE_PROVIDERS = {
    "vllm": {"api_url": "http://127.0.0.1:8000/v1"},
    "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    "ollama": {"api_url": "http://127.0.0.1:11434"},
    "custom": {"api_url": "http://127.0.0.1:8000/v1/chat/completions"},
    "openai": {"api_key": "test-key", "api_base_url": "https://api.openai.com/v1"},
}


#: Review r10e-rev4 (the reviewer's probe16): shapes **s** refuses, with an
#: authority the Console default writer took.
_REFUSED_PATHS_AND_PORTS = (
    "http://127.0.0.1:8000/v1/models/v1",
    "http://127.0.0.1:8000/v1/api/v1/generate",
    "http://127.0.0.1:8000/a//b",
    "http://127.0.0.1:8000/v1/../x",
    "http://127.0.0.1:8000/v1%2Fx",
    "http://127.0.0.1:0/v1",
    "http://127.0.0.1:8000/v1#frag",
    "http://127.0.0.1:8000/v1?api_key=sk-secret123",
    "http://127.0.0.1:8000/v1/chat/completions/v1",
)


@pytest.mark.parametrize("provider", sorted(_DEFAULT_WRITE_PROVIDERS))
@pytest.mark.parametrize(
    ("endpoint", "written"),
    [
        *((key, False) for key in _KEY_SHAPED_ENDPOINTS),
        ("gpu-box.lan:8000/v1", False),
        *((value, False) for value in _REFUSED_PATHS_AND_PORTS),
        ("http://127.0.0.1:8001/v1", True),
    ],
)
def test_make_default_never_writes_an_endpoint_the_contract_refuses(
    tmp_path, monkeypatch, _reset_console_default_intents, provider, endpoint, written
):
    """TASK-38964 review r10e-rev3: Save endpoint & use model takes any
    authority the preview parses, a pasted key included ("Remote/unknown").
    Nothing the contract refuses may reach the config file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "api_settings": {provider: dict(_DEFAULT_WRITE_PROVIDERS[provider])},
                "chat_defaults": {"provider": "anthropic", "model": "old-model"},
            }
        ),
        encoding="utf-8",
    )
    select_config_source(monkeypatch, config_path, vars(defaults_module))
    before = config_path.read_bytes()

    outcome = defaults_module.apply_console_default_intent(
        defaults_module.ConsoleDefaultMutationIntent(
            generation=1,
            action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            provider_config_key=provider,
            literal_model_id="some-model",
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
            values={"temperature": 0.25},
            endpoint_patch=defaults_module.ConsoleEndpointPatch(
                endpoint, provider, True, True
            ),
        )
    )

    if written:
        assert outcome.runtime_published is True, outcome
        saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert endpoint in saved["api_settings"][provider].values()
    else:
        assert outcome.file_replaced is False
        assert config_path.read_bytes() == before
    # The writer and **s** decide alike (review r10e-rev4).
    assert written is (_saved_endpoint(provider, endpoint) is not None)


@pytest.mark.parametrize("key", [k for k in _KEY_SHAPED_ENDPOINTS if "." not in k])
def test_the_default_recovery_summary_never_names_a_pasted_key(key):
    """TASK-38964 review r10e-rev3: Chat settings' "Not written to disk"
    summary named the endpoint through the preview, which took a bare key
    for a remote authority and showed it lower-cased."""
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    intent = defaults_module.ConsoleDefaultMutationIntent(
        generation=1,
        action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        provider_config_key="vllm",
        literal_model_id="vllm-model",
        field_mask=FULL_MODEL_DEFAULT_FIELDS,
        values={},
        endpoint_patch=defaults_module.ConsoleEndpointPatch(key, "vllm", True, True),
    )
    modal = SimpleNamespace(
        _default_durability_state=SimpleNamespace(
            recovery_intent=intent,
            failure_phase=defaults_module.ConsoleDefaultSavePhase.BEFORE_REPLACE,
        )
    )

    summary = ConsoleSettingsModal._default_recovery_summary(modal)

    assert summary.startswith("Not written to disk")
    assert key.lower()[:12] not in summary.lower()
    assert defaults_module.format_console_endpoint_preview(key) is None
