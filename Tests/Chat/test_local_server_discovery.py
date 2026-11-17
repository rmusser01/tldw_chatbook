"""Pure local-server discovery tests (task-188) using httpx.MockTransport."""

import errno
import itertools
import re
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
    REMOTE_ENDPOINT_SCHEME_REQUIRED,
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


#: TASK-39150: bracketed zone ids, ``%25``-encoded and raw.
_PROJECT_ZONE_ID_ENDPOINTS = (
    "http://[fe80::1%25en0]:5001",
    "http://[fe80::1%en0]:5001",
    "http://[fe80::1%25en0]:5001/v1",
    "https://[fe80::1%25eth0]/v1",
)


def _project_instruction_endpoint_callers(endpoint):
    """What each project-instruction identity caller makes of ``endpoint``."""
    from tldw_chatbook.Chat import console_exchange_export
    from tldw_chatbook.Chat.console_chat_controller import (
        build_project_instruction_dispatch_notice,
    )
    from tldw_chatbook.Chat.console_exchange_capture import build_request_capture
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderStreamSignals,
    )
    from tldw_chatbook.Chat.console_trace_final_values import (
        verify_provider_request_shadow,
    )
    from tldw_chatbook.Chat.console_trace_models import new_opaque_id
    from tldw_chatbook.Chat.console_trace_provenance import (
        ProviderRequestProvenance,
        SavedRevisionTraceProvenance,
    )

    seen = {}
    locator = fingerprint_canonical_locator("workspace:/39150")
    for name, call in (
        ("identity", lambda: canonical_provider_endpoint_identity(endpoint)),
        ("notice_key", lambda: project_instruction_notice_key(locator, "vllm", endpoint)),
    ):
        try:
            seen[name] = call()
        except ValueError as error:
            seen[name] = f"ValueError: {error}"
    seen["label"] = sanitized_destination_label("vLLM", endpoint)
    notice = build_project_instruction_dispatch_notice(
        SimpleNamespace(
            startup_source_metadata=None,
            startup_source=None,
            primary_delivery=SimpleNamespace(outcomes=()),
            warning_codes=(),
        ),
        session_id="session-39150",
        resolution=SimpleNamespace(provider="vLLM", execution_key="vllm", base_url=endpoint),
    )
    seen["controller_label"] = notice.destination_label
    signals = ConsoleProviderStreamSignals(exchange_capture_enabled=True)
    call = signals.new_usage_call()
    call.begin_exchange(
        provider="vllm", model="m", endpoint=endpoint, request={}, omitted_keys=()
    )
    call.close_exchange()
    seen["gateway_capture"] = signals.exchange_captures()[0].endpoint
    seen["exchange_capture"] = build_request_capture({"api_base_url": endpoint})[0][
        "api_base_url"
    ]
    seen["exchange_export"] = console_exchange_export._endpoint(endpoint)
    saved = SavedRevisionTraceProvenance(new_opaque_id())
    values = {
        "api_endpoint": "vllm",
        "messages_payload": [{"role": "user", "content": "safe"}],
        "model": "m",
    }
    shadow = verify_provider_request_shadow(
        actual_kwargs=values,
        expected_kwargs=dict(values),
        provenance=ProviderRequestProvenance(messages=(saved,), messages_payload=(saved,)),
        project_handler_kwargs=lambda projected: {
            "input_data": projected["messages_payload"],
            "model": projected["model"],
        },
        endpoint_identity=endpoint,
    )
    seen["trace_final_values"] = (
        shadow.endpoint_identity if shadow.available else shadow.as_content_free_record()
    )
    return seen


@pytest.mark.parametrize("endpoint", _PROJECT_ZONE_ID_ENDPOINTS)
def test_a_zone_id_endpoint_is_refused_by_project_instructions_like_a_refused_host(
    endpoint,
):
    """TASK-39150 AC#1/AC#2: the project-instruction parser accepted any IP
    literal first, so a zone id kept its zone in the destination label and the
    canonical identity, against the 2026-11-06 refuse-everywhere ruling. Every
    caller now treats it exactly as it treats ``-bad.corp``."""
    refused = _project_instruction_endpoint_callers("http://-bad.corp:5001/v1")
    assert refused == {
        "identity": "ValueError: invalid provider endpoint",
        "notice_key": "ValueError: invalid provider endpoint",
        "label": "vLLM (invalid endpoint)",
        "controller_label": "vLLM (invalid endpoint)",
        "gateway_capture": "[invalid endpoint]",
        "exchange_capture": "[invalid endpoint]",
        "exchange_export": "[invalid endpoint]",
        "trace_final_values": {"available": False, "omission_reason": "sanitizer_failed"},
    }

    assert _project_instruction_endpoint_callers(endpoint) == refused


@pytest.mark.parametrize(
    ("endpoint", "identity", "label", "key"),
    [
        (
            "http://[fe80::1]:5001",
            "http://[fe80::1]:5001",
            "vLLM (http://[fe80::1]:5001)",
            "f8df26aa2ff15278ae691e9b5c26d32013a8a05f398a04ef6221f9c82008cee4",
        ),
        (
            "http://[FE80::0001]:5001/v1/",
            "http://[fe80::1]:5001/v1",
            "vLLM (http://[fe80::1]:5001)",
            "6a7b4bac3614846f43292b135a1c0f9f2b69b585a7cf85a366a4e0dbd4964210",
        ),
        (
            "http://[fd12:3456::9]:8080/v1",
            "http://[fd12:3456::9]:8080/v1",
            "vLLM (http://[fd12:3456::9]:8080)",
            "0c71cec0889e1901ae14d66b056cdd9fbaca61e45c9ed7f27e6ea55032dd16c6",
        ),
    ],
)
def test_zone_less_link_local_and_ula_literals_keep_their_project_identity(
    endpoint, identity, label, key
):
    """TASK-39150 AC#3: only the zone is refused. Keys measured at ae9f9e8d21."""
    seen = _project_instruction_endpoint_callers(endpoint)

    assert seen["identity"] == identity
    assert seen["label"] == seen["controller_label"] == label
    assert seen["notice_key"] == key
    assert seen["gateway_capture"] == seen["exchange_export"] == identity
    assert seen["trace_final_values"] == identity


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


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (_ZONE_ID_HOST, "unknown"),
        ("[fe80::1%en0]", "unknown"),
        ("localhost..", "unknown"),  # review r12s-rev1 M3: refused as typed
        ("[fe80::1]", "private_network"),
        ("[fd12:3456::9]", "private_network"),
    ],
)
def test_console_egress_reads_a_zone_id_as_an_endpoint_it_cannot_use(host, expected):
    """TASK-39137: the owner's 2026-11-06 ruling refuses zone ids everywhere,
    but Console's Library egress classifier read one as a private network and
    kept the zone in its identity. Without a zone, link-local and ULA literals
    stay private."""
    from tldw_chatbook.Chat.console_dispatch_checkpoint import ConsoleEgressClass
    from tldw_chatbook.Chat.console_library_destination import (
        resolve_console_destination,
    )
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
    )

    destination = resolve_console_destination(
        ConsoleProviderResolution(
            provider="vllm",
            base_url=f"http://{host}:5001/v1",
            model="m",
            ready=True,
        )
    )
    assert destination.egress_class is ConsoleEgressClass(expected)
    if expected == "unknown":
        assert destination.endpoint_identity == "external/unknown"
    else:
        assert destination.endpoint_identity == f"http://{host}:5001"


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


#: TASK-39042: loopback endpoints typed without a scheme, which **s** saves.
_SCHEMELESS_LOOPBACK_ENDPOINTS = ("localhost:8080", "127.0.0.1:8080/v1", "[::1]:8080/v1")
#: TASK-39042: a synthetic ZAI-shaped key (``<32 hex>.<16>``) pasted into an
#: endpoint field, with ``/v1`` appended and with an ``sk-`` prefix.
_DOTTED_KEY = "0123456789abcdef0123456789abcdef.ABCDEFGHijklmnop"
_DOTTED_KEY_ENDPOINTS = (_DOTTED_KEY, f"{_DOTTED_KEY}/v1", f"sk-{_DOTTED_KEY}")


@pytest.mark.parametrize("provider", ["vllm", "ollama", "custom"])
@pytest.mark.parametrize("endpoint", _SCHEMELESS_LOOPBACK_ENDPOINTS)
def test_chat_settings_reads_a_schemeless_loopback_endpoint_as_the_save_does(
    provider, endpoint
):
    """TASK-39042 AC#1: readiness ran ``validate_url``, which needs a typed
    scheme, so Chat settings read "Invalid URL" for ``localhost:8080`` and
    could not save what **s** saves."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
        validate_console_session_settings,
    )

    settings = ConsoleSessionSettings(provider=provider, model="m", base_url=endpoint)
    app_config = {"api_settings": {provider: dict(_DEFAULT_WRITE_PROVIDERS[provider])}}
    readiness = build_console_settings_readiness(
        settings, app_config=app_config, environ={}
    )

    assert _saved_endpoint(provider, endpoint) is not None
    assert readiness.label != "Invalid URL", readiness
    assert readiness.blocker != "endpoint_invalid", readiness
    assert validate_console_session_settings(settings, app_config=app_config) == []


#: TASK-39151 review: the Endpoint line's copy for an invalid endpoint.
_ADD_SCHEME_ROW = "Endpoint: Invalid (add http:// or https://)"
_INVALID_ROW = "Endpoint: Invalid"


@pytest.mark.parametrize(
    ("endpoint", "row"),
    [
        ("gpu-box.lan:8080", _ADD_SCHEME_ROW),
        ("192.168.1.20:8080", _ADD_SCHEME_ROW),
        *[(key, _INVALID_ROW) for key in _DOTTED_KEY_ENDPOINTS],
    ],
)
def test_llamacpp_readiness_refuses_a_remote_endpoint_typed_without_a_scheme(
    endpoint, row
):
    """TASK-39042 AC#2: llama.cpp's ``normalize_llamacpp_base_url`` added
    ``http://`` first, so readiness read Ready for a remote host the save
    refuses ("Remote endpoint URLs must include an HTTP or HTTPS scheme.")."""
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsContextEstimate,
        build_console_settings_readiness,
        build_console_settings_summary_state,
    )

    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="m", base_url=endpoint
    )
    readiness = build_console_settings_readiness(
        settings,
        app_config={"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}},
        environ={},
    )

    assert _saved_endpoint("llama_cpp", endpoint) is None
    assert readiness.label != "Ready", readiness
    assert readiness.blocker == "endpoint_invalid", readiness
    # AC#3: the Chat settings Endpoint row never names the value. Rewritten on
    # purpose by TASK-39151 AC#3: an address is told to add its scheme and a
    # key-shaped value reads plain "Invalid", not "Not configured".
    assert build_console_settings_summary_state(
        settings, ConsoleSettingsContextEstimate(None, None, ""), readiness
    ).endpoint_row == row


@pytest.mark.parametrize("endpoint", _DOTTED_KEY_ENDPOINTS)
def test_a_schemeless_dotted_key_is_never_shown_as_a_host(endpoint):
    """TASK-39042 AC#3: a value without a scheme displayed as itself whenever
    ``http://<value>`` resolved, so a pasted ZAI-shaped key read as a host,
    lower-cased, in the display, the unsaved copy and the preview."""
    assert safe_endpoint_display(endpoint) == "invalid endpoint"
    assert local_discovery_module.endpoint_display(endpoint) == "invalid endpoint"
    copy = unsaved_endpoint_copy(endpoint, {"api_url": "http://127.0.0.1:9099"})
    assert "Selected endpoint: invalid endpoint." in copy
    assert defaults_module.parse_console_endpoint_preview(endpoint) is None
    assert defaults_module.format_console_endpoint_preview(endpoint) is None
    for text in (copy, safe_endpoint_display(endpoint)):
        assert _DOTTED_KEY.lower()[:12] not in text.lower()


@pytest.mark.parametrize(
    ("endpoint", "shown"),
    [
        ("http://gpu-box.lan/v1", "http://gpu-box.lan/v1"),
        ("https://api.example.com", "https://api.example.com"),
        ("localhost:8080", "http://localhost:8080"),
        ("127.0.0.1/v1", "http://127.0.0.1/v1"),
        ("[::1]:8080/v1", "http://[::1]:8080/v1"),
        ("local-service:9090/v1", "local-service:9090/v1"),
        ("gpu-box.lan:8080", "gpu-box.lan:8080"),
    ],
)
def test_values_with_a_scheme_a_port_or_a_loopback_host_display_as_before(
    endpoint, shown
):
    """TASK-39042 AC#4: only a schemeless, port-less remote value changed."""
    assert safe_endpoint_display(endpoint) == shown


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
        # TASK-39042: Chat settings read these as "Invalid URL".
        *((value, True) for value in _SCHEMELESS_LOOPBACK_ENDPOINTS),
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
        # TASK-39042, rewritten on purpose: the writer wrote the typed value
        # ("localhost:8080" with no scheme); it writes what **s** writes.
        assert _saved_endpoint(provider, endpoint) in (
            saved["api_settings"][provider].values()
        )
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "typed", "sent"),
    [
        ("vllm", "localhost:8080", "http://localhost:8080"),
        ("ollama", "127.0.0.1:8080", "http://127.0.0.1:8080"),
        ("local_vllm", "[::1]:8080", "http://[::1]:8080"),
        ("local_ollama", "localhost:8080", "http://localhost:8080"),
        ("local_llamafile", "localhost:8080", "http://localhost:8080"),
        ("strata", "localhost:8080", "http://localhost:8080"),
        ("qwencloud", "localhost:8080/v1", "http://localhost:8080/v1"),
    ],
)
async def test_a_schemeless_loopback_session_endpoint_reaches_the_adapter_with_http(
    provider, typed, sent
):
    """TASK-39042 review C1: readiness took ``localhost:8080`` for every
    URL-based provider, but the send passed it to the adapter as typed, and
    requests raised ``InvalidSchema`` (QwenCloud was blocked at send)."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.provider_setup_persistence import (
        persisted_endpoint_for_key,
        provider_endpoint_key,
    )

    key = provider_endpoint_key(provider)
    saved = persisted_endpoint_for_key(provider, key, typed)
    calls: list[dict] = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {provider: {key: saved, "model": "m", "api_key": "k"}}
        },
        environ={},
        chat_api_call_fn=adapter,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(
            provider=provider, explicit_model="m", base_url=typed, streaming=False
        )
    )

    assert resolution.ready, resolution.visible_copy
    [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "x"}])]
    assert calls[0]["api_base_url"] == sent


def test_a_chat_settings_draft_with_a_schemeless_loopback_endpoint_restores():
    """TASK-39042 review C1: the suspended-draft snapshot checked the session
    endpoint with ``validate_url``, so a draft holding ``localhost:8080``
    failed to restore and "Configure credential…" could not open."""
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsDraftSnapshot,
    )

    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(
            provider="vllm", model="m", base_url="localhost:8080"
        ),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-base-url": "localhost:8080"},
        provider_model_drafts={"vllm": "m"},
        provider_base_url_drafts={"vllm": "localhost:8080"},
        active_view="model",
        scroll_anchor=0,
        focus_control_id=None,
        disclosure_state={"advanced_generation": False, "connection_details": True},
    )
    restored = ConsoleSettingsDraftSnapshot.from_mapping(snapshot.to_mapping())

    assert restored is not None
    assert restored.settings.base_url == "localhost:8080"


@pytest.mark.parametrize(
    "typed",
    [
        "localhost:8080@gpu-box.lan:9000",
        "127.0.0.1:1@203.0.113.9",
        "[::1]:8080@evil.example",
    ],
)
def test_a_loopback_spelling_before_user_info_is_never_read_as_loopback(typed):
    """TASK-39042 review r11g-rev2: ``read_schemeless_loopback`` checked the
    whole netloc, so ``localhost:8080@gpu-box.lan:9000`` became
    ``http://localhost:8080@gpu-box.lan:9000``, whose host is gpu-box.lan."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )
    from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
        read_schemeless_loopback,
    )

    assert read_schemeless_loopback(typed) == typed
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="vllm", model="m", base_url=typed),
        app_config={"api_settings": {"vllm": {"api_url": "http://127.0.0.1:8000/v1"}}},
        environ={},
    )
    assert readiness.blocker == "endpoint_invalid", readiness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "configured", "sent"),
    [
        ("aphrodite", "localhost:8080/v1", "http://localhost:8080/v1"),
        ("oobabooga", "127.0.0.1:5000/v1", "http://127.0.0.1:5000/v1"),
        ("tabbyapi", "[::1]:5000/v1", "http://[::1]:5000/v1"),
        (
            "koboldcpp",
            "localhost:5001/api/v1/generate",
            "http://localhost:5001/api/v1/generate",
        ),
    ],
)
async def test_a_hand_edited_schemeless_loopback_endpoint_is_sent_as_readiness_reads_it(
    provider, configured, sent
):
    """TASK-39042 review r11g-rev2: these four adapters got no
    ``api_base_url`` and read their own config raw, so a hand-edited
    ``localhost:8080/v1`` that readiness accepts (and the gateway pinned as
    ``http://``) would reach the server without its scheme. The adapter takes
    ``api_base_url`` before its config, so the stub sees the URL it would use."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key

    key = provider_endpoint_key(provider)
    calls: list[dict] = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {
            "api_settings": {provider: {key: configured, "model": "m", "api_key": "k"}}
        },
        environ={},
        chat_api_call_fn=adapter,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider=provider, explicit_model="m", streaming=False)
    )

    assert resolution.ready, resolution.visible_copy
    [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "x"}])]
    assert calls[0].get("api_base_url") == sent


#: TASK-39151: remote llama.cpp endpoints typed without a scheme (save refuses).
_REMOTE_SCHEMELESS_LLAMACPP = ("gpu-box.lan:8080", "192.168.1.20:8080", "llama:8080")


def _llamacpp_source(source, endpoint):
    """Session settings and config holding ``endpoint`` in one llama.cpp place."""
    config = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}
    settings = ConsoleSessionSettings(provider="llama_cpp", model="m")
    if source == "session":
        settings = ConsoleSessionSettings(provider="llama_cpp", model="m", base_url=endpoint)
    elif source == "config":
        config["api_settings"]["llama_cpp"]["api_url"] = endpoint
    else:
        config["custom_endpoints"] = {
            "gpu": {"display_name": "GPU box", "family": "llama_cpp", "base_url": endpoint}
        }
        settings = ConsoleSessionSettings(provider="custom-ep:gpu", model="m")
    return settings, config


async def _resolve_llamacpp_send(settings, config, seen):
    """Resolve a send as Console does, through a transport that records URLs."""
    from tldw_chatbook.Chat.console_chat_controller import (
        build_console_provider_selection_from_settings,
    )

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_openai_models_payload("m"))
        return httpx.Response(200, json={"status": "ok"})

    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        environ={},
        http_client=_client(handler),
    )
    selection = build_console_provider_selection_from_settings(
        settings, app_config=config, workspace_context=None
    )
    return await gateway.resolve_for_send(selection)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", _REMOTE_SCHEMELESS_LLAMACPP)
@pytest.mark.parametrize("source", ["session", "config", "custom_entry"])
async def test_llamacpp_never_probes_a_remote_endpoint_typed_without_a_scheme(
    source, endpoint, monkeypatch
):
    """TASK-39151 AC#1: both ``normalize_llamacpp_base_url`` copies added
    ``http://`` to any value without a scheme, so the send-time resolution
    probed ``http://gpu-box.lan:8080/health`` and read Ready, whether the value
    came from the session, the config or a custom endpoint entry."""
    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    settings, config = _llamacpp_source(source, endpoint)
    seen: list[str] = []

    resolution = await _resolve_llamacpp_send(settings, config, seen)

    assert seen == []
    assert resolution.ready is False


@pytest.mark.parametrize("endpoint", _REMOTE_SCHEMELESS_LLAMACPP)
@pytest.mark.parametrize("place", ["api_settings", "console_override", "environment"])
def test_llamacpp_readiness_refuses_a_held_endpoint_typed_without_a_scheme(
    place, endpoint
):
    """TASK-39151 AC#2: with no session endpoint, readiness never looked at the
    endpoint the send would use, so a hand-edited ``gpu-box.lan:8080`` read
    Ready."""
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsContextEstimate,
        build_console_settings_readiness,
        build_console_settings_summary_state,
    )

    config = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}
    environ = {}
    if place == "api_settings":
        config["api_settings"]["llama_cpp"]["api_url"] = endpoint
    elif place == "console_override":
        config["console"] = {"llama_cpp_base_url_override": endpoint}
    else:
        environ["TLDW_CONSOLE_LLAMA_CPP_BASE_URL"] = endpoint

    settings = ConsoleSessionSettings(provider="llama_cpp", model="m")
    readiness = build_console_settings_readiness(
        settings, app_config=config, environ=environ
    )

    assert readiness.label != "Ready", readiness
    assert readiness.blocker == "endpoint_invalid", readiness
    # Review r13s-rev1: the Endpoint line agrees, not "Not configured".
    assert build_console_settings_summary_state(
        settings, ConsoleSettingsContextEstimate(None, None, ""), readiness
    ).endpoint_row == _ADD_SCHEME_ROW


@pytest.mark.asyncio
async def test_a_saved_chat_with_a_remote_schemeless_llamacpp_endpoint_sends_nothing(
    monkeypatch,
):
    """TASK-39151 AC#3: a chat saved before TASK-39042 can hold
    ``gpu-box.lan:8080`` as its session endpoint. Its next send probed that
    host as ``http://``, and its Endpoint row read "Not configured"."""
    import json
    from dataclasses import asdict

    from tldw_chatbook.Chat.console_provider_endpoints import (
        invalid_llamacpp_base_url_copy,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsContextEstimate,
        build_console_settings_readiness,
        build_console_settings_summary_state,
        parse_persisted_console_session_settings,
    )

    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    stored = ConsoleSessionSettings(
        provider="llama_cpp", model="m", base_url="gpu-box.lan:8080"
    )
    settings = parse_persisted_console_session_settings(
        json.dumps({"console_session_settings": {"version": 1, **asdict(stored)}})
    )
    assert settings == stored
    config = {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}}
    seen: list[str] = []

    resolution = await _resolve_llamacpp_send(settings, config, seen)

    assert seen == []
    assert resolution.ready is False
    # TASK-40300 review r14ep-rev2 M-b: the copy names the contract's reason.
    assert resolution.visible_copy == invalid_llamacpp_base_url_copy(
        REMOTE_ENDPOINT_SCHEME_REQUIRED
    )
    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    row = build_console_settings_summary_state(
        settings, ConsoleSettingsContextEstimate(None, None, ""), readiness
    ).endpoint_row
    assert row == _ADD_SCHEME_ROW


_NO_LONGER_CONFIGURED = (
    "Provider blocked: this custom endpoint is no longer configured. "
    "Restore its entry or choose another provider before sending."
)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["removed", "skipped"])
async def test_readiness_for_a_missing_custom_endpoint_says_to_restore_its_entry(
    entry, monkeypatch
):
    """TASK-40302 AC#1: a chat on a ``custom-ep:`` id whose entry was removed,
    or skipped at load (TASK-39151's schemeless remote llama.cpp entry), read
    "Provider blocked: 'custom_ep:gpu' is not available in Console yet." with
    the label "Unknown": a missing feature and a mangled internal id, while
    the send says to restore the entry."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    settings, config = _llamacpp_source("custom_entry", "gpu-box.lan:8080")
    if entry == "removed":
        del config["custom_endpoints"]

    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    resolution = await _resolve_llamacpp_send(settings, config, [])

    # TASK-40300 review r14ep-rev2 M-d: an entry refused at load says why and
    # where to fix it; one that is gone says to restore it.
    expected = _NO_LONGER_CONFIGURED if entry == "removed" else (
        "Provider blocked: this custom endpoint is not loaded: "
        f"{REMOTE_ENDPOINT_SCHEME_REQUIRED} Fix it in Settings ▸ Custom endpoints."
    )
    assert readiness.detail == resolution.visible_copy == expected
    assert readiness.native_send_supported is False
    assert readiness.recovery_action == "select_provider"
    shown = (readiness.label, readiness.detail, readiness.provider_display_name)
    assert not any("custom_ep" in text or "custom-ep" in text for text in shown), shown


def test_a_provider_console_does_not_support_keeps_its_copy():
    """TASK-40302 AC#2."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="future_provider", model="m"),
        app_config={"api_settings": {}},
        environ={},
    )

    assert readiness.blocker == "provider_unsupported"
    assert readiness.detail == (
        "Provider blocked: 'future_provider' is not available in Console yet. "
        "Choose a supported provider."
    )


@pytest.mark.parametrize(
    ("endpoint", "row"),
    [
        *[
            (address, _ADD_SCHEME_ROW)
            for address in (
                "gpu-box.lan:8080/v1",
                "192.168.1.20",
                "[fd12::9]:8080",
                "LOCALHOST:8080",
                "127.0.0.2:8080",
            )
        ],
        *[
            (value, _INVALID_ROW)
            for value in (
                "sk-proj-abcdefghijklmnopqrstuvwx",
                "gpu-box.lan",
                "ftp://gpu-box.lan:8080",
                # Review r13s-rev2: user info or a query is never told to add
                # a scheme; the contract refuses it for that, not the scheme.
                "user:pw@gpu-box.lan:8080",
                ":secretpw@gpu-box.lan:8080",
                "gpu-box.lan:8080?key=sk-x",
            )
        ],
    ],
)
def test_the_endpoint_line_tells_only_an_address_to_add_a_scheme(endpoint, row):
    """TASK-39151 review r13s-rev1: the Endpoint line told any value without a
    scheme to add one, a pasted key included, which would then read Ready and
    be probed as a host. Only a host and port, or an IP literal, that the
    contract refuses for lacking a scheme is told to add one, and adding it is
    then enough (``LOCALHOST:8080`` and ``127.0.0.2:8080`` too: only the exact
    loopback spellings may leave it out)."""
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsContextEstimate,
        build_console_settings_readiness,
        build_console_settings_summary_state,
    )

    settings = ConsoleSessionSettings(provider="llama_cpp", model="m", base_url=endpoint)
    readiness = build_console_settings_readiness(
        settings,
        app_config={"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}},
        environ={},
    )

    assert readiness.blocker == "endpoint_invalid", readiness
    assert build_console_settings_summary_state(
        settings, ConsoleSettingsContextEstimate(None, None, ""), readiness
    ).endpoint_row == row
    if row == _ADD_SCHEME_ROW:
        assert resolve_provider_endpoint("llama_cpp", f"http://{endpoint}").errors == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("typed", "base"),
    [
        ("localhost:8080", "http://localhost:8080"),
        ("127.0.0.1:8080/v1", "http://127.0.0.1:8080"),
        ("[::1]:8080", "http://[::1]:8080"),
        ("http://gpu-box.lan:8080", "http://gpu-box.lan:8080"),
        ("https://gpu-box.lan:8443/v1", "https://gpu-box.lan:8443"),
    ],
)
@pytest.mark.parametrize("source", ["session", "config", "custom_entry"])
async def test_llamacpp_loopback_or_schemed_endpoints_resolve_as_before(
    source, typed, base, monkeypatch
):
    """TASK-39151 AC#4: only a remote value without a scheme changed."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    settings, config = _llamacpp_source(source, typed)
    seen: list[str] = []

    resolution = await _resolve_llamacpp_send(settings, config, seen)

    assert seen == [f"{base}/health"]
    assert resolution.ready is True, resolution.visible_copy
    assert resolution.base_url == base
    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    assert readiness.blocker is None, readiness


@pytest.mark.parametrize(
    ("endpoint", "sent"),
    [
        *[(address, None) for address in _REMOTE_SCHEMELESS_LLAMACPP],
        ("localhost:8080/v1", "http://localhost:8080"),
        ("http://gpu-box.lan:8080/v1", "http://gpu-box.lan:8080"),
    ],
)
def test_chat_with_llama_refuses_a_remote_endpoint_typed_without_a_scheme(
    endpoint, sent, monkeypatch
):
    """TASK-39151 review r13s-rev1: outside Console, ``chat_with_llama`` kept
    a hand-edited ``gpu-box.lan:8080`` schemeless and ``requests`` failed it as
    a network error ("No connection adapters were found"). It is a
    configuration error, raised before anything is sent."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    calls: list[dict] = []
    monkeypatch.setattr(
        local_calls,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: calls.append(kwargs) or "ok",
    )
    # No config read: the endpoint comes from the argument.
    monkeypatch.setattr(
        local_calls,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={"api_settings": {}}),
    )
    messages = [{"role": "user", "content": "x"}]

    if sent is None:
        with pytest.raises(ChatConfigurationError, match=r"http:// or https://"):
            local_calls.chat_with_llama(
                messages, api_url=endpoint, model="m", api_key_resolved=True
            )
        assert calls == []
    else:
        local_calls.chat_with_llama(
            messages, api_url=endpoint, model="m", api_key_resolved=True
        )
        assert [call["api_base_url"] for call in calls] == [sent]


@pytest.mark.parametrize(
    "endpoint", ["ftp://user:hunter2@gpu-box.lan:8080", "htp://gpu-box.lan:8080/v1"]
)
@pytest.mark.parametrize(
    "provider", ["llama_cpp", "local_llamacpp", "local_llamafile", "strata", "polystrata"]
)
def test_chat_with_llama_refuses_a_non_http_scheme_without_echoing_it(
    provider, endpoint, monkeypatch
):
    """TASK-39151 review r13s-rev2: M5's guard checked only for "://", so an
    ``ftp://`` or ``htp://`` endpoint reached ``requests`` and failed as a 503
    whose message echoed the whole URL, user info included. Every provider
    that dispatches to ``chat_with_llama`` now refuses it as a configuration
    error that names no part of the value, before anything is sent."""
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    assert API_CALL_HANDLERS[provider] is local_calls.chat_with_llama
    calls: list[dict] = []
    monkeypatch.setattr(
        local_calls,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: calls.append(kwargs) or "ok",
    )
    monkeypatch.setattr(
        local_calls,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={"api_settings": {}}),
    )

    with pytest.raises(ChatConfigurationError, match=r"http:// or https://") as caught:
        local_calls.chat_with_llama(
            [{"role": "user", "content": "x"}],
            api_url=endpoint,
            model="m",
            provider_name=provider,
            api_key_resolved=True,
        )

    assert calls == []
    for part in ("hunter2", "user:", "gpu-box", "ftp", "htp"):
        assert part not in str(caught.value)


#: TASK-40300: one value per rule the endpoint contract refuses that Console's
#: send and readiness still accepted (``validate_url`` and the scheme rule).
_CONTRACT_REFUSED_BASES = {
    "zone-id": "http://[fe80::1%25en0]:8000",
    "leading-hyphen": "http://-bad.corp:8000",
    "digit-host": "http://127.1:8000",
    "idn": "http://bücher.example:8000",
    "double-dot": "http://localhost..:8000",
    "query": "http://127.0.0.1:8000?mode=fast",
    "fragment": "http://127.0.0.1:8000#part",
    # Review r14ep-rev1 M2.
    "user-info": "http://user:pw@127.0.0.1:8000",
}
#: Values the contract accepts, which resolve as before (AC#3).
_CONTRACT_ACCEPTED_BASES = (
    "http://gpu-box.lan:8000",
    "localhost:8000",
    "http://kobold_cpp:8000",
    "http://[fe80::1]:8000",
    "http://[fd12:3456::9]:8000",
)
#: The providers TASK-40300 names; a custom entry exists for three families.
_ENDPOINT_PROVIDERS = ("vllm", "ollama", "custom", "openai", "koboldcpp", "aphrodite", "llama_cpp")
_ENTRY_FAMILIES = {"custom": "openai_compatible", "ollama": "ollama", "llama_cpp": "llama_cpp"}
_SAFE_LOOPBACK = "http://127.0.0.1:9000"


def _with_path(provider, base):
    """``base`` with the provider's route, before any query or fragment."""
    path = "/api/v1/generate" if provider == "koboldcpp" else "/v1"
    if provider == "llama_cpp":
        path = ""
    cut = min((base.index(c) for c in "?#" if c in base), default=len(base))
    return base[:cut] + path + base[cut:]


def _endpoint_case(provider, source, endpoint):
    """Settings, config and a send selection holding ``endpoint`` in one place."""
    from tldw_chatbook.Chat.provider_setup_persistence import provider_endpoint_key

    key = provider_endpoint_key(provider)
    table = {key: _with_path(provider, _SAFE_LOOPBACK), "model": "m"}
    if provider == "openai":
        table["api_key"] = "test-key-not-real"
    config = {"api_settings": {provider: table}}
    settings = ConsoleSessionSettings(provider=provider, model="m")
    pinned = False
    if source == "session":
        settings = ConsoleSessionSettings(provider=provider, model="m", base_url=endpoint)
        pinned = provider != "llama_cpp"  # the live-only path past the unsaved guard
    elif source == "config":
        table[key] = endpoint
    else:
        config["custom_endpoints"] = {
            "box": {
                "display_name": "Box",
                "family": _ENTRY_FAMILIES[provider],
                "base_url": endpoint,
            }
        }
        settings = ConsoleSessionSettings(provider="custom-ep:box", model="m")
    return settings, config, pinned


async def _resolve_endpoint_send(settings, config, pinned, seen):
    """Resolve a send as Console does; every request lands in ``seen``."""
    from tldw_chatbook.Chat.console_chat_controller import (
        build_console_provider_selection_from_settings,
    )

    def handler(request):
        seen.append(str(request.url))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=_openai_models_payload("m"))
        return httpx.Response(200, json={"status": "ok"})

    def adapter(**kwargs):
        seen.append(f"adapter:{kwargs.get('api_base_url') or kwargs.get('api_url')}")
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        environ={},
        http_client=_client(handler),
        chat_api_call_fn=adapter,
    )
    selection = build_console_provider_selection_from_settings(
        settings, app_config=config, workspace_context=None
    )
    if pinned:
        selection = ConsoleProviderSelection(
            provider=selection.provider,
            base_url=settings.base_url,
            explicit_model="m",
            streaming=False,
            base_url_is_pinned=True,
        )
    resolution = await gateway.resolve_for_send(selection)
    if resolution.ready and not resolution.execution_key.startswith("llama"):
        [c async for c in gateway.stream_chat(resolution, [{"role": "user", "content": "x"}])]
    return resolution


def _endpoint_cases():
    for provider in _ENDPOINT_PROVIDERS:
        for source in ("session", "config", "custom_entry"):
            if source == "custom_entry" and provider not in _ENTRY_FAMILIES:
                continue
            yield provider, source


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_CONTRACT_REFUSED_BASES))
@pytest.mark.parametrize(("provider", "source"), list(_endpoint_cases()))
async def test_an_endpoint_the_contract_refuses_is_never_ready_or_contacted(
    provider, source, shape, monkeypatch
):
    """TASK-40300 AC#1/AC#2/AC#4: Console decided endpoints with
    ``validate_url`` and the scheme rule only, so a zone id, a leading-hyphen
    host, ``127.1``, an IDN, ``localhost..``, a query or a fragment from the
    session, the config or a custom entry read Ready and was probed (llama.cpp)
    or sent to (the rest), and a refused custom entry fell back to its raw
    value. The save, Discover and the display already refused all of them."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    endpoint = _with_path(provider, _CONTRACT_REFUSED_BASES[shape])
    assert resolve_provider_endpoint(provider, endpoint).errors, endpoint
    settings, config, pinned = _endpoint_case(provider, source, endpoint)
    seen: list[str] = []

    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    resolution = await _resolve_endpoint_send(settings, config, pinned, seen)

    assert seen == []
    assert resolution.ready is False
    assert readiness.native_send_supported is False
    if source == "custom_entry":
        assert resolution.base_url != endpoint  # AC#4: no raw fallback
        # Review r14ep-rev2 M-d: refused at load, so it says why and where.
        assert readiness.detail == resolution.visible_copy
        assert resolution.visible_copy.startswith(
            "Provider blocked: this custom endpoint is not loaded: Endpoint"
        ) and resolution.visible_copy.endswith("Fix it in Settings ▸ Custom endpoints.")
    else:
        assert readiness.label == "Invalid URL", readiness


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "source", "base"),
    [
        (provider, source, base)
        for provider, source in _endpoint_cases()
        for base in _CONTRACT_ACCEPTED_BASES
        # Unchanged: the registry refuses any value typed without a scheme
        # for its openai_compatible and ollama families.
        if not (source == "custom_entry" and provider != "llama_cpp" and "://" not in base)
    ],
)
async def test_an_endpoint_the_contract_accepts_resolves_as_before(
    provider, source, base, monkeypatch
):
    """TASK-40300 AC#3: schemed hosts, schemeless loopback spellings, compose
    names and zone-less link-local or ULA literals still resolve and send."""
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    monkeypatch.delenv("TLDW_CONSOLE_LLAMA_CPP_BASE_URL", raising=False)
    endpoint = _with_path(provider, base)
    assert resolve_provider_endpoint(provider, endpoint).errors == ()
    settings, config, pinned = _endpoint_case(provider, source, endpoint)
    seen: list[str] = []

    resolution = await _resolve_endpoint_send(settings, config, pinned, seen)

    assert resolution.ready is True, resolution.visible_copy
    assert seen, "the probe or the send was never made"
    if not pinned:  # a pinned session endpoint is not the saved one
        readiness = build_console_settings_readiness(
            settings, app_config=config, environ={}
        )
        assert readiness.blocker is None, readiness


@pytest.mark.parametrize(
    "endpoint",
    [
        "gpu-box.lan:8080",
        "192.168.1.20:8080/v1",
        "http://localhost..:5001",
        "http://example.com..:5001/v1",
        "http://127.1:5001",
        "https://bücher.example/v1",
    ],
)
def test_project_instructions_give_what_the_send_refuses_the_refused_identity(endpoint):
    """TASK-40300 AC#5: ``_parse_endpoint`` added ``http://`` to any schemeless
    endpoint and ``_canonical_hostname`` stripped every trailing dot, so a
    remote schemeless endpoint and a ``..`` host took the clean host's
    identity and consent key. Like the send, both are refused now, as are
    ``127.1`` and an IDN (TASK-39150 AC#4 kept those only while the send
    accepted them)."""
    assert _project_instruction_endpoint_callers(endpoint) == (
        _project_instruction_endpoint_callers("http://-bad.corp:5001/v1")
    )


@pytest.mark.parametrize(
    ("endpoint", "identity"),
    [
        ("http://localhost.:5001", "http://localhost:5001"),
        ("https://example.com./v1", "https://example.com/v1"),
        ("localhost:5001/v1", "http://localhost:5001/v1"),
        ("[::1]:5001", "http://[::1]:5001"),
    ],
)
def test_project_instructions_fold_one_trailing_dot_and_read_loopback_as_http(
    endpoint, identity
):
    """TASK-40300 AC#5: exactly one trailing dot folds, as in the contract,
    and the loopback spellings typed without a scheme read as ``http://``."""
    seen = _project_instruction_endpoint_callers(endpoint)

    assert seen["identity"] == identity
    assert len(seen["notice_key"]) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:8080?token=endpoint-secret", "localhost:8080/v1#endpoint-secret"],
)
async def test_a_llamacpp_send_never_drops_a_query_to_reach_the_server(base_url):
    """TASK-40300 AC#2: ``normalize_llamacpp_base_url`` rebuilt the URL with an
    empty query, so a send for a base URL the contract refuses went to the
    server anyway. The send refuses it before any request, naming no part of
    it."""
    from Tests.Chat.test_console_provider_gateway import _capture_on_prepared_request
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleRequestRoute,
        ConsoleTraceCaptureMode,
    )
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    seen: list[str] = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    resolution = ConsoleProviderResolution(
        provider="llama_cpp",
        base_url=base_url,
        model="m",
        ready=True,
        execution_key="llama_cpp",
        streaming=False,
    )
    bundles: list = []
    async with _client(handler) as client:
        gateway = ConsoleProviderGateway(http_client=client, trace_shadow_sink=bundles.append)
        prepared = _capture_on_prepared_request(gateway, resolution)
        with pytest.raises((ValueError, ChatConfigurationError)) as caught:
            [
                chunk
                async for chunk in gateway.stream_chat(
                    resolution,
                    prepared,
                    route=ConsoleRequestRoute.FRESH,
                    capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
                )
            ]

    assert seen == []
    assert "endpoint-secret" not in str(caught.value)
    # Review r14ep-rev1 M3: refused before the trace shadow is built.
    assert bundles == []


@pytest.mark.parametrize("shape", sorted(_CONTRACT_REFUSED_BASES))
def test_chat_with_llama_refuses_what_the_contract_refuses_without_echoing_it(
    shape, monkeypatch
):
    """TASK-40300: ``chat_with_llama`` shares the llama.cpp normalizer, which
    no longer drops a query or fragment, so it decides by the contract too;
    nothing is sent and the message names no part of the value."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    calls: list[dict] = []
    monkeypatch.setattr(
        local_calls,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: calls.append(kwargs) or "ok",
    )
    monkeypatch.setattr(
        local_calls,
        "get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={"api_settings": {}}),
    )
    endpoint = _CONTRACT_REFUSED_BASES[shape]

    with pytest.raises(ChatConfigurationError) as caught:
        local_calls.chat_with_llama(
            [{"role": "user", "content": "x"}],
            api_url=endpoint,
            model="m",
            provider_name="llama_cpp",
            api_key_resolved=True,
        )

    assert calls == []
    for part in (
        "fe80", "bad.corp", "127.1", "cher", "localhost", "mode=fast", "part", "pw@"
    ):
        assert part not in str(caught.value)


def _dispatch_resolution(provider, base_url, **fields):
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution

    return ConsoleProviderResolution(
        provider=provider,
        base_url=base_url,
        model="m",
        ready=True,
        readiness_key=fields.pop("readiness_key", provider),
        execution_key=fields.pop("execution_key", provider),
        streaming=False,
        **fields,
    )


def _recording_gateway(seen):
    def adapter(**kwargs):
        seen.append(f"adapter:{kwargs.get('api_base_url')}")
        return {"choices": [{"message": {"content": "ok"}}]}

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    return ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {}},
        environ={},
        http_client=_client(handler),
        chat_api_call_fn=adapter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_CONTRACT_REFUSED_BASES))
@pytest.mark.parametrize("provider", _ENDPOINT_PROVIDERS)
async def test_a_dispatch_never_reaches_an_endpoint_the_contract_refuses(provider, shape):
    """TASK-40300 review r14ep-rev1 I1: only ``resolve_for_send`` checked the
    endpoint, and an agent sub-call's overlay (a frozen spawn target or a run
    snapshot) replaces ``base_url`` afterwards, so ``127.1``, ``-bad.corp`` or
    ``?token=`` reached the adapter for every provider but llama.cpp. The
    stream and the auxiliary dispatch both refuse, with the contract's reason."""
    from tldw_chatbook.Chat.console_provider_gateway import AuxiliaryCompletionRequest

    endpoint = _with_path(provider, _CONTRACT_REFUSED_BASES[shape])
    seen: list[str] = []
    gateway = _recording_gateway(seen)
    resolution = _dispatch_resolution(provider, endpoint)

    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    with pytest.raises(ChatConfigurationError) as streamed:
        [c async for c in gateway.stream_chat(resolution, [{"role": "user", "content": "x"}])]
    with pytest.raises(ChatConfigurationError) as auxiliary:
        await gateway.complete_auxiliary(
            AuxiliaryCompletionRequest(
                resolution=resolution,
                messages=({"role": "user", "content": "x"},),
                response_format=None,
                max_output_tokens=8,
            )
        )

    assert seen == []
    for caught in (streamed, auxiliary):
        assert caught.value.status_code is None  # nothing reached the provider
        assert "Provider blocked" in str(caught.value)
        for part in ("fe80", "bad.corp", "127.1", "cher", "mode=fast", "part", "pw@"):
            assert part not in str(caught.value)


@pytest.mark.parametrize("shape", sorted(_CONTRACT_REFUSED_BASES))
@pytest.mark.parametrize("provider", ["vllm", "openai", "llama_cpp"])
def test_a_spawn_target_never_freezes_an_endpoint_the_contract_refuses(provider, shape):
    """TASK-40300 review r14ep-rev1 I1: ``freeze_spawn_target`` froze any
    endpoint, which the run's dispatch overlay then sent to."""
    from tldw_chatbook.Agents.agent_routing import (
        RoutingError,
        SpawnTarget,
        freeze_spawn_target,
    )

    endpoint = _with_path(provider, _CONTRACT_REFUSED_BASES[shape])
    target = SpawnTarget(provider, "m", endpoint, (), "override")

    with pytest.raises(RoutingError) as caught:
        freeze_spawn_target({"api_settings": {}}, target)

    assert caught.value.code == "provider_not_ready"
    assert caught.value.level == "override"
    frozen = freeze_spawn_target(
        {"api_settings": {}}, SpawnTarget(provider, "m", _with_path(provider, _SAFE_LOOPBACK), (), "override")
    )
    assert frozen.base_url == _with_path(provider, _SAFE_LOOPBACK)


#: TASK-40300, owner decision 2026-11-13 ("Accept them for custom endpoints"):
#: documented OpenAI-compatible bases whose v1 is not the last path segment.
_CUSTOM_HOSTED_BASES = (
    "https://api.deepinfra.com/v1/openai",
    "https://gateway.ai.cloudflare.com/v1/acct123/gw1/compat",
)


def _entry_config(base_url, family="openai_compatible", slug="box"):
    return {
        "api_settings": {},
        "custom_endpoints": {
            slug: {"display_name": "Box", "family": family, "base_url": base_url}
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", _CUSTOM_HOSTED_BASES)
async def test_an_openai_compatible_custom_endpoint_keeps_a_v1_that_is_not_last(base_url):
    """TASK-40300 owner decision 2026-11-13: the registry, readiness, the send,
    the dispatch and agent routing all read an OpenAI-compatible custom
    endpoint as its custom-hosted engine does, so these bases load, read
    Ready and send to ``<base>/chat/completions``; the generic ``custom``
    provider still refuses them (its reading has no API base of its own)."""
    from tldw_chatbook.Agents.agent_routing import SpawnTarget, freeze_spawn_target
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )
    from tldw_chatbook.Chat.custom_endpoint_registry import load_custom_endpoints

    config = _entry_config(base_url)
    assert list(load_custom_endpoints(config)) == ["box"]
    settings = ConsoleSessionSettings(provider="custom-ep:box", model="m")
    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    assert readiness.blocker is None, readiness
    seen: list[str] = []
    gateway = _recording_gateway(seen)
    gateway._config_provider = lambda: config
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="custom-ep:box", explicit_model="m", streaming=False)
    )
    assert resolution.ready is True, resolution.visible_copy
    [c async for c in gateway.stream_chat(resolution, [{"role": "user", "content": "x"}])]
    assert seen and seen[0].startswith(f"adapter:{base_url}"), seen
    freeze_spawn_target(config, SpawnTarget("custom-ep:box", "m", base_url, (), "override"))

    assert resolve_provider_endpoint("custom", base_url).errors == (
        "Endpoint path has ambiguous API suffixes.",
    )


@pytest.mark.parametrize(
    "path",
    [
        "/v1/completions",
        # Review r14ep-rev3: the relaxed reading took these routes as bases.
        "/v1/embeddings",
        "/v1/responses",
        "/v1/chat",
        "/v1/chat/completions/extra",
        "/v1/openai/embeddings",
    ],
)
def test_an_api_route_is_not_a_custom_endpoint_base(path):
    """TASK-40300 owner decision 2026-11-13: a route such as the legacy
    ``/v1/completions`` is not an API base (a chat would post to
    ``/v1/completions/chat/completions``), so it stays refused, with the
    contract's reason."""
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        custom_endpoint_refusals,
        load_custom_endpoints,
        validate_entry,
    )

    config = _entry_config(f"http://localhost:8000{path}")

    assert load_custom_endpoints(config) == {}
    assert custom_endpoint_refusals(config) == {
        "box": "Endpoint path has ambiguous API suffixes."
    }
    assert validate_entry("Box", "openai_compatible", f"http://localhost:8000{path}") == [
        "Endpoint path has ambiguous API suffixes."
    ]


def test_a_refused_custom_endpoint_is_listed_in_settings_with_its_reason():
    """TASK-40300 owner decision 2026-11-13: an entry still refused at load was
    only logged, so it vanished from Settings and could not be fixed there.
    It is listed after the valid ones with the contract's reason, which never
    names the value, and Edit and Delete can reach it."""
    from tldw_chatbook.Chat.custom_endpoint_registry import load_custom_endpoints
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        custom_endpoint_rows,
    )

    config = _entry_config("http://127.0.0.1:8000/v1")
    config["custom_endpoints"]["leaky"] = {
        "display_name": "Leaky",
        "family": "openai_compatible",
        "base_url": "http://127.0.0.1:8000/v1?token=not-a-real-secret",
    }

    rows = custom_endpoint_rows(config)

    assert [row.key for row in rows] == ["custom-ep:box", "custom-ep:leaky"]
    assert rows[1].label == "Leaky"
    assert rows[1].value == (
        "OpenAI-compatible · not loaded: Endpoint URL must not include a query. · 0 models"
    )
    assert "not-a-real-secret" not in rows[1].value
    assert list(load_custom_endpoints(config)) == ["box"]
    assert list(load_custom_endpoints(config, include_refused=True)) == ["box", "leaky"]


def _summary_calls(provider, api_settings, **extra):
    """Resolve and run one approval summary; return it and the calls made."""
    from tldw_chatbook.Chat import permission_summary_service as summary

    config = {
        "api_settings": api_settings,
        "permission_summary": {"mode": "always", "provider": provider, "model": "m"},
        **extra,
    }
    calls: list[dict] = []
    resolution = summary.resolve_permission_summary(config, environ={})
    out = summary.summarize_pending_round(
        resolution,
        [{"role": "user", "content": "hi"}],
        [{"tool_name": "t", "server_label": "", "description": "", "arguments_summary": ""}],
        call_fn=lambda **kwargs: calls.append(kwargs)
        or {"choices": [{"message": {"content": "ok"}}]},
    )
    return resolution, out, calls


#: Review r14ep-rev3 I-1: the table each adapter reads its endpoint from. The
#: summary looked settings up by the dispatch name, found nothing for these
#: but vLLM, and left the call unpinned, so the adapter sent to its own value.
_SUMMARY_ADAPTER_TABLES = (
    ("vllm", "vllm", "api_url"),
    ("local_llm", "local-llm", "api_url"),
    ("local_llm", "local-llm", "api_ip"),
    ("custom-openai-api", "custom", "api_url"),
    ("oobabooga", "ooba_api", "api_ip"),
)


@pytest.mark.parametrize("shape", sorted(_CONTRACT_REFUSED_BASES))
@pytest.mark.parametrize(("provider", "table", "key"), _SUMMARY_ADAPTER_TABLES)
def test_the_approval_summary_never_calls_an_endpoint_the_contract_refuses(
    provider, table, key, shape
):
    """TASK-40300 review r14ep-rev2 I2: the approval-round summarizer checked
    only the provider's credential, then called ``chat_api_call`` with no
    endpoint, so the adapter read and POSTed to a refused configured value."""
    endpoint = _with_path("vllm", _CONTRACT_REFUSED_BASES[shape])

    resolution, out, calls = _summary_calls(
        provider, {table: {key: endpoint, "model": "m"}}
    )

    assert resolution.active is False
    assert out is None and calls == []


@pytest.mark.parametrize("host", ["127.1", "-bad.corp", "bücher.example", "localhost.."])
def test_the_approval_summary_never_calls_an_mlx_host_the_contract_refuses(host):
    """Review r14ep-rev3 I-1: MLX-LM builds its endpoint from ``host`` and
    ``port``, which the summary never read, so ``host = "127.1"`` was sent to."""
    resolution, out, calls = _summary_calls("mlx_lm", {"mlx_lm": {"host": host}})

    assert resolution.active is False
    assert out is None and calls == []


@pytest.mark.parametrize(
    ("provider", "api_settings", "extra", "pinned"),
    [
        ("vllm", {"vllm": {"api_url": "http://127.0.0.1:8000/v1"}}, {}, "http://127.0.0.1:8000/v1"),
        # Review r14ep-rev3 I-1: the adapter's own table, checked and pinned.
        ("local_llm", {"local-llm": {"api_ip": "http://127.0.0.1:8000/v1"}}, {}, "http://127.0.0.1:8000/v1"),
        ("oobabooga", {"ooba_api": {"api_ip": "http://127.0.0.1:5000/v1"}}, {}, "http://127.0.0.1:5000/v1"),
        ("mlx_lm", {"mlx_lm": {"port": 9000}}, {}, "http://127.0.0.1:9000/v1"),
        (
            "custom-openai-api-2",
            {},
            {"custom_openai_api_2": {"api_ip": "http://127.0.0.1:5000/v1/chat/completions"}},
            "http://127.0.0.1:5000/v1/chat/completions",
        ),
        # Nothing configured: the adapter's own default, never unpinned.
        ("openai", {"openai": {"api_key": "sk-not-a-real-key-0000"}}, {}, "https://api.openai.com/v1"),
        ("together", {"together": {"api_key": "not-a-real-key-0000"}}, {}, "https://api.together.xyz/v1"),
        # Review r14ep-rev4 M-2: chat_with_google never reads a top-level [google].
        (
            "google",
            {"google": {"api_key": "not-a-real-key-0000"}},
            {"google": {"api_base_url": "http://127.0.0.1:9/v1beta"}},
            "https://generativelanguage.googleapis.com/v1beta",
        ),
    ],
)
def test_the_approval_summary_pins_the_endpoint_it_checked(provider, api_settings, extra, pinned):
    """TASK-40300 review r14ep-rev2 I2: an accepted endpoint is the one the
    call is pinned to, not whatever the adapter reads (review r14ep-rev3 I-1:
    always pinned)."""
    resolution, _out, calls = _summary_calls(provider, api_settings, **extra)

    assert resolution.active is True
    assert [call["api_base_url"] for call in calls] == [pinned]


@pytest.mark.parametrize(
    ("user", "pinned"),
    [
        ({}, "http://127.0.0.1:8080/v1"),
        ({"api_settings": {"local_mlx_lm": {"port": 9000}}}, "http://127.0.0.1:9000/v1"),
    ],
)
def test_the_approval_summary_pins_mlx_host_and_port_not_its_api_url(user, pinned):
    """Review r14ep-rev4 C-1: the summary pinned ``[api_settings.local_mlx_lm]
    api_url``, the shipped placeholder MLX-LM ignores."""
    from tldw_chatbook.config import _merge_user_config_over_shipped_defaults

    api_settings = _merge_user_config_over_shipped_defaults(user)["api_settings"]
    resolution, _out, calls = _summary_calls("local_mlx_lm", api_settings)

    assert resolution.active is True
    assert [call["api_base_url"] for call in calls] == [pinned]


def test_the_approval_summary_is_off_with_no_endpoint_to_send_to():
    """Review r14ep-rev3 I-1: with nothing configured and no built-in endpoint
    there is nothing checked to pin, so no summary is requested."""
    resolution, out, calls = _summary_calls("vllm", {})

    assert resolution.active is False
    assert out is None and calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "ready", "pinned"),
    [
        ("http://127.1:8000/v1", False, None),
        ("http://127.0.0.1:8000/v1?token=not-a-real-secret", False, None),
        ("http://127.0.0.1:8000/v1", True, "http://127.0.0.1:8000/v1"),
    ],
)
async def test_the_image_context_call_never_reaches_a_refused_endpoint(
    base_url, ready, pinned
):
    """TASK-40300 review r14ep-rev2 I2: ``/image``'s conversation-context call
    took only the resolution's execution key, so the adapter read its own
    config; it now refuses what a dispatch refuses and pins the endpoint."""
    from Tests.Chat.test_console_generation_actions import (
        _bare_generation_screen,
        _imagegen_cfg,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    screen = _bare_generation_screen(ConsoleChatStore())
    screen._build_console_provider_selection = lambda: "selection"

    class _FakeGateway:
        async def resolve_for_send(self, _selection):
            return _dispatch_resolution("vllm", base_url)

    screen._ensure_console_provider_gateway = lambda: _FakeGateway()

    options = await screen._image._console_generate_image_llm_context_options(
        _imagegen_cfg()
    )

    assert options.provider_ready is ready
    if ready:
        assert options.api_base_url == pinned
        from dataclasses import replace

        from tldw_chatbook.Chat.console_generate_image import (
            compose_llm_context_prompt,
        )

        calls: list[dict] = []
        compose_llm_context_prompt(
            [("user", "a red fox at dawn")],
            replace(
                options,
                chat_call=lambda **kwargs: calls.append(kwargs)
                or {"choices": [{"message": {"content": "a red fox"}}]},
            ),
        )
        assert [call.get("api_base_url") for call in calls] == [pinned]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pasted",
    [
        "https://api.deepinfra.com/v1/openai/models",
        "https://api.deepinfra.com/v1/openai/chat/completions",
    ],
)
async def test_a_relaxed_custom_endpoint_pasted_with_a_route_sends_to_its_base(pasted):
    """TASK-40300 review r14ep-rev2 M-a: a relaxed entry's raw value reached the
    engine, so a base pasted with ``/models`` sent to
    ``.../models/chat/completions``. The entry is read as its engine reads it."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection

    config = _entry_config(pasted)
    seen: list[str] = []
    gateway = _recording_gateway(seen)
    gateway._config_provider = lambda: config
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="custom-ep:box", explicit_model="m", streaming=False)
    )
    assert resolution.ready is True, resolution.visible_copy
    [c async for c in gateway.stream_chat(resolution, [{"role": "user", "content": "x"}])]

    assert [s.removesuffix("/chat/completions") for s in seen] == [
        "adapter:https://api.deepinfra.com/v1/openai"
    ]


@pytest.mark.asyncio
async def test_llamacpp_readiness_and_send_name_the_contracts_reason():
    """TASK-40300 review r14ep-rev2 M-b: llama.cpp readiness and the send said
    only "invalid llama.cpp base URL", while the guide promised the reason."""
    from tldw_chatbook.Chat.console_provider_endpoints import (
        invalid_llamacpp_base_url_copy,
    )
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    settings, config = _llamacpp_source("session", "http://127.1:8080")
    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    resolution = await _resolve_llamacpp_send(settings, config, [])

    expected = invalid_llamacpp_base_url_copy(
        "Endpoint URL must include a valid host and port."
    )
    assert readiness.detail == resolution.visible_copy == expected


class _ReferencingStore:
    """One live session on ``custom-ep:box`` (a minimal Console store)."""

    def __init__(self, base_url=None):
        self.replaced = []
        self.base_url = base_url

    def sessions(self):
        return [SimpleNamespace(id="s1", settings=SimpleNamespace(provider="custom-ep:box"))]

    def session_settings(self, _session_id):
        return ConsoleSessionSettings(provider="custom-ep:box", model="m", base_url=self.base_url)

    def replace_session_settings(self, session_id, settings):
        self.replaced.append((session_id, settings))


_KEEPS_WORKING = "'Box' keeps working"
_FIX_WITH_EDIT = "Fix it with Edit"
_OWN_URL = "keep their own endpoint URL"


@pytest.mark.parametrize(
    ("family", "base_url", "session_url", "says"),
    [
        ("openai_compatible", "https://api.deepinfra.com/v1/openai", None, _KEEPS_WORKING),
        ("openai_compatible", "https://gateway.ai.cloudflare.com/v1/acct123/gw1/compat", None, _KEEPS_WORKING),
        ("openai_compatible", "http://127.0.0.1:8000/v1?token=not-a-real-secret", None, _FIX_WITH_EDIT),
        # Review r14ep-rev3 I-2: only the OpenAI-compatible family was checked,
        # and only the entry's URL, not the one each conversation keeps.
        ("ollama", "http://127.0.0.1:11434?token=not-a-real-secret", None, _FIX_WITH_EDIT),
        ("llama_cpp", "http://127.1:8080", None, _FIX_WITH_EDIT),
        ("openai_compatible", "http://127.0.0.1:8000/v1", "https://api.deepinfra.com/v1/openai", _KEEPS_WORKING),
        # Review r14ep-rev4 M-1: the entry is fine; Edit cannot fix a
        # conversation's own URL.
        ("openai_compatible", "http://127.0.0.1:8000/v1", "http://127.1:8000/v1", _OWN_URL),
        ("ollama", "http://127.0.0.1:11434", "http://127.1:11434", _OWN_URL),
    ],
)
def test_detach_refuses_to_move_conversations_onto_a_url_the_custom_provider_refuses(
    family, base_url, session_url, says
):
    """TASK-40300 review r14ep-rev2 M-e (coordinator ruling): Detach moves an
    entry's conversations to the family provider, each with the endpoint it
    keeps; one that provider refuses is never moved. Nothing changes, and the
    copy names the entry, never the URL, and says whether the entry keeps
    working or Edit fixes it."""
    from tldw_chatbook.UI.Screens.settings_provider_view_model import (
        DetachRefused,
        detach_and_delete_entry,
    )

    store = _ReferencingStore(session_url)
    config = _entry_config(base_url, family=family)

    with pytest.raises(DetachRefused) as caught:
        detach_and_delete_entry(config, store, "custom-ep:box")

    assert store.replaced == []
    copy = caught.value.user_copy
    assert copy.startswith("Detach blocked:") and "'Box'" in copy
    assert [phrase for phrase in (_KEEPS_WORKING, _FIX_WITH_EDIT, _OWN_URL) if phrase in copy] == [says]
    assert "deepinfra" not in copy and "not-a-real-secret" not in copy and "127.1" not in copy


def test_detach_moves_conversations_whose_own_url_the_custom_provider_accepts(monkeypatch):
    """Review r14ep-rev3 I-2: an entry with a base only a named endpoint
    accepts was refused even when every conversation keeps its own URL, which
    plain ``custom`` accepts."""
    from tldw_chatbook.UI.Screens import settings_provider_view_model as view_model

    deleted = []
    monkeypatch.setattr(
        view_model,
        "delete_settings_from_cli_config",
        lambda section, keys: deleted.append((section, keys)) or True,
    )
    store = _ReferencingStore("http://127.0.0.1:8000/v1")

    view_model.detach_and_delete_entry(
        _entry_config("https://api.deepinfra.com/v1/openai"), store, "custom-ep:box"
    )

    assert [(sid, s.provider, s.base_url) for sid, s in store.replaced] == [
        ("s1", "custom", "http://127.0.0.1:8000/v1")
    ]
    assert deleted == [("custom_endpoints", ["box"])]


@pytest.mark.parametrize(
    ("params", "copy"),
    [
        (
            {},
            "Provider blocked: this custom endpoint is not loaded: Endpoint URL "
            "must not include a query. Fix it in Settings ▸ Custom endpoints.",
        ),
        (
            {"temperature": 5},
            "Provider blocked: this custom endpoint is not loaded: 'temperature' "
            "must be between 0.0 and 2.0. Fix it in config.toml (see 'Custom "
            "endpoints' in the User Guide's Settings page), or delete the entry "
            "in Settings ▸ Custom endpoints.",
        ),
        (
            {"bogus": 1},
            "Provider blocked: this custom endpoint is not loaded: unknown "
            "sampling param 'bogus'. Fix it in config.toml (see 'Custom "
            "endpoints' in the User Guide's Settings page), or delete the entry "
            "in Settings ▸ Custom endpoints.",
        ),
    ],
)
def test_a_custom_endpoint_not_loaded_says_where_its_reason_can_be_fixed(params, copy):
    """Review r14ep-rev3 Mi-1: the copy ran the reason into the next sentence
    and sent a sampling-params refusal to Settings, which cannot edit params.
    TASK-41146: the config.toml advice names the guide section that lists
    the params keys and ranges."""
    from tldw_chatbook.Chat.custom_endpoint_registry import missing_custom_endpoint_copy

    url = "http://127.0.0.1:8000/v1?token=not-a-real-secret" if not params else "http://127.0.0.1:8000/v1"
    config = _entry_config(url)
    config["custom_endpoints"]["box"]["params"] = params

    assert missing_custom_endpoint_copy(config, "custom-ep:box") == copy


def _nested(path, value):
    """``value`` at ``path`` in an otherwise empty config."""
    for key in reversed(path):
        value = {key: value}
    return {"api_settings": {}, **value}


#: Review r14ep-rev3 I-3 and its follow-up: where an adapter reads its
#: endpoint when Console's table has none; a send went unpinned to it.
_OWN_TABLE_ADAPTERS = [
    ("aphrodite", ("api_settings", "aphrodite_api", "api_url")),
    ("oobabooga", ("api_settings", "ooba_api", "api_ip")),
    ("custom_2", ("custom_openai_api_2", "api_ip")),
    # Review r14ep-rev4 I-a.
    ("local_llm", ("api_settings", "local-llm", "api_ip")),
]


def _merged(user):
    """``user`` merged over the shipped template, as every config view is."""
    from tldw_chatbook.config import _merge_user_config_over_shipped_defaults

    return _merge_user_config_over_shipped_defaults(user)


async def _assert_refused_without_a_request(provider, config):
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    settings = ConsoleSessionSettings(provider=provider, model="m")
    seen: list[str] = []

    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    resolution = await _resolve_endpoint_send(settings, config, False, seen)

    assert readiness.label == "Invalid URL", readiness
    assert resolution.ready is False
    assert seen == []


async def _assert_ready_and_pinned(provider, config, pinned):
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )

    settings = ConsoleSessionSettings(provider=provider, model="m")
    seen: list[str] = []

    readiness = build_console_settings_readiness(settings, app_config=config, environ={})
    resolution = await _resolve_endpoint_send(settings, config, False, seen)

    assert readiness.label == "Ready", readiness
    assert resolution.ready is True, resolution.visible_copy
    assert [s for s in seen if s.startswith("adapter:")] == [f"adapter:{pinned}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(_CONTRACT_REFUSED_BASES))
@pytest.mark.parametrize(("provider", "path"), _OWN_TABLE_ADAPTERS)
async def test_console_never_sends_to_a_refused_endpoint_in_the_adapters_own_table(
    provider, path, shape
):
    """Review r14ep-rev3 I-3: with the refused URL only in
    ``[api_settings.aphrodite_api] api_url``, ``[api_settings.ooba_api]
    api_ip``, Custom OpenAI API 2's ``api_ip`` or Local LLM's ``api_ip``,
    readiness and the send were Ready with no endpoint, and the adapter
    POSTed to it."""
    config = _nested(path, _with_path(provider, _CONTRACT_REFUSED_BASES[shape]))

    await _assert_refused_without_a_request(provider, config)


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider", "path"), _OWN_TABLE_ADAPTERS)
async def test_console_checks_and_pins_an_endpoint_in_the_adapters_own_table(
    provider, path
):
    """Review r14ep-rev3 I-3: a config that sets only the adapter's own table
    keeps working; the send is pinned to the endpoint Console checked."""
    await _assert_ready_and_pinned(
        provider, _nested(path, "http://127.0.0.1:5000/v1"), "http://127.0.0.1:5000/v1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "table"),
    [
        ("aphrodite", "aphrodite"),
        ("oobabooga", "oobabooga"),
        ("custom_2", "custom_2"),
        ("local_llm", "local-llm"),
        ("tabbyapi", "tabbyapi"),
        ("koboldcpp", "koboldcpp"),
    ],
)
async def test_own_table_adapters_keep_their_shipped_endpoint(provider, table):
    """Review r14ep-rev4 C-1: on the shipped template, as a real config is
    built, each adapter keeps the endpoint it was pinned to before."""
    config = _merged({})

    await _assert_ready_and_pinned(provider, config, config["api_settings"][table]["api_url"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user", "pinned"),
    [
        ({}, "http://127.0.0.1:8080/v1"),
        ({"api_settings": {"local_mlx_lm": {"host": "localhost", "port": 9000}}}, "http://localhost:9000/v1"),
    ],
)
async def test_mlx_sends_to_its_own_host_and_port_never_the_shipped_api_url(user, pinned):
    """Review r14ep-rev4 C-1: round 4b pinned ``[api_settings.local_mlx_lm]
    api_url``, which the shipped template sets to a placeholder and MLX-LM
    ignores, so every MLX send went there. Console checks and pins the URL MLX-LM
    builds from its host and port, or its 8080 default. (TASK-41145: the
    template no longer ships that placeholder; one saved earlier stays.)"""
    config = _merged(user)
    config["api_settings"]["local_mlx_lm"]["api_url"] = "http://localhost:5678/v1/chat/completions"
    assert config["api_settings"]["local_mlx_lm"]["api_url"] != pinned

    await _assert_ready_and_pinned("mlx_lm", config, pinned)


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.1", "-bad.corp", "bücher.example", "localhost.."])
async def test_console_never_sends_to_a_refused_mlx_host(host):
    """Review r14ep-rev3 follow-up and rev4 C-1: MLX-LM builds its endpoint from
    ``[api_settings.local_mlx_lm] host`` and ``port``, so ``host = "127.1"``
    is refused even with the shipped ``api_url`` merged in."""
    await _assert_refused_without_a_request(
        "mlx_lm", _merged({"api_settings": {"local_mlx_lm": {"host": host, "port": 8080}}})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", ["http://127.1:8000/v1", "http://127.0.0.1:8000/v1"])
async def test_huggingface_console_checks_and_pins_its_legacy_table(legacy):
    """Review r14ep-rev4 M-4: with ``[api_settings.huggingface] api_base_url``
    blank, Console checked the built-in URL and left the call unpinned, and
    the adapter posted to ``huggingface_api.api_base_url``."""
    config = _merged(
        {"api_settings": {"huggingface": {"api_base_url": "", "api_key": "hf_notarealkey0000000000"}}}
    )
    config["huggingface_api"] = {"api_base_url": legacy, "use_router_url_format": False}

    if legacy == "http://127.1:8000/v1":
        await _assert_refused_without_a_request("huggingface", config)
    else:
        await _assert_ready_and_pinned("huggingface", config, legacy)


_HF_BUILTIN = "https://api-inference.huggingface.co/v1"


def _hf_config(legacy, *, router=False, router_base_url="http://127.1:9000/hf"):
    """A merged config whose HuggingFace table has no endpoint, with a legacy one."""
    config = _merged(
        {"api_settings": {"huggingface": {"api_base_url": "", "api_key": "hf_notarealkey0000000000"}}}
    )
    config["huggingface_api"] = {
        "api_base_url": legacy,
        "use_router_url_format": router,
        "router_base_url": router_base_url,
    }
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "chat_endpoint", "pinned"),
    [
        # The legacy URL is the built-in one: it was checked, then left unpinned.
        (_hf_config(_HF_BUILTIN), None, _HF_BUILTIN),
        # A chat endpoint equal to the built-in one went unpinned, so the
        # adapter posted to a refused legacy URL.
        (_hf_config("http://127.1:8000/v1"), _HF_BUILTIN, _HF_BUILTIN),
        # Router mode reads the router base, which is checked and pinned.
        (
            _hf_config(_HF_BUILTIN, router=True, router_base_url="https://router.huggingface.co/hf-inference"),
            None,
            "https://router.huggingface.co/hf-inference",
        ),
    ],
)
async def test_huggingface_is_always_pinned_to_the_endpoint_console_checked(
    config, chat_endpoint, pinned
):
    """Review r14ep-rev5: HuggingFace was unpinned whenever Console's endpoint
    was the built-in one, and ``chat_with_huggingface`` then read
    ``huggingface_api`` (or its ``router_base_url``) unchecked."""
    settings = ConsoleSessionSettings(provider="huggingface", model="m", base_url=chat_endpoint)
    seen: list[str] = []

    resolution = await _resolve_endpoint_send(settings, config, chat_endpoint is not None, seen)

    assert resolution.ready is True, resolution.visible_copy
    assert [s for s in seen if s.startswith("adapter:")] == [f"adapter:{pinned}"]


@pytest.mark.asyncio
async def test_huggingface_router_mode_never_reaches_an_unchecked_router_base():
    """Review r14ep-rev5: in router mode the adapter posted to a refused
    ``huggingface_api.router_base_url`` while Console checked the built-in URL."""
    await _assert_refused_without_a_request("huggingface", _hf_config(_HF_BUILTIN, router=True))


@pytest.mark.asyncio
@pytest.mark.parametrize(("ooba_api", "ready"), [("http://127.1:5000/v1", False), ("http://127.0.0.1:5000/v1", True)])
async def test_a_blanked_shipped_endpoint_falls_back_to_the_adapters_own_table(ooba_api, ready):
    """Review r14ep-rev5 (a): on the merged config, a user who blanks the shipped
    ``[api_settings.oobabooga] api_url`` gets ``[api_settings.ooba_api] api_ip``,
    checked and pinned."""
    config = _merged({"api_settings": {"oobabooga": {"api_url": ""}, "ooba_api": {"api_ip": ooba_api}}})

    if ready:
        await _assert_ready_and_pinned("oobabooga", config, ooba_api)
    else:
        await _assert_refused_without_a_request("oobabooga", config)


def _frozen_endpoint(config, provider):
    from tldw_chatbook.Agents.agent_routing import SpawnTarget, freeze_spawn_target

    return freeze_spawn_target(config, SpawnTarget(provider, "m", None, (), "preset")).base_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user", "pinned"),
    [
        ({}, "http://127.0.0.1:8080/v1"),
        ({"api_settings": {"local_mlx_lm": {"host": "localhost", "port": 9000}}}, "http://localhost:9000/v1"),
    ],
)
async def test_a_sub_agent_frozen_on_mlx_posts_to_its_host_and_port(user, pinned):
    """Review r14ep-rev5: ``freeze_spawn_target`` froze ``[api_settings.
    local_mlx_lm] api_url``, the shipped placeholder, and a same-provider child,
    which overlays the frozen endpoint on its own resolution, posted there."""
    import dataclasses

    config = _merged(user)
    frozen = _frozen_endpoint(config, "local_mlx_lm")
    pins: list[str] = []

    def adapter(**kwargs):
        pins.append(kwargs.get("api_base_url"))
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(
        config_provider=lambda: config,
        environ={},
        http_client=_client(lambda request: httpx.Response(200, json=_openai_models_payload("m"))),
        chat_api_call_fn=adapter,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="local_mlx_lm", explicit_model="m", streaming=False)
    )
    child = dataclasses.replace(resolution, base_url=frozen)
    [c async for c in gateway.stream_chat(child, [{"role": "user", "content": "x"}])]

    assert frozen == pinned
    assert pins == [pinned]


def test_a_sub_agent_target_freezes_the_endpoint_the_send_checks():
    """Review r14ep-rev5: a frozen target took only Console's table, so an
    adapter's own table and HuggingFace's legacy one were never frozen, and a
    refused legacy URL froze as the built-in one."""
    from tldw_chatbook.Agents.agent_routing import RoutingError

    ooba = _merged({"api_settings": {"oobabooga": {"api_url": ""}, "ooba_api": {"api_ip": "http://127.0.0.1:5000/v1"}}})

    assert _frozen_endpoint(ooba, "oobabooga") == "http://127.0.0.1:5000/v1"
    with pytest.raises(RoutingError):
        _frozen_endpoint(_hf_config("http://127.1:8000/v1"), "huggingface")


def test_a_refused_entry_renames_and_counts_but_offers_no_duplicate():
    """TASK-40300 review r14ep-rev2 M-c: Rename checked the whole entry, so a
    refused one failed with its URL reason; the section count left it out."""
    from tldw_chatbook.Chat.custom_endpoint_registry import (
        validate_display_name,
        validate_entry,
    )
    from tldw_chatbook.UI.Settings_Modules.providers_models_card import (
        custom_endpoints_summary,
    )

    config = _entry_config("http://127.0.0.1:8000/v1?token=not-a-real-secret")

    assert validate_display_name("New name") == []
    assert validate_entry("New name", "openai_compatible", config["custom_endpoints"]["box"]["base_url"])
    assert validate_display_name(" ") == ["Display name is required."]
    screen = SimpleNamespace(_custom_endpoints_view_config=lambda: config)
    assert custom_endpoints_summary(screen).startswith("1 named endpoint ·")


#: TASK-41132: hosts ``inet_aton`` reads as an IPv4 address through hex or
#: octal parts, or fewer than four parts.
_NUMERIC_IPV4_HOSTS = ("0x7f.1", "0x7f000001", "0177.0.0.1", "0X7F.0.0.1")


@pytest.mark.parametrize("host", _NUMERIC_IPV4_HOSTS)
def test_a_host_inet_aton_reads_as_an_address_is_refused_like_127_1(host):
    """TASK-41132: ``127.1`` is refused because a resolver expands it into an
    address its dotted form hides. ``0x7f.1`` and ``0x7f000001`` expand the
    same way, but passed as DNS names, so loopback rules and identity
    comparisons saw a different host from the one dialled."""
    import socket

    from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
        canonical_endpoint_host,
        endpoint_refusal,
    )

    assert socket.inet_aton(host) == socket.inet_aton("127.0.0.1")
    assert canonical_endpoint_host(host) is None
    refused = endpoint_refusal("vllm", "http://127.1:8000/v1")
    assert endpoint_refusal("vllm", f"http://{host}:8000/v1") == refused != ""
    assert endpoint_refusal("custom", f"http://{host}:8000/v1", custom_endpoint=True) == refused


@pytest.mark.parametrize("host", ["0xdeadbeef.example.com", "0x7f.lan", "1.example.com"])
def test_a_name_with_a_number_like_label_is_still_a_name(host):
    """TASK-41132: only a host every label of which is a number is read as an
    address; a name that merely has one keeps working."""
    import socket

    from tldw_chatbook.Provider_Contract.provider_endpoint_contract import (
        canonical_endpoint_host,
        endpoint_refusal,
    )

    with pytest.raises(OSError):
        socket.inet_aton(host)
    assert canonical_endpoint_host(host) == host
    assert endpoint_refusal("vllm", f"http://{host}:8000/v1") == ""


#: TASK-41143: Settings saves a custom slot's readiness key, and only the
#: Console's alias table maps it onto the slot's chat handler.
_CUSTOM_SLOTS = (
    ("custom", "custom-openai-api", "http://127.0.0.1:1234/v1"),
    ("custom_2", "custom-openai-api-2", "http://127.0.0.1:5678/v1"),
)


@pytest.mark.parametrize(("provider", "handler", "endpoint"), _CUSTOM_SLOTS)
def test_permission_summaries_activate_on_a_ready_custom_slot(provider, handler, endpoint):
    """TASK-41143: a Permission summaries provider saved as ``custom`` or
    ``custom_2`` stayed off: ``chat_dispatch_name`` could not map it, and
    nothing said why. With the slot's endpoint saved where Settings saves it,
    the summary runs on the slot's handler and pins that endpoint."""
    from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
    from tldw_chatbook.Library.ingest_analysis import chat_dispatch_name

    api_settings = _merged({"api_settings": {provider: {"api_url": endpoint}}})["api_settings"]
    assert get_provider_readiness(provider, {"api_settings": api_settings}, environ={}).ready

    resolution, out, calls = _summary_calls(provider, api_settings)

    assert chat_dispatch_name(provider) == handler
    assert resolution.active is True
    assert resolution.dispatch_name == handler
    assert out == "ok"
    assert [call["api_base_url"] for call in calls] == [endpoint]


@pytest.mark.parametrize(("provider", "handler", "endpoint"), _CUSTOM_SLOTS)
def test_ingest_analysis_resolves_a_custom_slot_to_its_handler(provider, handler, endpoint):
    """TASK-41143: a ready ``[analysis_defaults] provider = "custom"`` was
    refused as "not supported for ingest analysis"."""
    from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider

    config = _merged(
        {
            "analysis_defaults": {"provider": provider, "model": "m"},
            "api_settings": {provider: {"api_url": endpoint}},
        }
    )

    resolution = resolve_ingest_analysis_provider(config, environ={})

    assert resolution.ready is True, resolution.short_reason
    assert resolution.dispatch_name == handler


@pytest.mark.parametrize(("provider", "handler", "endpoint"), _CUSTOM_SLOTS)
def test_dreams_resolves_a_custom_slot_to_its_handler(provider, handler, endpoint, monkeypatch):
    """TASK-41143: Dreams falls back to ``[chat_defaults] provider``, saved as
    a readiness key, and raised "Dreams provider/model unavailable" for
    ``custom`` and ``custom_2``."""
    from tldw_chatbook.Dreams import story_service

    captured: list[dict] = []
    monkeypatch.setattr(
        story_service,
        "chat_api_call",
        lambda **kwargs: captured.append(kwargs) or {"choices": [{"message": {"content": "ok"}}]},
    )
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default,
    )
    monkeypatch.setattr(
        story_service,
        "load_cli_config_and_ensure_existence",
        lambda **kw: _merged(
            {
                "chat_defaults": {"provider": provider, "model": "m"},
                "api_settings": {provider: {"api_url": endpoint}},
            }
        ),
    )

    chat = story_service.resolve_dreams_chat()
    chat(messages_payload=[{"role": "user", "content": "hi"}], streaming=False)

    assert [call["api_endpoint"] for call in captured] == [handler]


@pytest.mark.parametrize("session_url", [None, "http://127.0.0.1:8000/v1"])
def test_detach_never_moves_conversations_to_an_unknown_family(session_url, monkeypatch):
    """TASK-41144: an entry whose family Chatbook does not know (hand-written
    in config.toml) moved its conversations to a provider named after the
    family string, which no adapter serves, so every later send from them
    failed. Detach now refuses with a reason; nothing moves or is deleted,
    and the copy names the entry, never the family string or its URL."""
    from tldw_chatbook.UI.Screens import settings_provider_view_model as view_model

    deleted = []
    monkeypatch.setattr(
        view_model,
        "delete_settings_from_cli_config",
        lambda section, keys: deleted.append((section, keys)) or True,
    )
    store = _ReferencingStore(session_url)

    with pytest.raises(view_model.DetachRefused) as caught:
        view_model.detach_and_delete_entry(
            _entry_config("http://127.0.0.1:8000/v1", family="vllm_native"), store, "custom-ep:box"
        )

    assert store.replaced == [] and deleted == []
    copy = caught.value.user_copy
    assert copy.startswith("Detach blocked:") and "'Box'" in copy
    assert "vllm_native" not in copy and "127.0.0.1" not in copy


def test_an_unknown_family_entry_with_no_conversations_still_deletes(monkeypatch):
    """TASK-41144: with nothing to move, Detach only deletes the entry."""
    from tldw_chatbook.UI.Screens import settings_provider_view_model as view_model

    deleted = []
    monkeypatch.setattr(
        view_model,
        "delete_settings_from_cli_config",
        lambda section, keys: deleted.append((section, keys)) or True,
    )

    view_model.detach_and_delete_entry(
        _entry_config("http://127.0.0.1:8000/v1", family="vllm_native"), None, "custom-ep:box"
    )

    assert deleted == [("custom_endpoints", ["box"])]


# --- TASK-41131: chat_api_call callers outside Console -----------------------

#: The refused shapes, plus TASK-41132's hexadecimal host.
_DISPATCH_REFUSED_BASES = {**_CONTRACT_REFUSED_BASES, "hex-host": "http://0x7f.1:8000"}
#: Each local and custom dispatch name, the table and key its adapter reads
#: when the call pins no endpoint, and its route (``_with_path``).
_DISPATCH_OWN_ENDPOINTS = (
    ("local-llm", ("api_settings", "local-llm"), "api_url", "vllm"),
    ("vllm", ("api_settings", "vllm"), "api_url", "vllm"),
    ("local_vllm", ("api_settings", "local_vllm"), "api_url", "vllm"),
    ("ollama", ("api_settings", "ollama"), "api_url", "vllm"),
    ("local_ollama", ("api_settings", "local_ollama"), "api_url", "vllm"),
    ("oobabooga", ("api_settings", "ooba_api"), "api_ip", "vllm"),
    ("tabbyapi", ("api_settings", "tabby_api"), "api_url", "vllm"),
    ("aphrodite", ("api_settings", "aphrodite_api"), "api_url", "vllm"),
    ("custom-openai-api", ("api_settings", "custom"), "api_url", "vllm"),
    ("custom-openai-api-2", ("custom_openai_api_2",), "api_ip", "vllm"),
    ("koboldcpp", ("api_settings", "koboldcpp"), "api_url", "koboldcpp"),
    ("llama_cpp", ("api_settings", "llama_cpp"), "api_url", "llama_cpp"),
    ("custom-hosted", ("api_settings", "custom"), "api_url", "vllm"),
    ("together", ("api_settings", "together"), "api_base_url", "vllm"),
)
_ECHO_PARTS = ("fe80", "bad.corp", "127.1", "cher", "mode=fast", "part", "pw@", "0x7f")


class _RecordingSession:
    """A requests session that records each POST and answers 200."""

    def __init__(self, posts):
        self.posts = posts

    def post(self, url, **_kwargs):
        self.posts.append(url)
        return SimpleNamespace(
            status_code=200,
            headers={},
            text="",
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "results": [{"text": "ok"}],
            },
            close=lambda: None,
        )

    def close(self):
        pass


def _local_world(monkeypatch, config):
    """The local adapters and the hosted engine read ``config``; each POST
    lands in the list returned, and none reaches a socket."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls
    from tldw_chatbook.LLM_Calls import hosted_chat
    from tldw_chatbook.LLM_Calls import hosted_provider_engine as engine
    from tldw_chatbook.LLM_Calls import provider_sessions

    snapshot = SimpleNamespace(values=config)
    posts: list[str] = []
    monkeypatch.setattr(local_calls, "get_runtime_config_snapshot", lambda: snapshot)
    monkeypatch.setattr(local_calls, "load_settings", lambda *a, **k: config)
    monkeypatch.setattr(engine, "get_runtime_config_snapshot", lambda: snapshot)
    for module in (local_calls, hosted_chat):
        monkeypatch.setattr(
            module, "create_pooled_default_session", lambda **_kw: _RecordingSession(posts)
        )
    monkeypatch.setattr(provider_sessions, "get_session", lambda _key, factory: factory())
    return posts


def _contract_reason(dispatch, endpoint):
    from tldw_chatbook.Provider_Contract.provider_endpoint_contract import endpoint_refusal

    key = {"koboldcpp": "koboldcpp", "llama_cpp": "llama_cpp"}.get(dispatch, "custom")
    if dispatch in {"custom-hosted", "together"}:
        key = dispatch
    return endpoint_refusal(key, endpoint)


def _dispatch(name, **kwargs):
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call

    return chat_api_call(
        name,
        [{"role": "user", "content": "x"}],
        api_key="not-a-real-key-0000",
        model="m",
        streaming=False,
        **kwargs,
    )


@pytest.mark.parametrize(("dispatch", "table", "key", "route"), _DISPATCH_OWN_ENDPOINTS)
def test_chat_api_call_never_reaches_an_endpoint_the_contract_refuses(
    dispatch, table, key, route, monkeypatch
):
    """TASK-41131: only Console checked endpoints, so the same configured or
    pinned endpoint reached the network from every other ``chat_api_call``
    caller. Each local and custom adapter now refuses it, from its own table
    or pinned, before any request, with the contract's reason."""
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    for shape, pinned in itertools.product(sorted(_DISPATCH_REFUSED_BASES), (False, True)):
        endpoint = _with_path(route, _DISPATCH_REFUSED_BASES[shape])
        config: dict = {}
        node = config
        for part in table:
            node = node.setdefault(part, {})
        node[key] = _with_path(route, _SAFE_LOOPBACK) if pinned else endpoint
        posts = _local_world(monkeypatch, config)

        with pytest.raises(ChatConfigurationError) as caught:
            _dispatch(dispatch, api_base_url=endpoint if pinned else None)

        assert posts == [], (shape, pinned)
        reason = _contract_reason(dispatch, endpoint)
        assert reason and str(caught.value).endswith(reason), (shape, pinned, str(caught.value))
        assert caught.value.status_code is None
        assert not [part for part in _ECHO_PARTS if part in str(caught.value)], shape


@pytest.mark.parametrize("dispatch", ["mlx_lm", "local_mlx_lm"])
def test_chat_api_call_never_reaches_an_mlx_host_the_contract_refuses(dispatch, monkeypatch):
    """TASK-41131: MLX-LM builds its URL from ``host`` and ``port``, which no
    non-Console caller checked."""
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    for host in ("127.1", "-bad.corp", "bücher.example", "localhost..", "0x7f.1"):
        posts = _local_world(monkeypatch, {"api_settings": {dispatch: {"host": host, "port": 8000}}})

        with pytest.raises(ChatConfigurationError):
            _dispatch(dispatch)

        assert posts == [], host


def test_dreams_never_sends_to_an_endpoint_the_contract_refuses(monkeypatch):
    """TASK-41131: Dreams calls ``chat_api_call`` unpinned, so the adapter
    sent to whatever its table held."""
    from tldw_chatbook.Dreams import story_service
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default,
    )
    for shape, base in sorted(_DISPATCH_REFUSED_BASES.items()):
        endpoint = _with_path("vllm", base)
        config = {"api_settings": {"vllm": {"api_url": endpoint}}}
        posts = _local_world(monkeypatch, config)
        monkeypatch.setattr(
            story_service,
            "load_cli_config_and_ensure_existence",
            lambda config=config, **kw: {
                **config, "chat_defaults": {"provider": "vllm", "model": "m"}
            },
        )
        chat = story_service.resolve_dreams_chat()

        with pytest.raises(ChatConfigurationError) as caught:
            chat(messages_payload=[{"role": "user", "content": "hi"}], streaming=False)

        assert posts == [], shape
        assert str(caught.value).endswith(_contract_reason("vllm", endpoint)), shape


def test_summarization_never_sends_to_an_endpoint_the_contract_refuses(monkeypatch):
    """TASK-41131: the summarizers call ``chat_api_call`` unpinned; the refusal
    comes back in-band, with the contract's reason."""
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib as summarization

    monkeypatch.setattr(summarization, "_readiness_credential", lambda _endpoint: (None, None))
    monkeypatch.setattr(
        summarization, "get_cli_setting", lambda section, key, default=None: default
    )
    for shape, base in sorted(_DISPATCH_REFUSED_BASES.items()):
        endpoint = _with_path("vllm", base)
        posts = _local_world(
            monkeypatch, {"api_settings": {"vllm": {"api_url": endpoint, "model": "m"}}}
        )

        out = summarization._summarize_via_chat_api("vllm", None, "text", "p", None, None, False)

        assert posts == [], shape
        assert out.startswith("vLLM Summarization: Unexpected error occurred:"), shape
        assert out.endswith(_contract_reason("vllm", endpoint)), shape


#: TASK-41131 keeps-working checks: what each local and custom dispatch posts
#: to with the shipped config (and the adapter's own table, where the
#: template ships none), before and after the contract check.
_SHIPPED_DISPATCH_POSTS = (
    ("local-llm", {}, "http://localhost:8000/v1/chat/completions"),
    ("vllm", {}, "http://localhost:8000/v1/chat/completions"),
    ("local_vllm", {}, "http://localhost:8008/v1/chat/completions"),
    ("ollama", {}, "http://localhost:11434/v1/chat/completions"),
    ("local_ollama", {}, "http://localhost:5678/v1/chat/completions"),
    ("custom-openai-api", {}, "http://localhost:1234/v1/chat/completions"),
    ("koboldcpp", {}, "http://localhost:5001/api/v1/generate"),
    ("llama_cpp", {}, "http://localhost:8080/v1/chat/completions"),
    ("local_llamacpp", {}, "http://localhost:8001/v1/chat/completions"),
    ("local_llamafile", {}, "http://localhost:8001/v1/chat/completions"),
    ("strata", {}, "http://127.0.0.1:8080/v1/chat/completions"),
    ("local_mlx_lm", {}, "http://127.0.0.1:8080/v1/chat/completions"),
    (
        "oobabooga",
        {"api_settings": {"ooba_api": {"api_ip": "http://127.0.0.1:5000/v1"}}},
        "http://127.0.0.1:5000/v1/chat/completions",
    ),
    (
        "tabbyapi",
        {"api_settings": {"tabby_api": {"api_url": "http://127.0.0.1:5000"}}},
        "http://127.0.0.1:5000/v1/chat/completions",
    ),
    (
        "aphrodite",
        {"api_settings": {"aphrodite_api": {"api_url": "http://gpu-box.lan:2242/v1"}}},
        "http://gpu-box.lan:2242/v1/chat/completions",
    ),
    (
        # TASK-41145 owner decision 2026-11-16: the slot Settings writes.
        "custom-openai-api-2",
        {"api_settings": {"custom_2": {"api_url": "http://kobold_cpp:5678/v1/chat/completions"}}},
        "http://kobold_cpp:5678/v1/chat/completions",
    ),
    (
        "vllm",
        {"api_settings": {"vllm": {"api_url": "http://[fe80::1]:8000/v1"}}},
        "http://[fe80::1]:8000/v1/chat/completions",
    ),
    (
        "local_mlx_lm",
        {"api_settings": {"local_mlx_lm": {"host": "gpu-box.lan", "port": 9000}}},
        "http://gpu-box.lan:9000/v1/chat/completions",
    ),
)


def test_an_ordinary_local_config_still_posts_where_it_did(monkeypatch):
    """TASK-41131: the check refuses only what the contract refuses; every
    shipped endpoint, and the adapters' own tables, post where they did."""
    for dispatch, user, posted in _SHIPPED_DISPATCH_POSTS:
        posts = _local_world(monkeypatch, _merged(user))

        reply = _dispatch(dispatch)

        assert posts == [posted], dispatch
        assert reply["choices"][0]["message"]["content"] == "ok", dispatch


def test_every_engine_preset_still_resolves_its_shipped_base():
    """TASK-41131: the hosted engine's base resolution is checked too; every
    preset's shipped base, and a relaxed custom-hosted base, still resolve."""
    from tldw_chatbook.LLM_Calls.hosted_provider_engine import resolve_hosted_request
    from tldw_chatbook.provider_registry import ALL_RECORDS, RECORDS_BY_KEY

    config = _merged({})
    resolved = {
        record.key: resolve_hosted_request(
            record,
            explicit_api_key="not-a-real-key-0000",
            explicit_model="m",
            app_config=config,
            environ={},
        ).base_url
        for record in ALL_RECORDS
        if record.engine_driven and record.default_base_url
    }
    shipped = {
        key: config["api_settings"].get(key, {}).get("api_base_url") or RECORDS_BY_KEY[key].default_base_url
        for key in resolved
    }

    assert resolved == shipped
    relaxed = resolve_hosted_request(
        RECORDS_BY_KEY["custom-hosted"],
        explicit_api_key="not-a-real-key-0000",
        explicit_base_url="https://api.deepinfra.com/v1/openai",
        explicit_model="m",
        app_config=config,
        environ={},
    )
    assert relaxed.base_url == "https://api.deepinfra.com/v1/openai"


def test_the_shared_local_transport_keeps_its_recovery_wrapper():
    """Review r15ep-rev1 M-1: inserting the refusal helper between
    ``@_provider_recovery.unqualified`` and the transport moved the decorator
    onto the helper, so local sends and streams lost recovery admission."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls

    transport = local_calls._chat_with_openai_compatible_local_server
    assert transport.__wrapped__.__name__ == "_chat_with_openai_compatible_local_server"
    assert not hasattr(local_calls._refuse_a_refused_endpoint, "__wrapped__")


#: Review r15ep-rev1 M-2: the name each refusal shows and the provider key it
#: carries.
_REFUSAL_NAMES = {
    "local-llm": ("Local-LLM", "local-llm"),
    "vllm": ("vLLM", "vllm"),
    "local_vllm": ("vLLM", "local_vllm"),
    "ollama": ("Ollama", "ollama"),
    "local_ollama": ("Ollama", "local_ollama"),
    "oobabooga": ("Oobabooga (OpenAI Extension)", "oobabooga"),
    "tabbyapi": ("TabbyAPI", "tabbyapi"),
    "aphrodite": ("Aphrodite Engine", "aphrodite"),
    "custom-openai-api": ("Custom OpenAI", "custom-openai-api"),
    "custom-openai-api-2": ("Custom OpenAI API 2", "custom-openai-api-2"),
    "koboldcpp": ("KoboldCpp", "koboldcpp"),
    "llama_cpp": ("Llama.cpp", "llama_cpp"),
    "custom-hosted": ("Custom Hosted", "custom-hosted"),
    "together": ("Together", "together"),
}


@pytest.mark.parametrize(("dispatch", "table", "key", "route"), _DISPATCH_OWN_ENDPOINTS)
def test_a_refusal_names_the_adapter_and_carries_its_key(dispatch, table, key, route, monkeypatch):
    """Review r15ep-rev1 M-2: the copy read "Custom_openai_api_2 API URL" and
    "mlx_lm API URL", and the provider was a display name."""
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    config: dict = {}
    node = config
    for part in table:
        node = node.setdefault(part, {})
    node[key] = _with_path(route, "http://127.1:8000")
    _local_world(monkeypatch, config)

    with pytest.raises(ChatConfigurationError) as caught:
        _dispatch(dispatch)

    name, provider = _REFUSAL_NAMES[dispatch]
    assert str(caught.value).startswith(f"{name} API ")
    assert caught.value.provider == provider


@pytest.mark.parametrize("dispatch", ["mlx_lm", "local_mlx_lm"])
def test_an_mlx_refusal_names_mlx_lm_and_carries_its_key(dispatch, monkeypatch):
    """Review r15ep-rev1 M-2."""
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    _local_world(monkeypatch, {"api_settings": {dispatch: {"host": "127.1", "port": 8000}}})

    with pytest.raises(ChatConfigurationError) as caught:
        _dispatch(dispatch)

    assert str(caught.value).startswith("MLX-LM API URL: ")
    assert caught.value.provider == dispatch


@pytest.mark.asyncio
async def test_an_eval_never_retries_an_endpoint_refusal():
    """Review r15ep-rev1 I-2: Evals retried a contract refusal as an API
    error and reported it as a connection failure; nothing can succeed on a
    retry, so it fails at once, non-retryable, with the contract's reason."""
    from tldw_chatbook.Evals.eval_runner import ErrorHandler
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls

    attempts: list[int] = []

    async def operation():
        attempts.append(1)
        local_calls._refuse_a_refused_endpoint("vLLM", "http://127.1:8000/v1")

    with pytest.raises(Exception) as caught:
        await ErrorHandler(max_retries=3, retry_delay=0.001).with_retry(
            operation, "s1", provider="vllm"
        )

    context = caught.value.context
    assert attempts == [1]
    assert context.is_retryable is False
    assert _contract_reason("vllm", "http://127.1:8000/v1") in context.message
    assert "connect" not in context.message.lower()


@pytest.mark.parametrize(
    "configured",
    [
        "http://127.1:8080",
        "http://0x7f.1:8080/v1",
        "http://user:pw@127.0.0.1:8080",
        "http://-bad.corp:8080/completion",
        "http://127.0.0.1:8080/v1?token=not-a-real-secret",
    ],
)
def test_the_word_bench_never_targets_an_endpoint_the_contract_refuses(configured):
    """Review r15ep-rev1 I-3: the Evals word bench posts to
    ``[api_settings.llama_cpp] api_url`` through its own httpx client, so a
    refused URL was benched; normalising to the root even dropped the query.
    A refused URL now reads as not configured."""
    from tldw_chatbook.UI.Evals import sample_bench

    app_config = {"api_settings": {"llama_cpp": {"api_url": configured}}}

    assert sample_bench.configured_llama_cpp_url(app_config) is None


@pytest.mark.parametrize(
    "configured",
    [
        "http://localhost:8080/completion",
        # The bench's own route, which the llama.cpp chat reading refuses as
        # ambiguous; the bench sends to the root (test_evals_empty_states).
        "http://localhost:8080/v1/completions",
    ],
)
def test_the_word_bench_still_targets_an_accepted_endpoint(configured):
    """Review r15ep-rev1 I-3 control: the check reads the URL the bench
    sends to, its root, with the host, user info and query typed."""
    from tldw_chatbook.UI.Evals import sample_bench

    app_config = {"api_settings": {"llama_cpp": {"api_url": configured}}}

    assert sample_bench.configured_llama_cpp_url(app_config) == "http://localhost:8080"


#: Owner decision 2026-11-16 (TASK-41145): Custom OpenAI API 2 reads its
#: endpoint where Settings writes it.
_CUSTOM_2_SLOT = {
    "api_settings": {
        "custom_2": {"api_url": "http://127.0.0.1:5678/v1", "api_key": "not-a-real-key-0000"}
    }
}
_CUSTOM_2_POST = "http://127.0.0.1:5678/v1/chat/completions"


def test_a_dreams_send_on_custom_2_reaches_the_endpoint_settings_saved(monkeypatch):
    """Review r15ep-rev1 I-1: custom_2 resolved ready and mapped to its
    handler, then every Dreams send failed with "api_ip required", because
    the handler read only the top-level ``[custom_openai_api_2] api_ip``."""
    from tldw_chatbook.Dreams import story_service

    config = _merged({**_CUSTOM_2_SLOT, "chat_defaults": {"provider": "custom_2", "model": "m"}})
    posts = _local_world(monkeypatch, config)
    monkeypatch.setattr(
        "tldw_chatbook.Dreams.settings.get_cli_setting",
        lambda section, key, default=None: default,
    )
    monkeypatch.setattr(story_service, "load_cli_config_and_ensure_existence", lambda **kw: config)

    reply = story_service.resolve_dreams_chat()(
        messages_payload=[{"role": "user", "content": "hi"}], streaming=False
    )

    assert posts == [_CUSTOM_2_POST]
    assert reply["choices"][0]["message"]["content"] == "ok"


def test_an_ingest_analysis_on_custom_2_reaches_the_endpoint_settings_saved(monkeypatch):
    """Review r15ep-rev1 I-1: the ingest analysis call (unpinned) failed the
    same way, reported only as an analysis warning."""
    from tldw_chatbook.Library.ingest_analysis import resolve_ingest_analysis_provider
    from tldw_chatbook.Local_Ingestion import local_file_ingestion

    config = _merged({**_CUSTOM_2_SLOT, "analysis_defaults": {"provider": "custom_2", "model": "m"}})
    posts = _local_world(monkeypatch, config)
    resolution = resolve_ingest_analysis_provider(config, environ={})

    text, failure = local_file_ingestion._run_chat_analysis(
        api_name=resolution.dispatch_name,
        api_key=resolution.api_key,
        content="text",
        custom_prompt="p",
        system_prompt="s",
        analysis_call={"model": "m"},
    )

    assert posts == [_CUSTOM_2_POST]
    assert (text, failure) == ("ok", None)


@pytest.mark.parametrize(
    "legacy",
    [
        # Set explicitly: the merged view carries it as the slot's api_url.
        "http://legacy-box.lan:5000/v1/chat/completions",
        # At its code default: the bridge leaves the slot alone and the
        # adapter falls back to load_settings' ``custom_openai_api_2.api_ip``.
        "http://127.0.0.1:5000/v1/chat/completions",
    ],
)
def test_custom_2_still_reads_its_legacy_api_ip_when_the_slot_has_none(
    legacy, tmp_path, monkeypatch
):
    """Owner decision 2026-11-16: the legacy ``[API] custom_openai2_api_ip``
    (``load_settings`` builds ``custom_openai_api_2.api_ip`` from it) stays the
    fallback, so a config that set only it keeps sending there. Review
    r15ep-rev2 M-2: through a real scratch file, not a patched loader."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls
    from tldw_chatbook.LLM_Calls import provider_sessions

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "API": {"custom_openai2_api_ip": legacy},
                "api_settings": {"custom_2": {"api_url": ""}},
            }
        ),
        encoding="utf-8",
    )
    select_config_source(monkeypatch, config_path, vars(local_calls))
    posts: list[str] = []
    monkeypatch.setattr(
        local_calls, "create_pooled_default_session", lambda **_kw: _RecordingSession(posts)
    )
    monkeypatch.setattr(provider_sessions, "get_session", lambda _key, factory: factory())

    _dispatch("custom-openai-api-2")

    assert posts == [legacy]


def test_custom_2_without_an_endpoint_names_the_setting_to_fill(monkeypatch):
    """Review r15ep-rev2 M-1: the copy named only the legacy ``api_ip``."""
    from tldw_chatbook.Provider_Contract import ChatConfigurationError

    config = _merged({})
    config["api_settings"]["custom_2"]["api_url"] = ""
    _local_world(monkeypatch, config)

    with pytest.raises(ChatConfigurationError) as caught:
        _dispatch("custom-openai-api-2")

    assert "[api_settings.custom_2] api_url" in str(caught.value)


# --- TASK-41145, owner decision 2026-11-16: "Settings edits host/port" -------

#: Typed MLX-LM endpoints, and the host and port each is saved as.
_MLX_ACCEPTED = (
    ("http://gpu-box.lan:9000/v1", "gpu-box.lan", 9000),
    ("http://gpu-box.lan:9000", "gpu-box.lan", 9000),
    ("http://gpu-box.lan:9000/v1/", "gpu-box.lan", 9000),
    ("localhost:9000", "localhost", 9000),
    ("http://[::1]:9000/v1", "[::1]", 9000),
)


_MLX_REFUSED_ENDPOINTS = (
    "https://gpu-box.lan:9000/v1",
    "http://gpu-box.lan:9000/v1/chat/completions",
    "http://gpu-box.lan:9000/mlx/v1",
    "http://user:pw@gpu-box.lan:9000/v1",
    "http://127.1:9000/v1",
)


def _mlx_refused():
    """Each refused MLX-LM endpoint and the copy that refuses it."""
    from tldw_chatbook.Chat.provider_setup_persistence import (
        MLX_HTTPS_REFUSAL,
        MLX_PATH_REFUSAL,
    )

    return dict(
        zip(
            _MLX_REFUSED_ENDPOINTS,
            (
                MLX_HTTPS_REFUSAL,
                MLX_PATH_REFUSAL,
                MLX_PATH_REFUSAL,
                "Endpoint URL must not include user information.",
                "Endpoint URL must include a valid host and port.",
            ),
            strict=True,
        )
    )


def _applied(config, mutation):
    """``config`` with a provider setup mutation's writes and deletes applied."""
    import copy

    out = copy.deepcopy(config)
    for section, values in mutation.section_values.items():
        node = out
        for part in section.split("."):
            node = node.setdefault(part, {})
        node.update(values)
    for section, keys in mutation.delete_keys.items():
        node = out
        for part in section.split("."):
            node = node.get(part, {})
        for key in keys:
            node.pop(key, None)
    return out


def _mlx_draft(endpoint):
    return ProviderSetupDraft(
        provider="local_mlx_lm",
        model="m",
        endpoint=endpoint,
        credential_source="none",
        credential_revision=0,
        draft_generation=1,
    )


@pytest.mark.parametrize(("endpoint", "host", "port"), _MLX_ACCEPTED)
def test_saving_an_mlx_endpoint_writes_the_host_and_port_the_adapter_reads(
    endpoint, host, port, monkeypatch
):
    """TASK-41145: Settings' (and first-run setup's) save wrote the endpoint
    to ``[api_settings.local_mlx_lm] api_url``, which ``chat_with_mlx_lm``
    never reads, so chats still went to 127.0.0.1:8080. The save now writes
    ``host`` and ``port``, leaves ``api_url`` alone, and the adapter calls
    the saved address."""
    config = _merged({})
    placeholder = config["api_settings"]["local_mlx_lm"].get("api_url")

    mutation = build_provider_setup_mutation(_mlx_draft(endpoint), config)
    written = mutation.section_values["api_settings.local_mlx_lm"]
    saved = _applied(config, mutation)
    posts = _local_world(monkeypatch, saved)
    _dispatch("local_mlx_lm")

    assert (written["host"], written["port"]) == (host, port)
    assert "api_url" not in written
    assert saved["api_settings"]["local_mlx_lm"].get("api_url") == placeholder
    assert posts == [f"http://{host}:{port}/v1/chat/completions"]


def test_an_mlx_save_refuses_what_host_and_port_cannot_express():
    """TASK-41145: https, a path other than /v1 and user info cannot be
    saved as a host and port; the save refuses with copy that says why."""
    from tldw_chatbook.Chat.provider_setup_persistence import mlx_endpoint_refusal

    for endpoint, copy_ in _mlx_refused().items():
        assert mlx_endpoint_refusal(endpoint) == copy_, endpoint
        with pytest.raises(ValueError, match=re.escape(copy_)):
            build_provider_setup_mutation(_mlx_draft(endpoint), _merged({}))


def test_first_run_setup_reads_and_writes_mlx_host_and_port():
    """TASK-41145: first-run setup showed the template's unused ``api_url``
    placeholder as MLX-LM's endpoint and saved the typed one to it."""
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    config = _merged({"api_settings": {"local_mlx_lm": {"host": "gpu-box.lan", "port": 9000}}})

    def draft(endpoint):
        return wizard_state.FirstRunProviderDraft(
            "local_mlx_lm", endpoint, wizard_state.ProviderCredentialDraft("none", "", 0)
        )

    shown = wizard_state.resolve_first_run_provider_draft(draft(""), config)
    mutation = wizard_state.build_first_run_provider_commit(
        draft("http://other-box.lan:9100/v1"), "m", config
    )

    assert shown.endpoint == "http://gpu-box.lan:9000/v1"
    written = mutation.section_values["api_settings.local_mlx_lm"]
    assert (written["host"], written["port"]) == ("other-box.lan", 9100)
    assert "api_url" not in written
    for endpoint, copy_ in _mlx_refused().items():
        with pytest.raises(ValueError, match=re.escape(copy_)):
            wizard_state.build_first_run_provider_commit(draft(endpoint), "m", config)


@pytest.mark.parametrize("endpoint", ["http://gpu-box.lan:9000/v1", *_MLX_REFUSED_ENDPOINTS])
def test_make_default_writes_mlx_host_and_port(
    tmp_path, monkeypatch, _reset_console_default_intents, endpoint
):
    """TASK-41145: Chat settings' Make default wrote MLX-LM's endpoint to the
    ``api_url`` its adapter ignores. It writes ``host`` and ``port``, and
    refuses what they cannot express, writing nothing."""
    config_path = tmp_path / "config.toml"
    placeholder = "http://localhost:5678/v1/chat/completions"
    config_path.write_text(
        toml.dumps(
            {
                "api_settings": {"local_mlx_lm": {"api_url": placeholder}},
                "chat_defaults": {"provider": "openai", "model": "old-model"},
            }
        ),
        encoding="utf-8",
    )
    select_config_source(monkeypatch, config_path, vars(defaults_module))
    before = config_path.read_bytes()
    refusal = _mlx_refused().get(endpoint, "")

    outcome = defaults_module.apply_console_default_intent(
        defaults_module.ConsoleDefaultMutationIntent(
            generation=1,
            action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            provider_config_key="local_mlx_lm",
            literal_model_id="mlx-model",
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
            values={"temperature": 0.25},
            endpoint_patch=defaults_module.ConsoleEndpointPatch(
                endpoint, "local_mlx_lm", True, True
            ),
        )
    )

    assert defaults_module.console_default_endpoint_refusal("local_mlx_lm", endpoint) == refusal
    if refusal:
        assert outcome.file_replaced is False
        assert config_path.read_bytes() == before
    else:
        assert outcome.runtime_published is True, outcome
        saved = tomllib.loads(config_path.read_text(encoding="utf-8"))["api_settings"]["local_mlx_lm"]
        assert (saved["host"], saved["port"], saved["api_url"]) == ("gpu-box.lan", 9000, placeholder)


def test_the_shipped_mlx_table_holds_the_host_and_port_its_adapter_reads():
    """TASK-41145 AC#2: the template shipped an ``api_url`` placeholder
    (localhost:5678) that MLX-LM never reads, which Settings then showed as
    its endpoint. It ships the adapter's own defaults instead."""
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    table = DEFAULT_CONFIG_FROM_TOML["api_settings"]["local_mlx_lm"]

    assert "api_url" not in table
    assert (table["host"], table["port"]) == ("127.0.0.1", 8080)


# --- TASK-41145 ruling 2026-11-17: an explicit legacy custom_2 endpoint is not
# silently redirected -----------------------------------------------------------

_LEGACY_CUSTOM_2 = "http://legacy-box.lan:5000/v1/chat/completions"
_CUSTOM_2_PLACEHOLDER = "http://localhost:5678/v1/chat/completions"


@pytest.mark.parametrize(
    ("user", "sent"),
    [
        # Legacy set explicitly, the slot still the template's placeholder.
        ({"API": {"custom_openai2_api_ip": _LEGACY_CUSTOM_2}}, _LEGACY_CUSTOM_2),
        # ...or blanked: the legacy endpoint is kept.
        (
            {
                "API": {"custom_openai2_api_ip": _LEGACY_CUSTOM_2},
                "api_settings": {"custom_2": {"api_url": ""}},
            },
            _LEGACY_CUSTOM_2,
        ),
        # A slot the user edited wins.
        (
            {
                "API": {"custom_openai2_api_ip": _LEGACY_CUSTOM_2},
                "api_settings": {"custom_2": {"api_url": "http://slot-box.lan:5678/v1"}},
            },
            "http://slot-box.lan:5678/v1/chat/completions",
        ),
        # The legacy key at its shipped default is not an explicit choice.
        (
            {"API": {"custom_openai2_api_ip": "http://127.0.0.1:5000/v1/chat/completions"}},
            _CUSTOM_2_PLACEHOLDER,
        ),
    ],
)
def test_an_explicit_legacy_custom_2_endpoint_is_not_redirected(user, sent, monkeypatch):
    """TASK-41145 ruling 2026-11-17: reading ``[api_settings.custom_2] api_url``
    first moved a profile that set ``[API] custom_openai2_api_ip`` onto the
    template's localhost:5678 placeholder. The merged view (every reader's)
    keeps the explicit legacy endpoint until the slot holds a real one."""
    from tldw_chatbook.Chat.console_provider_endpoints import adapter_own_endpoint

    config = _merged(user)
    assert config["api_settings"]["custom_2"]["api_url"] in {sent, sent.removesuffix("/chat/completions")}
    posts = _local_world(monkeypatch, config)

    _dispatch("custom-openai-api-2")

    assert posts == [sent]
    own = adapter_own_endpoint("custom-openai-api-2", config)
    assert own.removesuffix("/chat/completions") == sent.removesuffix("/chat/completions")
    assert user.get("api_settings", {}).get("custom_2", {}).get("api_url") != _LEGACY_CUSTOM_2


def test_the_loaded_settings_keep_an_explicit_legacy_custom_2_endpoint(tmp_path, monkeypatch):
    """TASK-41145 ruling 2026-11-17: ``load_settings`` (the adapter's read) is
    built from the same merged view; the file itself is never written."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps({"API": {"custom_openai2_api_ip": _LEGACY_CUSTOM_2}}), encoding="utf-8"
    )
    before = config_path.read_bytes()
    config_module = select_config_source(monkeypatch, config_path)

    settings = config_module.load_settings(force_reload=True)

    assert settings["api_settings"]["custom_2"]["api_url"] == _LEGACY_CUSTOM_2
    assert config_path.read_bytes() == before


#: Review r15ep-rev2: an MLX-LM table as the old template wrote it.
_OLD_TEMPLATE_MLX = {
    "api_settings": {
        "local_mlx_lm": {
            "api_url": "http://localhost:5678/v1/chat/completions",
            "model": "custom-model-gamma",
        }
    }
}


@pytest.mark.parametrize(
    ("user", "configured"),
    [
        ({}, []),
        (_OLD_TEMPLATE_MLX, []),
        ({"api_settings": {"local_mlx_lm": {"host": "gpu-box.lan", "port": 9000}}}, ["local_mlx_lm"]),
    ],
)
def test_mlx_counts_as_configured_only_for_a_host_and_port_the_user_set(user, configured):
    """Review r15ep-rev2 I-2: Settings' "Configured" list compared the
    endpoint keys, so the old template's ignored ``api_url`` placeholder
    (the new template ships none) made MLX-LM read Configured in every
    profile written from it, and a host and port the user set did not."""
    from tldw_chatbook.UI.Screens.settings_provider_view_model import configured_provider_keys
    from tldw_chatbook.UI.Screens.settings_screen import PROVIDER_ENDPOINT_KEYS

    catalog = [SimpleNamespace(readiness_key="local_mlx_lm")]

    keys = configured_provider_keys(catalog, _merged(user), PROVIDER_ENDPOINT_KEYS, environ={})

    assert sorted(keys) == configured


@pytest.mark.parametrize(
    ("user", "listed"),
    [
        ({}, "http://127.0.0.1:8080/v1"),
        (_OLD_TEMPLATE_MLX, "http://127.0.0.1:8080/v1"),
        (
            {"api_settings": {"local_mlx_lm": {"host": "gpu-box.lan", "port": 9000}}},
            "http://gpu-box.lan:9000/v1",
        ),
    ],
)
def test_mlx_discovery_lists_the_host_and_port_its_send_uses(user, listed):
    """Review r15ep-rev2: model discovery and auto-refresh read MLX-LM's
    ``api_url``, so a fresh or host-and-port profile listed nothing and an
    old-template profile listed the ignored localhost:5678 placeholder."""
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )

    service = LocalLLMProviderCatalogService.__new__(LocalLLMProviderCatalogService)

    endpoint = service._resolve_endpoint(
        provider_key="local_mlx_lm", saved_settings=_merged(user), staged_settings=None
    )
    draft = service._resolve_endpoint(
        provider_key="local_mlx_lm",
        saved_settings=_merged(user),
        staged_settings={"api_settings": {"local_mlx_lm": {"api_url": "http://draft-box.lan:9100/v1"}}},
    )

    assert endpoint == listed
    assert draft == "http://draft-box.lan:9100/v1"
