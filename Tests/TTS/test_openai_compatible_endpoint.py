# test_openai_compatible_endpoint.py
# Description: TASK-2260 — custom (OpenAI-compatible) TTS endpoints must receive the
# configured model/voice unmodified and must not require an API key; behavior against
# the default OpenAI endpoint must not change.

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
from tldw_chatbook.TTS.openai_compatible_config import (
    OpenAIAuthenticationMode,
    normalize_openai_authentication_mode,
    normalize_openai_compatible_endpoint,
    openai_destination_fingerprint,
)

# Network opt-in (task-15111): this module talks to an in-process HTTP
# server on an ephemeral loopback port.
# The autouse guard in Tests/conftest.py denies egress by default; every address
# these tests reach is a port this process itself is listening on.
pytestmark = pytest.mark.allow_network

CUSTOM_URL = "http://127.0.0.1:8123/v1/audio/speech"


@pytest.mark.parametrize(
    ("raw", "speech", "origin", "catalog", "official"),
    (
        (
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/v1/audio/speech",
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/v1/models",
            False,
        ),
        (
            "http://127.0.0.1:8765/v1/",
            "http://127.0.0.1:8765/v1/audio/speech",
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/v1/models",
            False,
        ),
        (
            "http://127.0.0.1:8765/v1/models",
            "http://127.0.0.1:8765/v1/audio/speech",
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/v1/models",
            False,
        ),
        (
            "http://127.0.0.1:8765/v1/chat/completions",
            "http://127.0.0.1:8765/v1/audio/speech",
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/v1/models",
            False,
        ),
        (
            "http://127.0.0.1:8765/chat/completions",
            "http://127.0.0.1:8765/audio/speech",
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/models",
            False,
        ),
        (
            "http://127.0.0.1:8765/v1/audio/speech/",
            "http://127.0.0.1:8765/v1/audio/speech",
            "http://127.0.0.1:8765",
            "http://127.0.0.1:8765/v1/models",
            False,
        ),
        (
            "https://API.OpenAI.com:443/V1/AUDIO/SPEECH/",
            "https://api.openai.com/v1/audio/speech",
            "https://api.openai.com",
            "https://api.openai.com/v1/models",
            True,
        ),
        (
            "https://EXAMPLE.test/Custom/Speech/",
            "https://example.test/Custom/Speech/",
            "https://example.test",
            None,
            False,
        ),
    ),
)
def test_endpoint_plan(raw, speech, origin, catalog, official) -> None:
    endpoint = normalize_openai_compatible_endpoint(raw)

    assert endpoint.speech_url == speech
    assert endpoint.origin == origin
    assert endpoint.catalog_url == catalog
    assert endpoint.official is official


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "relative/speech",
        "ftp://example.test/speech",
        "https://user:secret@example.test/speech",
        "https://example.test/speech?token=secret",
        "https://example.test/speech?",
        "https://example.test/speech#fragment",
        "https://example.test/speech#",
        "https://example.test/speech\nInjected: value",
        " https://example.test/speech",
        "https://example.test/speech ",
        "https://example.test/speech path",
        "http://127.0.0.1:8765/v1/audio/speechshttp://127.0.0.1:8765/v1/audio/speech",
        "https://example.test:0/speech",
        "https://example.test:65536/speech",
        "https://example.test:not-a-port/speech",
    ),
)
def test_endpoint_rejects_ambiguous_or_unsafe_values(raw) -> None:
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        normalize_openai_compatible_endpoint(raw)


def test_authentication_mode_is_explicit_and_official_fails_closed() -> None:
    custom = normalize_openai_compatible_endpoint(CUSTOM_URL)
    official = normalize_openai_compatible_endpoint("https://api.openai.com")

    assert (
        normalize_openai_authentication_mode("none", endpoint=custom)
        is OpenAIAuthenticationMode.NONE
    )
    assert (
        normalize_openai_authentication_mode("api_key", endpoint=custom)
        is OpenAIAuthenticationMode.API_KEY
    )
    assert (
        normalize_openai_authentication_mode(None, endpoint=custom)
        is OpenAIAuthenticationMode.API_KEY
    )
    assert (
        normalize_openai_authentication_mode("NONE", endpoint=custom)
        is OpenAIAuthenticationMode.API_KEY
    )
    with pytest.raises(ValueError, match="API key"):
        normalize_openai_authentication_mode("none", endpoint=official)


@pytest.mark.parametrize(
    "raw",
    (
        "https://api.openai.com./v1/audio/speech",
        "https://api.openai.com.../v1/audio/speech",
        "https://api\u3002openai\uff0ecom\uff61/v1/audio/speech",
    ),
)
def test_official_hostname_equivalents_require_api_key(raw) -> None:
    endpoint = normalize_openai_compatible_endpoint(raw)

    assert endpoint.origin == "https://api.openai.com"
    assert endpoint.speech_url == "https://api.openai.com/v1/audio/speech"
    assert endpoint.official is True
    with pytest.raises(ValueError, match="API key"):
        normalize_openai_authentication_mode("none", endpoint=endpoint)


def test_destination_fingerprint_uses_provider_and_normalized_origin_only() -> None:
    first = normalize_openai_compatible_endpoint("HTTP://EXAMPLE.test:80/one/speech")
    second = normalize_openai_compatible_endpoint("http://example.test/two/speech")
    same_destination = openai_destination_fingerprint("openai", first)

    assert same_destination == openai_destination_fingerprint("openai", second)
    assert same_destination != openai_destination_fingerprint("another", second)
    assert re.fullmatch(r"[0-9a-f]{64}", same_destination)
    assert first.origin not in same_destination
    assert "speech" not in same_destination


def test_idn_and_ipv6_equivalents_share_origin_and_fingerprint() -> None:
    unicode_host = normalize_openai_compatible_endpoint(
        "https://b\u00fccher.example./custom/speech"
    )
    punycode_host = normalize_openai_compatible_endpoint(
        "https://xn--bcher-kva.example/another/speech"
    )
    expanded_ipv6 = normalize_openai_compatible_endpoint(
        "https://[2001:0DB8:0:0:0:0:0:1]/custom/speech"
    )
    compressed_ipv6 = normalize_openai_compatible_endpoint(
        "https://[2001:db8::1]/another/speech"
    )

    assert unicode_host.origin == "https://xn--bcher-kva.example"
    assert unicode_host.origin == punycode_host.origin
    assert openai_destination_fingerprint(
        "openai", unicode_host
    ) == openai_destination_fingerprint("openai", punycode_host)
    assert expanded_ipv6.origin == "https://[2001:db8::1]"
    assert expanded_ipv6.origin == compressed_ipv6.origin
    assert openai_destination_fingerprint(
        "openai", expanded_ipv6
    ) == openai_destination_fingerprint("openai", compressed_ipv6)


@pytest.mark.parametrize(
    "raw",
    (
        "http://[fe80::1%eth0]:8765/v1",
        "http://[fe80::1%25eth0]:8765/v1",
        "http://[fe80::1%2525eth0]:8765/v1",
    ),
)
def test_scoped_ipv6_endpoints_are_rejected(raw) -> None:
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        normalize_openai_compatible_endpoint(raw)


@pytest.mark.parametrize(
    "raw",
    (
        "https://example.test/v1//",
        "https://example.test//v1",
        "https://example.test/v1///audio/speech",
        "https://example.test/custom//speech",
    ),
)
def test_repeated_path_separators_are_rejected(raw) -> None:
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        normalize_openai_compatible_endpoint(raw)


@pytest.mark.parametrize(
    "raw",
    (
        "https://example.test/custom/../speech",
        "https://example.test/custom/./speech",
        "https://example.test/../speech",
        "https://example.test/./speech",
        "https://example.test/custom\\speech",
        "https://example.test/custom/%",
        "https://example.test/custom/%2",
        "https://example.test/custom/%GG",
        "https://example.test/custom/%2f/speech",
        "https://example.test/custom/%2F/speech",
        "https://example.test/custom/%5c/speech",
        "https://example.test/custom/%5C/speech",
        "https://example.test/custom/%2e/speech",
        "https://example.test/custom/%2E%2e/speech",
        "https://example.test/custom/.%2e/speech",
        "https://example.test/custom/%2e./speech",
        "https://example.test/custom/%00/speech",
        "https://example.test/custom/%0a/speech",
        "https://example.test/custom/%7f/speech",
        "https://example.test/custom/%C2%80/speech",
    ),
)
def test_endpoint_rejects_path_normalization_ambiguity(raw) -> None:
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        normalize_openai_compatible_endpoint(raw)


def test_safe_percent_escapes_are_canonical_without_decoding_path_semantics() -> None:
    lower = normalize_openai_compatible_endpoint(
        "https://example.test/custom/%7evoice/%2eprofile"
    )
    upper = normalize_openai_compatible_endpoint(
        "https://example.test/custom/%7Evoice/%2Eprofile"
    )

    assert lower.speech_url == "https://example.test/custom/%7Evoice/%2Eprofile"
    assert lower.catalog_url is None
    assert upper == lower
    assert openai_destination_fingerprint(
        "openai", lower
    ) == openai_destination_fingerprint("openai", upper)


def test_unknown_valid_speech_path_declares_no_catalog_operation() -> None:
    endpoint = normalize_openai_compatible_endpoint(
        "http://127.0.0.1:8765/custom/speech"
    )

    assert endpoint.speech_url == "http://127.0.0.1:8765/custom/speech"
    assert endpoint.catalog_url is None


@pytest.mark.parametrize(
    "raw",
    (
        "https://example..test/v1",
        "https://-invalid.example/v1",
        "https://invalid-.example/v1",
    ),
)
def test_invalid_dns_hostnames_are_rejected(raw) -> None:
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        normalize_openai_compatible_endpoint(raw)


def test_canonical_ipv4_spellings_share_origin_and_fingerprint() -> None:
    canonical = normalize_openai_compatible_endpoint(
        "http://127.0.0.1:8765/custom/speech"
    )
    rooted = normalize_openai_compatible_endpoint(
        "http://127.0.0.1.:8765/another/speech"
    )

    assert canonical.origin == "http://127.0.0.1:8765"
    assert rooted.origin == canonical.origin
    assert openai_destination_fingerprint(
        "openai", rooted
    ) == openai_destination_fingerprint("openai", canonical)


@pytest.mark.parametrize(
    "raw",
    (
        "http://127.1:8765/v1",
        "http://127.0.1:8765/v1",
        "http://2130706433:8765/v1",
        "http://017700000001:8765/v1",
        "http://0x7f000001:8765/v1",
        "http://0177.0.0.1:8765/v1",
        "http://127.00.0.1:8765/v1",
        "http://0x7f.0.0.1:8765/v1",
        "http://0x7f.0x0.0x0.0x1:8765/v1",
    ),
)
def test_ambiguous_numeric_ipv4_hostnames_are_rejected(raw) -> None:
    with pytest.raises(ValueError, match="OpenAI-compatible endpoint"):
        normalize_openai_compatible_endpoint(raw)


@pytest.mark.parametrize(
    ("raw", "origin"),
    (
        ("https://123-service.example/v1", "https://123-service.example"),
        ("https://127.example/v1", "https://127.example"),
        ("https://v1.2-example.test/v1", "https://v1.2-example.test"),
    ),
)
def test_dns_names_containing_digits_and_hyphens_remain_valid(raw, origin) -> None:
    endpoint = normalize_openai_compatible_endpoint(raw)

    assert endpoint.origin == origin


def _capture_transport(requests: list[httpx.Request]) -> httpx.MockTransport:
    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"audio")

    return httpx.MockTransport(respond)


async def _generate(backend: OpenAITTSBackend, **request_overrides) -> list[bytes]:
    request = OpenAISpeechRequest(
        model=request_overrides.pop("model", "tts-1"),
        input="hello",
        voice=request_overrides.pop("voice", "alloy"),
        response_format=request_overrides.pop("response_format", "wav"),
        **request_overrides,
    )
    try:
        return [chunk async for chunk in backend.generate_speech_stream(request)]
    finally:
        await backend.close()


def _backend_with_transport(
    config: dict, requests: list[httpx.Request]
) -> OpenAITTSBackend:
    backend = OpenAITTSBackend(config)
    original_client = backend.client
    backend.client = httpx.AsyncClient(transport=_capture_transport(requests))
    # The constructor's real client is replaced before any request; close it eagerly
    # via the event loop the test runs on.
    backend._original_client_for_cleanup = original_client
    return backend


async def _close_replaced_client(backend: OpenAITTSBackend) -> None:
    original = getattr(backend, "_original_client_for_cleanup", None)
    if original is not None:
        await original.aclose()


@pytest.mark.asyncio
async def test_allowed_custom_path_reaches_httpx_as_validated_raw_path() -> None:
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {
            "OPENAI_BASE_URL": "https://example.test/custom/%7evoice/%2eprofile",
            "OPENAI_AUTH_MODE": "none",
        },
        requests,
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend, model="pocket-tts", voice="alba")

    assert chunks == [b"audio"]
    assert requests[0].url.raw_path == b"/custom/%7Evoice/%2Eprofile"


@pytest.mark.asyncio
async def test_custom_endpoint_passes_model_and_voice_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom endpoints define their own model/voice names — no coercion."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": CUSTOM_URL,
            "OPENAI_AUTH_MODE": "api_key",
        },
        requests,
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend, model="pocket-tts", voice="marius")

    assert chunks == [b"audio"]
    payload = json.loads(requests[0].content)
    assert payload["model"] == "pocket-tts"
    assert payload["voice"] == "marius"


@pytest.mark.asyncio
async def test_custom_endpoint_without_api_key_sends_no_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keyless local servers work; no Authorization header is fabricated."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {"OPENAI_BASE_URL": CUSTOM_URL, "OPENAI_AUTH_MODE": "none"}, requests
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend)

    assert chunks == [b"audio"]
    assert "authorization" not in requests[0].headers


@pytest.mark.asyncio
async def test_custom_endpoint_with_key_still_sends_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured key is still sent to custom endpoints that need one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": CUSTOM_URL,
            "OPENAI_AUTH_MODE": "api_key",
        },
        requests,
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend)

    assert chunks == [b"audio"]
    assert requests[0].headers["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_none_auth_does_not_read_or_send_any_credentials(monkeypatch) -> None:
    def reject_lookup(*_args, **_kwargs):
        raise AssertionError("credential lookup is forbidden in none mode")

    monkeypatch.setattr("tldw_chatbook.TTS.backends.openai.os.getenv", reject_lookup)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.backends.openai.get_cli_setting", reject_lookup
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence", reject_lookup
    )
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8765",
            "OPENAI_AUTH_MODE": "none",
            "OPENAI_API_KEY": "must-not-be-read-either",
        },
        requests,
    )
    await _close_replaced_client(backend)

    chunks = await _generate(
        backend,
        model="pocket-tts",
        voice="alba",
        response_format="flac",
    )

    assert chunks == [b"audio"]
    assert str(requests[0].url) == "http://127.0.0.1:8765/v1/audio/speech"
    assert "authorization" not in requests[0].headers
    assert json.loads(requests[0].content) == {
        "model": "pocket-tts",
        "input": "hello",
        "voice": "alba",
        "response_format": "flac",
        "speed": 1.0,
    }


@pytest.mark.asyncio
async def test_default_endpoint_still_coerces_model_and_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterization pin: official-endpoint coercion is unchanged."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport({"OPENAI_API_KEY": "test-key"}, requests)
    await _close_replaced_client(backend)

    chunks = await _generate(backend, model="pocket-tts", voice="marius")

    assert chunks == [b"audio"]
    assert str(requests[0].url) == "https://api.openai.com/v1/audio/speech"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "tts-1"
    assert payload["voice"] == "alloy"


@pytest.mark.asyncio
async def test_default_endpoint_still_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterization pin: the official endpoint still refuses to run keyless."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport({}, requests)
    await _close_replaced_client(backend)
    backend.api_key = None

    with pytest.raises(ValueError, match="not configured"):
        await _generate(backend)

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "official_variant",
    (
        "https://api.openai.com/v1/audio/speech/",
        "HTTPS://API.OpenAI.com/v1/audio/speech",
        "https://api.openai.com:443/v1/audio/speech",
    ),
)
async def test_official_endpoint_variants_keep_guardrails(
    monkeypatch: pytest.MonkeyPatch, official_variant: str
) -> None:
    """A cosmetic rewrite of the official URL (slash, case, default port) must not
    be misclassified as a custom endpoint and lose model/voice coercion."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": official_variant}, requests
    )
    await _close_replaced_client(backend)

    assert backend.is_custom_endpoint is False
    chunks = await _generate(backend, model="pocket-tts", voice="marius")

    assert chunks == [b"audio"]
    payload = json.loads(requests[0].content)
    assert payload["model"] == "tts-1"
    assert payload["voice"] == "alloy"


@pytest.mark.asyncio
async def test_custom_endpoint_does_not_forward_organization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured OpenAI org ID is OpenAI account metadata and must not leak to
    third-party OpenAI-compatible servers."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": CUSTOM_URL,
            "OPENAI_ORG_ID": "org-test",
        },
        requests,
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend)

    assert chunks == [b"audio"]
    assert "openai-organization" not in requests[0].headers


@pytest.mark.asyncio
async def test_official_endpoint_still_sends_organization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The org header keeps working for the real OpenAI endpoint."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {"OPENAI_API_KEY": "test-key", "OPENAI_ORG_ID": "org-test"}, requests
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend)

    assert chunks == [b"audio"]
    assert requests[0].headers["OpenAI-Organization"] == "org-test"


@pytest.mark.asyncio
async def test_keyless_openai_compatible_server_over_real_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end against a real local HTTP server shaped like pocket-tts:
    keyless, custom model and voice, OpenAI-compatible /v1/audio/speech route."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    received: dict = {}

    class SpeechHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received["path"] = self.path
            received["payload"] = json.loads(body)
            received["authorization"] = self.headers.get("Authorization")
            audio = b"RIFFfake-wav-bytes"
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)

        def log_message(self, *args) -> None:  # keep test output quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), SpeechHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAITTSBackend(
            {
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1/audio/speech",
                "OPENAI_AUTH_MODE": "none",
            }
        )

        chunks = await _generate(backend, model="pocket-tts", voice="marius")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert b"".join(chunks) == b"RIFFfake-wav-bytes"
    assert received["path"] == "/v1/audio/speech"
    assert received["payload"]["model"] == "pocket-tts"
    assert received["payload"]["voice"] == "marius"
    assert received["payload"]["response_format"] == "wav"
    assert received["authorization"] is None
