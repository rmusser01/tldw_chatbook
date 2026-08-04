# test_openai_compatible_endpoint.py
# Description: TASK-2260 — custom (OpenAI-compatible) TTS endpoints must receive the
# configured model/voice unmodified and must not require an API key; behavior against
# the default OpenAI endpoint must not change.

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend

CUSTOM_URL = "http://127.0.0.1:8123/v1/audio/speech"


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
        response_format="wav",
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
async def test_custom_endpoint_passes_model_and_voice_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom endpoints define their own model/voice names — no coercion."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    backend = _backend_with_transport(
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": CUSTOM_URL}, requests
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
    backend = _backend_with_transport({"OPENAI_BASE_URL": CUSTOM_URL}, requests)
    await _close_replaced_client(backend)
    # Config-file fallbacks may resolve a key in some environments; this test is about
    # the keyless path, so force the resolved key empty.
    backend.api_key = None

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
        {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": CUSTOM_URL}, requests
    )
    await _close_replaced_client(backend)

    chunks = await _generate(backend)

    assert chunks == [b"audio"]
    assert requests[0].headers["Authorization"] == "Bearer test-key"


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
                "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1/audio/speech"
            }
        )
        backend.api_key = None

        chunks = await _generate(backend, model="pocket-tts", voice="marius")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert b"".join(chunks) == b"RIFFfake-wav-bytes"
    assert received["path"] == "/v1/audio/speech"
    assert received["payload"]["model"] == "pocket-tts"
    assert received["payload"]["voice"] == "marius"
    assert received["authorization"] is None
