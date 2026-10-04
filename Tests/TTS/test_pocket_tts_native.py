"""TASK-34100.8: PocketTTS's native ``/tts`` API through the OpenAI-compatible slot.

The official pocket-tts server (Kyutai, PyPI ``pocket-tts``) serves
``POST /tts`` with form fields ``text`` and ``voice_url`` and streams a WAV
whose RIFF and data sizes are placeholders. It answers 404 on the OpenAI
route ``/v1/audio/speech`` the wizard's PocketTTS preset used to target, so the
preselected voice could never speak. These tests pin the dialect against a
local server shaped exactly like that one (paths, form fields and the
placeholder header were read from pocket-tts 3.3.0 and captured from a real
``pocket-tts serve`` run).
"""

from __future__ import annotations

import io
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import pytest

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
from tldw_chatbook.TTS.pocket_tts_native import (
    POCKET_TTS_DEFAULT_ENDPOINT,
    is_pocket_tts_native_url,
    pocket_tts_form,
    repair_streamed_wav,
)
from tldw_chatbook.TTS.sample_audio_validation import wav_has_complete_frames

pytestmark = [pytest.mark.allow_network, pytest.mark.bootstrap_profile]


def _streamed_wav(frames: int = 2400) -> bytes:
    """A WAV the way pocket-tts streams it: real PCM, placeholder sizes."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x01\x00" * frames)
    body = bytearray(output.getvalue())
    # The bytes captured from a real `pocket-tts serve` /tts reply carry
    # 0x77359424 as the RIFF size and 0x77359400 as the data size.
    body[4:8] = (0x77359424).to_bytes(4, "little")
    body[40:44] = (0x77359400).to_bytes(4, "little")
    return bytes(body)


def test_default_endpoint_is_the_servers_own_route_and_port() -> None:
    assert POCKET_TTS_DEFAULT_ENDPOINT == "http://127.0.0.1:8000/tts"
    assert is_pocket_tts_native_url(POCKET_TTS_DEFAULT_ENDPOINT)


@pytest.mark.parametrize(
    ("url", "native"),
    (
        ("http://127.0.0.1:8766/tts", True),
        ("http://localhost:8000/tts", True),
        ("http://127.0.0.1:8765/v1/audio/speech", False),
        ("https://api.openai.com/v1/audio/speech", False),
        ("http://127.0.0.1:8000/tts/extra", False),
        ("http://127.0.0.1:8000/TTS", False),
        ("not a url", False),
    ),
)
def test_only_an_exact_tts_path_selects_the_native_dialect(url, native) -> None:
    assert is_pocket_tts_native_url(url) is native


def test_form_names_the_voice_only_when_one_is_given() -> None:
    assert pocket_tts_form("Hello", "alba") == {"text": "Hello", "voice_url": "alba"}
    assert pocket_tts_form("Hello", "  ") == {"text": "Hello"}


def test_streamed_wav_placeholder_sizes_are_rewritten_to_the_real_length() -> None:
    streamed = _streamed_wav()
    assert wav_has_complete_frames(streamed) is False  # what the app saw before

    repaired = repair_streamed_wav(streamed)

    assert wav_has_complete_frames(repaired) is True
    assert repaired[44:] == streamed[44:]


def test_a_well_formed_or_foreign_body_is_returned_unchanged() -> None:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 10)
    complete = output.getvalue()

    assert repair_streamed_wav(complete) == complete
    assert repair_streamed_wav(b"ID3 not a wav") == b"ID3 not a wav"


class _PocketTTSHandler(BaseHTTPRequestHandler):
    """The real server's routes: POST /tts (form) and nothing at /v1/audio/speech."""

    received: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/tts":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"detail":"Not Found"}')
            return
        type(self).received.append(
            {
                "content_type": self.headers.get("Content-Type", ""),
                "fields": {
                    key: values[0]
                    for key, values in parse_qs(body.decode("utf-8")).items()
                },
                "authorization": self.headers.get("Authorization"),
            }
        )
        audio = _streamed_wav()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.end_headers()
        self.wfile.write(audio)

    def log_message(self, *_args) -> None:
        return


@pytest.fixture
def pocket_tts_server():
    _PocketTTSHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _PocketTTSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


async def _speak(backend: OpenAITTSBackend, **overrides) -> bytes:
    request = OpenAISpeechRequest(
        model=overrides.pop("model", "pocket-tts"),
        input=overrides.pop("input", "Hello from Chatbook."),
        voice=overrides.pop("voice", "alba"),
        response_format=overrides.pop("response_format", "wav"),
    )
    try:
        return b"".join(
            [chunk async for chunk in backend.generate_speech_stream(request)]
        )
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_backend_speaks_the_native_form_and_returns_a_playable_wav(
    pocket_tts_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAITTSBackend(
        {"OPENAI_BASE_URL": f"{pocket_tts_server}/tts", "OPENAI_AUTH_MODE": "none"}
    )

    audio = await _speak(backend)

    assert wav_has_complete_frames(audio) is True
    [request] = _PocketTTSHandler.received
    assert request["content_type"].startswith("application/x-www-form-urlencoded")
    assert request["fields"] == {"text": "Hello from Chatbook.", "voice_url": "alba"}
    assert request["authorization"] is None


@pytest.mark.asyncio
async def test_backend_refuses_a_non_wav_format_for_pocket_tts(
    pocket_tts_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAITTSBackend(
        {"OPENAI_BASE_URL": f"{pocket_tts_server}/tts", "OPENAI_AUTH_MODE": "none"}
    )

    with pytest.raises(ValueError, match="WAV"):
        await _speak(backend, response_format="mp3")
    assert _PocketTTSHandler.received == []


@pytest.mark.asyncio
async def test_the_openai_route_on_a_pocket_tts_server_is_still_a_404(
    pocket_tts_server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the old preset's route reaches nothing on this server."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAITTSBackend(
        {
            "OPENAI_BASE_URL": f"{pocket_tts_server}/v1/audio/speech",
            "OPENAI_AUTH_MODE": "none",
        }
    )

    with pytest.raises(ValueError, match="404"):
        await _speak(backend)
