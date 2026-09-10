"""Adapter regressions from pinned Higgs V2 and AllTalk V2 runtime probes."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from loguru import logger

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.TTS.backends.alltalk import AllTalkTTSBackend
from tldw_chatbook.TTS.backends.higgs import HiggsAudioTTSBackend
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_VOICES,
    LEGACY_VOICE_OPTIONS,
    legacy_catalog,
)

# AllTalk f16117e95b540e9bbbd8247b49ca6c6b1350b172, tts_server.py,
# OpenAIInput.validate_voice. /api/voices instead lists native engine voices.
_PINNED_ALLTALK_VOICES = ("alloy", "echo", "fable", "nova", "onyx", "shimmer")


def test_alltalk_catalog_and_default_match_pinned_endpoint_contract():
    model = legacy_catalog("alltalk").models[0]
    assert model.voices == _PINNED_ALLTALK_VOICES
    assert LEGACY_DEFAULT_VOICES["alltalk"] == "alloy"
    assert tuple(voice for _label, voice in LEGACY_VOICE_OPTIONS["alltalk"]) == (
        _PINNED_ALLTALK_VOICES
    )


@asynccontextmanager
async def _alltalk_client(handler):
    backend = AllTalkTTSBackend({"ALLTALK_TTS_URL": "http://alltalk.test"})
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        yield backend
    finally:
        await backend.close()


async def _synthesize(backend: AllTalkTTSBackend, voice: str) -> bytes:
    request = OpenAISpeechRequest(
        input="A short provider contract check.", voice=voice, response_format="wav"
    )
    return b"".join([chunk async for chunk in backend.generate_speech_stream(request)])


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", (*_PINNED_ALLTALK_VOICES, "default"))
async def test_alltalk_voice_request_satisfies_pinned_openai_schema(voice):
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["voice"] not in _PINNED_ALLTALK_VOICES:
            return httpx.Response(422, json={"error": "Unsupported voice"})
        return httpx.Response(200, content=b"accepted speech bytes")

    async with _alltalk_client(handle) as backend:
        assert await _synthesize(backend, voice) == b"accepted speech bytes"


@pytest.mark.asyncio
async def test_alltalk_custom_voice_is_opaque_when_endpoint_accepts_it():
    def handle(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["voice"] != "StudioVoice-17":
            return httpx.Response(422, json={"error": "Unknown custom voice"})
        return httpx.Response(200, content=b"custom speech bytes")

    async with _alltalk_client(handle) as backend:
        assert await _synthesize(backend, "StudioVoice-17") == b"custom speech bytes"


@pytest.mark.asyncio
async def test_alltalk_unsupported_custom_voice_is_not_silently_remapped():
    submitted = []

    def handle(request: httpx.Request) -> httpx.Response:
        submitted.append(json.loads(request.content)["voice"])
        return httpx.Response(422, json={"error": "Unsupported voice"})

    async with _alltalk_client(handle) as backend:
        with pytest.raises(ValueError):
            await _synthesize(backend, "p225")

    assert submitted == ["p225"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (200, 503))
async def test_alltalk_initialization_awaits_the_http_health_request(status_code):
    requested = []
    captured = []

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append((request.method, request.url.path))
        return httpx.Response(
            status_code,
            json={"status": "success", "voices": ["private-native-voice.wav"]},
        )

    sink = logger.add(captured.append, format="{message}")
    try:
        async with _alltalk_client(handle) as backend:
            await backend.initialize()
    finally:
        logger.remove(sink)

    assert requested == [("GET", "/api/voices")]
    assert "private-native-voice.wav" not in "".join(captured)
    if status_code == 503:
        assert "status=503" in "".join(captured)


@pytest.mark.asyncio
async def test_alltalk_initialization_handles_actual_connection_failure_safely():
    requested = []
    captured = []

    def handle(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        raise httpx.ConnectError("PRIVATE_CONNECTION_DETAIL", request=request)

    sink = logger.add(captured.append, format="{message}")
    try:
        async with _alltalk_client(handle) as backend:
            await backend.initialize()
    finally:
        logger.remove(sink)

    assert requested == ["/api/voices"]
    assert "PRIVATE_CONNECTION_DETAIL" not in "".join(captured)


@pytest.mark.asyncio
async def test_alltalk_advertises_endpoint_aliases_instead_of_native_speakers():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "voices": ["p225"]})

    async with _alltalk_client(handle) as backend:
        assert tuple(await backend.list_voices()) == _PINNED_ALLTALK_VOICES


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype_name", ("float32", "float16", "bfloat16"))
async def test_higgs_configured_dtype_reaches_pinned_v2_constructor(
    tmp_path, dtype_name
):
    torch = pytest.importorskip("torch")

    # Exact keyword shape from boson-ai/higgs-audio@05a145bb490501b534563bf51bf2f7aa2326b271.
    # Only the external model constructor is replaced; the production async
    # loader, argument selection, worker ownership, and cleanup remain real.
    class PinnedV2Engine:
        def __init__(
            self,
            model_name_or_path,
            audio_tokenizer_name_or_path,
            tokenizer_name_or_path=None,
            device="cuda",
            torch_dtype="auto",
            kv_cache_lengths=(1024, 4096, 8192),
        ):
            self.dtype = torch_dtype

    backend = HiggsAudioTTSBackend(
        {
            "HIGGS_MODEL_PATH": "unloaded-model",
            "HIGGS_AUDIO_TOKENIZER_PATH": "unloaded-tokenizer",
            "HIGGS_DEVICE": "cpu",
            "HIGGS_DTYPE": dtype_name,
            "HIGGS_VOICE_SAMPLES_DIR": str(tmp_path),
            "HIGGS_ENABLE_VOICE_CLONING": False,
        }
    )
    backend._boson_multimodal = object()
    backend._higgs_serve_engine = PinnedV2Engine
    try:
        await backend.load_model()
        assert backend.serve_engine.dtype is getattr(torch, dtype_name)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_higgs_retains_legacy_dtype_constructor_support(tmp_path):
    torch = pytest.importorskip("torch")

    class LegacyEngine:
        def __init__(self, model_name_or_path, device="cpu", dtype=None):
            self.dtype = dtype

    backend = HiggsAudioTTSBackend(
        {
            "HIGGS_MODEL_PATH": "unloaded-model",
            "HIGGS_DEVICE": "cpu",
            "HIGGS_DTYPE": "float32",
            "HIGGS_VOICE_SAMPLES_DIR": str(tmp_path),
            "HIGGS_ENABLE_VOICE_CLONING": False,
        }
    )
    backend._boson_multimodal = object()
    backend._higgs_serve_engine = LegacyEngine
    try:
        await backend.load_model()
        assert backend.serve_engine.dtype is torch.float32
    finally:
        await backend.close()


def test_alltalk_settings_default_matches_endpoint_without_changing_migration():
    from tldw_chatbook.TTS.studio_preferences import _LEGACY_DEFAULTS
    from tldw_chatbook.UI.Speech.speech_settings_model import SETTING_CONFIG_SOURCES

    assert SETTING_CONFIG_SOURCES["alltalk-voice-input"][2] == "alloy"
    assert _LEGACY_DEFAULTS["ALLTALK_TTS_VOICE_DEFAULT"] == "female_01.wav"


@pytest.mark.asyncio
async def test_mounted_alltalk_settings_picker_offers_pinned_endpoint_aliases():
    from textual.app import App, ComposeResult
    from textual.widgets import Select

    from tldw_chatbook.UI.Speech.speech_settings_mixin import SpeechSettingsMixin

    class SettingsHost(SpeechSettingsMixin, App):
        def compose(self) -> ComposeResult:
            yield Select([], id="default-voice-select")

    app = SettingsHost()
    async with app.run_test() as pilot:
        app._update_default_voice_options("alltalk")
        await pilot.pause()
        selector = app.query_one("#default-voice-select", Select)
        assert selector.value == "alloy"
        assert (
            tuple(value for _, value in selector._options if isinstance(value, str))
            == _PINNED_ALLTALK_VOICES
        )


def test_shipped_alltalk_default_is_an_endpoint_alias():
    import tomllib
    from pathlib import Path

    config = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "config.toml").read_text()
    )
    assert config["TTSSettings"]["ALLTALK_TTS_VOICE_DEFAULT"] == "alloy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, expected",
    [
        ({"ALLTALK_TTS_VOICE_DEFAULT": "female_01.wav"}, "alloy"),
        ({"ALLTALK_TTS_VOICE_DEFAULT": "nova"}, "nova"),
        ({"ALLTALK_TTS_VOICE_DEFAULT": "narrator.wav"}, "narrator.wav"),
        ({"ALLTALK_TTS_VOICE": "female_01.wav"}, "female_01.wav"),
    ],
)
async def test_alltalk_inherited_default_and_explicit_voice_are_distinct(
    config, expected
):
    submitted = []

    def handler(request):
        submitted.append(json.loads(request.content)["voice"])
        return httpx.Response(200, content=b"speech")

    backend = AllTalkTTSBackend({"ALLTALK_TTS_URL": "http://alltalk.test", **config})
    await backend.client.aclose()
    backend.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await _synthesize(backend, "default") == b"speech"
        assert await _synthesize(backend, "female_01.wav") == b"speech"
        assert submitted == [expected, "female_01.wav"]
    finally:
        await backend.close()
