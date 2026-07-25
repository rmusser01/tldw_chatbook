from __future__ import annotations

import hashlib
import logging
import re
import struct
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from loguru import logger

import tldw_chatbook.TTS as tts
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import (
    TTSOperationError,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.legacy_bridge import LEGACY_ROUTES
from tldw_chatbook.TTS.TTS_Generation import TTSService

GUIDE_PATH = Path(__file__).parents[2] / "Docs/Development/TTS/TTS_MODULE_GUIDE.md"


class _PrivacyStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body

    async def aclose(self) -> None:
        return


def _privacy_response(
    body: bytes,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    extensions: dict[str, Any] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        extensions=extensions,
        stream=_PrivacyStream(body),
    )


def _exception_graph(error: BaseException) -> list[BaseException]:
    pending = [error]
    seen: set[int] = set()
    graph: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.append(current)
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return graph


def test_tts_package_exports_only_stable_adapter_service_api() -> None:
    expected = {
        "NormalizationOptions",
        "OpenAISpeechRequest",
        "ProgressSink",
        "ProviderHealth",
        "TTSAudioResponse",
        "TTSModelInfo",
        "TTSOperationCode",
        "TTSOperationError",
        "TTSProgress",
        "TTSProviderCatalog",
        "TTSProviderDescriptor",
        "TTSRequest",
        "TTSService",
        "bind_tts_service",
        "close_tts_resources",
        "get_tts_service",
        "reset_tts_service_binding",
    }
    forbidden = {
        "BackendRegistry",
        "LegacyBackendHost",
        "LegacyTTSAdapter",
        "OpenAITTSBackend",
        "TTSBackendBase",
        "TTSBackendManager",
    }

    assert set(tts.__all__) == expected
    assert all(hasattr(tts, name) for name in expected)
    assert all(not hasattr(tts, name) for name in forbidden)


def test_tts_guide_documents_exact_legacy_routes_and_working_example() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")
    architecture = guide.split("### TTS adapter service", 1)[1].split(
        "### Module Structure", 1
    )[0]
    normalized_architecture = " ".join(architecture.split())
    usage = guide.split("### Programmatic Usage", 1)[1].split(
        "### Event System Integration", 1
    )[0]
    routes = guide.split("### Exact legacy route allowlist", 1)[1].split(
        "### Audio Formats", 1
    )[0]
    documented_routes = dict(
        re.findall(r"^- `([^`]+)` → `([^`]+)`$", routes, re.MULTILINE)
    )

    assert (
        "Native adapters use canonical provider IDs and "
        "`TTSService.synthesize()`." in normalized_architecture
    )
    assert (
        "Until `audio_cpp` lands, all currently registered providers are "
        "compatibility adapters and callers use `generate_audio_stream()` "
        "with an enumerated legacy internal model ID." in normalized_architecture
    )
    assert documented_routes == LEGACY_ROUTES
    assert 'internal_model_id = "openai_official_tts-1"' in usage
    assert "generate_audio_stream(request, internal_model_id)" in usage
    assert "tts_service.synthesize(" not in usage


@pytest.mark.asyncio
async def test_openai_backend_never_logs_api_key_details(monkeypatch) -> None:
    secret = "sk-UniquePrefix-Extremely-Private-Suffix"
    messages: list[str] = []
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        lambda: {"api_settings": {"openai": {"api_key": secret}}},
    )

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    backend = None
    try:
        backend = OpenAITTSBackend(config={})
    finally:
        if backend is not None:
            await backend.close()
        logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert secret not in rendered
    assert secret[:10] not in rendered
    assert secret[-10:] not in rendered
    assert hashlib.sha256(secret.encode()).hexdigest() not in rendered
    assert "API key length" not in rendered


@pytest.mark.asyncio
async def test_stts_settings_save_logs_names_and_destinations_not_secrets(
    monkeypatch,
) -> None:
    secrets = {
        "openai_api_key": "sk-OpenAI-UniquePrefix-PrivateSuffix",
        "elevenlabs_api_key": "xi-ElevenLabs-UniquePrefix-PrivateSuffix",
    }
    saved_batches: list[dict[str, dict[str, str]]] = []
    messages: list[str] = []

    class App:
        def notify(self, message: str, *, severity: str) -> None:
            messages.append(f"{severity}: {message}")

    handler = STTSEventHandler(App())

    class Service:
        async def reconfigure_provider(self, provider_id: str, config: object) -> None:
            del provider_id, config

    def save_settings(section_values: dict[str, dict[str, str]]) -> bool:
        saved_batches.append(section_values)
        return True

    handler._stts_service = Service()
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_settings,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        lambda *, force_reload=False: {
            "COMPREHENSIVE_CONFIG_RAW": {"API": secrets, "app_tts": {}}
        },
    )

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await handler.handle_settings_save(STTSSettingsSaveEvent(secrets))
    finally:
        logger.remove(sink_id)

    assert saved_batches == [
        {
            "API": {
                "openai_api_key": secrets["openai_api_key"],
                "elevenlabs_api_key": secrets["elevenlabs_api_key"],
            }
        }
    ]
    rendered = "\n".join(messages)
    assert "Saved openai_api_key to [API].openai_api_key" in rendered
    assert "Saved elevenlabs_api_key to [API].elevenlabs_api_key" in rendered
    for secret in secrets.values():
        assert secret not in rendered
        assert secret[:12] not in rendered
        assert secret[-12:] not in rendered
        assert str(len(secret)) not in rendered
        assert hashlib.sha256(secret.encode()).hexdigest() not in rendered
    assert "length" not in rendered.lower()


@pytest.mark.asyncio
async def test_stts_settings_save_does_not_echo_secret_from_writer_error(
    monkeypatch,
) -> None:
    secret = "sk-WriterError-UniquePrefix-PrivateSuffix"
    messages: list[str] = []

    class App:
        def notify(self, message: str, *, severity: str) -> None:
            messages.append(f"{severity}: {message}")

    handler = STTSEventHandler(App())

    def fail_to_save(section_values: dict[str, dict[str, str]]) -> None:
        raise RuntimeError(f"could not save {section_values['API']['openai_api_key']}")

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        fail_to_save,
    )

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await handler.handle_settings_save(
            STTSSettingsSaveEvent({"openai_api_key": secret})
        )
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert "Failed to save settings" in rendered
    assert secret not in rendered
    assert secret[:12] not in rendered
    assert secret[-12:] not in rendered
    assert str(len(secret)) not in rendered
    assert hashlib.sha256(secret.encode()).hexdigest() not in rendered
    assert "length" not in rendered.lower()


@pytest.mark.asyncio
async def test_stts_settings_save_does_not_echo_reconfiguration_error_secret(
    monkeypatch,
) -> None:
    secret = "sk-Reconfigure-UniquePrefix-PrivateSuffix"
    messages: list[str] = []
    saved_batches: list[dict[str, dict[str, str]]] = []
    get_service_calls = 0

    class App:
        def notify(self, message: str, *, severity: str) -> None:
            messages.append(f"{severity}: {message}")

    handler = STTSEventHandler(App())

    class Service:
        async def reconfigure_provider(
            self,
            provider_id: str,
            config: object,
        ) -> None:
            del provider_id, config
            raise RuntimeError(f"rejected credential {secret}")

    def save_settings(section_values: dict[str, dict[str, str]]) -> bool:
        saved_batches.append(section_values)
        return True

    async def get_bound_service() -> Service:
        nonlocal get_service_calls
        get_service_calls += 1
        return Service()

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_settings,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        lambda *, force_reload=False: {
            "COMPREHENSIVE_CONFIG_RAW": {
                "API": {"openai_api_key": secret},
                "app_tts": {},
            }
        },
    )
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_bound_service,
    )

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await handler.handle_settings_save(
            STTSSettingsSaveEvent({"openai_api_key": secret})
        )
    finally:
        logger.remove(sink_id)

    assert saved_batches == [{"API": {"openai_api_key": secret}}]
    assert get_service_calls == 1
    rendered = "\n".join(messages)
    assert "Failed to reconfigure TTS providers: openai" in rendered
    assert "Settings saved, but some TTS providers could not be updated" in rendered
    assert "rejected credential" not in rendered
    assert secret not in rendered
    assert secret[:12] not in rendered
    assert secret[-12:] not in rendered
    assert str(len(secret)) not in rendered
    assert hashlib.sha256(secret.encode()).hexdigest() not in rendered
    assert "length" not in rendered.lower()


@pytest.mark.asyncio
async def test_audio_cpp_service_boundary_never_exposes_private_http_or_request_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_url = "http://private-audio-origin-sentinel.invalid:8181"
    text = "PRIVATE_SYNTHESIS_TEXT_SENTINEL"
    invalid_model = "PRIVATE_INVALID_MODEL_SENTINEL"
    invalid_voice = "PRIVATE_INVALID_VOICE_SENTINEL\u0000"
    raw_config_value = "PRIVATE_RAW_CONFIG_VALUE_SENTINEL"
    remote_body = "PRIVATE_REMOTE_BODY_SENTINEL"
    remote_reason = "PRIVATE_REMOTE_REASON_SENTINEL"
    remote_cookie = "PRIVATE_REMOTE_COOKIE_SENTINEL"
    speech_calls = 0
    created: list[AudioCppAdapter] = []
    loguru_messages: list[str] = []

    fixture_dir = Path(__file__).parent / "fixtures/audio_cpp_http_v1"
    health = (fixture_dir / "health.json").read_bytes()
    models = (fixture_dir / "models.json").read_bytes()
    model_id = "pocket-tts"
    samples = b"\x00\x00\x01\x00"
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        1,
        24_000,
        48_000,
        2,
        16,
    )
    riff_payload = b"WAVE" + fmt + struct.pack("<4sI", b"data", len(samples)) + samples
    wav = b"RIFF" + struct.pack("<I", len(riff_payload)) + riff_payload

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal speech_calls
        if request.url.path == "/health":
            return _privacy_response(health)
        if request.url.path == "/v1/models":
            return _privacy_response(models)

        speech_calls += 1
        if speech_calls == 1:
            logging.getLogger("httpx").info(
                "HTTP Request: POST %s body=%s",
                request.url,
                remote_body,
            )
            logging.getLogger("httpcore.http11").debug(
                "response headers cookie=%s reason=%s",
                remote_cookie,
                remote_reason,
            )
            body = (
                b'{"error":{"message":"'
                + remote_body.encode()
                + b'","type":"server_error"}}'
            )
            return _privacy_response(
                body,
                status=503,
                headers={"Set-Cookie": f"private={remote_cookie}; Path=/"},
                extensions={"reason_phrase": remote_reason.encode()},
            )
        return _privacy_response(
            wav,
            headers={"Content-Type": "audio/wav"},
        )

    def factory(config: Mapping[str, Any]) -> AudioCppAdapter:
        assert config["private_test_value"] == raw_config_value
        adapter = AudioCppAdapter(
            AudioCppConfig.from_mapping(config),
            transport=httpx.MockTransport(respond),
        )
        created.append(adapter)
        return adapter

    def speech_request(
        *,
        model: str = model_id,
        voice: str | None = None,
    ) -> TTSRequest:
        return TTSRequest(
            provider_id="audio_cpp",
            model_id=model,
            text=text,
            voice=voice,
            response_format="wav",
        )

    initial_config: dict[str, Any] = {
        **AudioCppConfig(base_url=base_url).to_mapping(),
        "private_test_value": raw_config_value,
    }
    service = TTSService(
        TTSAdapterRegistry(
            specs=(
                TTSProviderSpec(
                    descriptor=TTSProviderDescriptor(
                        provider_id="audio_cpp",
                        display_name="audio.cpp",
                        native=True,
                    ),
                    factory=factory,
                    initial_config=initial_config,
                    exclusive_reconfigure=True,
                ),
            ),
            aliases={},
        )
    )
    errors: list[TTSOperationError] = []
    response = None
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(loguru_messages.append, level="DEBUG", format="{message}")
    try:
        requests = (
            speech_request(voice=invalid_voice),
            speech_request(model=invalid_model),
            speech_request(),
        )
        for request in requests:
            with pytest.raises(TTSOperationError) as captured:
                await service.synthesize(request)
            errors.append(captured.value)

        response = await service.synthesize(speech_request())
        assert [chunk async for chunk in response.byte_stream] == [wav]
        catalog = await service.get_catalog("audio_cpp")
        retained_metadata = dict(response.metadata)
        assert len(created[0]._client.cookies) == 0
    finally:
        if response is not None:
            await response.aclose()
        await service.close()
        await service.wait_closed()
        logger.remove(sink_id)

    assert [error.code for error in errors] == [
        "request_invalid",
        "model_invalid",
        "connection_unavailable",
    ]
    assert all(_exception_graph(error) == [error] for error in errors)
    public_output = " ".join(
        (
            caplog.text,
            "\n".join(loguru_messages),
            repr([(error.args, str(error)) for error in errors]),
            repr([_exception_graph(error) for error in errors]),
            repr(catalog),
            repr(retained_metadata),
        )
    )
    private_values = (
        base_url,
        text,
        invalid_model,
        "PRIVATE_INVALID_VOICE_SENTINEL",
        raw_config_value,
        remote_body,
        remote_reason,
        remote_cookie,
    )
    assert all(value not in public_output for value in private_values)
