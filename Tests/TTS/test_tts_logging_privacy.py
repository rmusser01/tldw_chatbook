from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import struct
import sys
import traceback
from collections.abc import AsyncIterator, Awaitable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
from loguru import logger

import tldw_chatbook.TTS as tts
from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend
from tldw_chatbook.TTS.adapter_registry import (
    ReconfigureResult,
    TTSAdapterRegistry,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSAudioResponse,
    TTSConfigurationRevisionError,
    TTSOperationError,
    TTSProviderDescriptor,
    TTSProviderReconfiguringError,
    TTSProviderSpec,
    TTSProviderUnavailableError,
    TTSRequest,
)
from tldw_chatbook.TTS.adapters import audio_cpp as audio_cpp_module
from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.legacy_bridge import LEGACY_ROUTES
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import (
    TTSService,
    TTSSettingsPublicationTicket,
)

GUIDE_PATH = Path(__file__).parents[2] / "Docs/Development/TTS/TTS_MODULE_GUIDE.md"
_TEST_WAIT_SECONDS = 2.0


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


async def _capture_bounded_cleanup(
    awaitable: Awaitable[Any],
    errors: list[BaseException],
) -> None:
    task = asyncio.ensure_future(awaitable)
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=_TEST_WAIT_SECONDS,
        )
    except BaseException as error:
        errors.append(error)
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True),
                    timeout=_TEST_WAIT_SECONDS,
                )
            except BaseException as join_error:
                errors.append(join_error)


async def _cleanup_audio_cpp_privacy_resources(
    service: TTSService,
    adapters: list[AudioCppAdapter],
    response: TTSAudioResponse | None,
) -> list[BaseException]:
    errors: list[BaseException] = []
    if response is not None:
        await _capture_bounded_cleanup(response.aclose(), errors)
    await _capture_bounded_cleanup(service.close(), errors)
    await _capture_bounded_cleanup(service.wait_closed(), errors)
    for adapter in adapters:
        await _capture_bounded_cleanup(adapter.close(), errors)

    for adapter in adapters:
        privacy_filter = adapter._httpx_privacy_filter
        leaked = False
        for logger_name in audio_cpp_module._HTTP_LOGGER_NAMES:
            http_logger = logging.getLogger(logger_name)
            if privacy_filter in http_logger.filters:
                leaked = True
                http_logger.removeFilter(privacy_filter)
        if leaked:
            errors.append(AssertionError("audio.cpp HTTP privacy filter leaked"))
    return errors


def test_tts_package_exports_only_stable_adapter_service_api() -> None:
    expected = {
        "NormalizationOptions",
        "OpenAISpeechRequest",
        "ProgressSink",
        "ProviderHealth",
        "STTSGeneratedAudio",
        "STTSPlaygroundRequest",
        "TTSAudioResponse",
        "TTSConfigMutation",
        "TTSModelInfo",
        "TTSOperationCode",
        "TTSOperationError",
        "TTSPreferencesSnapshot",
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
        "`audio_cpp` is the first native adapter. It is registered first, by the "
        "exact canonical ID `audio_cpp`, with display label `audio.cpp` and no alias."
        in normalized_architecture
    )
    assert (
        "The following six entries remain unchanged behind the temporary "
        "compatibility bridge: `openai`, `elevenlabs`, `kokoro`, `chatterbox`, "
        "`higgs`, and `alltalk`." in normalized_architecture
    )
    assert documented_routes == LEGACY_ROUTES
    assert 'internal_model_id = "openai_official_tts-1"' in usage
    assert "generate_audio_stream(request, internal_model_id)" in usage
    assert "tts_service.synthesize(" not in usage


@pytest.mark.asyncio
async def test_registry_admission_errors_expose_no_configuration_values() -> None:
    private_values = (
        "http://private-audio-origin.invalid:8181",
        "PRIVATE_AUDIO_CPP_CREDENTIAL",
        "PRIVATE_REPLACEMENT_VALUE",
    )
    factory = FakeAdapterFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                {
                    "origin": private_values[0],
                    "credential": private_values[1],
                },
                exclusive=True,
            ),
        ),
        aliases={},
    )
    errors: list[BaseException] = []
    await registry.reconfigure_provider(
        "audio_cpp",
        {"value": private_values[2]},
    )
    with pytest.raises(TTSConfigurationRevisionError) as revision:
        await registry.acquire("audio_cpp", expected_revision=1)
    errors.append(revision.value)

    await registry.seal_provider_unavailable("audio_cpp")
    with pytest.raises(TTSProviderUnavailableError) as unavailable:
        await registry.acquire("audio_cpp", expected_revision=1)
    errors.append(unavailable.value)

    assert (
        await registry.reconfigure_provider("audio_cpp", {"revision": 3})
        is ReconfigureResult.CHANGED
    )
    lease = await registry.acquire("audio_cpp", expected_revision=3)
    reconfigure = asyncio.create_task(
        registry.reconfigure_provider("audio_cpp", {"revision": 4})
    )
    await asyncio.sleep(0)
    with pytest.raises(TTSProviderReconfiguringError) as pending:
        await registry.acquire("audio_cpp", expected_revision=2)
    errors.append(pending.value)
    await lease.release()
    assert await reconfigure is ReconfigureResult.CHANGED
    await registry.close()

    exception_graphs = [_exception_graph(error) for error in errors]
    rendered = " ".join(
        (
            repr([(type(error), error.args) for error in errors]),
            repr(exception_graphs),
            "\n".join("".join(traceback.format_exception(error)) for error in errors),
        )
    )
    assert [type(error) for error in errors] == [
        TTSConfigurationRevisionError,
        TTSProviderUnavailableError,
        TTSProviderReconfiguringError,
    ]
    assert [str(error) for error in errors] == [
        "TTS provider configuration changed: audio_cpp",
        "TTS provider is unavailable: audio_cpp",
        "TTS provider is reconfiguring: audio_cpp",
    ]
    assert all(graph == [error] for graph, error in zip(exception_graphs, errors))
    assert all(private_value not in rendered for private_value in private_values)


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
    from tldw_chatbook import config as config_module

    secrets = {
        "openai_api_key": "sk-OpenAI-UniquePrefix-PrivateSuffix",
        "elevenlabs_api_key": "xi-ElevenLabs-UniquePrefix-PrivateSuffix",
    }
    saved_batches: list[dict[str, dict[str, Any]]] = []
    saved_deletes: list[dict[str, tuple[str, ...]]] = []
    attempted_configs: list[tuple[str, dict[str, Any]]] = []
    messages: list[str] = []
    posted_messages: list[object] = []
    current_settings: dict[str, Any] = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "API": {},
            "app_tts": {},
        }
    }

    class App:
        def notify(self, message: str, *, severity: str) -> None:
            messages.append(f"{severity}: {message}")

        def post_message(self, message: object) -> bool:
            posted_messages.append(message)
            return True

    class CapturingRegistry(TTSAdapterRegistry):
        async def begin_reconfigure_provider(
            self,
            provider_id: str,
            config: Mapping[str, Any],
            *,
            generation: int | None = None,
        ) -> Any:
            attempted_configs.append((provider_id, deepcopy(dict(config))))
            return await super().begin_reconfigure_provider(
                provider_id,
                config,
                generation=generation,
            )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    registry = CapturingRegistry(
        specs=tuple(
            provider_spec(provider_id, FakeAdapterFactory(provider_id))
            for provider_id in ("openai", "elevenlabs")
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot.from_settings(current_settings),
    )
    handler = STTSEventHandler(App())
    handler._stts_service = service

    def apply_settings(
        section_values: Mapping[str, Mapping[Any, Any]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> Any:
        saved_batches.append(
            {
                section: deepcopy(dict(values))
                for section, values in section_values.items()
            }
        )
        saved_deletes.append(deepcopy(dict(delete_keys)))
        return config_module.ConfigMutationResult(True, True, None)

    monkeypatch.setattr(config_module, "settings", current_settings)
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        apply_settings,
    )

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await asyncio.wait_for(
            handler.handle_settings_save(STTSSettingsSaveEvent(secrets)),
            timeout=_TEST_WAIT_SECONDS,
        )
    finally:
        logger.remove(sink_id)
        await asyncio.wait_for(service.close(), timeout=_TEST_WAIT_SECONDS)
        await asyncio.wait_for(service.wait_closed(), timeout=_TEST_WAIT_SECONDS)

    assert len(saved_batches) == 1
    assert saved_batches[0]["API"] == secrets
    assert saved_deletes == [{}]
    assert service.preferences_generation() == 1
    assert registry.configuration_revision("openai") == 2
    assert registry.configuration_revision("elevenlabs") == 2
    assert [provider_id for provider_id, _config in attempted_configs] == [
        "openai",
        "elevenlabs",
    ]
    configs_by_provider = dict(attempted_configs)
    assert (
        configs_by_provider["openai"]["app_config"]["openai_api"]["api_key"]
        == secrets["openai_api_key"]
    )
    assert (
        configs_by_provider["elevenlabs"]["app_config"]["elevenlabs_api"]["api_key"]
        == secrets["elevenlabs_api_key"]
    )
    assert len(posted_messages) == 2
    rendered = "\n".join(messages)
    assert "Saved openai_api_key to [API].openai_api_key" in rendered
    assert "Saved elevenlabs_api_key to [API].elevenlabs_api_key" in rendered
    assert "information: Settings saved successfully!" in rendered
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
    from tldw_chatbook import config as config_module

    secret = "sk-WriterError-UniquePrefix-PrivateSuffix"
    log_messages: list[str] = []
    notifications: list[tuple[str, str]] = []
    posted_messages: list[object] = []
    attempted_batches: list[dict[str, dict[str, Any]]] = []
    captured_ticket: TTSSettingsPublicationTicket | None = None
    current_settings: dict[str, Any] = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "API": {},
            "app_tts": {},
        }
    }

    class App:
        def notify(self, message: str, *, severity: str) -> None:
            notifications.append((message, severity))

        def post_message(self, message: object) -> bool:
            posted_messages.append(message)
            return True

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    registry = TTSAdapterRegistry(
        specs=(provider_spec("openai", FakeAdapterFactory("openai")),),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot.from_settings(current_settings),
    )
    original_begin = service.begin_preferences_publication

    def begin_publication(*args: Any, **kwargs: Any) -> TTSSettingsPublicationTicket:
        nonlocal captured_ticket
        captured_ticket = original_begin(*args, **kwargs)
        return captured_ticket

    service.begin_preferences_publication = begin_publication  # type: ignore[method-assign]
    handler = STTSEventHandler(App())
    handler._stts_service = service

    def fail_to_save(
        section_values: Mapping[str, Mapping[Any, Any]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> Any:
        del delete_keys
        attempted_batches.append(
            {
                section: deepcopy(dict(values))
                for section, values in section_values.items()
            }
        )
        raise RuntimeError(f"could not save {section_values['API']['openai_api_key']}")

    monkeypatch.setattr(config_module, "settings", current_settings)
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        fail_to_save,
    )

    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    completion = None
    try:
        await asyncio.wait_for(
            handler.handle_settings_save(
                STTSSettingsSaveEvent({"openai_api_key": secret})
            ),
            timeout=_TEST_WAIT_SECONDS,
        )
        assert captured_ticket is not None
        completion = await asyncio.wait_for(
            asyncio.shield(captured_ticket.completion),
            timeout=_TEST_WAIT_SECONDS,
        )
    finally:
        logger.remove(sink_id)
        await asyncio.wait_for(service.close(), timeout=_TEST_WAIT_SECONDS)
        await asyncio.wait_for(service.wait_closed(), timeout=_TEST_WAIT_SECONDS)

    assert len(attempted_batches) == 1
    assert attempted_batches[0]["API"] == {"openai_api_key": secret}
    assert completion is not None
    assert completion.published is False
    assert completion.persistence.file_replaced is False
    assert completion.persistence.caches_reloaded is False
    assert completion.persistence.failure_phase == "before_replace"
    assert completion.provider_statuses == {"openai": "unchanged"}
    assert captured_ticket is not None
    assert captured_ticket.completion.exception() is None
    assert registry.configuration_revision("openai") == 1
    assert posted_messages == []
    assert notifications == [("Failed to save settings", "error")]
    rendered = "\n".join(
        [
            *log_messages,
            *(f"{severity}: {message}" for message, severity in notifications),
            repr(completion),
        ]
    )
    assert "Failed to save settings" in rendered
    assert "could not save" not in rendered
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
    from tldw_chatbook import config as config_module

    secret = "sk-Reconfigure-UniquePrefix-PrivateSuffix"
    log_messages: list[str] = []
    notifications: list[tuple[str, str]] = []
    posted_messages: list[object] = []
    saved_batches: list[dict[str, dict[str, Any]]] = []
    attempted_configs: list[tuple[str, dict[str, Any]]] = []
    captured_ticket: TTSSettingsPublicationTicket | None = None
    current_settings: dict[str, Any] = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "API": {},
            "app_tts": {},
        }
    }

    class App:
        def notify(self, message: str, *, severity: str) -> None:
            notifications.append((message, severity))

        def post_message(self, message: object) -> bool:
            posted_messages.append(message)
            return True

    class SecretFailingRegistry(TTSAdapterRegistry):
        async def begin_reconfigure_provider(
            self,
            provider_id: str,
            config: Mapping[str, Any],
            *,
            generation: int | None = None,
        ) -> Any:
            del generation
            attempted_configs.append((provider_id, deepcopy(dict(config))))
            raise RuntimeError(f"rejected credential {secret}")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    registry = SecretFailingRegistry(
        specs=(provider_spec("openai", FakeAdapterFactory("openai")),),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot.from_settings(current_settings),
    )
    original_begin = service.begin_preferences_publication

    def begin_publication(*args: Any, **kwargs: Any) -> TTSSettingsPublicationTicket:
        nonlocal captured_ticket
        captured_ticket = original_begin(*args, **kwargs)
        return captured_ticket

    service.begin_preferences_publication = begin_publication  # type: ignore[method-assign]
    handler = STTSEventHandler(App())
    handler._stts_service = service

    def apply_settings(
        section_values: Mapping[str, Mapping[Any, Any]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> Any:
        del delete_keys
        saved_batches.append(
            {
                section: deepcopy(dict(values))
                for section, values in section_values.items()
            }
        )
        return config_module.ConfigMutationResult(True, True, None)

    monkeypatch.setattr(config_module, "settings", current_settings)
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        apply_settings,
    )

    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")
    completion = None
    try:
        await asyncio.wait_for(
            handler.handle_settings_save(
                STTSSettingsSaveEvent({"openai_api_key": secret})
            ),
            timeout=_TEST_WAIT_SECONDS,
        )
        assert captured_ticket is not None
        completion = await asyncio.wait_for(
            asyncio.shield(captured_ticket.completion),
            timeout=_TEST_WAIT_SECONDS,
        )
    finally:
        logger.remove(sink_id)
        await asyncio.wait_for(service.close(), timeout=_TEST_WAIT_SECONDS)
        await asyncio.wait_for(service.wait_closed(), timeout=_TEST_WAIT_SECONDS)

    assert len(saved_batches) == 1
    assert saved_batches[0]["API"] == {"openai_api_key": secret}
    assert attempted_configs[0][0] == "openai"
    assert attempted_configs[0][1]["app_config"]["openai_api"]["api_key"] == secret
    assert completion is not None
    assert completion.provider_statuses == {"openai": "unavailable"}
    assert captured_ticket is not None
    assert captured_ticket.completion.exception() is None
    assert posted_messages == []
    assert notifications == [
        (
            "Settings saved, but TTS is unavailable. Retry/Reconnect.",
            "error",
        )
    ]
    rendered = "\n".join(
        [
            *log_messages,
            *(f"{severity}: {message}" for message, severity in notifications),
            repr(completion),
        ]
    )
    assert "Saved openai_api_key to [API].openai_api_key" in rendered
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
        assert "cookie" not in request.headers
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
    response: TTSAudioResponse | None = None
    caplog.set_level(logging.DEBUG)
    sink_id = logger.add(loguru_messages.append, level="DEBUG", format="{message}")
    try:
        try:
            async with asyncio.timeout(_TEST_WAIT_SECONDS):
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
        finally:
            primary_error = sys.exception()
            cleanup_errors = await _cleanup_audio_cpp_privacy_resources(
                service,
                created,
                response,
            )
            if primary_error is None and cleanup_errors:
                raise cleanup_errors[0]
    finally:
        logger.remove(sink_id)

    assert [error.code for error in errors] == [
        "request_invalid",
        "model_invalid",
        "connection_unavailable",
    ]
    exception_graphs = [_exception_graph(error) for error in errors]
    assert all(graph == [error] for graph, error in zip(exception_graphs, errors))
    exception_notes = [
        getattr(node, "__notes__", ()) for graph in exception_graphs for node in graph
    ]
    rendered_tracebacks = [
        "".join(traceback.format_exception(error)) for error in errors
    ]
    public_output = " ".join(
        (
            caplog.text,
            "\n".join(loguru_messages),
            repr([(error.args, str(error)) for error in errors]),
            repr(exception_graphs),
            repr(exception_notes),
            "\n".join(rendered_tracebacks),
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
