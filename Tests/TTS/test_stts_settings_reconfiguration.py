from __future__ import annotations

import asyncio
import threading
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

from Tests.TTS.adapter_fakes import FakeAdapter, FakeAdapterFactory, provider_spec
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSProviderConfigurationChanged,
    STTSEventHandler,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.adapter_registry import ReconfigureResult, TTSAdapterRegistry
from tldw_chatbook.TTS.audio_cpp_config import (
    AudioCppConfig,
    project_audio_cpp_config,
)
from tldw_chatbook.TTS.legacy_bridge import (
    legacy_provider_config,
    legacy_provider_specs,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import (
    TTSService,
    TTSSettingsPersistenceOutcome,
    TTSSettingsPublication,
    TTSSettingsPublicationTicket,
)


class RecordingFactory(FakeAdapterFactory):
    def __init__(self, provider_id: str) -> None:
        super().__init__(provider_id)
        self.configs: list[dict[str, Any]] = []

    def __call__(self, config: Mapping[str, Any]) -> FakeAdapter:
        self.configs.append(deepcopy(dict(config)))
        return super().__call__(config)


class RecordingApp:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []
        self.messages: list[object] = []

    def notify(self, message: str, *, severity: str) -> None:
        self.notifications.append((message, severity))

    def post_message(self, message: object) -> bool:
        self.messages.append(message)
        return True


class ImmediatePublicationService:
    """Small handler-contract fake whose publication task owns async work."""

    def __init__(self, reconfigure_provider: Any | None = None) -> None:
        self.reconfigure_provider = reconfigure_provider or AsyncMock(
            return_value=ReconfigureResult.CHANGED
        )
        self._generation = 0
        self._published_generation = 0
        self._revisions: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def begin_preferences_publication(
        self,
        preferences: TTSPreferencesSnapshot,
        provider_configs: Mapping[str, Mapping[str, Any]],
        persistence: Any,
        **_kwargs: Any,
    ) -> TTSSettingsPublicationTicket:
        self._generation += 1
        generation = self._generation
        foreground: asyncio.Future[TTSSettingsPublication] = (
            asyncio.get_running_loop().create_future()
        )

        async def run() -> TTSSettingsPublication:
            async with self._lock:
                outcome = await asyncio.to_thread(persistence)
                statuses: dict[str, str] = {}
                if outcome.file_replaced:
                    for provider_id, config in provider_configs.items():
                        try:
                            result = await self.reconfigure_provider(
                                provider_id,
                                config,
                            )
                        except BaseException:
                            statuses[provider_id] = "unavailable"
                        else:
                            if result is ReconfigureResult.UNCHANGED:
                                statuses[provider_id] = "unchanged"
                            elif result is ReconfigureResult.SUPERSEDED:
                                statuses[provider_id] = "superseded"
                            else:
                                statuses[provider_id] = "applied"
                                self._revisions[provider_id] = (
                                    self._revisions.get(provider_id, 1) + 1
                                )
                    self._published_generation = generation
                else:
                    statuses = {
                        provider_id: "unchanged" for provider_id in provider_configs
                    }
                publication = TTSSettingsPublication(
                    generation=generation,
                    preferences=preferences,
                    persistence=outcome,
                    provider_statuses=statuses,  # type: ignore[arg-type]
                    provider_revisions=self._revisions,
                    published=outcome.file_replaced,
                )
                foreground.set_result(publication)
                return publication

        completion = asyncio.create_task(run())
        return TTSSettingsPublicationTicket(generation, foreground, completion)

    def preferences_generation(self) -> int:
        return self._published_generation


def _mutation_outcome(
    *,
    file_replaced: bool = True,
    caches_reloaded: bool = True,
    failure_phase: str | None = None,
) -> TTSSettingsPersistenceOutcome:
    return TTSSettingsPersistenceOutcome(
        file_replaced=file_replaced,
        caches_reloaded=caches_reloaded,
        failure_phase=failure_phase,  # type: ignore[arg-type]
    )


PROVIDER_SETTING_KEYS = {
    "openai": ("openai_api_key",),
    "elevenlabs": (
        "elevenlabs_api_key",
        "ELEVENLABS_DEFAULT_MODEL",
        "ELEVENLABS_OUTPUT_FORMAT",
        "ELEVENLABS_VOICE_STABILITY",
        "ELEVENLABS_SIMILARITY_BOOST",
        "ELEVENLABS_STYLE",
        "ELEVENLABS_USE_SPEAKER_BOOST",
    ),
    "kokoro": (
        "KOKORO_DEVICE_DEFAULT",
        "KOKORO_USE_ONNX",
        "KOKORO_ONNX_MODEL_PATH_DEFAULT",
        "KOKORO_ONNX_VOICES_JSON_DEFAULT",
        "KOKORO_MAX_TOKENS",
        "KOKORO_ENABLE_VOICE_MIXING",
        "KOKORO_TRACK_PERFORMANCE",
    ),
    "higgs": (
        "HIGGS_MODEL_PATH",
        "HIGGS_VOICE_SAMPLES_DIR",
        "HIGGS_DEVICE",
        "HIGGS_ENABLE_FLASH_ATTN",
        "HIGGS_DTYPE",
        "HIGGS_MAX_REFERENCE_DURATION",
        "HIGGS_DEFAULT_LANGUAGE",
        "HIGGS_ENABLE_VOICE_CLONING",
        "HIGGS_ENABLE_MULTI_SPEAKER",
        "HIGGS_SPEAKER_DELIMITER",
        "HIGGS_TRACK_PERFORMANCE",
        "HIGGS_MAX_NEW_TOKENS",
        "HIGGS_TEMPERATURE",
        "HIGGS_TOP_P",
        "HIGGS_REPETITION_PENALTY",
    ),
}


def test_settings_save_event_copies_mapping_and_carries_preferences() -> None:
    preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    provider_settings = {
        "audio_cpp": {
            "mode": "external",
            "base_url": "http://127.0.0.1:8080",
        }
    }

    event = STTSSettingsSaveEvent(
        MappingProxyType(provider_settings),
        preferences=preferences,
    )
    provider_settings["audio_cpp"]["base_url"] = "http://mutated.invalid"

    assert event.settings["audio_cpp"]["base_url"] == "http://127.0.0.1:8080"
    assert event.preferences is preferences


def test_settings_save_event_defaults_to_provider_settings_only() -> None:
    event = STTSSettingsSaveEvent({"audio_cpp": AudioCppConfig().to_mapping()})

    assert event.preferences is None


@pytest.mark.asyncio
async def test_provider_setting_reconfigures_only_current_materialized_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_snapshot = {
        "API": {"openai_api_key": "old"},
        "app_tts": {"default_format": "mp3"},
    }
    current_settings = {"COMPREHENSIVE_CONFIG_RAW": initial_snapshot}
    openai_factory = RecordingFactory("openai")
    kokoro_factory = RecordingFactory("kokoro")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "openai",
                openai_factory,
                legacy_provider_config("openai", initial_snapshot),
            ),
            provider_spec(
                "kokoro",
                kokoro_factory,
                legacy_provider_config("kokoro", initial_snapshot),
            ),
        ),
        aliases={},
    )
    service = TTSService(registry)
    for provider_id in ("openai", "kokoro"):
        lease = await registry.acquire(provider_id)
        await lease.release()

    handler = STTSEventHandler(RecordingApp())
    handler._stts_service = service
    saved_batches: list[dict[str, dict[str, object]]] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_batches.append(deepcopy(dict(section_values)))
        return _mutation_outcome()

    initialize_stts = AsyncMock(side_effect=AssertionError("service rebuilt"))
    get_bound_service = AsyncMock(side_effect=AssertionError("accessor used"))
    monkeypatch.setattr(handler, "initialize_stts", initialize_stts)
    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr("tldw_chatbook.config.settings", current_settings)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_bound_service,
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent({"openai_api_key": "new"}))

    assert saved_batches[0]["API"] == {"openai_api_key": "new"}
    assert openai_factory.instances[0].close_calls == 1
    assert kokoro_factory.instances[0].close_calls == 0
    assert registry.configuration_revision("openai") == 2
    assert registry.configuration_revision("kokoro") == 1
    replacement = await registry.acquire("openai")
    await replacement.release()
    assert openai_factory.configs[-1] == legacy_provider_config(
        "openai",
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "API": {"openai_api_key": "new"},
                "app_tts": {
                    "default_provider": "openai",
                    "default_model_mode": "exact",
                    "default_voice_mode": "exact",
                    "default_format": "mp3",
                    "default_speed": 1.0,
                    "default_model": "tts-1-hd",
                    "default_voice": "shimmer",
                },
                "tts_settings": {
                    "default_tts_provider": "openai",
                    "default_openai_tts_output_format": "mp3",
                    "default_openai_tts_speed": 1.0,
                    "default_openai_tts_model": "tts-1-hd",
                    "default_tts_voice": "shimmer",
                },
            }
        },
    )
    assert openai_factory.calls == 2
    assert kokoro_factory.calls == 1
    initialize_stts.assert_not_awaited()
    get_bound_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_mixed_provider_save_retires_only_effectively_changed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "environment-elevenlabs-key")
    initial_raw = {
        "API": {
            "openai_api_key": "stable-openai-key",
            "elevenlabs_api_key": "stored-elevenlabs-key-before",
        },
        "app_tts": {
            "ELEVENLABS_DEFAULT_MODEL": "eleven_multilingual_v2",
            "KOKORO_DEVICE_DEFAULT": "cpu",
        },
        "HiggsSettings": {"device": "cpu"},
    }
    effective_raw = deepcopy(initial_raw)
    effective_raw["API"]["elevenlabs_api_key"] = "stored-elevenlabs-key-after"
    effective_raw["app_tts"]["KOKORO_DEVICE_DEFAULT"] = "mps"

    def normalized(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "COMPREHENSIVE_CONFIG_RAW": raw,
            "APP_TTS_CONFIG": {"KOKORO_DEVICE_DEFAULT": "cpu"},
            "openai_api": {"api_key": "stable-openai-key"},
            "elevenlabs_api": {"api_key": "environment-elevenlabs-key"},
        }

    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            normalized(initial_raw),
            manager_factory=Mock(side_effect=AssertionError("manager constructed")),
        ),
        aliases={},
    )
    materialized: dict[str, Any] = {}
    for provider_id in ("elevenlabs", "kokoro", "higgs"):
        lease = await registry.acquire(provider_id)
        materialized[provider_id] = lease.adapter
        await lease.release()

    service = TTSService(registry)
    handler = STTSEventHandler(RecordingApp())
    handler._stts_service = service
    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        Mock(return_value=_mutation_outcome()),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        normalized(initial_raw),
    )
    event_settings = {
        "openai_api_key": "stable-openai-key",
        "elevenlabs_api_key": "stored-elevenlabs-key-after",
        "KOKORO_DEVICE_DEFAULT": "mps",
        "HIGGS_DEVICE": "cpu",
    }

    await handler.handle_settings_save(STTSSettingsSaveEvent(event_settings))

    assert registry.configuration_revision("openai") == 1
    assert registry.configuration_revision("elevenlabs") == 1
    assert registry.configuration_revision("kokoro") == 2
    assert registry.configuration_revision("higgs") == 1
    assert (
        legacy_provider_config(
            "kokoro",
            normalized(effective_raw),
        )["app_config"]["app_tts"]["KOKORO_DEVICE_DEFAULT"]
        == "mps"
    )
    assert registry._slots["openai"].active is None
    assert materialized["elevenlabs"].host._closed is False
    assert materialized["kokoro"].host._closed is True
    assert materialized["higgs"].host._closed is False

    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_recognized_keys_reconfigure_each_candidate_once_in_provider_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = STTSEventHandler(RecordingApp())
    reconfigure_provider = AsyncMock()
    handler._stts_service = ImmediatePublicationService(reconfigure_provider)
    snapshot = {"API": {"openai_api_key": "secret"}}
    event_settings = {
        key: "value"
        for provider_id in reversed(tuple(PROVIDER_SETTING_KEYS))
        for key in PROVIDER_SETTING_KEYS[provider_id]
    }
    event_settings["default_provider"] = "openai"
    saved_batches: list[dict[str, dict[str, object]]] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_batches.append(deepcopy(dict(section_values)))
        return _mutation_outcome()

    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        {"COMPREHENSIVE_CONFIG_RAW": snapshot},
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent(event_settings))

    assert saved_batches[0]["API"] == {
        "elevenlabs_api_key": "value",
        "openai_api_key": "value",
    }
    assert saved_batches[0]["app_tts"]["default_provider"] == "openai"
    assert saved_batches[0]["app_tts"]["default_model_mode"] == "exact"
    assert saved_batches[0]["tts_settings"]["default_tts_provider"] == "openai"
    assert [call_.args[0] for call_ in reconfigure_provider.await_args_list] == list(
        PROVIDER_SETTING_KEYS
    )
    assert (
        reconfigure_provider.await_args_list[0].args[1]["app_config"]["openai_api"][
            "api_key"
        ]
        == "value"
    )


@pytest.mark.asyncio
async def test_defaults_only_save_reloads_once_without_reconfiguring_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    reconfigure_provider = AsyncMock()
    handler._stts_service = ImmediatePublicationService(reconfigure_provider)
    saved_batches: list[dict[str, dict[str, object]]] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_batches.append(deepcopy(dict(section_values)))
        return _mutation_outcome()

    initialize_stts = AsyncMock(side_effect=AssertionError("service rebuilt"))
    monkeypatch.setattr(handler, "initialize_stts", initialize_stts)
    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {}}},
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "default_provider": "kokoro",
                "default_voice": "af_heart",
                "default_model": "model",
                "default_format": "wav",
                "default_speed": 1.25,
                "unrecognized": "ignored",
            }
        )
    )

    assert saved_batches[0]["app_tts"] == {
        "default_provider": "kokoro",
        "default_voice": "af_heart",
        "default_model": "model",
        "default_format": "wav",
        "default_speed": 1.25,
        "default_model_mode": "exact",
        "default_voice_mode": "exact",
    }
    assert saved_batches[0]["tts_settings"] == {
        "default_tts_provider": "kokoro",
        "default_tts_voice": "af_heart",
        "default_openai_tts_model": "model",
        "default_openai_tts_output_format": "wav",
        "default_openai_tts_speed": 1.25,
    }
    reconfigure_provider.assert_not_awaited()
    initialize_stts.assert_not_awaited()
    assert app.notifications == [
        ("Settings saved successfully!", "information"),
    ]


@pytest.mark.asyncio
async def test_failed_atomic_batch_stops_before_reload_and_reconfiguration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    reconfigure_provider = AsyncMock()
    handler._stts_service = ImmediatePublicationService(reconfigure_provider)
    saved_batches: list[dict[str, dict[str, object]]] = []

    def fail_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_batches.append(deepcopy(dict(section_values)))
        return _mutation_outcome(
            file_replaced=False,
            caches_reloaded=False,
            failure_phase="before_replace",
        )

    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        fail_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        {"COMPREHENSIVE_CONFIG_RAW": {}},
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "default_provider": "kokoro",
                "openai_api_key": "secret",
                "KOKORO_DEVICE_DEFAULT": "cpu",
            }
        )
    )

    assert saved_batches[0]["API"] == {"openai_api_key": "secret"}
    assert saved_batches[0]["app_tts"]["default_provider"] == "kokoro"
    assert saved_batches[0]["app_tts"]["KOKORO_DEVICE_DEFAULT"] == "cpu"
    reconfigure_provider.assert_not_awaited()
    assert app.notifications == [("Failed to save settings", "error")]


@pytest.mark.asyncio
async def test_provider_failure_does_not_skip_later_candidate_or_expose_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sk-Reconfigure-Private-Value"
    app = RecordingApp()
    attempts: list[str] = []

    class Service:
        async def reconfigure_provider(
            self,
            provider_id: str,
            config: object,
        ) -> None:
            del config
            attempts.append(provider_id)
            if provider_id == "openai":
                raise RuntimeError(f"rejected credential {secret}")

    handler = STTSEventHandler(app)
    handler._stts_service = ImmediatePublicationService(Service().reconfigure_provider)
    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        lambda _section_values, *, delete_keys: _mutation_outcome(),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "API": {"openai_api_key": secret},
                "app_tts": {"KOKORO_DEVICE_DEFAULT": "cpu"},
            }
        },
    )
    messages: list[str] = []

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await handler.handle_settings_save(
            STTSSettingsSaveEvent(
                {
                    "openai_api_key": secret,
                    "KOKORO_DEVICE_DEFAULT": "cpu",
                }
            )
        )
    finally:
        logger.remove(sink_id)

    assert attempts == ["openai", "kokoro"]
    assert app.notifications == [
        (
            "Settings saved, but TTS is unavailable. Retry/Reconnect.",
            "error",
        )
    ]
    rendered = "\n".join(messages + [message for message, _ in app.notifications])
    assert "Retry/Reconnect" in rendered
    assert "rejected credential" not in rendered
    assert secret not in rendered


@pytest.mark.asyncio
async def test_connection_and_local_provider_settings_persist_and_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    reconfigure_provider = AsyncMock()
    handler._stts_service = ImmediatePublicationService(reconfigure_provider)
    saved_batches: list[dict[str, dict[str, object]]] = []
    snapshot = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "OPENAI_BASE_URL": "http://127.0.0.1:9000/v1/audio/speech",
                "OPENAI_ORG_ID": "",
                "CHATTERBOX_DEVICE": "cpu",
                "ALLTALK_TTS_URL_DEFAULT": "http://127.0.0.1:7851",
            }
        }
    }

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_batches.append(deepcopy(dict(section_values)))
        return _mutation_outcome()

    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        snapshot,
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "OPENAI_BASE_URL": "http://127.0.0.1:9000/v1/audio/speech",
                "OPENAI_ORG_ID": "",
                "CHATTERBOX_DEVICE": "cpu",
                "ALLTALK_TTS_URL_DEFAULT": "http://127.0.0.1:7851",
            }
        )
    )

    assert (
        saved_batches[0]["app_tts"].items()
        >= {
            "OPENAI_BASE_URL": "http://127.0.0.1:9000/v1/audio/speech",
            "OPENAI_ORG_ID": "",
            "CHATTERBOX_DEVICE": "cpu",
            "ALLTALK_TTS_URL_DEFAULT": "http://127.0.0.1:7851",
        }.items()
    )
    assert [call_.args[0] for call_ in reconfigure_provider.await_args_list] == [
        "openai",
        "chatterbox",
        "alltalk",
    ]
    assert app.notifications == [
        ("Settings saved successfully!", "information"),
    ]


@pytest.mark.asyncio
async def test_concurrent_settings_saves_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    first_reconfigure_started = asyncio.Event()
    allow_first_reconfigure = asyncio.Event()
    reconfigure_calls = 0

    async def reconfigure_provider(_provider_id: str, _config: object) -> None:
        nonlocal reconfigure_calls
        reconfigure_calls += 1
        if reconfigure_calls == 1:
            first_reconfigure_started.set()
            await allow_first_reconfigure.wait()

    handler._stts_service = ImmediatePublicationService(reconfigure_provider)
    saved_values: list[str] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_values.append(str(section_values["API"]["openai_api_key"]))
        return _mutation_outcome()

    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        {"COMPREHENSIVE_CONFIG_RAW": {"API": {}}},
    )

    first = asyncio.create_task(
        handler.handle_settings_save(STTSSettingsSaveEvent({"openai_api_key": "first"}))
    )
    await first_reconfigure_started.wait()
    second = asyncio.create_task(
        handler.handle_settings_save(
            STTSSettingsSaveEvent({"openai_api_key": "second"})
        )
    )
    await asyncio.sleep(0)

    assert saved_values == ["first"]

    allow_first_reconfigure.set()
    await asyncio.gather(first, second)

    assert saved_values == ["first", "second"]
    assert reconfigure_calls == 2


@pytest.mark.asyncio
async def test_audio_cpp_save_persists_nested_plain_mapping_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = AudioCppConfig(
        base_url="https://voice.example.test:8443",
        connect_timeout_seconds=2.5,
        synthesis_timeout_seconds=45,
        max_input_characters=1234,
        max_response_bytes=1_048_576,
        max_metadata_bytes=4096,
        max_catalog_models=12,
        max_voices_per_model=34,
        max_identifier_characters=128,
    ).to_mapping()
    expected = deepcopy(candidate)
    effective = {
        "COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": deepcopy(expected)}}
    }
    saved_batches: list[dict[str, dict[str, object]]] = []
    reconfigure_provider = AsyncMock(return_value=ReconfigureResult.CHANGED)
    service = ImmediatePublicationService(reconfigure_provider)
    service.get_catalog = AsyncMock(side_effect=AssertionError("catalog requested"))
    service.get_voices = AsyncMock(side_effect=AssertionError("voices requested"))
    service.synthesize = AsyncMock(side_effect=AssertionError("synthesis requested"))
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> TTSSettingsPersistenceOutcome:
        assert delete_keys == {}
        saved_batches.append(deepcopy(dict(section_values)))
        return _mutation_outcome()

    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        effective,
    )

    event = STTSSettingsSaveEvent({"audio_cpp": candidate})
    candidate["base_url"] = "http://mutated.invalid"
    await handler.handle_settings_save(event)

    assert type(saved_batches[0]["app_tts"]["audio_cpp"]) is dict
    assert saved_batches[0]["app_tts"]["audio_cpp"] == expected
    assert saved_batches[0]["app_tts"]["default_model_mode"] == "exact"
    reconfigure_provider.assert_awaited_once_with(
        "audio_cpp",
        project_audio_cpp_config(effective).to_mapping(),
    )
    service.get_catalog.assert_not_awaited()
    service.get_voices.assert_not_awaited()
    service.synthesize.assert_not_awaited()
    assert len(app.messages) == 1
    changed = app.messages[0]
    assert isinstance(changed, STTSProviderConfigurationChanged)
    assert changed.provider_id == "audio_cpp"
    assert changed.configuration_revision == 2


@pytest.mark.asyncio
async def test_unchanged_audio_cpp_save_emits_no_configuration_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = AudioCppConfig().to_mapping()
    effective = {
        "COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": deepcopy(candidate)}}
    }
    reconfigure_provider = AsyncMock(return_value=ReconfigureResult.UNCHANGED)
    service = ImmediatePublicationService(reconfigure_provider)
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service
    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        Mock(return_value=_mutation_outcome()),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        effective,
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent({"audio_cpp": candidate}))

    reconfigure_provider.assert_awaited_once()
    assert app.messages == []


@pytest.mark.asyncio
async def test_changed_audio_cpp_config_retires_only_audio_cpp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AudioCppConfig().to_mapping()
    replacement = AudioCppConfig(base_url="http://127.0.0.1:18080").to_mapping()
    audio_cpp_factory = RecordingFactory("audio_cpp")
    legacy_factory = RecordingFactory("openai")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                audio_cpp_factory,
                original,
                exclusive=True,
            ),
            provider_spec("openai", legacy_factory, {}),
        ),
        aliases={},
    )
    service = TTSService(registry)
    for provider_id in ("audio_cpp", "openai"):
        lease = await registry.acquire(provider_id)
        await lease.release()

    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service
    effective = {
        "COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": deepcopy(replacement)}}
    }
    monkeypatch.setattr(
        "tldw_chatbook.config.apply_settings_mutation_to_cli_config",
        Mock(return_value=_mutation_outcome()),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.settings",
        effective,
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent({"audio_cpp": replacement})
    )

    assert audio_cpp_factory.instances[0].close_calls == 1
    assert audio_cpp_factory.calls == 1
    assert legacy_factory.instances[0].close_calls == 0
    assert registry.configuration_revision("audio_cpp") == 2
    assert registry.configuration_revision("openai") == 1
    assert len(app.messages) == 1
    assert isinstance(app.messages[0], STTSProviderConfigurationChanged)

    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_settings_handler_merges_preferences_before_retained_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    original = AudioCppConfig().to_mapping()
    replacement = AudioCppConfig(
        base_url="http://127.0.0.1:18080",
    ).to_mapping()
    preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="first_available",
        model_id=None,
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    current_settings = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "app_tts": {
                "audio_cpp": deepcopy(original),
                "default_provider": "openai",
                "default_model": "tts-1",
                "default_voice": "alloy",
            },
            "tts_settings": {
                "default_tts_provider": "openai",
                "default_openai_tts_model": "tts-1",
                "default_tts_voice": "alloy",
            },
        }
    }
    factory = RecordingFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                original,
                exclusive=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot.from_settings(current_settings),
    )
    old_lease = await registry.acquire("audio_cpp")
    await old_lease.release()
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service
    main_thread_id = threading.get_ident()
    publication_begun = False
    persisted_on_thread: int | None = None
    captured_sets: dict[str, dict[str, Any]] = {}
    captured_deletes: dict[str, tuple[str, ...]] = {}
    original_begin = service.begin_preferences_publication

    def begin_publication(*args: Any, **kwargs: Any) -> Any:
        nonlocal publication_begun
        publication_begun = True
        return original_begin(*args, **kwargs)

    def apply_mutation(
        section_values: Mapping[str, Mapping[Any, Any]],
        *,
        delete_keys: Mapping[str, tuple[str, ...]],
    ) -> Any:
        nonlocal persisted_on_thread
        assert publication_begun is True
        persisted_on_thread = threading.get_ident()
        captured_sets.update(deepcopy(dict(section_values)))
        captured_deletes.update(deepcopy(dict(delete_keys)))
        return SimpleNamespace(
            file_replaced=True,
            caches_reloaded=True,
            failure_phase=None,
        )

    service.begin_preferences_publication = begin_publication  # type: ignore[method-assign]
    monkeypatch.setattr(config_module, "settings", current_settings)
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        apply_mutation,
    )
    monkeypatch.setattr(
        config_module,
        "save_settings_to_cli_config",
        Mock(side_effect=AssertionError("legacy settings writer used")),
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {"audio_cpp": replacement},
            preferences=preferences,
        )
    )

    assert persisted_on_thread is not None
    assert persisted_on_thread != main_thread_id
    assert captured_sets["app_tts"]["audio_cpp"] == replacement
    assert captured_sets["app_tts"]["default_provider"] == "audio_cpp"
    assert captured_sets["app_tts"]["default_model_mode"] == "first_available"
    assert captured_sets["app_tts"]["default_voice_mode"] == "server_default"
    assert captured_deletes == {
        "app_tts": ("default_model", "default_voice"),
        "tts_settings": (
            "default_openai_tts_model",
            "default_tts_voice",
        ),
    }
    assert service.preferences_snapshot() == preferences
    assert registry.configuration_revision("audio_cpp") == 2
    assert old_lease.adapter.close_calls == 1
    assert len(app.messages) == 1
    assert isinstance(app.messages[0], STTSProviderConfigurationChanged)
    assert app.notifications == [
        ("Settings saved successfully!", "information"),
    ]
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_audio_cpp_pending_save_returns_without_cancelling_active_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    original = AudioCppConfig().to_mapping()
    replacement = AudioCppConfig(
        base_url="http://127.0.0.1:18080",
    ).to_mapping()
    preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="exact",
        model_id="model",
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    factory = RecordingFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                original,
                exclusive=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(registry, preferences_snapshot=preferences)
    response = await service.synthesize_default(text="active speech")
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service
    captured_ticket: TTSSettingsPublicationTicket | None = None
    original_begin = service.begin_preferences_publication

    def begin_with_zero_timeout(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_ticket
        kwargs["foreground_timeout_seconds"] = 0
        captured_ticket = original_begin(*args, **kwargs)
        return captured_ticket

    service.begin_preferences_publication = begin_with_zero_timeout  # type: ignore[method-assign]
    monkeypatch.setattr(
        config_module,
        "settings",
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": deepcopy(original)}}},
    )
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        Mock(return_value=_mutation_outcome()),
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {"audio_cpp": replacement},
            preferences=preferences,
        )
    )

    assert app.notifications == [
        ("Saved — applying after current speech", "information")
    ]
    assert app.messages == []
    assert factory.instances[0].close_calls == 0
    assert [chunk async for chunk in response.byte_stream] == [b"audio"]
    await response.aclose()
    assert captured_ticket is not None
    await asyncio.shield(captured_ticket.completion)
    if handler._active_tasks:
        await asyncio.gather(*tuple(handler._active_tasks))

    assert factory.instances[0].close_calls == 1
    assert len(app.messages) == 1
    assert isinstance(app.messages[0], STTSProviderConfigurationChanged)
    await service.close()
    await service.wait_closed()


@pytest.mark.asyncio
async def test_cache_reload_failure_still_publishes_runtime_with_safe_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    secret = "http://private-config-value.invalid"
    original = AudioCppConfig().to_mapping()
    replacement = AudioCppConfig(base_url=secret).to_mapping()
    old_preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="exact",
        model_id="old-model",
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    new_preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="exact",
        model_id="new-model",
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )
    factory = RecordingFactory("audio_cpp")
    registry = TTSAdapterRegistry(
        specs=(
            provider_spec(
                "audio_cpp",
                factory,
                original,
                exclusive=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(registry, preferences_snapshot=old_preferences)
    lease = await registry.acquire("audio_cpp")
    await lease.release()
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service
    monkeypatch.setattr(
        config_module,
        "settings",
        {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {"audio_cpp": deepcopy(original)}}},
    )
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        Mock(
            return_value=_mutation_outcome(
                file_replaced=True,
                caches_reloaded=False,
                failure_phase="cache_reload",
            )
        ),
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {"audio_cpp": replacement},
            preferences=new_preferences,
        )
    )

    assert service.preferences_snapshot() == new_preferences
    assert registry.configuration_revision("audio_cpp") == 2
    assert len(app.messages) == 1
    rendered = repr(app.notifications)
    assert "saved" in rendered.lower()
    assert "runtime updated" in rendered.lower()
    assert "restart recommended" in rendered.lower()
    assert secret not in rendered
    await service.close()
    await service.wait_closed()


def test_audio_cpp_mapping_serializes_as_nested_toml_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    config_path = tmp_path / "config.toml"
    candidate = AudioCppConfig(
        base_url="http://127.0.0.1:18080",
        max_catalog_models=42,
    ).to_mapping()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.save_settings_to_cli_config(
        {"app_tts": {"audio_cpp": dict(candidate)}}
    )

    persisted = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["app_tts"]["audio_cpp"] == candidate
