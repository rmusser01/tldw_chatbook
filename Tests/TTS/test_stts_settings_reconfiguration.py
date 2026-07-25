from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, call

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
from tldw_chatbook.TTS.TTS_Generation import TTSService


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


@pytest.mark.asyncio
async def test_provider_setting_reconfigures_only_current_materialized_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_snapshot = {
        "API": {"openai_api_key": "old"},
        "app_tts": {"default_format": "mp3"},
    }
    effective_settings = {
        "COMPREHENSIVE_CONFIG_RAW": {
            "API": {"openai_api_key": "new"},
            "app_tts": {"default_format": "wav"},
        },
        "APP_TTS_CONFIG": {"default_format": "ignored"},
    }
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
    reloads: list[bool] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
    ) -> bool:
        saved_batches.append(deepcopy(dict(section_values)))
        return True

    def load_effective_settings(*, force_reload: bool = False) -> dict[str, Any]:
        reloads.append(force_reload)
        return effective_settings

    initialize_stts = AsyncMock(side_effect=AssertionError("service rebuilt"))
    get_bound_service = AsyncMock(side_effect=AssertionError("accessor used"))
    monkeypatch.setattr(handler, "initialize_stts", initialize_stts)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr("tldw_chatbook.config.load_settings", load_effective_settings)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_bound_service,
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent({"openai_api_key": "new"}))

    assert saved_batches == [{"API": {"openai_api_key": "new"}}]
    assert reloads == [False]
    assert openai_factory.instances[0].close_calls == 1
    assert kokoro_factory.instances[0].close_calls == 0
    assert registry.configuration_revision("openai") == 2
    assert registry.configuration_revision("kokoro") == 1
    replacement = await registry.acquire("openai")
    await replacement.release()
    assert openai_factory.configs[-1] == legacy_provider_config(
        "openai",
        effective_settings,
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
        "tldw_chatbook.config.save_settings_to_cli_config",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        Mock(return_value=normalized(effective_raw)),
    )
    event_settings = {
        key: "unchanged" for keys in PROVIDER_SETTING_KEYS.values() for key in keys
    }
    event_settings.update(
        {
            "openai_api_key": "stable-openai-key",
            "elevenlabs_api_key": "stored-elevenlabs-key-after",
            "ELEVENLABS_DEFAULT_MODEL": "eleven_multilingual_v2",
            "KOKORO_DEVICE_DEFAULT": "mps",
            "HIGGS_DEVICE": "cpu",
        }
    )

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
    handler._stts_service = SimpleNamespace(
        reconfigure_provider=reconfigure_provider,
    )
    snapshot = {"API": {"openai_api_key": "secret"}}
    event_settings = {
        key: "value"
        for provider_id in reversed(tuple(PROVIDER_SETTING_KEYS))
        for key in PROVIDER_SETTING_KEYS[provider_id]
    }
    event_settings["default_provider"] = "openai"
    saved_batches: list[dict[str, dict[str, object]]] = []
    load_calls: list[bool] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
    ) -> bool:
        saved_batches.append(deepcopy(dict(section_values)))
        return True

    def load_effective_settings(*, force_reload: bool = False) -> dict[str, Any]:
        load_calls.append(force_reload)
        return {"COMPREHENSIVE_CONFIG_RAW": snapshot}

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        load_effective_settings,
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent(event_settings))

    assert saved_batches == [
        {
            "HiggsSettings": {
                "model_path": "value",
                "voice_samples_dir": "value",
                "device": "value",
                "enable_flash_attn": "value",
                "dtype": "value",
                "max_reference_duration": "value",
                "default_language": "value",
                "enable_voice_cloning": "value",
                "enable_multi_speaker": "value",
                "speaker_delimiter": "value",
                "track_performance": "value",
                "max_new_tokens": "value",
                "temperature": "value",
                "top_p": "value",
                "repetition_penalty": "value",
            },
            "app_tts": {
                "KOKORO_DEVICE_DEFAULT": "value",
                "KOKORO_USE_ONNX": "value",
                "KOKORO_ONNX_MODEL_PATH_DEFAULT": "value",
                "KOKORO_ONNX_VOICES_JSON_DEFAULT": "value",
                "KOKORO_MAX_TOKENS": "value",
                "KOKORO_ENABLE_VOICE_MIXING": "value",
                "KOKORO_TRACK_PERFORMANCE": "value",
                "ELEVENLABS_DEFAULT_MODEL": "value",
                "ELEVENLABS_OUTPUT_FORMAT": "value",
                "ELEVENLABS_VOICE_STABILITY": "value",
                "ELEVENLABS_SIMILARITY_BOOST": "value",
                "ELEVENLABS_STYLE": "value",
                "ELEVENLABS_USE_SPEAKER_BOOST": "value",
                "default_provider": "openai",
            },
            "API": {
                "elevenlabs_api_key": "value",
                "openai_api_key": "value",
            },
            "tts_settings": {"default_tts_provider": "openai"},
        }
    ]
    assert load_calls == [False]
    assert reconfigure_provider.await_args_list == [
        call(
            provider_id,
            legacy_provider_config(
                provider_id,
                {"COMPREHENSIVE_CONFIG_RAW": snapshot},
            ),
        )
        for provider_id in PROVIDER_SETTING_KEYS
    ]


@pytest.mark.asyncio
async def test_defaults_only_save_reloads_once_without_reconfiguring_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    reconfigure_provider = AsyncMock()
    handler._stts_service = SimpleNamespace(
        reconfigure_provider=reconfigure_provider,
    )
    saved_batches: list[dict[str, dict[str, object]]] = []
    reloads: list[bool] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
    ) -> bool:
        saved_batches.append(deepcopy(dict(section_values)))
        return True

    def load_effective_settings(*, force_reload: bool = False) -> dict[str, Any]:
        reloads.append(force_reload)
        return {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {}}}

    initialize_stts = AsyncMock(side_effect=AssertionError("service rebuilt"))
    monkeypatch.setattr(handler, "initialize_stts", initialize_stts)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr("tldw_chatbook.config.load_settings", load_effective_settings)

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

    assert saved_batches == [
        {
            "app_tts": {
                "default_provider": "kokoro",
                "default_voice": "af_heart",
                "default_model": "model",
                "default_format": "wav",
                "default_speed": 1.25,
            },
            "tts_settings": {
                "default_tts_provider": "kokoro",
                "default_tts_voice": "af_heart",
                "default_openai_tts_model": "model",
                "default_openai_tts_output_format": "wav",
                "default_openai_tts_speed": 1.25,
            },
        }
    ]
    assert reloads == [False]
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
    handler._stts_service = SimpleNamespace(
        reconfigure_provider=reconfigure_provider,
    )
    load_effective_settings = Mock()
    saved_batches: list[dict[str, dict[str, object]]] = []

    def fail_batch(
        section_values: Mapping[str, Mapping[object, object]],
    ) -> bool:
        saved_batches.append(deepcopy(dict(section_values)))
        return False

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        fail_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        load_effective_settings,
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

    assert saved_batches == [
        {
            "app_tts": {
                "default_provider": "kokoro",
                "KOKORO_DEVICE_DEFAULT": "cpu",
            },
            "tts_settings": {"default_tts_provider": "kokoro"},
            "API": {"openai_api_key": "secret"},
        }
    ]
    load_effective_settings.assert_not_called()
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
    handler._stts_service = Service()
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda _section_values: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        Mock(side_effect=AssertionError("per-setting writer used")),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        lambda: {
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
            "Settings saved, but some TTS providers could not be updated",
            "error",
        )
    ]
    rendered = "\n".join(messages + [message for message, _ in app.notifications])
    assert "Failed to reconfigure TTS providers: openai" in rendered
    assert "rejected credential" not in rendered
    assert secret not in rendered


@pytest.mark.asyncio
async def test_connection_and_local_provider_settings_persist_and_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    reconfigure_provider = AsyncMock()
    handler._stts_service = SimpleNamespace(
        reconfigure_provider=reconfigure_provider,
    )
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
    ) -> bool:
        saved_batches.append(deepcopy(dict(section_values)))
        return True

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        Mock(return_value=snapshot),
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

    assert saved_batches == [
        {
            "app_tts": {
                "OPENAI_BASE_URL": "http://127.0.0.1:9000/v1/audio/speech",
                "OPENAI_ORG_ID": "",
                "CHATTERBOX_DEVICE": "cpu",
                "ALLTALK_TTS_URL_DEFAULT": "http://127.0.0.1:7851",
            }
        }
    ]
    assert [call_.args[0] for call_ in reconfigure_provider.await_args_list] == [
        "openai",
        "chatterbox",
        "alltalk",
    ]
    for provider_id, call_ in zip(
        ("openai", "chatterbox", "alltalk"),
        reconfigure_provider.await_args_list,
    ):
        assert call_.args[1] == legacy_provider_config(provider_id, snapshot)
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

    handler._stts_service = SimpleNamespace(
        reconfigure_provider=reconfigure_provider,
    )
    saved_values: list[str] = []

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
    ) -> bool:
        saved_values.append(str(section_values["API"]["openai_api_key"]))
        return True

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        lambda: {"COMPREHENSIVE_CONFIG_RAW": {"API": {}}},
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
    service = SimpleNamespace(
        reconfigure_provider=AsyncMock(return_value=ReconfigureResult.CHANGED),
        configuration_revision=Mock(return_value=2),
        get_catalog=AsyncMock(side_effect=AssertionError("catalog requested")),
        get_voices=AsyncMock(side_effect=AssertionError("voices requested")),
        synthesize=AsyncMock(side_effect=AssertionError("synthesis requested")),
    )
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service

    def save_batch(
        section_values: Mapping[str, Mapping[object, object]],
    ) -> bool:
        saved_batches.append(deepcopy(dict(section_values)))
        return True

    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_batch,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        Mock(return_value=effective),
    )

    event = STTSSettingsSaveEvent({"audio_cpp": candidate})
    candidate["base_url"] = "http://mutated.invalid"
    await handler.handle_settings_save(event)

    assert type(saved_batches[0]["app_tts"]["audio_cpp"]) is dict
    assert saved_batches == [{"app_tts": {"audio_cpp": expected}}]
    service.reconfigure_provider.assert_awaited_once_with(
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
    service = SimpleNamespace(
        reconfigure_provider=AsyncMock(return_value=ReconfigureResult.UNCHANGED),
        configuration_revision=Mock(side_effect=AssertionError("revision requested")),
    )
    app = RecordingApp()
    handler = STTSEventHandler(app)
    handler._stts_service = service
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        Mock(return_value=effective),
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent({"audio_cpp": candidate}))

    service.reconfigure_provider.assert_awaited_once()
    service.configuration_revision.assert_not_called()
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
        "tldw_chatbook.config.save_settings_to_cli_config",
        Mock(return_value=True),
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        Mock(return_value=effective),
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
