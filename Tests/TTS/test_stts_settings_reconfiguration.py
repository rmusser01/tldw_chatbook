from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import pytest

from Tests.TTS.adapter_fakes import FakeAdapter, FakeAdapterFactory, provider_spec
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
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

    def notify(self, message: str, *, severity: str) -> None:
        self.notifications.append((message, severity))


PROVIDER_SETTING_KEYS = {
    "openai": ("openai_api_key",),
    "elevenlabs": (
        "elevenlabs_api_key",
        "elevenlabs_voice_stability",
        "elevenlabs_similarity_boost",
        "elevenlabs_style",
        "elevenlabs_use_speaker_boost",
    ),
    "kokoro": (
        "kokoro_device",
        "kokoro_use_onnx",
        "kokoro_model_path",
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
        "HIGGS_TOP_K",
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
                {"app_config": initial_snapshot},
            ),
            provider_spec(
                "kokoro",
                kokoro_factory,
                {"app_config": initial_snapshot},
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
    saved: list[tuple[str, str, str]] = []
    reloads: list[bool] = []

    def save_setting(section: str, setting_name: str, value: str) -> None:
        saved.append((section, setting_name, value))

    def load_effective_settings(*, force_reload: bool = False) -> dict[str, Any]:
        reloads.append(force_reload)
        return effective_settings

    initialize_stts = AsyncMock(side_effect=AssertionError("service rebuilt"))
    get_bound_service = AsyncMock(side_effect=AssertionError("accessor used"))
    monkeypatch.setattr(handler, "initialize_stts", initialize_stts)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        save_setting,
    )
    monkeypatch.setattr("tldw_chatbook.config.load_settings", load_effective_settings)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_bound_service,
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent({"openai_api_key": "new"}))

    assert saved == [("API", "openai_api_key", "new")]
    assert reloads == [True]
    assert openai_factory.instances[0].close_calls == 1
    assert kokoro_factory.instances[0].close_calls == 0
    assert registry.configuration_revision("openai") == 2
    assert registry.configuration_revision("kokoro") == 1
    replacement = await registry.acquire("openai")
    await replacement.release()
    assert openai_factory.configs[-1] == {
        "app_config": effective_settings["COMPREHENSIVE_CONFIG_RAW"]
    }
    assert openai_factory.calls == 2
    assert kokoro_factory.calls == 1
    initialize_stts.assert_not_awaited()
    get_bound_service.assert_not_awaited()


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

    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        lambda _section, _setting_name, _value: None,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        lambda *, force_reload=False: {
            "COMPREHENSIVE_CONFIG_RAW": snapshot,
        },
    )

    await handler.handle_settings_save(STTSSettingsSaveEvent(event_settings))

    normalized_snapshot = {**snapshot, "app_tts": {}}
    assert reconfigure_provider.await_args_list == [
        call(provider_id, {"app_config": normalized_snapshot})
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
    saved: list[tuple[str, str, object]] = []
    reloads: list[bool] = []

    def save_setting(section: str, setting_name: str, value: object) -> None:
        saved.append((section, setting_name, value))

    def load_effective_settings(*, force_reload: bool = False) -> dict[str, Any]:
        reloads.append(force_reload)
        return {"COMPREHENSIVE_CONFIG_RAW": {"app_tts": {}}}

    initialize_stts = AsyncMock(side_effect=AssertionError("service rebuilt"))
    monkeypatch.setattr(handler, "initialize_stts", initialize_stts)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        save_setting,
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

    assert saved == [
        ("app_tts", "default_provider", "kokoro"),
        ("tts_settings", "default_tts_provider", "kokoro"),
        ("app_tts", "default_voice", "af_heart"),
        ("tts_settings", "default_tts_voice", "af_heart"),
        ("app_tts", "default_model", "model"),
        ("tts_settings", "default_openai_tts_model", "model"),
        ("app_tts", "default_format", "wav"),
        ("tts_settings", "default_openai_tts_output_format", "wav"),
        ("app_tts", "default_speed", 1.25),
        ("tts_settings", "default_openai_tts_speed", 1.25),
    ]
    assert reloads == [True]
    reconfigure_provider.assert_not_awaited()
    initialize_stts.assert_not_awaited()
    assert app.notifications == [
        ("Settings saved successfully!", "information"),
    ]


@pytest.mark.asyncio
async def test_explicit_false_save_stops_before_reload_and_reconfiguration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = RecordingApp()
    handler = STTSEventHandler(app)
    reconfigure_provider = AsyncMock()
    handler._stts_service = SimpleNamespace(
        reconfigure_provider=reconfigure_provider,
    )
    load_effective_settings = Mock()
    saved: list[tuple[str, str, str]] = []

    def save_setting(section: str, setting_name: str, value: str) -> bool | None:
        saved.append((section, setting_name, value))
        return False if setting_name == "KOKORO_DEVICE" else None

    monkeypatch.setattr(
        "tldw_chatbook.config.save_setting_to_cli_config",
        save_setting,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_settings",
        load_effective_settings,
    )

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "openai_api_key": "secret",
                "kokoro_device": "cpu",
            }
        )
    )

    assert saved == [
        ("API", "openai_api_key", "secret"),
        ("app_tts", "KOKORO_DEVICE", "cpu"),
    ]
    load_effective_settings.assert_not_called()
    reconfigure_provider.assert_not_awaited()
    assert app.notifications == [
        (
            "Failed to save kokoro_device to [app_tts].KOKORO_DEVICE",
            "error",
        )
    ]
