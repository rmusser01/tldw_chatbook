from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSSettingsSaveEvent,
)


class _App:
    def __init__(self) -> None:
        self.notices: list[tuple[str, str]] = []

    def notify(self, message: str, *, severity: str) -> None:
        self.notices.append((message, severity))


class _Service:
    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def reconfigure_provider(
        self,
        provider_id: str,
        config: dict[str, Any],
    ) -> None:
        self.calls.append((provider_id, config))
        if provider_id in self.failures:
            raise RuntimeError(f"secret refresh failure for {provider_id}")


def _install_config_cycle(
    monkeypatch: pytest.MonkeyPatch,
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[dict[str, dict[str, Any]]]:
    snapshots = iter((before, after))
    saved: list[dict[str, dict[str, Any]]] = []

    def load_config(*, force_reload: bool = False) -> dict[str, Any]:
        del force_reload
        return deepcopy(next(snapshots))

    def save_settings(batch: dict[str, dict[str, Any]]) -> bool:
        saved.append(deepcopy(batch))
        return True

    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        load_config,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_settings,
    )
    return saved


@pytest.mark.asyncio
async def test_settings_save_maps_exact_ui_keys_into_one_atomic_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before: dict[str, Any] = {}
    after = {
        "app_tts": {
            "default_provider": "kokoro",
            "ELEVENLABS_VOICE_STABILITY": 0.6,
            "KOKORO_DEVICE_DEFAULT": "cuda",
            "KOKORO_USE_ONNX": False,
        },
        "tts_settings": {"default_tts_provider": "kokoro"},
        "HiggsSettings": {"repetition_penalty": 1.2},
    }
    saved = _install_config_cycle(monkeypatch, before, after)
    service = _Service()
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        AsyncMock(return_value=service),
    )
    handler = STTSEventHandler(_App())

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "default_provider": "kokoro",
                "ELEVENLABS_VOICE_STABILITY": 0.6,
                "KOKORO_DEVICE_DEFAULT": "cuda",
                "KOKORO_USE_ONNX": False,
                "HIGGS_REPETITION_PENALTY": 1.2,
            }
        )
    )

    assert saved == [
        {
            "app_tts": {
                "default_provider": "kokoro",
                "ELEVENLABS_VOICE_STABILITY": 0.6,
                "KOKORO_DEVICE_DEFAULT": "cuda",
                "KOKORO_USE_ONNX": False,
            },
            "tts_settings": {"default_tts_provider": "kokoro"},
            "HiggsSettings": {"repetition_penalty": 1.2},
        }
    ]
    assert [provider_id for provider_id, _ in service.calls] == [
        "elevenlabs",
        "kokoro",
        "higgs",
    ]
    assert all(config == {"app_config": after} for _, config in service.calls)


@pytest.mark.asyncio
async def test_failed_atomic_settings_write_stops_before_reload_or_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = 0

    def load_config(*, force_reload: bool = False) -> dict[str, Any]:
        nonlocal load_calls
        del force_reload
        load_calls += 1
        return {"app_tts": {"OPENAI_BASE_URL": "https://old.test/speech"}}

    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        load_config,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda _batch: False,
    )
    get_service = AsyncMock()
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_service,
    )
    app = _App()
    handler = STTSEventHandler(app)

    await handler.handle_settings_save(
        STTSSettingsSaveEvent({"OPENAI_BASE_URL": "https://new.test/speech"})
    )

    assert load_calls == 1
    get_service.assert_not_awaited()
    assert app.notices == [("Failed to save settings", "error")]


@pytest.mark.asyncio
async def test_routing_defaults_do_not_retire_live_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        "app_tts": {
            "default_provider": "openai",
            "KOKORO_USE_ONNX": True,
        }
    }
    after = {
        "app_tts": {
            "default_provider": "kokoro",
            "KOKORO_USE_ONNX": False,
        },
        "tts_settings": {"default_tts_provider": "kokoro"},
    }
    _install_config_cycle(monkeypatch, before, after)
    get_service = AsyncMock()
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_service,
    )
    app = _App()
    handler = STTSEventHandler(app)

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "default_provider": "kokoro",
                "KOKORO_USE_ONNX": False,
            }
        )
    )

    get_service.assert_not_awaited()
    assert app.notices == [("Settings saved successfully!", "information")]


@pytest.mark.asyncio
async def test_settings_refresh_attempts_every_changed_provider_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before: dict[str, Any] = {}
    after = {
        "app_tts": {
            "OPENAI_BASE_URL": "https://example.test/speech",
            "ELEVENLABS_DEFAULT_MODEL": "eleven_turbo_v2",
        },
        "HiggsSettings": {"device": "cpu"},
    }
    _install_config_cycle(monkeypatch, before, after)
    service = _Service({"openai", "higgs"})
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        AsyncMock(return_value=service),
    )
    app = _App()
    handler = STTSEventHandler(app)

    await handler.handle_settings_save(
        STTSSettingsSaveEvent(
            {
                "OPENAI_BASE_URL": "https://example.test/speech",
                "ELEVENLABS_DEFAULT_MODEL": "eleven_turbo_v2",
                "HIGGS_DEVICE": "cpu",
            }
        )
    )

    assert [provider_id for provider_id, _ in service.calls] == [
        "openai",
        "elevenlabs",
        "higgs",
    ]
    assert app.notices == [
        (
            "Settings saved, but TTS refresh failed for: openai, higgs",
            "warning",
        )
    ]
    assert "secret" not in app.notices[0][0]


@pytest.mark.asyncio
async def test_concurrent_settings_saves_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        (
            {"app_tts": {"OPENAI_BASE_URL": "https://zero.test/speech"}},
            {"app_tts": {"OPENAI_BASE_URL": "https://one.test/speech"}},
            {"app_tts": {"OPENAI_BASE_URL": "https://one.test/speech"}},
            {"app_tts": {"OPENAI_BASE_URL": "https://two.test/speech"}},
        )
    )
    saved: list[dict[str, dict[str, Any]]] = []

    def load_config(*, force_reload: bool = False) -> dict[str, Any]:
        del force_reload
        return deepcopy(next(snapshots))

    def save_settings(batch: dict[str, dict[str, Any]]) -> bool:
        saved.append(deepcopy(batch))
        return True

    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        load_config,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        save_settings,
    )
    first_refresh_started = asyncio.Event()
    allow_first_refresh = asyncio.Event()

    class BlockingService(_Service):
        async def reconfigure_provider(
            self,
            provider_id: str,
            config: dict[str, Any],
        ) -> None:
            await super().reconfigure_provider(provider_id, config)
            if len(self.calls) == 1:
                first_refresh_started.set()
                await allow_first_refresh.wait()

    service = BlockingService()
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        AsyncMock(return_value=service),
    )
    handler = STTSEventHandler(_App())

    first = asyncio.create_task(
        handler.handle_settings_save(
            STTSSettingsSaveEvent({"OPENAI_BASE_URL": "https://one.test/speech"})
        )
    )
    await first_refresh_started.wait()
    second = asyncio.create_task(
        handler.handle_settings_save(
            STTSSettingsSaveEvent({"OPENAI_BASE_URL": "https://two.test/speech"})
        )
    )
    await asyncio.sleep(0)

    assert len(saved) == 1

    allow_first_refresh.set()
    await asyncio.gather(first, second)

    assert len(saved) == 2
    assert [provider_id for provider_id, _ in service.calls] == [
        "openai",
        "openai",
    ]


@pytest.mark.asyncio
async def test_stts_initialization_only_retrieves_the_bound_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object()
    get_service = AsyncMock(return_value=service)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_service,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        lambda: pytest.fail("initialization must not rebuild compatibility config"),
    )
    handler = STTSEventHandler(_App())

    await handler.initialize_stts()

    get_service.assert_awaited_once_with()
    assert handler._stts_service is service
