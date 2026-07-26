from __future__ import annotations

import asyncio
import ast
import inspect
import textwrap
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Select, Static

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    _TTS_SETTING_BINDINGS,
)
from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSProviderCatalog,
)
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.UI import STTS_Window
from tldw_chatbook.UI.STTS_Window import TTSSettingsWidget
from tldw_chatbook.UI.stts_playground_catalog import (
    FIRST_AVAILABLE_MODEL_ID,
    SERVER_DEFAULT_VOICE_ID,
)

_PREFERENCE_PAYLOAD_KEYS = {
    "default_provider",
    "default_model",
    "default_voice",
    "default_format",
    "default_speed",
}


class _SettingsHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.saved_events: list[STTSSettingsSaveEvent] = []
        self.notices: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield TTSSettingsWidget()

    def post_message(self, message: Any) -> bool:
        if isinstance(message, STTSSettingsSaveEvent):
            self.saved_events.append(message)
            return True
        return super().post_message(message)

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        del title, timeout
        self.notices.append((message, severity))


def test_settings_binding_table_classifies_every_widget_payload_key() -> None:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(TTSSettingsWidget._save_settings))
    )
    payload_keys = {
        target.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == "settings"
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    }

    assert payload_keys == set(_TTS_SETTING_BINDINGS) - _PREFERENCE_PAYLOAD_KEYS


@pytest.fixture
def settings_config(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides: dict[tuple[str, str], Any] = {
        ("app_tts", "OPENAI_ORG_ID"): "org-existing",
        ("app_tts", "audio_cpp"): AudioCppConfig().to_mapping(),
    }

    def get_setting(section: str, key: str, default: Any = None) -> Any:
        return overrides.get((section, key), default)

    monkeypatch.setattr(STTS_Window, "get_cli_setting", get_setting)
    monkeypatch.setattr(
        TTSSettingsWidget, "_load_kokoro_voice_blends", lambda self: None
    )


@pytest.mark.asyncio
async def test_settings_selects_mount_with_canonical_values(
    settings_config: None,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()

        assert app.query_one("#default-provider-select", Select).value == "openai"
        voice_select = app.query_one("#default-voice-select", Select)
        assert voice_select.value == "alloy", voice_select._options
        assert app.query_one("#default-model-select", Select).value == "tts-1"
        assert (
            app.query_one("#elevenlabs-model-select", Select).value
            == "eleven_multilingual_v2"
        )
        assert app.query_one("#kokoro-device-select", Select).value == "cpu"
        assert app.query_one("#higgs-device-select", Select).value == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_preferences", "expected_model", "expected_voice"),
    (
        (
            {
                "default_model_mode": "first_available",
                "default_model": "stale-model",
                "default_voice_mode": "server_default",
                "default_voice": "stale-voice",
            },
            FIRST_AVAILABLE_MODEL_ID,
            SERVER_DEFAULT_VOICE_ID,
        ),
        (
            {
                "default_model": "",
                "default_voice": "",
            },
            FIRST_AVAILABLE_MODEL_ID,
            SERVER_DEFAULT_VOICE_ID,
        ),
        (
            {
                "default_model": "Legacy.Model/Exact",
                "default_voice": "Legacy.Voice/Exact",
            },
            "Legacy.Model/Exact",
            "Legacy.Voice/Exact",
        ),
    ),
)
async def test_audio_cpp_mount_uses_one_read_only_preference_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    stored_preferences: dict[str, object],
    expected_model: object,
    expected_voice: object,
) -> None:
    from tldw_chatbook import config as config_module

    stored = {
        ("app_tts", "default_provider"): "audio_cpp",
        ("app_tts", "default_format"): "wav",
        ("app_tts", "default_speed"): 1.0,
        ("app_tts", "audio_cpp"): AudioCppConfig().to_mapping(),
        **{("app_tts", key): value for key, value in stored_preferences.items()},
    }
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: stored.get((section, key), default),
    )
    monkeypatch.setattr(
        TTSSettingsWidget,
        "_load_kokoro_voice_blends",
        lambda self: None,
    )
    parse_preferences = Mock(wraps=TTSPreferencesSnapshot.from_settings)
    monkeypatch.setattr(
        TTSPreferencesSnapshot,
        "from_settings",
        parse_preferences,
    )
    configuration_write = Mock(
        side_effect=AssertionError("mount must not write configuration")
    )
    for helper_name in (
        "apply_settings_mutation_to_cli_config",
        "save_settings_to_cli_config",
        "save_setting_to_cli_config",
        "delete_settings_from_cli_config",
    ):
        monkeypatch.setattr(config_module, helper_name, configuration_write)
    get_service = AsyncMock(
        side_effect=AssertionError("mount must not materialize the TTS service")
    )
    monkeypatch.setattr(STTS_Window, "get_tts_service", get_service)
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()

        assert app.query_one("#default-provider-select", Select).value == "audio_cpp"
        assert app.query_one("#default-model-select", Select).value == expected_model
        assert app.query_one("#default-voice-select", Select).value == expected_voice
        assert app.query_one("#default-format-select", Select).value == "wav"
        assert app.query_one("#default-format-select", Select).disabled is True
        assert app.query_one("#default-speed-input", Input).value == "1.0"
        assert app.query_one("#default-speed-input", Input).disabled is True

    assert parse_preferences.call_count == 1
    configuration_write.assert_not_called()
    get_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_cpp_stored_defaults_mount_and_save_without_nulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {
        ("app_tts", "default_provider"): "audio_cpp",
        ("app_tts", "default_model"): "<opaque:model>",
        ("app_tts", "default_voice"): "[voice]",
        ("app_tts", "default_format"): "wav",
        ("app_tts", "audio_cpp"): AudioCppConfig().to_mapping(),
    }
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: stored.get((section, key), default),
    )
    monkeypatch.setattr(
        TTSSettingsWidget,
        "_load_kokoro_voice_blends",
        lambda self: None,
    )
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        widget = app.query_one(TTSSettingsWidget)

        assert app.query_one("#default-provider-select", Select).value == "audio_cpp"
        assert app.query_one("#default-model-select", Select).value == "<opaque:model>"
        assert app.query_one("#default-voice-select", Select).value == "[voice]"

        widget._save_settings()
        await pilot.pause()

        event = app.saved_events[-1]
        assert event.preferences is not None
        assert event.preferences.provider_id == "audio_cpp"
        assert event.preferences.model_mode == "exact"
        assert event.preferences.model_id == "<opaque:model>"
        assert event.preferences.voice_mode == "exact"
        assert event.preferences.voice_id == "[voice]"
        assert event.preferences.response_format == "wav"
        assert event.preferences.speed == 1.0
        assert _PREFERENCE_PAYLOAD_KEYS.isdisjoint(event.settings)


@pytest.mark.asyncio
async def test_audio_cpp_settings_preserve_sentinel_shaped_remote_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote_model_id = str(FIRST_AVAILABLE_MODEL_ID)
    remote_voice_id = str(SERVER_DEFAULT_VOICE_ID)
    stored = {
        ("app_tts", "default_provider"): "audio_cpp",
        ("app_tts", "default_model_mode"): "exact",
        ("app_tts", "default_model"): remote_model_id,
        ("app_tts", "default_voice_mode"): "exact",
        ("app_tts", "default_voice"): remote_voice_id,
        ("app_tts", "default_format"): "wav",
        ("app_tts", "audio_cpp"): AudioCppConfig().to_mapping(),
    }
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: stored.get((section, key), default),
    )
    monkeypatch.setattr(
        TTSSettingsWidget,
        "_load_kokoro_voice_blends",
        lambda self: None,
    )
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        model_select = app.query_one("#default-model-select", Select)
        voice_select = app.query_one("#default-voice-select", Select)

        assert model_select.value == remote_model_id
        assert voice_select.value == remote_voice_id
        voice_values = tuple(value for _label, value in voice_select._options)
        assert SERVER_DEFAULT_VOICE_ID in voice_values
        assert remote_voice_id in voice_values
        assert SERVER_DEFAULT_VOICE_ID != remote_voice_id

        app.query_one(TTSSettingsWidget)._save_settings()
        await pilot.pause()

    event = app.saved_events[-1]
    assert event.preferences is not None
    assert event.preferences.model_mode == "exact"
    assert event.preferences.model_id == remote_model_id
    assert event.preferences.voice_mode == "exact"
    assert event.preferences.voice_id == remote_voice_id
    assert _PREFERENCE_PAYLOAD_KEYS.isdisjoint(event.settings)


@pytest.mark.asyncio
async def test_selecting_audio_cpp_defaults_uses_non_materializing_sentinels(
    settings_config: None,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        provider = app.query_one("#default-provider-select", Select)
        provider.value = "audio_cpp"
        await pilot.pause()

        assert app.query_one("#default-model-select", Select).value == (
            FIRST_AVAILABLE_MODEL_ID
        )
        assert app.query_one("#default-voice-select", Select).value == (
            SERVER_DEFAULT_VOICE_ID
        )
        assert app.query_one("#default-format-select", Select).value == "wav"
        assert app.query_one("#default-format-select", Select).disabled is True
        assert app.query_one("#default-speed-input", Input).value == "1.0"
        assert app.query_one("#default-speed-input", Input).disabled is True

        app.query_one(TTSSettingsWidget)._save_settings()
        await pilot.pause()

        event = app.saved_events[-1]
        assert event.preferences is not None
        assert event.preferences.provider_id == "audio_cpp"
        assert event.preferences.model_mode == "first_available"
        assert event.preferences.model_id is None
        assert event.preferences.voice_mode == "server_default"
        assert event.preferences.voice_id is None
        assert "default_model" not in event.settings
        assert "default_voice" not in event.settings


@pytest.mark.asyncio
async def test_settings_save_posts_canonical_values_and_explicit_openai_resets(
    settings_config: None,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        widget = app.query_one(TTSSettingsWidget)
        assert app.query_one("#openai-org-id-input", Input).value == "org-existing"
        app.query_one(
            "#openai-base-url-input", Input
        ).value = "https://api.openai.com/v1/audio/speech"
        app.query_one("#openai-org-id-input", Input).value = ""

        widget._save_settings()
        await pilot.pause()

        assert app.saved_events, app.notices
        event = app.saved_events[-1]
        settings = event.settings
        assert event.preferences is not None
        assert event.preferences.provider_id == "openai"
        assert event.preferences.voice_mode == "exact"
        assert event.preferences.voice_id == "alloy"
        assert event.preferences.model_mode == "exact"
        assert event.preferences.model_id == "tts-1"
        assert event.preferences.response_format == "mp3"
        assert event.preferences.speed == 1.0
        assert _PREFERENCE_PAYLOAD_KEYS.isdisjoint(settings)
        assert settings["ELEVENLABS_DEFAULT_MODEL"] == "eleven_multilingual_v2"
        assert settings["KOKORO_DEVICE_DEFAULT"] == "cpu"
        assert settings["HIGGS_DEVICE"] == "auto"
        assert settings["OPENAI_BASE_URL"] == ("https://api.openai.com/v1/audio/speech")
        assert settings["OPENAI_ORG_ID"] == ""


@pytest.mark.asyncio
async def test_settings_widget_waits_for_handler_outcome_before_notifying(
    settings_config: None,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        app.query_one(TTSSettingsWidget)._save_settings()
        await pilot.pause()

        assert len(app.saved_events) == 1
        assert app.notices == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    (
        "relative/audio/speech",
        "ftp://example.test/audio/speech",
        "https://user:secret@example.test/audio/speech",
        "https://example.test/audio/speech#fragment",
    ),
)
async def test_settings_widget_rejects_unsafe_openai_base_urls_without_echoing_them(
    settings_config: None,
    base_url: str,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        app.query_one("#openai-base-url-input", Input).value = base_url

        app.query_one(TTSSettingsWidget)._save_settings()

        assert app.saved_events == []
        assert app.notices
        assert app.notices[-1][1] == "error"
        assert base_url not in app.notices[-1][0]


@pytest.mark.asyncio
async def test_settings_widget_does_not_echo_collection_error_details(
    settings_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del settings_config
    secret = "sk-WidgetCollectionError-PrivateSuffix"
    messages: list[str] = []
    app = _SettingsHost()

    def fail_normalization(_self: TTSSettingsWidget, _value: str) -> str:
        raise RuntimeError(f"invalid setting {secret}")

    monkeypatch.setattr(
        TTSSettingsWidget,
        "_normalize_openai_base_url",
        fail_normalization,
    )
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            app.query_one(TTSSettingsWidget)._save_settings()
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(messages)
    assert app.saved_events == []
    assert app.notices == [("Failed to save settings", "error")]
    assert secret not in rendered


@pytest.mark.asyncio
async def test_audio_cpp_settings_surface_is_external_only(
    settings_config: None,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        panel = app.query_one("#audio-cpp-settings", Collapsible)

        assert str(panel.query_one("#audio-cpp-mode-value", Static).render()) == (
            "External"
        )
        assert {
            widget.id for widget in panel.query(Input) if widget.id is not None
        } == {
            "audio-cpp-base-url-input",
            "audio-cpp-connect-timeout-input",
            "audio-cpp-synthesis-timeout-input",
            "audio-cpp-max-input-characters-input",
            "audio-cpp-max-response-bytes-input",
            "audio-cpp-max-metadata-bytes-input",
            "audio-cpp-max-catalog-models-input",
            "audio-cpp-max-voices-per-model-input",
            "audio-cpp-max-identifier-characters-input",
        }
        button_labels = {str(button.label) for button in panel.query(Button)}
        assert button_labels == {"Test Connection", "Refresh Models"}
        privacy_copy = str(
            panel.query_one("#audio-cpp-privacy-notice", Static).render()
        )
        assert "submitted text" in privacy_copy.lower()
        assert "configured server" in privacy_copy.lower()
        rendered_panel = " ".join(
            (
                str(panel.title),
                privacy_copy,
                *button_labels,
            )
        ).lower()
        for managed_term in (
            "binary path",
            "server.json",
            "start server",
            "restart",
            "managed log",
            "process control",
        ):
            assert managed_term not in rendered_panel


@pytest.mark.asyncio
async def test_audio_cpp_settings_save_posts_validated_defensive_plain_mapping(
    settings_config: None,
) -> None:
    del settings_config
    app = _SettingsHost()

    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        values = {
            "#audio-cpp-base-url-input": "https://voice.example.test:8443",
            "#audio-cpp-connect-timeout-input": "2.5",
            "#audio-cpp-synthesis-timeout-input": "45",
            "#audio-cpp-max-input-characters-input": "1234",
            "#audio-cpp-max-response-bytes-input": "1048576",
            "#audio-cpp-max-metadata-bytes-input": "4096",
            "#audio-cpp-max-catalog-models-input": "12",
            "#audio-cpp-max-voices-per-model-input": "34",
            "#audio-cpp-max-identifier-characters-input": "128",
        }
        for selector, value in values.items():
            app.query_one(selector, Input).value = value

        app.query_one(TTSSettingsWidget)._save_settings()
        await pilot.pause()

        assert len(app.saved_events) == 1
        candidate = app.saved_events[0].settings["audio_cpp"]
        assert type(candidate) is dict
        assert candidate == {
            "mode": "external",
            "base_url": "https://voice.example.test:8443",
            "connect_timeout_seconds": 2.5,
            "synthesis_timeout_seconds": 45.0,
            "max_input_characters": 1234,
            "max_response_bytes": 1048576,
            "max_metadata_bytes": 4096,
            "max_catalog_models": 12,
            "max_voices_per_model": 34,
            "max_identifier_characters": 128,
        }

        app.query_one(
            "#audio-cpp-base-url-input", Input
        ).value = "http://changed.invalid"
        assert candidate["base_url"] == "https://voice.example.test:8443"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "invalid_value"),
    (
        ("#audio-cpp-base-url-input", "relative/path"),
        ("#audio-cpp-base-url-input", "https://user:secret@example.test"),
        ("#audio-cpp-base-url-input", "https://example.test/path"),
        ("#audio-cpp-base-url-input", "https://example.test?secret=query"),
        ("#audio-cpp-base-url-input", "https://example.test#fragment"),
        ("#audio-cpp-connect-timeout-input", "0"),
        ("#audio-cpp-synthesis-timeout-input", "nan"),
        ("#audio-cpp-max-catalog-models-input", "1.5"),
        ("#audio-cpp-max-voices-per-model-input", "-1"),
        ("#audio-cpp-max-identifier-characters-input", "9" * 5000),
    ),
)
async def test_audio_cpp_settings_reject_invalid_values_without_echo(
    settings_config: None,
    selector: str,
    invalid_value: str,
) -> None:
    del settings_config
    app = _SettingsHost()
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        async with app.run_test(size=(180, 80)) as pilot:
            await pilot.pause()
            app.query_one(selector, Input).value = invalid_value
            app.query_one(TTSSettingsWidget)._save_settings()
            await pilot.pause()
    finally:
        logger.remove(sink_id)

    assert app.saved_events == []
    assert app.notices == [("Failed to save settings", "error")]
    rendered = "\n".join(messages + [message for message, _ in app.notices])
    assert invalid_value not in rendered


def _available_audio_cpp_catalog() -> TTSProviderCatalog:
    return TTSProviderCatalog(
        provider_id="audio_cpp",
        revision=9,
        health=ProviderHealth(state="available", fresh=True),
        models=(
            TTSModelInfo(
                model_id="opaque-model",
                display_name="Opaque model",
                family="test",
                upstream_mode="tts",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                omit_voice_uses_server_default=True,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_audio_cpp_test_and_refresh_are_explicit_saved_config_actions(
    settings_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del settings_config
    service = SimpleNamespace(
        configuration_revision=Mock(side_effect=(4, 4, 4, 4)),
        get_catalog=AsyncMock(return_value=_available_audio_cpp_catalog()),
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        AsyncMock(return_value=service),
    )
    app = _SettingsHost()

    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        await pilot.click("#audio-cpp-test-connection-btn")
        await app.workers.wait_for_complete()
        await pilot.click("#audio-cpp-refresh-models-btn")
        await app.workers.wait_for_complete()

    assert service.get_catalog.await_args_list == [
        call("audio_cpp", refresh=True),
        call("audio_cpp", refresh=True),
    ]
    assert app.notices == [
        ("audio.cpp connection is ready (1 model)", "information"),
        ("audio.cpp models refreshed (1 model)", "information"),
    ]


@pytest.mark.asyncio
async def test_audio_cpp_settings_discovery_discards_changed_revision(
    settings_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del settings_config
    service = SimpleNamespace(
        configuration_revision=Mock(side_effect=(7, 8)),
        get_catalog=AsyncMock(return_value=_available_audio_cpp_catalog()),
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        AsyncMock(return_value=service),
    )
    app = _SettingsHost()

    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        await pilot.click("#audio-cpp-test-connection-btn")
        await app.workers.wait_for_complete()

    assert app.notices == [
        ("audio.cpp settings changed; retry the check", "warning"),
    ]


@pytest.mark.asyncio
async def test_audio_cpp_settings_discovery_failure_rechecks_revision(
    settings_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del settings_config
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def get_catalog(
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        del provider_id, refresh
        request_started.set()
        await release_request.wait()
        raise RuntimeError("obsolete settings failed")

    service = SimpleNamespace(
        configuration_revision=Mock(side_effect=(7, 8)),
        get_catalog=get_catalog,
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        AsyncMock(return_value=service),
    )
    app = _SettingsHost()

    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        await pilot.click("#audio-cpp-test-connection-btn")
        await request_started.wait()
        release_request.set()
        await app.workers.wait_for_complete()

        status = str(app.query_one("#audio-cpp-discovery-status", Static).render())

    assert status == "Settings changed; retry"
    assert app.notices == [
        ("audio.cpp settings changed; retry the check", "warning"),
    ]


@pytest.mark.asyncio
async def test_superseded_settings_discovery_failure_cannot_overwrite_success(
    settings_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del settings_config
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_returned = asyncio.Event()
    call_count = 0

    async def get_catalog(
        provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        nonlocal call_count
        del provider_id, refresh
        call_count += 1
        if call_count == 1:
            first_started.set()
            try:
                await release_first.wait()
            except asyncio.CancelledError:
                await release_first.wait()
            raise RuntimeError("superseded action failed")
        second_returned.set()
        return _available_audio_cpp_catalog()

    service = SimpleNamespace(
        configuration_revision=Mock(return_value=7),
        get_catalog=get_catalog,
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        AsyncMock(return_value=service),
    )
    app = _SettingsHost()

    async with app.run_test(size=(180, 80)) as pilot:
        await pilot.pause()
        widget = app.query_one(TTSSettingsWidget)
        widget._discover_audio_cpp("test")
        await first_started.wait()
        widget._discover_audio_cpp("refresh")
        await second_returned.wait()
        await pilot.pause()

        release_first.set()
        await app.workers.wait_for_complete()
        await pilot.pause()

        status = str(app.query_one("#audio-cpp-discovery-status", Static).render())

    assert status == "audio.cpp models refreshed (1 model)"
    assert app.notices == [
        ("audio.cpp models refreshed (1 model)", "information"),
    ]
