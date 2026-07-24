from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.widgets import Input, Select

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSSettingsSaveEvent,
    _TTS_SETTING_BINDINGS,
)
from tldw_chatbook.UI import STTS_Window
from tldw_chatbook.UI.STTS_Window import TTSSettingsWidget


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

    assert payload_keys == set(_TTS_SETTING_BINDINGS)


@pytest.fixture
def settings_config(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides: dict[tuple[str, str], Any] = {
        ("app_tts", "OPENAI_ORG_ID"): "org-existing",
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
        settings = app.saved_events[-1].settings
        assert settings["default_provider"] == "openai"
        assert settings["default_voice"] == "alloy"
        assert settings["default_model"] == "tts-1"
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
