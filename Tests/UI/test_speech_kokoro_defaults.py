"""The live Lab request must retain configured Kokoro engine and voice."""

import pytest
from textual import on
from textual.app import App
from textual.widgets import Button, Select, Switch, TextArea

from Tests.UI.speech_playground_fixtures import FakeTTSService, _resolved, _wait_until
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.UI.Speech import speech_catalog_mixin
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.STTS_Window import _seed_axis_defaults


class KokoroHost(App):
    def __init__(self, voice):
        super().__init__()
        self.requests = []
        studio = StudioTTSPreferencesSnapshot()
        global_preferences = TTSPreferencesSnapshot.from_settings(
            {
                "app_tts": {
                    "default_provider": "kokoro",
                    "default_model_mode": "exact",
                    "default_model": "kokoro",
                    "default_voice_mode": "exact",
                    "default_voice": voice,
                    "default_format": "wav",
                }
            }
        )
        self.pane = SpeechPlaygroundPane(
            provider="kokoro",
            studio_preferences=studio,
            global_preferences=global_preferences,
            axis_defaults=_seed_axis_defaults(studio, global_preferences),
        )

    def compose(self):
        yield self.pane

    @on(STTSPlaygroundGenerateEvent)
    def capture_request(self, event):
        self.requests.append(event.request)


@pytest.fixture
def catalog(monkeypatch):
    service = FakeTTSService()
    monkeypatch.setattr(
        SpeechPlaygroundPane, "_tts_service_factory", lambda self: _resolved(service)
    )
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_fresh_engine_follows_settings_and_session_toggle_remains_explicit(
    monkeypatch, catalog, configured
):
    monkeypatch.setattr(
        speech_catalog_mixin,
        "get_cli_setting",
        lambda section, key, default=None: (
            configured if (section, key) == ("app_tts", "KOKORO_USE_ONNX") else default
        ),
    )
    host = KokoroHost("af_heart")
    async with host.run_test(size=(150, 65)) as pilot:
        await _wait_until(
            pilot, lambda: not host.pane.query_one("#tts-generate-btn", Button).disabled
        )
        switch = host.pane.query_one("#tts-kokoro-use-onnx", Switch)
        assert switch.value is configured
        assert host.pane.query_one("#tts-voice-select", Select).value == "af_heart"
        assert host.requests == [] and catalog.synthesize_calls == 0
        switch.value = not configured
        host.pane.query_one("#tts-text-input", TextArea).text = "A complete reply."
        host.pane.query_one("#tts-generate-btn", Button).press()
        await _wait_until(pilot, lambda: bool(host.requests))
        assert host.requests[0].options["use_onnx"] is not configured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice, language",
    [
        ("ef_dora", "es"),
        ("ff_siwis", "fr"),
        ("hf_alpha", "hi"),
        ("if_sara", "it"),
        ("pf_dora", "pt-br"),
        ("jf_alpha", "ja"),
        ("zf_xiaobei", "zh"),
    ],
)
async def test_official_language_voice_survives_fresh_catalog_projection(
    catalog, voice, language
):
    host = KokoroHost(voice)
    async with host.run_test(size=(150, 65)) as pilot:
        await _wait_until(
            pilot, lambda: not host.pane.query_one("#tts-generate-btn", Button).disabled
        )
        assert host.pane.query_one("#tts-voice-select", Select).value == voice
        assert host.requests == [] and catalog.synthesize_calls == 0
        host.pane.query_one("#tts-language-select", Select).value = language
        host.pane.query_one("#tts-text-input", TextArea).text = "Language test"
        host.pane.query_one("#tts-generate-btn", Button).press()
        await _wait_until(pilot, lambda: bool(host.requests))
        assert host.requests[0].voice_id == voice
        assert host.requests[0].options["language"] == language


@pytest.mark.asyncio
async def test_settings_default_picker_accepts_all_official_language_voices(
    tmp_path, monkeypatch
):
    from tldw_chatbook.UI.Speech import speech_settings_mixin

    monkeypatch.setattr(
        speech_settings_mixin, "kokoro_ui_blend_file", lambda: tmp_path / "absent.json"
    )

    class SettingsHost(speech_settings_mixin.SpeechSettingsMixin, App):
        def compose(self):
            yield Select([], id="default-voice-select")

    host = SettingsHost()
    async with host.run_test() as pilot:
        host._update_default_voice_options("kokoro")
        selector = host.query_one("#default-voice-select", Select)
        for voice in (
            "af_heart",
            "bf_emma",
            "ef_dora",
            "ff_siwis",
            "hf_alpha",
            "if_sara",
            "jf_alpha",
            "pf_dora",
            "zf_xiaobei",
        ):
            selector.value = voice
            await pilot.pause()
            assert selector.value == voice
        assert not (tmp_path / "absent.json").exists()
