"""Global voice policies must agree with the effective admission contract."""

import pytest
from textual.widgets import Select, Static

from Tests.UI.test_kokoro_playground_generation import _RecordingAdapter
from Tests.UI.test_settings_speech_tts_panel import (
    _audio_cpp_observation,
    _audio_cpp_state,
    _PanelHarness,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import TTSProviderDescriptor, TTSProviderSpec
from tldw_chatbook.TTS.legacy_catalogs import (
    LEGACY_DEFAULT_MODELS,
    LEGACY_DEFAULT_VOICES,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    GlobalSpeechTTSValidationError,
    load_global_speech_tts_state,
)
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)


def _legacy_state(provider="kokoro", voice_mode="exact"):
    return load_global_speech_tts_state(
        {
            "COMPREHENSIVE_CONFIG_RAW": {
                "app_tts": {
                    "default_provider": provider,
                    "default_model_mode": "exact",
                    "default_model": LEGACY_DEFAULT_MODELS[provider],
                    "default_voice_mode": voice_mode,
                    **(
                        {"default_voice": LEGACY_DEFAULT_VOICES[provider]}
                        if voice_mode == "exact"
                        else {}
                    ),
                    "default_format": "wav",
                }
            }
        }
    )


@pytest.mark.parametrize("provider", LEGACY_DEFAULT_MODELS)
def test_legacy_defaults_reject_server_default_voice_before_publication(provider):
    state = _legacy_state(provider, "server_default")
    with pytest.raises(GlobalSpeechTTSValidationError) as caught:
        state.defaults.snapshot()
    assert caught.value.field_id == "voice_mode"
    assert "Exact" in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("exact_audio_cpp", [False, True])
async def test_switch_to_kokoro_uses_its_own_admissible_model_and_voice(
    exact_audio_cpp,
):
    state = _audio_cpp_state(
        model_mode="exact" if exact_audio_cpp else "first_available",
        model_id="model-a" if exact_audio_cpp else None,
        voice_mode="exact" if exact_audio_cpp else "server_default",
        voice_id="voice-a" if exact_audio_cpp else None,
    )
    app = _PanelHarness(state=state, observation=_audio_cpp_observation())
    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-default-provider", Select).value = "kokoro"
        await pilot.pause()
        panel = app.query_one(SpeechTTSSettingsPanel)
        voice = app.query_one("#settings-speech-voice-policy", Select)
        assert voice._legal_values == {"exact"}
        assert voice.value == "exact"
        panel._collect_visible_state()
        preferences = panel.state.defaults.snapshot()
        assert preferences.model_id == "kokoro"
        assert preferences.voice_id == LEGACY_DEFAULT_VOICES["kokoro"]
        adapter = _RecordingAdapter("kokoro")
        registry = TTSAdapterRegistry(
            specs=(
                TTSProviderSpec(
                    descriptor=TTSProviderDescriptor(
                        provider_id="kokoro", display_name="Kokoro", native=False
                    ),
                    factory=lambda config: adapter,
                    initial_config={},
                    exclusive_reconfigure=True,
                ),
            ),
            aliases={},
        )
        service = TTSService(registry, preferences_snapshot=preferences)
        try:
            for _ in range(2):
                response = await service.synthesize_default(text="Speak this reply.")
                try:
                    assert (
                        b"".join([chunk async for chunk in response.byte_stream])
                        == b"test-audio"
                    )
                finally:
                    await response.aclose()
        finally:
            await service.close()
            await service.wait_closed()


@pytest.mark.asyncio
async def test_saved_invalid_kokoro_policy_remains_visible_and_can_be_repaired():
    app = _PanelHarness(
        state=_legacy_state(voice_mode="server_default"), configure_provider="kokoro"
    )
    async with app.run_test(size=(150, 60)) as pilot:
        panel = app.query_one(SpeechTTSSettingsPanel)
        voice = app.query_one("#settings-speech-voice-policy", Select)
        assert voice.value == "server_default"
        assert panel.request_save() is None
        await pilot.pause()
        assert not app.events
        error = app.query_one("#settings-speech-voice-policy-error", Static)
        assert "Exact" in str(error.renderable)
        voice.value = "exact"
        await pilot.pause()
        app.query_one("#settings-speech-voice-value", Select).value = "af_heart"
        await pilot.pause()
        status = app.query_one("#settings-speech-default-status", Static)
        assert "Unsaved" in str(status.renderable)
        assert panel.request_save() is not None
        await pilot.pause()
        assert len(app.events) == 1
        assert app.events[0].preferences.voice_mode == "exact"
        assert app.events[0].preferences.voice_id == "af_heart"


@pytest.mark.asyncio
async def test_return_to_saved_kokoro_restores_its_explicit_voice():
    state = _legacy_state()
    state.defaults.voice_id = "bf_emma"
    app = _PanelHarness(state=state)
    async with app.run_test(size=(150, 60)) as pilot:
        app.query_one("#settings-speech-default-provider", Select).value = "audio_cpp"
        await pilot.pause()
        app.query_one("#settings-speech-default-provider", Select).value = "kokoro"
        await pilot.pause()
        assert app.query_one("#settings-speech-voice-policy", Select).value == "exact"
        assert app.query_one("#settings-speech-voice-value", Select).value == "bf_emma"
        assert app.query_one("#settings-speech-model-value", Select).value == "kokoro"
