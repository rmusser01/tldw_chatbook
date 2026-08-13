"""The Studio editor exposes its complete, deliberately narrow inventory."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    STUDIO_TTS_PROVIDER_OPTION_KEYS,
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferencesSnapshot,
)
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID,
    SpeechTTSOwnershipScope,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import (
    STUDIO_ACTIONS,
    VOICE_BLEND_ACTIONS,
    VOICE_DESTINATION_ACTIONS,
    SpeechSettingsPane,
    VoiceBlendsPane,
)

STUDIO_SELECTION_CONTROLS = {
    "studio-tts-provider",
    "studio-tts-model-mode",
    "studio-tts-model-id",
    "studio-tts-voice-mode",
    "studio-tts-voice-id",
    "studio-tts-format",
    "studio-tts-speed",
}
STUDIO_OPTION_CONTROLS = {
    "chatterbox-exaggeration-input",
    "chatterbox-cfg-weight-input",
}


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(
            global_preferences=TTSPreferencesSnapshot.from_settings({}),
            load_result=StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.LOADED,
            ),
            id="speech-settings-pane",
        )


async def _mounted_ids() -> set[str]:
    app = _Harness()
    async with app.run_test(size=(160, 60)) as pilot:
        await pilot.pause()
        return {widget.id for widget in app.query("*") if widget.id}


@pytest.mark.asyncio
async def test_every_supported_studio_setting_and_action_is_mounted() -> None:
    mounted = await _mounted_ids()
    required = (
        STUDIO_SELECTION_CONTROLS
        | STUDIO_OPTION_CONTROLS
        | {action.id for action in STUDIO_ACTIONS}
        | {action.id for action in VOICE_DESTINATION_ACTIONS}
    )
    assert not required - mounted


def test_voice_action_constants_keep_destinations_separate_from_blend_operations() -> None:
    from tldw_chatbook.UI.Speech import speech_settings_pane

    assert not hasattr(speech_settings_pane, "VOICE_PROFILE_ACTIONS")
    assert {action.id for action in VOICE_DESTINATION_ACTIONS} == {
        "voice-profiles",
        "voice-blends",
    }
    assert {action.id for action in VOICE_BLEND_ACTIONS} == {
        "add-voice-blend-btn",
        "import-blends-btn",
        "export-blends-btn",
    }


@pytest.mark.asyncio
async def test_voice_blends_pane_mounts_every_blend_operation() -> None:
    class _BlendHarness(App[None]):
        def compose(self) -> ComposeResult:
            yield VoiceBlendsPane()

    app = _BlendHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not {
            action.id for action in VOICE_BLEND_ACTIONS
        } - {widget.id for widget in app.query("*") if widget.id}


def test_control_inventory_matches_the_request_option_contract() -> None:
    assert STUDIO_TTS_PROVIDER_OPTION_KEYS["chatterbox"] == {
        "exaggeration",
        "cfg_weight",
    }
    assert all(
        not options
        for provider, options in STUDIO_TTS_PROVIDER_OPTION_KEYS.items()
        if provider != "chatterbox"
    )


@pytest.mark.asyncio
async def test_global_configuration_controls_are_not_secondarily_mounted() -> None:
    mounted = await _mounted_ids()
    global_owned = {
        control_id
        for control_id, ownership in SPEECH_TTS_OWNERSHIP_BY_CONTROL_ID.items()
        if ownership.scope is SpeechTTSOwnershipScope.GLOBAL_CONFIGURATION
    }
    assert global_owned.isdisjoint(mounted)
