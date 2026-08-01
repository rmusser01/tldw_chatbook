"""Layout and action contracts for the Studio TTS Preferences pane."""

from __future__ import annotations

import pathlib

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select

from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferencesSnapshot,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import (
    STUDIO_ACTIONS,
    SpeechSettingsPane,
)

_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _Harness(App[None]):
    CSS_PATH = _BUNDLE

    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(
            global_preferences=TTSPreferencesSnapshot.from_settings({}),
            load_result=StudioTTSLoadResult(
                StudioTTSPreferencesSnapshot(),
                StudioTTSLoadState.LOADED,
            ),
            id="speech-settings-pane",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 60), (80, 24)])
async def test_primary_studio_action_is_reachable_without_scrolling(size) -> None:
    app = _Harness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        pane = app.query_one("#speech-settings-pane")
        save = app.query_one("#studio-tts-save-btn", Button)
        assert pane.region.contains_region(save.region), (
            f"Studio save below the fold at {size}: y={save.region.y}"
        )


@pytest.mark.asyncio
async def test_narrow_studio_actions_and_field_errors_remain_visible() -> None:
    app = _Harness()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        for action in STUDIO_ACTIONS:
            button = app.query_one(f"#{action.id}", Button)
            assert pane.region.contains_region(button.region), action.id

        app.query_one("#studio-tts-model-mode", Select).value = "exact"
        await pilot.pause()
        assert not await pane.save_preferences()
        error = app.query_one("#studio-tts-model-id-error")
        error.scroll_visible()
        await pilot.pause()
        assert pane.region.contains_region(error.region)
        assert error.region.width > 0


@pytest.mark.asyncio
async def test_only_selected_request_scoped_provider_tuning_is_shown() -> None:
    app = _Harness()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        chatterbox = app.query_one("#studio-tts-chatterbox-options")
        assert chatterbox.has_class("hidden")

        pane._apply_provider("chatterbox")
        assert not chatterbox.has_class("hidden")
        assert app.query_one("#chatterbox-exaggeration-input")
        assert app.query_one("#chatterbox-cfg-weight-input")

        pane._apply_provider("audio_cpp")
        assert chatterbox.has_class("hidden")
        assert app.query_one("#studio-tts-format", Select).disabled
        assert app.query_one("#studio-tts-speed").disabled


@pytest.mark.asyncio
async def test_studio_actions_keep_their_declared_ids() -> None:
    app = _Harness()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        for action in STUDIO_ACTIONS:
            assert app.query(f"#{action.id}"), f"{action.id} was renamed while mounting"
