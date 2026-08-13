"""Layout and action contracts for the Studio TTS Preferences pane."""

from __future__ import annotations

import pathlib

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Static, Switch

from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSLoadResult,
    StudioTTSLoadState,
    StudioTTSPreferencesSnapshot,
)
from tldw_chatbook.UI.Speech.speech_settings_pane import (
    STUDIO_ACTIONS,
    VOICE_DESTINATION_ACTIONS,
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

    def __init__(
        self,
        snapshot: StudioTTSPreferencesSnapshot | None = None,
    ) -> None:
        super().__init__()
        self._snapshot = snapshot or StudioTTSPreferencesSnapshot()

    def compose(self) -> ComposeResult:
        yield SpeechSettingsPane(
            global_preferences=TTSPreferencesSnapshot.from_settings({}),
            load_result=StudioTTSLoadResult(
                self._snapshot,
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
@pytest.mark.parametrize("size", [(160, 48), (80, 24)])
async def test_voice_destination_strip_has_computed_desktop_and_narrow_layout(
    size: tuple[int, int],
) -> None:
    app = _Harness()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        strip = app.query_one("#studio-tts-voice-destination-actions")
        buttons = [
            app.query_one(f"#{action.id}", Button)
            for action in VOICE_DESTINATION_ACTIONS
        ]
        strip.scroll_visible()
        await pilot.pause()

        if size[0] < 104:
            assert pane.has_class("studio-tts-settings-stacked")
            assert len({button.region.y for button in buttons}) == len(buttons)
            assert all(
                button.region.width == strip.content_region.width
                for button in buttons
            )
        else:
            assert not pane.has_class("studio-tts-settings-stacked")
            assert len({button.region.y for button in buttons}) == 1
            assert all(button.region.height == 1 for button in buttons)


def test_voice_destination_css_selector_is_synced_to_bundle() -> None:
    source = (
        _BUNDLE.parent / "features" / "_lab.tcss"
    ).read_text(encoding="utf-8")
    bundle = _BUNDLE.read_text(encoding="utf-8")
    for stylesheet in (source, bundle):
        assert "#studio-tts-voice-destination-actions" in stylesheet
        assert "#studio-tts-voice-profile-actions" not in stylesheet


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
async def test_auto_play_is_an_explicit_studio_only_preference() -> None:
    app = _Harness(StudioTTSPreferencesSnapshot(auto_play=True))

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        pane = app.query_one(SpeechSettingsPane)
        auto_play = app.query_one("#studio-tts-auto-play", Switch)
        state = app.query_one("#studio-tts-auto-play-state", Static)

        assert auto_play.value is True
        assert "On" in str(state.renderable)
        assert "only" in str(state.renderable).casefold()
        assert pane.is_dirty is False

        auto_play.value = False
        await pilot.pause()

        assert pane.is_dirty is True
        assert "Off" in str(state.renderable)
        candidate = pane._collect_candidate(show_errors=True)
        assert candidate is not None
        assert candidate.auto_play is False


@pytest.mark.asyncio
async def test_studio_actions_keep_their_declared_ids() -> None:
    app = _Harness()
    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        for action in STUDIO_ACTIONS:
            assert app.query(f"#{action.id}"), f"{action.id} was renamed while mounting"
