"""Process-local Speech navigation for character TTS profile actions."""

from __future__ import annotations

import pytest
from textual.app import App

from tldw_chatbook.TTS import TTSPlaygroundSelectionPreset
from tldw_chatbook.UI.Screens.stts_screen import STTSScreen
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.stts_profile_library import STTSProfileLibrary


def _preset() -> TTSPlaygroundSelectionPreset:
    return TTSPlaygroundSelectionPreset(
        provider_id="audio_cpp",
        model_id="roleplay/model",
        voice_id="roleplay-voice",
        response_format="wav",
        speed=1.0,
        options={},
        availability="available",
    )


class _SpeechHost(App[None]):
    def __init__(self, context: dict[str, object] | None = None) -> None:
        super().__init__()
        self.screen_under_test = STTSScreen(self)
        if context is not None:
            self.screen_under_test.apply_navigation_context(context)

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


async def _wait_until(pilot, predicate, *, attempts: int = 120) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_profile_library_navigation_waits_for_deferred_speech_body() -> None:
    app = _SpeechHost({"view": "profiles"})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                screen.stts_window is not None
                and screen.stts_window.current_view == "profiles"
                and len(screen.query(STTSProfileLibrary)) == 1
            ),
        )


@pytest.mark.asyncio
async def test_exact_playground_preset_survives_deferred_speech_body_mount() -> None:
    preset = _preset()
    app = _SpeechHost({"view": "playground", "profile_preset": preset})
    screen = app.screen_under_test

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: (
                len(screen.query(SpeechPlaygroundPane)) == 1
                and screen.query_one(SpeechPlaygroundPane)._profile_preset is preset
            ),
        )


@pytest.mark.asyncio
async def test_exact_preset_applies_to_an_already_open_playground() -> None:
    app = _SpeechHost()
    screen = app.screen_under_test
    preset = _preset()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: len(screen.query(SpeechPlaygroundPane)) == 1,
        )
        original = screen.query_one(SpeechPlaygroundPane)

        screen.apply_navigation_context(
            {"view": "playground", "profile_preset": preset}
        )
        await _wait_until(
            pilot,
            lambda: (
                len(screen.query(SpeechPlaygroundPane)) == 1
                and screen.query_one(SpeechPlaygroundPane) is original
                and screen.query_one(SpeechPlaygroundPane)._profile_preset is preset
            ),
        )


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"view": 1},
        {"view": "unknown"},
        {"view": "playground", "profile_preset": object()},
        {"view": "profiles", "profile_preset": _preset()},
    ],
)
def test_malformed_speech_navigation_context_is_rejected(
    context: dict[str, object],
) -> None:
    screen = STTSScreen(App())

    screen.apply_navigation_context(context)

    assert screen._pending_navigation_context is None
