"""Process-local Speech navigation for character TTS profile actions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.TTS import STTSGeneratedAudio, TTSPlaygroundSelectionPreset
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


def _artifact(tmp_path, operation_id: str) -> STTSGeneratedAudio:
    path = tmp_path / f"{operation_id}.wav"
    path.write_bytes(b"RIFFold-audio")
    return STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="old-model",
        voice_id="old-voice",
        source_text="Old result.",
        operation_id=operation_id,
        audio_format="wav",
        content_type="audio/wav",
        metadata={},
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


@pytest.mark.asyncio
async def test_exact_preset_retires_existing_playground_audio(tmp_path) -> None:
    app = _SpeechHost()
    screen = app.screen_under_test
    artifact = _artifact(tmp_path, "old-complete-operation")
    retire = Mock()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: len(screen.query(SpeechPlaygroundPane)) == 1,
        )
        playground = screen.query_one(SpeechPlaygroundPane)
        playground._store_delivered_artifact(artifact, announce=False)
        app._stts_handler = SimpleNamespace(retire_playground_context=retire)

        screen.apply_navigation_context(
            {"view": "playground", "profile_preset": _preset()}
        )
        await _wait_until(pilot, lambda: retire.call_count == 1)

        assert playground.current_audio_artifact is None
        assert playground.current_audio_file is None
        assert playground.query_one("#audio-play-btn", Button).disabled is True
        assert playground.query_one("#audio-export-btn", Button).disabled is True
        assert (
            str(playground.query_one("#audio-player-status", Static).renderable)
            == "Nothing loaded"
        )


@pytest.mark.asyncio
async def test_exact_preset_rejects_late_prior_generation_completion(tmp_path) -> None:
    app = _SpeechHost()
    screen = app.screen_under_test
    artifact = _artifact(tmp_path, "old-in-flight-operation")
    retire = Mock()

    async with app.run_test(size=(150, 55)) as pilot:
        await _wait_until(
            pilot,
            lambda: len(screen.query(SpeechPlaygroundPane)) == 1,
        )
        playground = screen.query_one(SpeechPlaygroundPane)
        playground._generation_operation_id = artifact.operation_id
        app._stts_handler = SimpleNamespace(retire_playground_context=retire)

        screen.apply_navigation_context(
            {"view": "playground", "profile_preset": _preset()}
        )
        await _wait_until(pilot, lambda: retire.call_count == 1)
        playground._generation_complete(artifact)

        assert playground.current_audio_artifact is None
        assert playground.current_audio_file is None
        assert playground.query_one("#audio-play-btn", Button).disabled is True
        assert playground.query_one("#audio-export-btn", Button).disabled is True


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
