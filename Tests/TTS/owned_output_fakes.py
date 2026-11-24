"""Native boundaries for tests exercising real Console audio ownership."""

import asyncio

import pytest

from Tests.TTS.test_audio_output_runs import FilePlayer
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
from tldw_chatbook.TTS import audio_output_runs, audio_player
from tldw_chatbook.TTS.audio_player import PlaybackState


class ConsoleFilePlayer(FilePlayer):
    """A private playback attempt with a chosen state after native launch."""

    def __init__(self, state=PlaybackState.FINISHED):
        super().__init__(external=state is PlaybackState.HANDED_OFF)
        self.state_after_play = state
        self.paths = []
        self.retired_token = None

    def play_prepared(self, plan, *, attempt_token, start_authority):
        self.paths.append(plan.file_path)
        self.retired_token = None
        if self.state_after_play is None:
            assert start_authority()
            self.retired_token = attempt_token
            return False
        if super().play_prepared(
            plan, attempt_token=attempt_token, start_authority=start_authority
        ):
            self.state = self.state_after_play
            return True
        return False

    def stop(self, *, expected_token):
        if self.state is PlaybackState.IDLE:
            self.stops.append(expected_token)
            return False
        if super().stop(expected_token=expected_token):
            self.retired_token = expected_token
            return True
        return False

    def retirement_confirmed(self, *, expected_token):
        return self.retired_token is not None and expected_token is self.retired_token


@pytest.fixture(autouse=True)
def forbid_native_audio(monkeypatch):
    """Fence hardware and legacy singleton seams before any handler runs."""

    def unexpected(*_args, **_kwargs):
        pytest.fail("Playback reached an unconfigured native audio boundary")

    monkeypatch.setattr(audio_player, "get_audio_player", unexpected)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", unexpected)
    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", unexpected)
    for name in (
        "play_audio_file",
        "stop_audio_playback_if_current",
        "StreamingPcmSink",
    ):
        if hasattr(tts_events, name):
            monkeypatch.setattr(tts_events, name, unexpected)


async def drain_handler_playback(handler):
    """Wait for the finite playback jobs dispatched by handler events."""
    await asyncio.gather(*tuple(handler._active_tasks))


async def wait_for_condition(predicate):
    async with asyncio.timeout(3):
        while not predicate():  # noqa: ASYNC110 - observe exact background state
            await asyncio.sleep(0.005)
