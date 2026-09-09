"""Device failure propagates through both public file-playback consumers."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.TTS_Events import test_spoken_feedback_streaming as streaming_fixture
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
from tldw_chatbook.TTS.audio_player import PlaybackState

handler = streaming_fixture.handler


def _player(path, state):
    player = Mock()
    player.play.return_value = True
    player.get_current_file.return_value = path
    player.get_state.return_value = state
    return player


@pytest.mark.parametrize(
    "state,timeout,expected",
    [
        (PlaybackState.ERROR, 0.03, False),
        (PlaybackState.ERROR, 0, False),
        (PlaybackState.IDLE, 0, False),
        (PlaybackState.PAUSED, 0, False),
        (PlaybackState.FINISHED, 0, True),
        (PlaybackState.PLAYING, 0, True),
    ],
)
def test_file_poll_accepts_only_finished_or_still_playing_at_timeout(
    state, timeout, expected
):
    path = Path("synthetic.wav")
    assert (
        tts_events._play_legacy_clip_and_await_completion(
            _player(path, state),
            path,
            timeout_seconds=timeout,
            poll_interval_seconds=0.005,
        )
        is expected
    )


@pytest.mark.asyncio
async def test_utterance_device_failure_never_reports_success(handler, monkeypatch):
    handler._tts_service = streaming_fixture._FakeService(
        streaming_fixture._FakeResponse(
            [b"ID3", b"synthetic"], audio_format="mp3", sample_rate=None
        )
    )
    player = _player(None, PlaybackState.ERROR)

    def play(path):
        player.get_current_file.return_value = path
        return True

    player.play.side_effect = play
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(
        tts_events, "_legacy_playback_timeout_seconds", lambda *_args: 0.03
    )
    release = AsyncMock(wraps=handler._try_secure_delete_tts_artifact)
    monkeypatch.setattr(handler, "_try_secure_delete_tts_artifact", release)
    results = []
    try:
        await handler.speak_utterance("A synthetic reply.", on_finished=results.append)
        assert results == [False]
        assert len(player.play.call_args_list) == 1
        artifact = player.play.call_args.args[0]
        await handler.cleanup_tts_resources()
        assert not artifact.exists()
        release.assert_awaited_once()
        assert release.await_args.args == (artifact,)
        assert release.await_args.kwargs["on_late_success"].args[-1] is None
        await handler.cleanup_tts_resources()
        release.assert_awaited_once()
        assert results == [False]
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_owned_playback_device_failure_releases_artifact_and_reports_failed(
    handler, monkeypatch, tmp_path
):
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic device-transport fixture")
    states = []
    lifecycle = tts_events.TTSPlaybackLifecycle(
        message_id="owned-reply",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )
    handler._audio_files[lifecycle.message_id] = artifact
    handler._audio_file_owners[lifecycle.message_id] = lifecycle
    player = _player(artifact, PlaybackState.ERROR)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 0.03)
    release = AsyncMock(wraps=handler._try_secure_delete_tts_artifact)
    monkeypatch.setattr(handler, "_try_secure_delete_tts_artifact", release)
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="play",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
        await handler._active_file_playback_task
        assert states == ["playing", "failed"]
        assert handler._active_file_playback_task is None
        assert handler._active_file_playback_owner is None
        assert handler._active_file_playback_stop is None
        assert handler._last_played is None
        assert not artifact.exists()
        assert handler._audio_files == {}
        assert handler._audio_file_owners == {}
        release.assert_awaited_once()
        assert release.await_args.args == (artifact,)
        assert release.await_args.kwargs["on_late_success"].args[-1] is lifecycle
        await handler.cleanup_tts_resources()
        release.assert_awaited_once()
    finally:
        await handler.cleanup_tts_resources()
