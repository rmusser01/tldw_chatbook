"""File playback failure, external Stop refusal and handoff artifact retention."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from Tests.TTS.owned_output_fakes import (  # noqa: F401
    ConsoleFilePlayer,
    drain_handler_playback,
    forbid_native_audio,
    wait_for_condition,
)
from Tests.TTS_Events import test_spoken_feedback_streaming as streaming_fixture
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
from tldw_chatbook.TTS import audio_output_runs, audio_player, playback_capability
from tldw_chatbook.TTS.audio_player import PlaybackState

handler = streaming_fixture.handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (PlaybackState.ERROR, False),
        (PlaybackState.IDLE, False),
        (PlaybackState.PAUSED, False),
        (PlaybackState.FINISHED, True),
        (PlaybackState.PLAYING, True),
        (PlaybackState.HANDED_OFF, True),
    ],
)
async def test_file_poll_accepts_only_finished_playing_or_handed_off(
    handler, monkeypatch, tmp_path, state, expected
):
    path = tmp_path / "synthetic.wav"
    path.write_bytes(b"RIFF synthetic fixture")
    player = ConsoleFilePlayer(state)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    attempt = handler._prepare_output("synthetic", None, path=path)

    assert await handler._play_file_output(attempt, None, 0) is expected
    assert player.paths == [path]
    if state in (PlaybackState.PLAYING, PlaybackState.PAUSED):
        assert attempt.run.lease.is_current()
        assert player.stops == []
        assert await handler.app.audio_output.stop(attempt.run.lease)
        await drain_handler_playback(handler)
    assert handler.app.audio_output.owner is None
    await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_utterance_device_failure_never_reports_success(handler, monkeypatch):
    handler._tts_service = streaming_fixture._FakeService(
        streaming_fixture._FakeResponse(
            [b"ID3", b"synthetic"], audio_format="mp3", sample_rate=None
        )
    )
    player = ConsoleFilePlayer(PlaybackState.ERROR)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    monkeypatch.setattr(
        playback_capability, "format_playable_locally", lambda _format: True
    )
    monkeypatch.setattr(
        audio_player, "find_player_for_format", lambda _format: "test-player"
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
        (artifact,) = player.paths
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
    player = ConsoleFilePlayer(PlaybackState.ERROR)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
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
        await drain_handler_playback(handler)
        assert states == ["playing", "failed"]
        assert handler._speech_outputs == []
        assert handler.app.audio_output.owner is None
        assert player.stops == [player.token]
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


def _hand_off_lifecycle(states):
    return tts_events.TTSPlaybackLifecycle(
        message_id="owned-reply",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )


@pytest.fixture
def external_player(monkeypatch):
    player = ConsoleFilePlayer(PlaybackState.HANDED_OFF)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 0.4)
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 0.25)
    return player


async def _play(handler, artifact, lifecycle=None, *, drain=True):
    message_id = lifecycle.message_id if lifecycle is not None else "legacy-reply"
    await handler._cache_audio_file(message_id, artifact, lifecycle)
    await handler.handle_tts_playback(
        tts_events.TTSPlaybackEvent(
            action="play", message_id=message_id, playback_lifecycle=lifecycle
        )
    )
    if drain:
        await drain_handler_playback(handler)
    return message_id


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [True, False])
async def test_stop_after_hand_off_is_refused_and_preserves_file(
    handler, external_player, tmp_path, owned
):
    artifact = tmp_path / "reply.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    states, outcomes = [], []
    lifecycle = _hand_off_lifecycle(states) if owned else None
    try:
        dispatched_at = time.monotonic()
        message_id = await _play(handler, artifact, lifecycle, drain=False)
        host = handler.app.audio_output
        await wait_for_condition(
            lambda: host.handoff_epoch is not None and host.owner is None
        )
        assert external_player.paths == [artifact]
        if owned:
            assert states == ["playing"]
        epoch = host.handoff_epoch
        native_retirements = list(external_player.stops)

        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="stop",
                message_id=message_id,
                playback_lifecycle=lifecycle,
                outcome_callback=outcomes.append,
            )
        )

        assert outcomes == [False]
        assert external_player.stops == native_retirements
        assert host.handoff_epoch is epoch
        assert artifact.exists()
        assert handler._audio_files == {message_id: artifact}
        if owned:
            assert states == ["playing"]
        await drain_handler_playback(handler)
        if owned:
            assert states == ["playing", "stopped"]
        assert artifact.exists()
        await wait_for_condition(lambda: not handler._audio_files)
        assert not artifact.exists()
        assert time.monotonic() - dispatched_at >= 0.4
        assert handler._audio_file_owners == {}
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_owned_hand_off_file_is_deleted_when_the_handler_shuts_down(
    handler, external_player, tmp_path, monkeypatch
):
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 60.0)
    try:
        await _play(handler, artifact, _hand_off_lifecycle([]))
        assert artifact.exists()
        assert len(handler._pending_legacy_cleanup_timers) == 1

        await handler.cleanup_tts_resources()

        assert not artifact.exists()
        assert not handler._pending_legacy_cleanup_timers
        assert not handler._audio_files
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_replacement_output_preserves_handed_off_file_for_its_bound(
    handler, external_player, tmp_path
):
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    host = handler.app.audio_output
    try:
        await _play(handler, artifact, _hand_off_lifecycle([]))
        epoch = host.handoff_epoch
        next_player = ConsoleFilePlayer(PlaybackState.PLAYING)
        next_run = await audio_output_runs.FileOutputRun.prepare(
            tmp_path / "next.wav", player=next_player
        )
        ticket = host.begin_intent()
        assert host.acknowledge_handoff(ticket, epoch)
        lease = await host.acquire("Replacement", next_run.retire, intent=ticket)
        try:
            assert await next_run.start(lease)
            assert artifact.exists()
            await wait_for_condition(lambda: not artifact.exists())
            assert lease.is_current()
            assert not next_player.stops
        finally:
            await host.stop(lease)
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_a_stop_during_the_hand_off_window_keeps_the_file_for_the_bound(
    handler, external_player, tmp_path
):
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    states, outcomes = [], []
    lifecycle = _hand_off_lifecycle(states)
    external_player.release.clear()
    stop = None
    try:
        await _play(handler, artifact, lifecycle, drain=False)
        assert await asyncio.to_thread(external_player.entered.wait, 3)
        (attempt,) = handler._speech_outputs
        stop = asyncio.create_task(
            handler.handle_tts_playback(
                tts_events.TTSPlaybackEvent(
                    action="stop",
                    message_id=lifecycle.message_id,
                    playback_lifecycle=lifecycle,
                    outcome_callback=outcomes.append,
                )
            )
        )
        await wait_for_condition(lambda: attempt.run.lease.revoked)
        assert not stop.done()
        assert artifact.exists()

        external_player.release.set()
        await stop
        await drain_handler_playback(handler)
        assert outcomes == [False]
        assert handler.app.audio_output.handoff_epoch is not None
        assert artifact.exists()
        await wait_for_condition(lambda: not artifact.exists())
    finally:
        external_player.release.set()
        if stop is not None:
            await stop
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_regenerating_speech_waits_out_the_replaced_clips_hand_off(
    handler, external_player, tmp_path
):
    old = tmp_path / "old.wav"
    new = tmp_path / "new.wav"
    old.write_bytes(b"RIFF synthetic hand-off fixture")
    new.write_bytes(b"RIFF synthetic regenerated fixture")
    lifecycle = _hand_off_lifecycle([])
    try:
        await _play(handler, old, lifecycle)
        await handler._cache_audio_file(lifecycle.message_id, new, lifecycle)

        assert handler._audio_files == {lifecycle.message_id: new}
        assert old.exists()
        await wait_for_condition(lambda: not old.exists())
        assert new.exists()
    finally:
        await handler.cleanup_tts_resources()
