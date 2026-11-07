"""Device failure propagates through both public file-playback consumers, and
a handed-off clip (TASK-39040) is neither reported stopped nor deleted early."""

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.TTS_Events import test_spoken_feedback_streaming as streaming_fixture
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
from tldw_chatbook.TTS import audio_player as audio_player_module
from tldw_chatbook.TTS.audio_player import PlaybackState

handler = streaming_fixture.handler


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


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


# TASK-39040 fix round 2 (Qodo). On `854e00b9e3`, `stop_audio_playback_if_
# current` discarded `SimpleAudioPlayer.stop()`'s False for a clip handed to
# the default app, so the Console accepted the Stop, reported the message
# "stopped" and deleted the clip while the default app was still playing it.
# New with that PR (the hand-off path is new); fixed here.


def _hand_off_lifecycle(states):
    return tts_events.TTSPlaybackLifecycle(
        message_id="owned-reply",
        request_id=1,
        validator=lambda: True,
        callback=states.append,
    )


def test_file_poll_treats_a_handed_off_clip_like_a_playing_one_at_the_bound():
    """A handed-off clip has no process to observe, so the poll runs to its
    bound and, as for PLAYING, assumes the clip played through rather than
    reporting the message failed."""
    path = Path("synthetic.wav")
    assert (
        tts_events._play_legacy_clip_and_await_completion(
            _player(path, PlaybackState.HANDED_OFF),
            path,
            timeout_seconds=0,
            poll_interval_seconds=0.005,
        )
        is True
    )


@pytest.mark.asyncio
async def test_owned_stop_after_a_hand_off_is_refused_and_the_file_outlives_it(
    handler, monkeypatch, tmp_path
):
    """Stop on a handed-off Console clip is refused (outcome False, the player
    is not told to stop, no "stopped" terminal, the owner keeps the slot), and
    the file is kept for `HAND_OFF_RETENTION_SECONDS` from the hand-off, past
    the poll bound, then deleted."""
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    states: list[str] = []
    lifecycle = _hand_off_lifecycle(states)
    handler._audio_files[lifecycle.message_id] = artifact
    handler._audio_file_owners[lifecycle.message_id] = lifecycle
    player = _player(artifact, PlaybackState.HANDED_OFF)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 0.2)
    monkeypatch.setattr(audio_player_module, "HAND_OFF_RETENTION_SECONDS", 0.6)
    outcomes: list[bool] = []
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="play",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
        await _wait_for(lambda: states == ["playing"])
        handed_off_at = time.monotonic()

        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="stop",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
                outcome_callback=outcomes.append,
            )
        )

        assert outcomes == [False], "a stop the app cannot perform was accepted"
        player.stop.assert_not_called()
        assert states == ["playing"], "reported stopped while the default app plays"
        assert artifact.exists(), "deleted under the default app"
        assert handler._active_file_playback_owner is lifecycle

        await handler._active_file_playback_task
        assert states == ["playing", "stopped"], "the poll bound ends the message"
        assert artifact.exists(), "deleted at the poll bound, inside the retention"
        assert handler._audio_files == {lifecycle.message_id: artifact}

        await _wait_for(lambda: not artifact.exists())
        assert time.monotonic() - handed_off_at >= 0.5
        await _wait_for(lambda: handler._audio_files == {})
        assert handler._audio_file_owners == {}
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_owned_hand_off_file_is_deleted_when_the_handler_shuts_down(
    handler, monkeypatch, tmp_path
):
    """The handler's shutdown is the existing cleanup point: a clip retained
    for a hand-off is deleted there, and no retention timer outlives it."""
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    states: list[str] = []
    lifecycle = _hand_off_lifecycle(states)
    handler._audio_files[lifecycle.message_id] = artifact
    handler._audio_file_owners[lifecycle.message_id] = lifecycle
    player = _player(artifact, PlaybackState.HANDED_OFF)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 0.05)
    monkeypatch.setattr(audio_player_module, "HAND_OFF_RETENTION_SECONDS", 60.0)
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="play",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
        await _wait_for(lambda: handler._active_file_playback_task is not None)
        await handler._active_file_playback_task
        assert artifact.exists()
        assert len(handler._pending_legacy_cleanup_timers) == 1

        await handler.cleanup_tts_resources()

        assert not artifact.exists()
        assert handler._pending_legacy_cleanup_timers == set()
        assert handler._audio_files == {}
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_legacy_stop_after_a_hand_off_keeps_the_file_for_the_bound(
    handler, monkeypatch, tmp_path
):
    """The legacy (no-lifecycle) play path: Stop is refused and the cached
    clip the stop branch would delete is kept for the bound instead."""
    artifact = tmp_path / "legacy.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    handler._audio_files["legacy-reply"] = artifact
    player = _player(artifact, PlaybackState.HANDED_OFF)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(audio_player_module, "HAND_OFF_RETENTION_SECONDS", 0.3)
    outcomes: list[bool] = []
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(action="play", message_id="legacy-reply")
        )
        assert player.play.call_args.args == (artifact,)

        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="stop",
                message_id="legacy-reply",
                outcome_callback=outcomes.append,
            )
        )

        assert outcomes == [False]
        player.stop.assert_not_called()
        assert artifact.exists(), "deleted under the default app"
        await _wait_for(lambda: not artifact.exists())
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_a_displacing_clip_does_not_delete_a_handed_off_file_early(
    handler, monkeypatch, tmp_path
):
    """The next clip takes the single-slot player (every `play()` stops the
    previous clip first), which ends the owned poll before its bound. The
    handed-off file is still kept for the bound from its hand-off, not
    deleted on displacement: the player has forgotten it by then, so the
    runner remembers the hand-off itself."""
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    states: list[str] = []
    lifecycle = _hand_off_lifecycle(states)
    handler._audio_files[lifecycle.message_id] = artifact
    handler._audio_file_owners[lifecycle.message_id] = lifecycle
    player = _player(artifact, PlaybackState.HANDED_OFF)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 5.0)
    monkeypatch.setattr(audio_player_module, "HAND_OFF_RETENTION_SECONDS", 0.4)
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="play",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
        await _wait_for(lambda: states == ["playing"])
        task = handler._active_file_playback_task

        # The next clip displaces ours: the player now holds another file.
        player.get_current_file.return_value = tmp_path / "next.wav"
        player.get_state.return_value = PlaybackState.PLAYING
        await task

        assert states == ["playing", "failed"], "displacement, not a finish"
        assert artifact.exists(), "deleted on displacement, under the default app"
        await _wait_for(lambda: not artifact.exists())
    finally:
        await handler.cleanup_tts_resources()


# TASK-39040 fix round 3 (scoped re-review of cda6992405). A Stop that lands
# while `play()` is still inside `os.startfile` (seconds, while the default
# app cold-starts) reached the worker's post-`play()` stop window before the
# hand-off was recorded: the worker called `stop()` (freeing the slot) and
# returned before `on_started`, the runner passed no retention, the live check
# saw IDLE, and the file was deleted under the default app. Also: regenerating
# speech for the same message replaced the cache entry and deleted the old,
# handed-off file at once. Both new with this PR; fixed here.


class _BlockingHandOffPlayer:
    """`play()` blocks until the test has dispatched Stop, then returns a
    handed-off clip: the reviewer's probe of the ShellExecute window."""

    def __init__(self, play_entered: threading.Event, release: threading.Event):
        self._play_entered = play_entered
        self._release = release
        self.current: Path | None = None
        self.state = PlaybackState.IDLE
        self.stop_calls = 0

    def play(self, path: Path) -> bool:
        self._play_entered.set()
        assert self._release.wait(5), "the test never released play()"
        self.current = path
        self.state = PlaybackState.HANDED_OFF
        return True

    def get_current_file(self) -> Path | None:
        return self.current

    def get_state(self) -> PlaybackState:
        return self.state

    def stop(self) -> bool:
        self.stop_calls += 1
        self.current = None
        self.state = PlaybackState.IDLE
        return False


@pytest.mark.asyncio
async def test_a_stop_during_the_hand_off_window_keeps_the_file_for_the_bound(
    handler, monkeypatch, tmp_path
):
    """Stop pressed while `play()` is still handing the clip to the default
    app: the worker stops nothing (there is no process), and the file is kept
    for the bound from the hand-off, then deleted. At press time the Console
    has already accepted the Stop and reported the message stopped, because
    `play()` has not returned and the app cannot yet know the clip will be
    handed off; the lifecycle is terminal-once, so that report stands (the
    hand-off toast is still raised when `play()` returns)."""
    artifact = tmp_path / "owned.wav"
    artifact.write_bytes(b"RIFF synthetic hand-off fixture")
    states: list[str] = []
    lifecycle = _hand_off_lifecycle(states)
    handler._audio_files[lifecycle.message_id] = artifact
    handler._audio_file_owners[lifecycle.message_id] = lifecycle
    play_entered = threading.Event()
    release_play = threading.Event()
    player = _BlockingHandOffPlayer(play_entered, release_play)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 5.0)
    monkeypatch.setattr(audio_player_module, "HAND_OFF_RETENTION_SECONDS", 0.4)
    outcomes: list[bool] = []
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="play",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
        await _wait_for(play_entered.is_set)

        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="stop",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
                outcome_callback=outcomes.append,
            )
        )
        # The press-time report, as documented above.
        assert outcomes == [True] and states == ["stopped"]

        release_play.set()
        await _wait_for(lambda: handler._active_file_playback_task is None)

        assert player.stop_calls == 0, "stopped a clip the default app has"
        assert artifact.exists(), "deleted under the default app"
        assert handler._audio_files == {lifecycle.message_id: artifact}
        await _wait_for(lambda: not artifact.exists())
    finally:
        release_play.set()
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_regenerating_speech_waits_out_the_replaced_clips_hand_off(
    handler, monkeypatch, tmp_path
):
    """A new artifact for the same message replaces a handed-off one in the
    cache; the replaced file is deleted at the hand-off bound, not now."""
    old = tmp_path / "old.wav"
    new = tmp_path / "new.wav"
    old.write_bytes(b"RIFF synthetic hand-off fixture")
    new.write_bytes(b"RIFF synthetic regenerated fixture")
    states: list[str] = []
    lifecycle = _hand_off_lifecycle(states)
    handler._audio_files[lifecycle.message_id] = old
    handler._audio_file_owners[lifecycle.message_id] = lifecycle
    player = _player(old, PlaybackState.HANDED_OFF)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: player
    )
    monkeypatch.setattr(tts_events, "_LEGACY_PLAYBACK_POLL_MAX_SECONDS", 5.0)
    monkeypatch.setattr(audio_player_module, "HAND_OFF_RETENTION_SECONDS", 0.4)
    try:
        await handler.handle_tts_playback(
            tts_events.TTSPlaybackEvent(
                action="play",
                message_id=lifecycle.message_id,
                playback_lifecycle=lifecycle,
            )
        )
        await _wait_for(lambda: states == ["playing"])

        await handler._cache_audio_file(lifecycle.message_id, new, lifecycle)

        assert handler._audio_files == {lifecycle.message_id: new}
        assert old.exists(), "the replaced clip was deleted under the default app"
        await _wait_for(lambda: not old.exists())
        assert new.exists()
    finally:
        await handler.cleanup_tts_resources()
