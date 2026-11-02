"""TASK-38581 regression: pause()/resume() must not deadlock the player.

`SimpleAudioPlayer` guarded its state with a non-reentrant
`threading.Lock`. On players without native pause (afplay -- the macOS
default -- and ffplay/paplay/pw-play/aplay on Linux), `pause()` takes the
lock and then calls `get_position()` and `stop()`, both of which re-acquire
it; `resume()` takes the lock and calls `play()`, which calls `stop()`.
With a plain `Lock` those acquisitions self-deadlock permanently: one
Pause click wedged that call forever, and because `AsyncAudioPlayer`
funnels every player call through a single-worker executor, it wedged all
later TTS audio with it. Runtime-proven by the 2026-11-02 audit (pause()
never returned within 3s; get_state() then blocked; the executor never
recovered).

These tests exercise the REAL player object in the afplay shape (forced
onto the instance so the test is deterministic on any host OS) with the
join-timeout pattern: the call runs on a helper thread and must finish
within 3s. Pre-fix, the `is_alive()` assertion fails fast instead of the
suite hanging. No real audio: `subprocess.Popen` is faked wherever a
spawn could happen, and the pause test needs no process at all.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.TTS.audio_player import (
    AudioPlayerInfo,
    PlaybackState,
    SimpleAudioPlayer,
)

pytestmark = pytest.mark.unit

#: How long a pause/resume call may take before it counts as deadlocked.
#: The post-fix calls return in microseconds; 3s leaves huge headroom
#: while keeping a pre-fix run failing fast instead of hanging the suite.
_DEADLOCK_TIMEOUT_S = 3.0


def _no_native_pause_player() -> SimpleAudioPlayer:
    """A real `SimpleAudioPlayer` forced into the afplay (macOS) shape.

    Constructing the real class keeps `_find_player`/`play`'s actual code
    paths in play; the instance attributes are then forced so the test is
    identical on Linux/Windows/CI hosts. Construction itself probes PATH
    only -- it never opens an audio device.
    """
    player = SimpleAudioPlayer()
    player._system = "Darwin"
    player._player_name = "afplay"
    player._player_cmd = ["/usr/bin/afplay"]
    player._supports_pause = False
    return player


def _run_with_deadlock_guard(call) -> tuple[bool, object]:
    """Run `call` on a helper thread; return (completed, result).

    The daemon thread plus bounded join means a deadlocked call shows up
    as `completed=False` after ~3s rather than hanging the test process.
    """
    outcome: dict[str, object] = {}

    def _run() -> None:
        outcome["value"] = call()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=_DEADLOCK_TIMEOUT_S)
    return (not thread.is_alive()), outcome.get("value")


def _guarded_cleanup(player: SimpleAudioPlayer) -> None:
    """`player.stop()` for test cleanup, itself deadlock-guarded.

    Pre-fix, a wedged pause/resume helper thread holds the player lock
    forever -- an unguarded `stop()` from the main thread would hang the
    whole suite on that same lock (exactly how the single-worker
    `AsyncAudioPlayer` executor stayed wedged in production). Guarded, a
    pre-fix run leaks one daemon thread and moves on.
    """
    _run_with_deadlock_guard(player.stop)


def _live_process_mock() -> MagicMock:
    """A fake player process that behaves like a live one.

    `poll()` reports running and `wait()` blocks until `kill()` fires --
    an instant-exit mock would let `_monitor_playback` flip the state to
    FINISHED before the test can assert PLAYING, a race real playback
    never exhibits (the clip runs for seconds).
    """
    exited = threading.Event()
    process = MagicMock()
    process.poll.return_value = None

    def _wait(*args, **kwargs) -> int:
        exited.wait(timeout=10)
        return 0

    process.kill.side_effect = exited.set
    process.wait.side_effect = _wait
    return process


def test_pause_on_no_native_pause_player_returns_promptly(tmp_path):
    """pause() on afplay re-enters the lock via get_position() and stop()."""
    player = _no_native_pause_player()
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake audio data")
    # PLAYING state as a live playback would leave it; the pause path
    # under test needs no real process -- stop() handles process=None.
    player._current = AudioPlayerInfo(
        file_path=clip,
        state=PlaybackState.PLAYING,
        start_time=time.time(),
    )

    completed, result = _run_with_deadlock_guard(player.pause)

    assert completed, (
        "pause() did not return within "
        f"{_DEADLOCK_TIMEOUT_S}s on a no-native-pause player (afplay): "
        "self-deadlock on the non-reentrant player lock (TASK-38581)"
    )
    assert result is True
    assert player.get_state() == PlaybackState.PAUSED


def test_resume_on_no_native_pause_player_returns_promptly(tmp_path, monkeypatch):
    """resume() on afplay re-enters the lock via play() -> stop()."""
    player = _no_native_pause_player()
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake audio data")
    player._current = AudioPlayerInfo(
        file_path=clip,
        state=PlaybackState.PAUSED,
        pause_time=time.time(),
    )
    # Fake the spawn: the resume path under test is the lock chain
    # resume() -> play() -> stop(); no real player may be launched here.
    fake_process = _live_process_mock()
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    try:
        completed, result = _run_with_deadlock_guard(player.resume)

        assert completed, (
            "resume() did not return within "
            f"{_DEADLOCK_TIMEOUT_S}s on a no-native-pause player (afplay): "
            "self-deadlock on the non-reentrant player lock (TASK-38581)"
        )
        assert result is True
        assert player.get_state() == PlaybackState.PLAYING
    finally:
        _guarded_cleanup(player)


def test_pause_then_resume_cycle_on_no_native_pause_player(tmp_path, monkeypatch):
    """Full user flow: Pause click, then Resume click, both return promptly.

    Mirrors the audit repro shape: pause() from PLAYING, then resume().
    After pause()'s internal stop() the tracked file_path is cleared (a
    pre-existing quirk), so the cycle test re-arms file_path before
    resume -- exactly what a live playback that resumed from restart
    needs.
    """
    player = _no_native_pause_player()
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fake audio data")
    player._current = AudioPlayerInfo(
        file_path=clip,
        state=PlaybackState.PLAYING,
        start_time=time.time(),
    )
    fake_process = _live_process_mock()
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    try:
        completed, paused = _run_with_deadlock_guard(player.pause)
        assert completed, "pause() deadlocked (TASK-38581)"
        assert paused is True
        assert player.get_state() == PlaybackState.PAUSED

        player._current.file_path = clip
        completed, resumed = _run_with_deadlock_guard(player.resume)
        assert completed, "resume() deadlocked (TASK-38581)"
        assert resumed is True
        assert player.get_state() == PlaybackState.PLAYING
    finally:
        _guarded_cleanup(player)
