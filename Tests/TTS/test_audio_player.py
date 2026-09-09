"""Regression test for task-4 review N1 (pre-existing, `audio_player.py`
last touched by `d9f060f0b`; surfaced while reviewing the hands-free-loop
utterance entry, since a legacy-path hands-free reply routes through
`SimpleAudioPlayer.play()`).

`play()` imported `time` at MODULE scope (`:10`) and again FUNCTION-LOCALLY
inside the Darwin/afplay branch (the pre-`Popen` delay). Python's compiler
sees any assignment target (including an `import` statement) anywhere in a
function body and treats that name as local for the WHOLE function -- so
`time` became a local name for all of `play()`, and any path that never
executes the Darwin/afplay branch (every other player: Linux mpv/mplayer/
ffplay/aplay/paplay, Windows) hit `UnboundLocalError` at
`self._current.start_time = time.time()`, caught by `play()`'s own broad
`except Exception` and silently returned as `False`. Consequence: every
legacy-path (the default response format for every provider except
`audio_cpp`) hands-free utterance returned `on_finished(False)` immediately
on Linux and Windows -- reply speech entirely silent on two of three
platforms.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tldw_chatbook.TTS.audio_player import (
    AudioPlayerInfo,
    PlaybackState,
    SimpleAudioPlayer,
)


def test_play_does_not_raise_unbound_local_error_on_a_non_darwin_afplay_path(
    tmp_path,
    monkeypatch,
):
    """Deterministic regardless of the host OS this test actually runs on:
    forces the player instance into a non-Darwin/afplay shape directly
    (bypassing `_find_player()`'s real OS/binary detection, which would
    otherwise need a real mpv/aplay/etc. binary present) and fakes
    `subprocess.Popen` so no real player process is spawned.
    """
    player = SimpleAudioPlayer()
    player._system = "Linux"
    player._player_name = "aplay"
    player._player_cmd = ["aplay", "-q"]
    player._supports_pause = False

    audio_file = tmp_path / "clip.wav"
    audio_file.write_bytes(b"fake audio data")

    fake_process = MagicMock()
    fake_process.poll.return_value = None
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    started = player.play(audio_file)

    assert started is True, (
        "a non-Darwin/afplay path must not raise UnboundLocalError on "
        "time.time() -- see the module-level vs. function-local `import "
        "time` collision this pins (task-4 review N1)"
    )
    assert player.get_current_file() == audio_file


def test_macos_opus_uses_a_compatible_player(tmp_path, monkeypatch):
    player = SimpleAudioPlayer()
    player._system = "Darwin"
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.shutil.which",
        lambda name: "/test/ffplay" if name == "ffplay" else None,
    )
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        process = MagicMock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        return process

    monkeypatch.setattr("tldw_chatbook.TTS.audio_player.subprocess.Popen", spawn)
    for suffix in ("opus", "wav"):
        path = tmp_path / f"reply.{suffix}"
        path.write_bytes(b"fixture")
        assert player.play(path)
    assert commands[0][0] == "/test/ffplay"
    assert "-autoexit" in commands[0] and "-nodisp" in commands[0]
    assert commands[1][0] == "/usr/bin/afplay"
    player.cleanup()


def test_macos_opus_without_compatible_player_refuses_playback(tmp_path, monkeypatch):
    player = SimpleAudioPlayer()
    player._system = "Darwin"
    monkeypatch.setattr("tldw_chatbook.TTS.audio_player.shutil.which", lambda _: None)
    popen = MagicMock()
    monkeypatch.setattr("tldw_chatbook.TTS.audio_player.subprocess.Popen", popen)
    path = tmp_path / "reply.opus"
    path.write_bytes(b"fixture")
    assert not player.play(path)
    popen.assert_not_called()


def test_player_nonzero_exit_is_failure():
    player = SimpleAudioPlayer()
    failed = MagicMock()
    failed.wait.return_value = 1
    player._current = AudioPlayerInfo(process=failed, state=PlaybackState.PLAYING)
    player._monitor_playback()
    assert player.get_state() == PlaybackState.ERROR


def test_previous_monitor_cannot_finish_next_clip():
    player = SimpleAudioPlayer()
    old, new = MagicMock(), MagicMock()
    player._current = AudioPlayerInfo(process=old, state=PlaybackState.PLAYING)

    def wait():
        player._current = AudioPlayerInfo(process=new, state=PlaybackState.PLAYING)
        return 0

    old.wait.side_effect = wait
    player._monitor_playback()
    assert player.get_state() == PlaybackState.PLAYING
    assert player._current.process is new
