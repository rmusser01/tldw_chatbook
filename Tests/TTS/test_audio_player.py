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

from tldw_chatbook.TTS.audio_player import SimpleAudioPlayer


def test_play_does_not_raise_unbound_local_error_on_a_non_darwin_afplay_path(
    tmp_path, monkeypatch,
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
