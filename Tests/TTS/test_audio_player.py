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

import ast
import os
import threading
from contextvars import ContextVar, copy_context
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.TTS import audio_player
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


# --- TASK-39040: the Windows fallback never reaches cmd.exe -------------------
#
# On dev before TASK-39040, a Windows host without the legacy
# `C:/Program Files/Windows Media Player/wmplayer.exe` played every clip with
# `["cmd", "/c", "start", "", "/wait", path]`. `subprocess.list2cmdline` quotes
# an argument only when it contains whitespace, so a space-free path such as
# `C:\briefings\a&calc&.mp3` reached cmd.exe with a bare `&`: command injection
# from a database row (the Artifacts screen plays `briefings` audio rows). This
# PR replaces the fallback with `os.startfile`. These tests run on any host:
# the platform, `Path.exists` for wmplayer.exe, `subprocess.Popen` and
# `os.startfile` are all patched, so no process starts and no shell runs.

#: Space-free names that cmd.exe would have parsed as command syntax.
_SHELL_METACHARACTER_NAMES = (
    "a&calc&.mp3",
    "a|calc.mp3",
    "x^calc.mp3",
    "a&b|c^d.wav",
    "%COMSPEC%.mp3",
)


def _windows_host(monkeypatch, *, wmplayer: bool = False):
    """Make `SimpleAudioPlayer` behave as on Windows, with no real process.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        wmplayer: Whether the legacy wmplayer.exe "exists".

    Returns:
        ``(spawned, opened)``: every argv handed to `subprocess.Popen`, and
        every ``(path, args, kwargs)`` call of `os.startfile`.
    """
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Windows")
    real_exists = Path.exists

    def exists(self, *args, **kwargs):
        if self.name.lower() == "wmplayer.exe":
            return wmplayer
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    spawned: list[list[str]] = []
    finished = threading.Event()
    monkeypatch.setattr(audio_player, "_audio_player_instance", None)

    def popen(argv, **kwargs):
        spawned.append([str(part) for part in argv])
        process = MagicMock()
        process.poll.side_effect = lambda: 0 if finished.is_set() else None
        process.wait.side_effect = lambda *a, **k: finished.wait(5) and 0
        process.terminate.side_effect = finished.set
        process.kill.side_effect = finished.set
        return process

    monkeypatch.setattr(audio_player.subprocess, "Popen", popen)
    opened: list[tuple[object, tuple, dict]] = []
    monkeypatch.setattr(
        os,
        "startfile",
        lambda path, *args, **kwargs: opened.append((path, args, kwargs)),
        raising=False,
    )
    return spawned, opened


def _is_cmd_exe_argv(argv: list[str]) -> bool:
    lowered = [part.lower() for part in argv]
    return any(part in {"cmd", "cmd.exe", "/c", "/k"} for part in lowered)


def test_windows_fallback_opens_the_clip_with_startfile_not_cmd_exe(
    tmp_path, monkeypatch
):
    """With no wmplayer.exe, the clip goes to `os.startfile` exactly as given,
    and no argv anywhere names cmd.exe or its `/c` switch."""
    spawned, opened = _windows_host(monkeypatch)
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fixture")

    assert SimpleAudioPlayer().play(clip) is True

    assert not any(_is_cmd_exe_argv(argv) for argv in spawned), spawned
    assert spawned == []
    assert opened == [(str(clip), (), {})]


@pytest.mark.parametrize("name", _SHELL_METACHARACTER_NAMES)
def test_windows_fallback_passes_shell_metacharacters_through_untouched(
    tmp_path, monkeypatch, name
):
    """`&`, `|`, `^` and `%VAR%` in a space-free name reach `os.startfile`
    byte for byte, with no quoting or escaping, and nothing reaches a process
    (so nothing reaches a shell)."""
    spawned, opened = _windows_host(monkeypatch)
    clip = tmp_path / name
    clip.write_bytes(b"fixture")

    assert SimpleAudioPlayer().play(clip) is True

    assert spawned == []
    assert [call[0] for call in opened] == [str(clip)]
    assert opened[0][0].endswith(name)


@pytest.mark.parametrize(
    "name", ("payload.bat", "payload.cmd", "payload.exe", "shortcut.lnk", "noext")
)
def test_windows_fallback_refuses_a_file_that_is_not_audio(tmp_path, monkeypatch, name):
    """ShellExecute runs whatever an extension is registered to run, so the
    hand-off accepts audio extensions only: a row naming a batch file is
    refused, never opened and never spawned."""
    spawned, opened = _windows_host(monkeypatch)
    target = tmp_path / name
    target.write_bytes(b"fixture")

    assert SimpleAudioPlayer().play(target) is False

    assert spawned == []
    assert opened == []


def test_windows_media_player_argv_is_unchanged_when_it_is_installed(
    tmp_path, monkeypatch
):
    """The legacy wmplayer.exe route keeps its list argv (no shell), and the
    hand-off is not used when that player exists."""
    spawned, opened = _windows_host(monkeypatch, wmplayer=True)
    clip = tmp_path / "a&calc&.mp3"
    clip.write_bytes(b"fixture")

    player = SimpleAudioPlayer()
    assert player.play(clip) is True
    player.stop()

    assert len(spawned) == 1
    assert Path(spawned[0][0]).name == "wmplayer.exe"
    assert spawned[0][1:] == ["/play", "/close", str(clip)]
    assert opened == []


def test_windows_hand_off_claims_no_control_it_does_not_have(tmp_path, monkeypatch):
    """Ruling (TASK-39040): `os.startfile` returns no process handle, so the
    player can neither pause, resume, stop nor watch a handed-off clip. It
    keeps the clip current in its own state, HANDED_OFF (fix round 2: a
    distinct state, so a caller can tell it from a clip with a process and
    stop polling for an end the app will never see), and it never reports a
    pause, resume or stop it did not perform. `stop()` releases the slot
    (returning False); deleting the file is the caller's decision, bounded
    by `HAND_OFF_RETENTION_SECONDS`."""
    _windows_host(monkeypatch)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"fixture")
    player = SimpleAudioPlayer()

    assert player.play(clip) is True

    assert player.get_current_file() == clip
    assert player.get_state() is PlaybackState.HANDED_OFF
    assert player.is_playing() is False, "no process: nothing the app can watch"
    assert player.pause() is False
    assert player.get_state() is PlaybackState.HANDED_OFF
    assert player.resume() is False
    assert player.stop() is False, "stop() must not claim it stopped the default app"
    assert player.get_current_file() is None
    assert player.get_state() is PlaybackState.IDLE
    assert audio_player.HAND_OFF_RETENTION_SECONDS == 300.0


class _NoticeRecordingApp:
    """Stands in for the running Textual app.

    `_thread_id` and `call_from_thread` mirror Textual 8.2.8 (`textual/app.py`:
    `_thread_init` records the constructing thread's ident, and
    `call_from_thread` refuses to run on that same thread). The fake runs the
    callback inline and records that it was routed, plus, per notice, whether
    `notify` ran on the app's thread.
    """

    def __init__(self) -> None:
        self.notices: list[tuple[str, dict, bool]] = []
        self.thread_calls: list[tuple[str, tuple, dict]] = []
        self._thread_id = threading.get_ident()

    def notify(self, message: str, **kwargs) -> None:
        on_app_thread = threading.get_ident() == self._thread_id
        self.notices.append((message, kwargs, on_app_thread))

    def call_from_thread(self, callback, *args, **kwargs):
        if self._thread_id == threading.get_ident():
            raise RuntimeError(
                "The `call_from_thread` method must run in a different thread "
                "from the app"
            )
        self.thread_calls.append((callback.__name__, args, kwargs))
        return callback(*args, **kwargs)


_NOTICE = "Playing in your default audio app. Pause or stop it there."


def _active_app_var() -> ContextVar:
    from textual._context import active_app

    return active_app


def test_windows_hand_off_tells_the_running_app_once_per_clip(tmp_path, monkeypatch):
    """Ruling (TASK-39040): the UI says so. Each hand-off raises one literal
    toast in the running app naming where playback went and that in-app pause
    and stop cannot reach it. From the app's own thread (a handler on the
    event loop) it is a direct `notify`; nothing is routed through
    `call_from_thread`, which would refuse its own thread."""
    _windows_host(monkeypatch)
    clip = tmp_path / "a&calc&.mp3"
    clip.write_bytes(b"fixture")
    app = _NoticeRecordingApp()
    token = _active_app_var().set(app)
    try:
        assert audio_player.play_audio_file(clip) is True
    finally:
        _active_app_var().reset(token)

    assert app.notices == [(_NOTICE, {"markup": False}, True)]
    assert app.thread_calls == []


def test_windows_hand_off_notice_crosses_the_async_player_thread(tmp_path, monkeypatch):
    """`AsyncAudioPlayer` (Speech playground, setup wizard) plays on its own
    executor thread. Qodo (fix round 2): the notice is routed through
    `app.call_from_thread(app.notify, ...)` from that thread, never a direct
    `notify` off the app's thread (on `854e00b9e3` it was direct)."""
    import asyncio

    _windows_host(monkeypatch)
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fixture")
    app = _NoticeRecordingApp()

    async def scenario() -> bool:
        token = _active_app_var().set(app)
        player = audio_player.AsyncAudioPlayer()
        try:
            return await player.play(clip)
        finally:
            _active_app_var().reset(token)
            await player.cleanup()

    assert asyncio.run(scenario()) is True
    assert app.thread_calls == [("notify", (_NOTICE,), {"markup": False})]
    assert app.notices == [(_NOTICE, {"markup": False}, False)]


def test_windows_hand_off_from_a_stopped_app_logs_instead_of_raising(
    tmp_path, monkeypatch
):
    """`call_from_thread` raises `RuntimeError("App is not running")` when the
    app has no loop (Textual 8.2.8 `app.py`, the first guard in
    `call_from_thread`). The hand-off still succeeds and nothing is toasted
    off the app's thread; the notice goes to the log."""

    class _StoppedApp(_NoticeRecordingApp):
        def call_from_thread(self, callback, *args, **kwargs):
            raise RuntimeError("App is not running")

    _windows_host(monkeypatch)
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fixture")
    app = _StoppedApp()
    token = _active_app_var().set(app)
    try:
        context = copy_context()
    finally:
        _active_app_var().reset(token)
    outcome: list[bool] = []
    worker = threading.Thread(
        target=lambda: outcome.append(context.run(audio_player.play_audio_file, clip))
    )
    worker.start()
    worker.join(5)

    assert outcome == [True]
    assert app.notices == []


async def test_windows_hand_off_toast_reaches_a_real_textual_app(tmp_path, monkeypatch):
    """The same notice through a real running Textual app, from the two
    contexts the app's play paths use: a handler on the event loop
    (Watchlists, Artifacts, Library) and `asyncio.to_thread` (Console). Qodo
    (fix round 2): both notices run `notify` on the app's thread; only the
    `to_thread` one is routed through `call_from_thread`."""
    import asyncio

    from textual.app import App

    class _Host(App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple[str, dict, bool]] = []
            self.routed: list[str] = []

        def notify(self, message, **kwargs):  # type: ignore[override]
            on_app_thread = threading.get_ident() == self._thread_id
            self.seen.append((str(message), kwargs, on_app_thread))
            return super().notify(message, **kwargs)

        def call_from_thread(self, callback, *args, **kwargs):  # type: ignore[override]
            self.routed.append(getattr(callback, "__name__", repr(callback)))
            return super().call_from_thread(callback, *args, **kwargs)

    _windows_host(monkeypatch)
    clip = tmp_path / "a&calc&.mp3"
    clip.write_bytes(b"fixture")
    host = _Host()
    async with host.run_test(notifications=True) as pilot:
        assert audio_player.play_audio_file(clip) is True
        assert await asyncio.to_thread(audio_player.play_audio_file, clip) is True
        await pilot.pause()

    notice = (_NOTICE, {"markup": False}, True)
    assert host.seen == [notice, notice]
    assert host.routed == ["notify"]


def _speech_host():
    """A `SpeechPlaybackMixin` host over the real `AsyncAudioPlayer`, with the
    DOM and transport-button sync faked."""
    from tldw_chatbook.UI.Speech.speech_playback_mixin import SpeechPlaybackMixin

    host = SpeechPlaybackMixin.__new__(SpeechPlaybackMixin)
    host.app = MagicMock()
    # The app's thread identity and cross-thread routing, as the real App
    # has them (the player's executor thread reaches `notify` through
    # `call_from_thread`).
    host.app._thread_id = threading.get_ident()
    host.app.call_from_thread = lambda callback, *args, **kwargs: callback(
        *args, **kwargs
    )
    host.app.audio_player = audio_player.AsyncAudioPlayer()
    host.init_playback_state()
    host.query_one = MagicMock(return_value=MagicMock())
    host._sync_idle_transport_actions = MagicMock()
    host._sync_active_transport_actions = MagicMock()
    host._current_result_status_copy = MagicMock(return_value="Ready")
    return host


async def _settle_speech_host(host) -> None:
    await host._retire_progress_timer()
    retire_retention = getattr(host, "_retire_hand_off_retention", None)
    if retire_retention is not None:
        await retire_retention()
    await host.app.audio_player.cleanup()


def _speech_messages(host) -> list[str]:
    return [str(call.args[0]) for call in host.app.notify.call_args_list]


def _deleting_release(clip: Path, released: list[bool]):
    """The playground's real release securely deletes its WAV copy; this one
    records whether the copy still existed when it was asked to."""

    def release() -> None:
        released.append(clip.exists())
        clip.unlink(missing_ok=True)

    return release


async def test_speech_playground_reports_no_pause_or_stop_after_a_hand_off(
    tmp_path, monkeypatch
):
    """Ruling (TASK-39040), at the surface with Pause and Stop buttons. After a
    hand-off the playground shows the notice, answers Pause with where to
    pause (it performed none), and does not say "Playback stopped" (on dev it
    did: stopping the cmd.exe wrapper left the default app playing)."""
    _windows_host(monkeypatch)
    clip = tmp_path / "speech.wav"
    clip.write_bytes(b"RIFF....WAVE")
    host = _speech_host()
    token = _active_app_var().set(host.app)
    try:
        await host._play_audio_async(clip, lambda: None)
        await host._pause_audio_async()
        assert await host._stop_audio_async() is True
    finally:
        _active_app_var().reset(token)
        await _settle_speech_host(host)

    assert _speech_messages(host) == [_NOTICE, _NOTICE]
    assert host.app.notify.call_args_list[-1].kwargs == {
        "severity": "warning",
        "markup": False,
    }


async def test_speech_playground_stop_after_a_hand_off_keeps_the_copy_for_the_bound(
    tmp_path, monkeypatch
):
    """Qodo (fix round 2): on `854e00b9e3`, Stop after a hand-off released the
    playground's WAV copy at once (`secure_delete_file`) while the default app
    was still opening it. Now Stop releases the in-app slot but keeps the copy
    for `HAND_OFF_RETENTION_SECONDS` (Ruling), then releases it; and it still
    never says "Playback stopped". This drives the Stop handler itself, with
    the copy leased the way `_play_audio_async` leases it."""
    import asyncio

    _windows_host(monkeypatch)
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 0.2)
    clip = tmp_path / "speech.wav"
    clip.write_bytes(b"RIFF....WAVE")
    released: list[bool] = []
    host = _speech_host()
    token = _active_app_var().set(host.app)
    try:
        assert await host.app.audio_player.play(clip) is True
        host._active_playback_release = _deleting_release(clip, released)

        assert await host._stop_audio_async() is True

        assert released == [] and clip.exists(), "Stop deleted a playing clip"
        assert host._active_playback_release is None, "the in-app slot is free"
        assert "Playback stopped" not in _speech_messages(host)
        assert await host.app.audio_player.get_state() is PlaybackState.IDLE
        await asyncio.sleep(0.5)
        assert released == [True] and not clip.exists(), "not released at the bound"
        assert host._hand_off_releases == {}
    finally:
        _active_app_var().reset(token)
        await _settle_speech_host(host)


async def test_speech_playground_progress_polling_ends_on_a_hand_off(
    tmp_path, monkeypatch
):
    """Qodo (fix round 2): on `854e00b9e3` a handed-off clip stayed PLAYING
    with no process to end it, so the Speech progress task polled every
    100 ms for as long as the pane lived and kept its WAV copy. Now the task
    ends on the hand-off: the progress bar and transport are hidden, the
    status says where the clip plays, and the copy gets the same bounded
    retention as Stop."""
    import asyncio

    _windows_host(monkeypatch)
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 0.2)
    clip = tmp_path / "speech.wav"
    clip.write_bytes(b"RIFF....WAVE")
    released: list[bool] = []
    host = _speech_host()
    token = _active_app_var().set(host.app)
    try:
        await host._play_audio_async(clip, _deleting_release(clip, released))
        timer = host._progress_timer_task
        assert timer is not None
        await asyncio.wait_for(timer, timeout=1.0)

        widget = host.query_one.return_value
        assert widget.update.call_args_list[-1].args == (_NOTICE,)
        class_calls = [
            (name, call.args)
            for name, call in (
                (c[0], c)
                for c in widget.method_calls
                if c[0] in {"add_class", "remove_class"}
            )
        ]
        assert class_calls[-1] == ("add_class", ("hidden",)), class_calls
        assert released == [] and clip.exists(), (
            "the copy was freed under the default app"
        )
        assert host._active_playback_release is None
        assert "Playback complete" not in _speech_messages(host)
        await asyncio.sleep(0.5)
        assert released == [True] and not clip.exists()
    finally:
        _active_app_var().reset(token)
        await _settle_speech_host(host)


async def test_speech_playground_exit_releases_a_retained_hand_off_copy_at_once(
    tmp_path, monkeypatch
):
    """The pane's unmount is the existing release point for its copy; a copy
    retained for a hand-off is released there without waiting out the bound
    (`on_unmount` calls `_retire_hand_off_retention`; the pane-level case in
    `Tests/UI/test_speech_playground_pane_lifecycle.py` drives the real
    unmount)."""
    _windows_host(monkeypatch)
    monkeypatch.setattr(audio_player, "HAND_OFF_RETENTION_SECONDS", 60.0)
    clip = tmp_path / "speech.wav"
    clip.write_bytes(b"RIFF....WAVE")
    released: list[bool] = []
    host = _speech_host()
    token = _active_app_var().set(host.app)
    try:
        assert await host.app.audio_player.play(clip) is True
        host._active_playback_release = _deleting_release(clip, released)
        assert await host._stop_audio_async() is True
        assert released == [] and len(host._hand_off_releases) == 1

        await host._retire_hand_off_retention()

        assert released == [True] and not clip.exists()
        assert host._hand_off_releases == {}
        await host._retire_hand_off_retention()
        assert released == [True], "a retained release must run once"
    finally:
        _active_app_var().reset(token)
        await _settle_speech_host(host)


# Fix round 1 (TASK-39040 review, Important 2 and Minor 3). The first commit
# checked only the GIVEN path's suffix: a link `clip.mp3 -> payload.exe`, an
# alternate data stream `x.bat:evil.mp3` and a UNC path all passed it.

_NOT_AUDIO = "not an audio file type"
_STREAM = "an alternate data stream"
_NETWORK = "a network or device path"

_HAND_OFF_REFUSED = (
    (
        r"C:\briefings\clip.mp3",
        r"C:\briefings\payload.exe",
        _NOT_AUDIO,
        "link-to-program",
    ),
    (r"C:\briefings\x.bat:evil.mp3", None, _STREAM, "alternate-data-stream"),
    (r"\\host\share\x.mp3", None, _NETWORK, "unc-path"),
    (r"C:\briefings\x.bat.", None, _NOT_AUDIO, "trailing-dot"),
    (r"C:\briefings\x.bat ", None, _NOT_AUDIO, "trailing-space"),
    (r"C:\briefings\x.mp3.bat", None, _NOT_AUDIO, "audio-name-batch-suffix"),
    (r"C:\briefings\clip.mp3", r"\\host\share\clip.mp3", _NETWORK, "link-to-unc"),
)


@pytest.mark.parametrize(
    ("given", "resolved", "rule"),
    [case[:3] for case in _HAND_OFF_REFUSED],
    ids=[case[3] for case in _HAND_OFF_REFUSED],
)
def test_hand_off_refusal_parses_windows_paths(given, resolved, rule):
    """The pure predicate, on `PureWindowsPath`, so Windows parsing (drives,
    UNC roots, streams, suffixes) is exercised on any host. Each case names
    the rule that refuses it."""
    from pathlib import PureWindowsPath

    from tldw_chatbook.TTS.windows_audio_launch import hand_off_refusal

    target = PureWindowsPath(resolved or given)
    assert hand_off_refusal(PureWindowsPath(given), target) == rule


@pytest.mark.parametrize(
    "given",
    (
        r"C:\briefings\clip.mp3",
        r"C:\briefings\X.MP3",
        r"C:\briefings\x.Mp3",
        "C:clip.wav",
    ),
)
def test_hand_off_refusal_accepts_audio_names_in_any_case(given):
    """Suffixes compare case-insensitively, as Windows does; a drive-relative
    path's drive colon is not a stream."""
    from pathlib import PureWindowsPath

    from tldw_chatbook.TTS.windows_audio_launch import hand_off_refusal

    path = PureWindowsPath(given)
    assert hand_off_refusal(path, path) is None


def test_windows_fallback_refuses_a_link_whose_target_is_not_audio(
    tmp_path, monkeypatch
):
    """Through `play()` with a real link: `clip.mp3 -> payload.exe` is
    resolved and refused, so ShellExecute never runs the program."""
    spawned, opened = _windows_host(monkeypatch)
    payload = tmp_path / "payload.exe"
    payload.write_bytes(b"MZ")
    link = tmp_path / "clip.mp3"
    try:
        link.symlink_to(payload)
    except OSError as exc:  # Windows without the symlink privilege
        pytest.skip(f"cannot create a symlink here: {exc}")

    assert SimpleAudioPlayer().play(link) is False

    assert spawned == []
    assert opened == []


def test_windows_fallback_accepts_an_upper_case_audio_suffix(tmp_path, monkeypatch):
    """`X.MP3` is audio: the case of the suffix does not matter."""
    spawned, opened = _windows_host(monkeypatch)
    clip = tmp_path / "X.MP3"
    clip.write_bytes(b"fixture")

    assert SimpleAudioPlayer().play(clip) is True

    assert spawned == []
    assert [call[0] for call in opened] == [str(clip)]


def test_windows_hand_off_failure_neither_toasts_nor_records_playing(
    tmp_path, monkeypatch
):
    """When `os.startfile` raises (no app is registered for the type), play()
    fails: no "Playing in your default audio app" toast, and the player
    records nothing as playing."""
    _windows_host(monkeypatch)

    def no_association(path, *args, **kwargs):
        raise OSError("no application is associated with this file")

    monkeypatch.setattr(os, "startfile", no_association, raising=False)
    clip = tmp_path / "clip.opus"
    clip.write_bytes(b"fixture")
    app = _NoticeRecordingApp()
    player = SimpleAudioPlayer()
    token = _active_app_var().set(app)
    try:
        assert player.play(clip) is False
    finally:
        _active_app_var().reset(token)

    assert app.notices == []
    assert player.get_state() is not PlaybackState.PLAYING
    assert player.get_current_file() is None


def test_no_audio_playback_module_builds_a_cmd_exe_argv():
    """A static guard over every audio playback package: no list or tuple
    literal names cmd.exe or its `/c` / `/k` switches. dev before TASK-39040
    had one, in `TTS/audio_player.py`."""
    package = Path(audio_player.__file__).resolve().parents[1]
    offenders: list[str] = []
    for root in ("TTS", "Audio", "Media_Playback"):
        for source in sorted((package / root).rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.List, ast.Tuple)):
                    continue
                values = [
                    element.value.lower()
                    for element in node.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                ]
                if any(value in {"cmd", "cmd.exe", "/c", "/k"} for value in values):
                    offenders.append(f"{source.relative_to(package)}:{node.lineno}")
    assert offenders == []


def test_windows_fallback_opens_the_validated_resolved_audio_path(
    tmp_path, monkeypatch
):
    spawned, opened = _windows_host(monkeypatch)
    nested = tmp_path / "nested"
    nested.mkdir()
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"fixture")
    alias = nested / ".." / clip.name
    assert alias != clip.resolve()

    assert SimpleAudioPlayer().play(alias) is True

    assert spawned == []
    assert [call[0] for call in opened] == [str(clip.resolve())]
