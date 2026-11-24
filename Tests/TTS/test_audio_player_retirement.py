"""Checked native retirement and exact file attempts. No audio devices used."""

import asyncio
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.TTS import audio_player, windows_audio_launch
from tldw_chatbook.TTS.audio_player import (
    AsyncAudioPlayer,
    PlaybackState,
    SimpleAudioPlayer,
)


class Child:
    def __init__(self, *, resistant=False):
        self.resistant = resistant
        self.exited = threading.Event()
        self.signals = []

    def poll(self):
        return 0 if self.exited.is_set() else None

    def wait(self, timeout=None):
        if timeout is None:
            assert self.exited.wait(4), "monitor never observed exit"
        elif not self.exited.is_set():
            raise subprocess.TimeoutExpired("player", timeout)
        return 0

    def terminate(self):
        self.signals.append("terminate")
        if not self.resistant:
            self.exited.set()

    def kill(self):
        self.signals.append("kill")
        if not self.resistant:
            self.exited.set()


@pytest.fixture
def linux(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: sys.executable if name == "aplay" else None,
    )
    path = tmp_path / "clip.wav"
    path.write_bytes(b"fixture")
    return SimpleAudioPlayer(), path


def test_failed_stop_retains_owner_and_blocks_replacement(linux, monkeypatch):
    player, path = linux
    child = Child(resistant=True)
    spawned = []

    def spawn(*args, **kwargs):
        spawned.append(child)
        return child

    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    assert player.play(path)
    try:
        assert not player.stop()
        assert player.get_current_file() == path
        assert player.get_state() is PlaybackState.BLOCKED
        assert not player.play(path)
        assert spawned == [child]
    finally:
        child.exited.set()
    assert player.stop()
    assert player.get_current_file() is None


def test_invalid_replacement_leaves_healthy_file_alone(linux, monkeypatch):
    player, path = linux
    child = Child()
    monkeypatch.setattr(audio_player.subprocess, "Popen", lambda *a, **kw: child)
    assert player.play(path)
    try:
        assert not player.play(path.with_name("missing.wav"))
        incompatible = path.with_suffix(".mp3")
        incompatible.touch()
        assert not player.play(incompatible)
        assert player.get_state() is PlaybackState.PLAYING
        assert child.signals == []
    finally:
        player.stop()


def test_prepared_file_replacement_or_revocation_cannot_displace_owner(
    linux, monkeypatch
):
    player, path = linux
    child = Child()
    monkeypatch.setattr(audio_player.subprocess, "Popen", lambda *a, **kw: child)
    candidate = path.with_name("next.wav")
    candidate.write_bytes(b"a")
    plan = player.prepare(candidate)
    assert plan is not None and not plan.external
    assert player.play(path)
    try:
        assert not player.play_prepared(plan, start_authority=lambda: False)
        candidate.write_bytes(b"different contents")
        assert not player.play_prepared(plan)
        assert player.get_current_file() == path
        assert child.signals == []
    finally:
        player.stop()


def test_prepared_windows_route_remains_the_authorized_route(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_audio_launch, "wmplayer_argv", lambda _: None)
    player = SimpleAudioPlayer()
    path = tmp_path / "clip.wav"
    path.touch()
    plan = player.prepare(path)
    assert plan is not None and plan.external
    monkeypatch.setattr(windows_audio_launch, "wmplayer_argv", lambda _: ["unexpected"])
    opened = []
    monkeypatch.setattr(audio_player.os, "startfile", opened.append, raising=False)
    monkeypatch.setattr(
        audio_player.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("changed launch route"),
    )
    assert player.play_prepared(plan)
    assert opened == [str(path)]


def test_windows_refuses_revoked_shell_open_after_validation(tmp_path, monkeypatch):
    path = tmp_path / "clip.wav"
    path.touch()
    revoked = threading.Event()

    def verify(_path):
        revoked.set()
        return True

    monkeypatch.setattr(windows_audio_launch, "can_hand_off", verify)
    opened = []
    monkeypatch.setattr(audio_player.os, "startfile", opened.append, raising=False)
    assert not windows_audio_launch.hand_off(
        path, start_authority=lambda: not revoked.is_set()
    )
    assert opened == []


def test_late_spawn_keeps_revoked_child_in_exact_custody(linux, monkeypatch):
    player, path = linux
    child = Child(resistant=True)
    entered, revoked = threading.Event(), threading.Event()
    token = object()

    def spawn(*args, **kwargs):
        entered.set()
        assert revoked.wait(3)
        return child

    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    with ThreadPoolExecutor(1) as workers:
        call = workers.submit(
            player.play,
            path,
            attempt_token=token,
            start_authority=lambda: not revoked.is_set(),
        )
        try:
            assert entered.wait(1)
        finally:
            revoked.set()
        assert not call.result(timeout=3)
    assert player.get_state(expected_token=token) is PlaybackState.BLOCKED
    assert child.signals == ["terminate", "kill"]
    child.exited.set()
    assert player.stop(expected_token=token)


def test_stale_monitor_and_stop_cannot_touch_same_filename(linux, monkeypatch):
    player, path = linux
    first, second = Child(), Child()
    children = iter((first, second))
    first_monitor, release_monitor = threading.Event(), threading.Event()
    real_wait = first.wait
    monitors = []

    def held_monitor(timeout=None):
        if timeout is None:
            monitors.append(threading.current_thread())
            first_monitor.set()
            assert release_monitor.wait(3)
        return real_wait(timeout)

    first.wait = held_monitor
    monkeypatch.setattr(
        audio_player.subprocess, "Popen", lambda *a, **kw: next(children)
    )
    old, new = object(), object()
    try:
        assert player.play(path, attempt_token=old)
        assert first_monitor.wait(1)
        assert player.play(path, attempt_token=new)
        assert not player.stop(expected_token=old)
        release_monitor.set()
        monitors[0].join(timeout=2)
        assert not monitors[0].is_alive()
        assert player.get_state(expected_token=new) is PlaybackState.PLAYING
        assert second.signals == []
    finally:
        release_monitor.set()
        player.stop()


def test_plan_changed_while_previous_owner_reaps_never_enters_native_spawn(
    linux, monkeypatch
):
    player, path = linux
    candidate = path.with_name("next.wav")
    candidate.touch()
    plan = player.prepare(candidate)
    entered, changed = threading.Event(), threading.Event()
    old, new = Child(), Child()
    spawned = []

    def spawn(*args, **kwargs):
        child = new if spawned else old
        spawned.append(child)
        return child

    def held_terminate():
        entered.set()
        assert changed.wait(2)
        old.exited.set()

    old.terminate = held_terminate
    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    assert player.play(path)
    try:
        with ThreadPoolExecutor(1) as workers:
            result = workers.submit(player.play_prepared, plan)
            try:
                assert entered.wait(1)
                candidate.write_bytes(b"replaced while waiting")
            finally:
                changed.set()
            assert not result.result(timeout=3)
        assert spawned == [old]
    finally:
        changed.set()
        player.stop()


def test_restart_on_resume_uses_original_prepared_file(linux, monkeypatch):
    player, path = linux
    child = Child()
    launches = []

    def spawn(*args, **kwargs):
        launches.append(child)
        return child

    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    token = object()
    assert player.play(path, attempt_token=token)
    assert player.pause(expected_token=token)
    assert player.get_current_file() == path
    path.write_bytes(b"different file")
    assert not player.resume(expected_token=token)
    assert launches == [child]
    assert player.stop(expected_token=token)


def test_revocation_during_mpv_connection_fences_resume_command(linux, monkeypatch):
    player, path = linux
    child, revoke = Child(), threading.Event()
    messages = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def connect(self, _path):
            if messages:  # pause already committed
                revoke.set()

        def send(self, body):
            messages.append(body)
            return len(body)

    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: sys.executable if name == "mpv" else None,
    )
    monkeypatch.setattr(audio_player.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(socket, "AF_UNIX", 1, raising=False)
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: Connection())
    token = object()
    try:
        assert player.play(
            path, attempt_token=token, start_authority=lambda: not revoke.is_set()
        )
        assert player.pause(expected_token=token)
        assert not player.resume(expected_token=token)
        assert len(messages) == 1
        assert b"true" in messages[0]
    finally:
        player.stop(expected_token=token)


@pytest.mark.parametrize("wmplayer", [False, True])
def test_late_delegation_remains_handed_off_even_when_toast_fails(
    tmp_path, monkeypatch, wmplayer
):
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Windows")
    path = tmp_path / "clip.wav"
    path.touch()
    child, token, revoked = Child(), object(), threading.Event()

    def launch(*args, **kwargs):
        revoked.set()
        return child

    monkeypatch.setattr(
        windows_audio_launch,
        "wmplayer_argv",
        lambda _: [sys.executable, str(path)] if wmplayer else None,
    )
    monkeypatch.setattr(audio_player.os, "startfile", launch, raising=False)
    monkeypatch.setattr(audio_player.subprocess, "Popen", launch)

    def failed_notice(*args):
        raise ValueError("notification failed")

    monkeypatch.setattr(windows_audio_launch, "_tell_running_app", failed_notice)
    player = SimpleAudioPlayer()
    try:
        assert player.play(
            path, attempt_token=token, start_authority=lambda: not revoked.is_set()
        )
        assert revoked.is_set()
        assert player.get_state(expected_token=token) is PlaybackState.HANDED_OFF
        assert not player.stop(expected_token=token)
        assert not player.stop(expected_token=token)
        assert child.signals == []
    finally:
        child.exited.set()


async def test_async_cleanup_keeps_loop_free_while_joining_executor(linux):
    player, _ = linux
    wrapper = AsyncAudioPlayer(player)
    loop = asyncio.get_running_loop()
    acknowledged = threading.Event()
    responsive = []

    class FinishingExecutor(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            if wait:
                loop.call_soon_threadsafe(acknowledged.set)
                responsive.append(acknowledged.wait(0.5))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    wrapper._executor = FinishingExecutor(max_workers=1)
    await wrapper.cleanup()
    assert responsive == [True]


def test_shell_identity_change_during_final_validation_never_opens(
    tmp_path, monkeypatch
):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"prepared")
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_audio_launch, "wmplayer_argv", lambda _: None)
    player = SimpleAudioPlayer()
    plan = player.prepare(path)
    verify = windows_audio_launch.can_hand_off
    opened = []

    def replace_while_validating(file):
        file.write_bytes(b"changed during validation")
        return verify(file)

    monkeypatch.setattr(windows_audio_launch, "can_hand_off", replace_while_validating)
    monkeypatch.setattr(audio_player.os, "startfile", opened.append, raising=False)
    assert plan is not None
    assert not player.play_prepared(plan)
    assert not opened


def test_removed_prepared_executable_preserves_playing_child(
    linux, tmp_path, monkeypatch
):
    player, path = linux
    executable = tmp_path / "aplay"
    executable.touch()
    executable.chmod(0o700)
    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: str(executable) if name == "aplay" else None,
    )
    old = Child()
    launches = []

    def spawn(argv, **kwargs):
        launches.append(argv[0])
        if not executable.exists():
            raise FileNotFoundError
        return old

    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    assert player.play(path)
    try:
        plan = player.prepare(path)
        assert plan is not None
        executable.unlink()
        assert not player.play_prepared(plan)
        assert player.get_state() is PlaybackState.PLAYING
        assert old.signals == []
        assert launches == [str(executable)]
        assert plan.argv[0] == str(executable)
    finally:
        player.stop()


def test_removed_wmplayer_preserves_prior_handoff(tmp_path, monkeypatch):
    path, executable = tmp_path / "clip.wav", tmp_path / "wmplayer.exe"
    path.touch()
    executable.touch()
    executable.chmod(0o700)
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Windows")
    monkeypatch.setattr(windows_audio_launch, "WMPLAYER_PATH", executable)
    player, old, launcher = SimpleAudioPlayer(), object(), Child()
    launched = []

    def spawn(argv, **kwargs):
        launched.append(argv)
        return launcher

    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    try:
        assert player.play(path, attempt_token=old)
        plan = player.prepare(path)
        assert plan is not None
        executable.unlink()
        assert not player.play_prepared(plan)
        assert player.get_state(expected_token=old) is PlaybackState.HANDED_OFF
        assert len(launched) == 1
    finally:
        launcher.exited.set()


def test_failed_owned_spawn_records_exact_error_attempt(linux, monkeypatch):
    player, path = linux
    token = object()

    def failed(*args, **kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(audio_player.subprocess, "Popen", failed)
    assert not player.play(path, attempt_token=token)
    assert player.get_state(expected_token=token) is PlaybackState.ERROR
    assert player.get_state(expected_token=object()) is PlaybackState.IDLE


@pytest.mark.parametrize("cancel_observer", [True, False])
async def test_queued_cleanup_survives_cancellation_and_joins_once(
    linux, monkeypatch, cancel_observer
):
    player, path = linux
    wrapper = AsyncAudioPlayer(player)
    entering, release = threading.Event(), threading.Event()
    cleaned, joined = threading.Event(), threading.Event()
    queued = asyncio.Event()
    child = Child()
    calls = []
    native_cleanup = player.cleanup

    def held_spawn(*args, **kwargs):
        entering.set()
        assert release.wait(4)
        return child

    def cleanup():
        native_cleanup()
        calls.append("cleanup")
        cleaned.set()

    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            if fn is cleanup:
                queued.set()
            return future

        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
            calls.append("joined")
            joined.set()

    monkeypatch.setattr(audio_player.subprocess, "Popen", held_spawn)
    monkeypatch.setattr(player, "cleanup", cleanup)
    wrapper._executor = ObservedExecutor(max_workers=1)
    play = asyncio.create_task(wrapper.play(path))
    try:
        assert await asyncio.to_thread(entering.wait, 2)
        first = asyncio.create_task(wrapper.cleanup())
        await asyncio.wait_for(queued.wait(), 2)
        if cancel_observer:
            first.cancel()
            await asyncio.sleep(0)  # deliver cancellation while play owns the worker
        else:
            second = asyncio.create_task(wrapper.cleanup())
            await asyncio.sleep(0)
        release.set()
        assert await play
        if cancel_observer:
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            await asyncio.gather(first, second)
        assert await asyncio.to_thread(joined.wait, 2)
        assert cleaned.is_set()
        assert calls == ["cleanup", "joined"]
        assert child.exited.is_set()
    finally:
        release.set()
        child.exited.set()
