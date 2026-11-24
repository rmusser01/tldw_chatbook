"""Exact retained child startup, completion and physical retirement."""

import asyncio
import os
import queue
import threading
from pathlib import Path
from dataclasses import replace

import pytest

from Tests.TTS.test_controlled_wav_profile import retained_files as retained_files
from tldw_chatbook.TTS import controlled_wav_playback as playback
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputOwnership

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.name != "nt", reason="native Windows profile"),
]


class Child:
    def __init__(self):
        self.stderr = self
        self.pipe = queue.Queue()
        self.returncode = None
        self.entered_read = threading.Event()
        self.done = threading.Event()
        self.closed = False
        self._handle = self.Handle()

    class Handle:
        def __init__(self):
            self.closed = False

        def Close(self):
            self.closed = True

    def read(self, size):
        self.entered_read.set()
        return self.pipe.get(timeout=5)

    def close(self):
        self.closed = True

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.done.wait(timeout):
            raise playback.subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def exit(self, code):
        self.returncode = code
        self.pipe.put(b"")
        self.done.set()

    def terminate(self):
        self.exit(1)

    kill = terminate


async def test_exact_eof_launches_no_player_and_closes_both_pins(retained):
    exe, artifact, bookmark = retained
    run = await playback.ControlledWavPlayback.prepare(
        artifact, replace(bookmark, frame=4800), executable=exe
    )
    assert run.already_at_end
    assert (await run.completion).kind.value == "already_at_end"
    assert await run.retire()
    artifact.path.unlink()
    exe.unlink()


@pytest.mark.parametrize("code,expected", [(0, "uncertain"), (1, "failed")])
async def test_early_exit_and_exact_argv(retained, monkeypatch, code, expected):
    exe, artifact, bookmark = retained
    child = Child()
    args = []
    supplied_env = []
    monkeypatch.setenv("FFREPORT", "file=private-player-argv.log")

    def launch(argv, **options):
        args.extend(argv)
        supplied_env.append(options["env"])
        child.exit(code)
        return child

    monkeypatch.setattr(playback, "_spawn", launch)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, replace(bookmark, frame=2400), executable=exe
    )
    host = AudioOutputOwnership()
    await run.start(await host.acquire("article", run.retire))
    result = await asyncio.wait_for(run.completion, 5)
    assert result.kind.value == expected and result.bookmark.frame == 2400
    assert Path(args[0]).samefile(exe) and Path(args[-1]).samefile(artifact.path)
    assert args[0].startswith("\\\\?\\Volume{") and args[-1].startswith(
        "\\\\?\\Volume{"
    )
    assert args[1:-1] == [
        "-nodisp",
        "-autoexit",
        "-stats",
        "-sync",
        "audio",
        "-ss",
        "0.050000",
    ]
    assert child.closed and child._handle.closed and host.owner is None
    assert "FFREPORT" not in supplied_env[0]


async def test_cancel_committed_start_retains_late_child_until_retirement(
    retained, monkeypatch
):
    exe, artifact, bookmark = retained
    entered, return_child = threading.Event(), threading.Event()
    child = Child()

    def launch(*args, **kwargs):
        entered.set()
        assert return_child.wait(5)
        return child

    monkeypatch.setattr(playback, "_spawn", launch)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    lease = await host.acquire("article", run.retire)
    started = asyncio.create_task(run.start(lease))
    assert await asyncio.to_thread(entered.wait, 3)
    started.cancel()
    with pytest.raises(asyncio.CancelledError):
        await started

    async def successor_retire():
        return True

    successor = asyncio.create_task(host.acquire("next", successor_retire))
    await asyncio.sleep(0)
    assert not successor.done()
    return_child.set()
    current = await asyncio.wait_for(successor, 5)
    assert child.done.is_set() and child.closed and current.is_current()
    assert (await run.completion).kind.value == "stopped"


@pytest.mark.parametrize("stop", [False, True])
async def test_terminal_padding_requires_spontaneous_exit_or_reports_stop(
    retained, monkeypatch, stop
):
    exe, artifact, bookmark = retained
    child = Child()
    read_complete = queue.Queue()

    # The reader requests its next block only after accepting the previous one.
    def read(size):
        read_complete.put(None)
        return child.pipe.get(timeout=5)

    child.read = read
    now = [10.0]
    monkeypatch.setattr(playback, "_now", lambda: now[0])
    monkeypatch.setattr(playback, "_spawn", lambda *a, **k: child)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    await run.start(await host.acquire("article", run.retire))
    await asyncio.to_thread(read_complete.get, True, 3)
    for source, stamp in (("0.01", 10.01), ("0.23", 10.23)):
        now[0] = stamp
        child.pipe.put(f"{source} M-A: 0.000 fd= 0 aq= 0KB\r".encode())
        await asyncio.to_thread(read_complete.get, True, 3)
    assert run.position().frame == 4800
    if stop:
        assert await run.retire()
    else:
        child.exit(0)
    result = await asyncio.wait_for(run.completion, 4)
    assert result.kind.value == ("stopped" if stop else "natural")
    assert result.evidence.last_physical_frame == 11040
    assert result.evidence.advanced_span_s == pytest.approx(0.22)
    assert result.bookmark.frame == 4800 and host.owner is None


async def test_guarded_stop_timeout_blocks_replacement_until_late_physical_reap(
    retained, monkeypatch
):
    exe, artifact, bookmark = retained
    child = Child()
    attempted = threading.Event()

    def refuse():
        attempted.set()
        raise OSError("simulated refusal")

    child.terminate = child.kill = refuse
    monkeypatch.setattr(playback, "_spawn", lambda *a, **k: child)
    monkeypatch.setattr(playback, "_STOP_WAIT_S", 0.05)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()

    async def successor_retire():
        return True

    await run.start(await host.acquire("article", run.retire))
    try:
        with pytest.raises(RuntimeError, match="blocked"):
            await host.acquire("next", successor_retire)
        assert attempted.is_set()
        assert host.owner == "article" and host.blocked and not run.completion.done()
    finally:
        child.exit(1)
    await asyncio.wait_for(run.completion, 4)
    assert host.owner is None and not host.blocked
    assert (await host.acquire("next", successor_retire)).is_current()


async def test_cancel_preparation_retires_late_pins(retained, monkeypatch):
    exe, artifact, bookmark = retained
    prepared, allow_return, closed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = playback.prepare_files

    def hold(*args):
        result = original(*args)
        pin = result[-1]
        original_close = pin.close

        def close():
            clean = original_close()
            closed.set()
            return clean

        pin.close = close
        prepared.set()
        assert allow_return.wait(4)
        return result

    monkeypatch.setattr(playback, "prepare_files", hold)
    pending = asyncio.create_task(
        playback.ControlledWavPlayback.prepare(artifact, bookmark, executable=exe)
    )
    assert await asyncio.to_thread(prepared.wait, 3)
    try:
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(OSError):
            artifact.path.unlink()
    finally:
        allow_return.set()
    assert await asyncio.to_thread(closed.wait, 3)
    artifact.path.unlink()


async def test_revoked_unstarted_lease_cannot_spawn(retained, monkeypatch):
    exe, artifact, bookmark = retained
    births = []
    monkeypatch.setattr(playback, "_spawn", lambda *a, **k: births.append(a))
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    lease = await host.acquire("article", run.retire)
    lease.revoke()
    with pytest.raises(playback.ControlledWavError):
        await run.start(lease)
    assert await run.retire()
    assert not births


async def test_startup_telemetry_deadline_reaps_child_without_inventing_position(
    retained, monkeypatch
):
    exe, artifact, bookmark = retained
    now = [50.0]
    child = Child()
    monkeypatch.setattr(playback, "_now", lambda: now[0])
    monkeypatch.setattr(playback, "_spawn", lambda *a, **k: child)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, replace(bookmark, frame=960), executable=exe
    )
    host = AudioOutputOwnership()
    await run.start(await host.acquire("article", run.retire))
    now[0] = 81.0
    result = await asyncio.wait_for(run.completion, 3)
    assert result.kind.value == "uncertain" and result.reason == "progress_unavailable"
    assert result.bookmark.frame == 960
    assert child.done.is_set() and host.owner is None


async def test_current_lease_cannot_launch_another_prepared_run(retained, monkeypatch):
    exe, artifact, bookmark = retained
    child, births = Child(), []

    def launch(*args, **kwargs):
        births.append(True)
        return child

    monkeypatch.setattr(playback, "_spawn", launch)
    first = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    other = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    lease = await host.acquire("article", first.retire)
    try:
        await first.start(lease)
        with pytest.raises(playback.ControlledWavError):
            await other.start(lease)
        assert await other.retire()
        assert births == [True] and lease.is_current() and host.owner == "article"
    finally:
        await first.retire()
        await other.retire()


async def test_held_launch_times_out_logically_but_keeps_physical_custody(
    retained, monkeypatch
):
    exe, artifact, bookmark = retained
    child, entered, proceed = Child(), threading.Event(), threading.Event()

    def launch(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return child

    monkeypatch.setattr(playback, "_spawn", launch)
    monkeypatch.setattr(playback, "STARTUP_S", 0.15, raising=False)
    monkeypatch.setattr(playback, "_STOP_WAIT_S", 0.05)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    lease = await host.acquire("article", run.retire)
    task = asyncio.create_task(run.start(lease))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        result = await asyncio.wait_for(run.completion, 1)
        assert result.kind.value == "uncertain" and result.bookmark.frame == 0
        assert (
            result.evidence.retired_at_s is None
            and lease.revoked
            and host.owner == "article"
        )
        await task
        assert not await run.retire()
    finally:
        proceed.set()
        await task
    monkeypatch.setattr(playback, "_STOP_WAIT_S", 1.5)
    assert await run.retire() and child.done.is_set() and host.owner is None


@pytest.mark.parametrize("known_child", [False, True])
async def test_ambiguous_process_birth_refuses_replacement(
    retained, monkeypatch, known_child
):
    exe, artifact, bookmark = retained
    # Hold the real pin objects for deliberate cleanup of a simulated birth.
    captured = []
    original = playback.prepare_files

    def prepare(*args):
        result = original(*args)
        captured.extend(result[1:])
        return result

    def partial_launch(*args, **kwargs):
        raise OSError("simulated failure following native process birth")

    monkeypatch.setattr(playback, "prepare_files", prepare)
    births = []
    if known_child:

        class PartialChild(Child):
            def __init__(self, *args, **kwargs):
                super().__init__()
                births.append(self)
                partial_launch()

        monkeypatch.setattr(playback.subprocess, "Popen", PartialChild)
    else:
        monkeypatch.setattr(playback, "_spawn", partial_launch)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    try:
        await run.start(await host.acquire("article", run.retire))
        result = await run.completion
        assert result.kind.value == "uncertain" and result.evidence.retired_at_s is None
        if known_child:
            (child,) = births
            assert child.done.is_set() and child._handle.closed and child.closed
        with pytest.raises(RuntimeError, match="blocked"):
            await host.acquire("other", run.retire)
        with pytest.raises(OSError):
            artifact.path.unlink()
    finally:
        for pin in captured:
            pin.close()


async def test_ambiguous_process_handle_close_keeps_custody_after_reap(
    retained, monkeypatch
):
    exe, artifact, bookmark = retained
    child = Child()
    child.exit(0)

    def refuse():
        raise OSError("ambiguous close")

    child._handle.Close = refuse
    monkeypatch.setattr(playback, "_spawn", lambda *a, **kw: child)
    run = await playback.ControlledWavPlayback.prepare(
        artifact, bookmark, executable=exe
    )
    host = AudioOutputOwnership()
    await run.start(await host.acquire("article", run.retire))
    result = await run.completion
    assert (
        result.evidence.reaped_at_s is not None and result.evidence.retired_at_s is None
    )
    assert not await run.retire() and host.owner == "article"
