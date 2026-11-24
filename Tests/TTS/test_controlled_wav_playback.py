"""Exact retained child startup, completion and physical retirement."""

import asyncio
import os
import queue
import threading
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
    assert args == [
        str(exe),
        "-nodisp",
        "-autoexit",
        "-stats",
        "-sync",
        "audio",
        "-ss",
        "0.050000",
        str(artifact.path),
    ]
    assert child.closed and host.owner is None
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
