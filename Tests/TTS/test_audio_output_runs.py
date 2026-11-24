"""App-owned attempts use exact native custody, independent of UI observers."""

import asyncio
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Media_Playback.player_pipeline import PlayerProbe, PlayerRun
from tldw_chatbook.TTS import audio_output_runs as runs
from tldw_chatbook.TTS.audio_output_ownership import (
    AudioOutputOwnership,
    OutputBlocked,
    OutputHandoffRequired,
)
from tldw_chatbook.TTS.audio_player import PlaybackState


class FilePlayer:
    def __init__(self, *, external=False):
        self.external = external
        self.state = PlaybackState.IDLE
        self.token = None
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()
        self.can_stop = True
        self.stops = []
        self.launches = 0
        self.worker_threads = []

    def prepare(self, path):
        self.worker_threads.append(threading.get_ident())
        return SimpleNamespace(file_path=path, external=self.external)

    def play_prepared(self, plan, *, attempt_token, start_authority):
        self.worker_threads.append(threading.get_ident())
        assert start_authority()
        self.token = attempt_token
        self.entered.set()
        assert self.release.wait(4)
        self.launches += 1
        self.state = (
            PlaybackState.HANDED_OFF if self.external else PlaybackState.PLAYING
        )
        return True

    def get_state(self, *, expected_token=None):
        self.worker_threads.append(threading.get_ident())
        return (
            self.state
            if expected_token is None or self.token is expected_token
            else PlaybackState.IDLE
        )

    def stop(self, *, expected_token):
        self.worker_threads.append(threading.get_ident())
        self.stops.append(expected_token)
        if expected_token is not self.token:
            return False
        if not self.can_stop:
            self.state = PlaybackState.BLOCKED
            return False
        self.state = PlaybackState.IDLE
        return not self.external

    def retirement_confirmed(self, *, expected_token):
        return False

    def pause(self, *, expected_token):
        assert expected_token is self.token
        self.state = PlaybackState.PAUSED
        return True

    def resume(self, *, expected_token):
        assert expected_token is self.token
        self.state = PlaybackState.PLAYING
        return True

    def get_position(self):
        return 0.25

    def get_duration(self):
        self.worker_threads.append(threading.get_ident())
        return 2.5


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():  # noqa: ASYNC110 - observe the public state of settling native work
            await asyncio.sleep(0.005)


async def test_same_path_attempts_stop_only_their_own_child(tmp_path):
    host = AudioOutputOwnership()
    a, b = FilePlayer(), FilePlayer()
    path = tmp_path / "same.wav"
    first = await runs.FileOutputRun.prepare(path, player=a)
    second = await runs.FileOutputRun.prepare(path, player=b)
    assert await first.start(await host.acquire("first", first.retire))
    assert await first.pause() and await first.resume()
    assert await first.get_position() == 0.25
    assert await first.get_duration() == 2.5
    lease = await host.acquire("second", second.retire)
    assert await second.start(lease)
    assert await first.stop()
    assert first.retirement_confirmed and host.owner == "second"
    assert a.token is not b.token and a.stops == [a.token] and not b.stops
    assert await second.get_state() is PlaybackState.PLAYING
    assert all(t != threading.get_ident() for t in a.worker_threads + b.worker_threads)
    assert await second.stop()


async def test_default_file_run_uses_private_native_player_and_validates_before_acquire(
    tmp_path, monkeypatch
):
    from Tests.TTS.test_audio_player_retirement import Child
    from tldw_chatbook.TTS import audio_player

    child = Child()
    monkeypatch.setattr(audio_player.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: sys.executable if name == "aplay" else None,
    )
    monkeypatch.setattr(audio_player.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(
        audio_player,
        "get_audio_player",
        lambda: pytest.fail("global file player accessed"),
    )
    host = AudioOutputOwnership()
    path = tmp_path / "private.wav"
    path.touch()
    file = await runs.FileOutputRun.prepare(path)
    assert file.path == path and not file.external
    assert await file.start(await host.acquire("private", file.retire))
    assert await runs.FileOutputRun.prepare(tmp_path / "missing.wav") is None
    assert host.owner == "private" and not child.signals
    assert await file.stop() and child.exited.is_set() and host.owner is None

    stale = await runs.FileOutputRun.prepare(path)
    path.write_bytes(b"changed while awaiting admission")
    assert not await stale.start(await host.acquire("stale", stale.retire))
    assert stale.retirement_confirmed and host.owner is None


async def test_second_file_run_cannot_use_another_runs_live_lease(tmp_path):
    host = AudioOutputOwnership()
    owner = await runs.FileOutputRun.prepare(
        tmp_path / "shared.wav", player=FilePlayer()
    )
    other_player = FilePlayer()
    other = await runs.FileOutputRun.prepare(
        tmp_path / "shared.wav", player=other_player
    )
    lease = await host.acquire("first", owner.retire)
    assert await owner.start(lease)
    try:
        assert not await other.start(lease)
        assert other_player.launches == 0
        assert lease.is_current() and host.owner == "first"
        assert await other.retire() and lease.is_current()
    finally:
        await owner.retire()


async def test_posix_file_exec_refusal_admits_successor(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from tldw_chatbook.TTS import audio_player

    monkeypatch.setattr(audio_player.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audio_player, "os", SimpleNamespace(**(vars(os) | {"name": "posix"}))
    )
    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: sys.executable if name == "aplay" else None,
    )
    popen = Mock(side_effect=FileNotFoundError("interpreter disappeared"))
    monkeypatch.setattr(audio_player.subprocess, "Popen", popen)
    path = tmp_path / "clip.wav"
    path.touch()
    host = AudioOutputOwnership()
    run = await runs.FileOutputRun.prepare(path)
    assert not await run.start(await host.acquire("file", run.retire))
    popen.assert_called_once()
    assert run.retirement_confirmed and host.owner is None and not host.blocked
    assert host.handoff_epoch is None
    assert (await host.acquire("next", run.retire)).is_current()


async def test_eager_stop_before_worker_entry_never_launches(tmp_path):
    loop = asyncio.get_running_loop()
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    lease = await host.acquire("file", run.retire)
    previous = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        start = asyncio.create_task(run.start(lease))
        retired = run.retire()
        assert lease.revoked
        assert not await start and await retired
        assert player.launches == 0 and host.owner is None
    finally:
        loop.set_task_factory(previous)


async def test_unobserved_late_file_start_reaps_and_releases_only_its_lease(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runs, "_RETIRE_TIMEOUT", 0.02)
    host, player = AudioOutputOwnership(), FilePlayer()
    player.release.clear()
    run = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    lease = await host.acquire("file", run.retire)
    start = asyncio.create_task(run.start(lease))
    try:
        assert await asyncio.to_thread(player.entered.wait, 2)
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start
        assert lease.revoked
        assert not await host.stop(lease)
        assert host.blocked and host.owner == "file"
        cancelled = asyncio.ensure_future(run.retire())
        cancelled.cancel()
        player.release.set()
        await wait_until(lambda: run.retirement_confirmed)
        assert host.owner is None and player.stops == [player.token]
    finally:
        player.release.set()


@pytest.mark.parametrize("cancel_waiter", [True, False])
async def test_late_os_handoff_records_epoch_before_pending_in_app_transfer(
    tmp_path, cancel_waiter
):
    host, player = AudioOutputOwnership(), FilePlayer(external=True)
    player.release.clear()
    run = await runs.FileOutputRun.prepare(tmp_path / "external.wav", player=player)
    lease = await host.acquire("external", run.retire, external=run.external)
    start = asyncio.create_task(run.start(lease))
    try:
        assert await asyncio.to_thread(player.entered.wait, 2)
        transfer = asyncio.create_task(host.acquire("inside", run.retire))
        await wait_until(lambda: lease.revoked)
        if cancel_waiter:
            start.cancel()
            with pytest.raises(asyncio.CancelledError):
                await start
        player.release.set()
        if not cancel_waiter:
            assert await start
        with pytest.raises(OutputHandoffRequired):
            await transfer
        assert host.handoff_epoch is not None and host.owner is None
        assert run.retirement_confirmed
    finally:
        player.release.set()


async def test_file_terminal_projection_survives_checked_natural_retirement(tmp_path):
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "done.wav", player=player)
    assert await run.start(await host.acquire("file", run.retire))
    player.state = PlaybackState.FINISHED
    await wait_until(lambda: run.retirement_confirmed)
    assert await run.get_state() is PlaybackState.FINISHED
    assert host.owner is None and player.stops == [player.token]


async def test_file_stop_joins_its_completion_observer(tmp_path):
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "done.wav", player=player)
    prior = asyncio.all_tasks()
    assert await run.start(await host.acquire("file", run.retire))
    assert await run.stop()
    assert not (asyncio.all_tasks() - prior)


@pytest.mark.parametrize("terminal", [PlaybackState.FINISHED, PlaybackState.ERROR])
async def test_public_terminal_before_next_monitor_poll_survives_retire(
    tmp_path, terminal
):
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    parked, release = asyncio.Event(), asyncio.Event()
    read = run.get_state

    async def hold_first_monitor_read():
        state = await read()
        if asyncio.current_task() is run._monitor and not parked.is_set():
            parked.set()
            await release.wait()
        return state

    run.get_state = hold_first_monitor_read
    try:
        assert await run.start(await host.acquire("file", run.retire))
        await asyncio.wait_for(parked.wait(), 2)
        player.state = terminal
        assert await run.get_state() is terminal
        stop = run.retire()
        release.set()
        assert await stop
        assert await run.get_state() is terminal
    finally:
        release.set()
        await run.retire()


@pytest.mark.parametrize("automatic", [False, True])
async def test_local_failed_retirement_blocks_unrelated_acquire(tmp_path, automatic):
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    lease = await host.acquire("file", run.retire)
    assert await run.start(lease)
    player.can_stop = False
    if automatic:
        player.state = PlaybackState.FINISHED
        await wait_until(lambda: run._retirement is not None)
        assert not await run._retirement
    else:
        assert not await run.retire()
    player.can_stop = True
    with pytest.raises(OutputBlocked):
        await host.acquire("unrelated", run.retire)
    assert host.blocked and host.owner == "file" and player.stops == [player.token]
    assert await host.stop(lease)
    assert not host.blocked and host.owner is None and len(player.stops) == 2


async def test_local_retirement_timeout_blocks_until_late_exact_cleanup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runs, "_RETIRE_TIMEOUT", 0.03)
    host, player = AudioOutputOwnership(), FilePlayer()
    entered, release = threading.Event(), threading.Event()
    stop = player.stop

    def delayed_stop(*, expected_token):
        entered.set()
        assert release.wait(4)
        return stop(expected_token=expected_token)

    player.stop = delayed_stop
    run = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    lease = await host.acquire("file", run.retire)
    assert await run.start(lease)
    try:
        retired = run.retire()
        assert await asyncio.to_thread(entered.wait, 2)
        assert not await retired
        assert host.blocked and host.owner == "file"
        with pytest.raises(OutputBlocked):
            await host.acquire("unrelated", run.retire)
    finally:
        release.set()
        assert await run._retirement
    assert player.stops == [player.token] and host.owner is None and not host.blocked


async def test_file_revoked_after_retained_launch_before_observer_resumes(tmp_path):
    loop = asyncio.get_running_loop()
    host, player = AudioOutputOwnership(), FilePlayer()
    player.release.clear()
    run = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    lease = await host.acquire("file", run.retire)
    start = loop.create_task(run.start(lease))
    try:
        assert await asyncio.to_thread(player.entered.wait, 2)
        # Revoke in the completed job's callback batch, before the shielded
        # caller can observe its native result.
        run._pending.add_done_callback(lambda _: lease.revoke())
        player.release.set()
        assert not await start
        assert player.launches == 1 and player.stops == [player.token]
        assert host.owner is None
    finally:
        player.release.set()
        await run.retire()


async def test_opaque_launch_exception_plus_idle_projection_keeps_file_lease(tmp_path):
    host, player = AudioOutputOwnership(), FilePlayer()

    def unknown(*args, **kwargs):
        raise FileNotFoundError("post-dispatch observation file disappeared")

    player.play_prepared = unknown
    run = await runs.FileOutputRun.prepare(tmp_path / "unknown.wav", player=player)
    lease = await host.acquire("file", run.retire)
    assert not await run.start(lease)
    assert not await host.stop(lease) and host.blocked
    assert not run.retirement_confirmed and host.owner == "file"


async def test_ordinary_accepted_owned_file_plus_missing_token_never_proves_retirement(
    tmp_path,
):
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "unknown.wav", player=player)
    lease = await host.acquire("file", run.retire)
    assert await run.start(lease)
    player.token, player.state = object(), PlaybackState.IDLE
    assert not await host.stop(lease)
    assert host.blocked and host.owner == "file"


async def test_negative_file_start_without_positive_receipt_cannot_release_on_idle(
    tmp_path,
):
    host, player = AudioOutputOwnership(), FilePlayer()
    player.play_prepared = lambda *args, **kwargs: False
    run = await runs.FileOutputRun.prepare(tmp_path / "unknown.wav", player=player)
    lease = await host.acquire("file", run.retire)
    assert not await run.start(lease)
    assert not await host.stop(lease) and host.owner == "file" and host.blocked


async def test_compensated_native_resume_releases_correct_owner(tmp_path, monkeypatch):
    from Tests.TTS.test_audio_player_retirement import Child
    from tldw_chatbook.TTS import audio_player

    path = tmp_path / "clip.wav"
    path.touch()
    host = AudioOutputOwnership()
    children = [Child(), Child()]
    spawned = []
    entered, release = threading.Event(), threading.Event()

    def spawn(*args, **kwargs):
        child = children[len(spawned)]
        spawned.append(child)
        if child is children[1]:
            entered.set()
            assert release.wait(3)
        return child

    monkeypatch.setattr(audio_player.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audio_player.shutil,
        "which",
        lambda name: sys.executable if name == "aplay" else None,
    )
    monkeypatch.setattr(audio_player.subprocess, "Popen", spawn)
    run = await runs.FileOutputRun.prepare(path)
    try:
        assert await run.start(await host.acquire("file", run.retire))
        assert await run.pause() and children[0].exited.is_set()
        restart = asyncio.create_task(run.resume())
        assert await asyncio.to_thread(entered.wait, 2)
        stop = run.stop()
        release.set()
        assert not await restart
        assert await stop and children[1].exited.is_set()
        assert host.owner is None and run.retirement_confirmed
    finally:
        release.set()
        for child in children:
            child.exited.set()


async def test_resistant_file_retains_custody_and_later_exact_retry_releases(tmp_path):
    host, player = AudioOutputOwnership(), FilePlayer()
    run = await runs.FileOutputRun.prepare(tmp_path / "resistant.wav", player=player)
    lease = await host.acquire("file", run.retire)
    assert await run.start(lease)
    player.can_stop = False
    assert not await host.stop(lease)
    assert host.blocked and not run.retirement_confirmed
    player.can_stop = True
    assert await host.stop(lease)
    assert host.owner is None and run.retirement_confirmed


class Stream:
    def __init__(self):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def abort(self):
        pass

    def close(self):
        self.closed = True


async def test_pcm_cancellation_during_factory_retires_late_allocation():
    host, stream = AudioOutputOwnership(), Stream()
    entered, release = threading.Event(), threading.Event()

    def build(**kwargs):
        entered.set()
        assert release.wait(4)
        return stream

    pcm = runs.PcmOutputRun(24000, 1, on_event=lambda _: None, stream_factory=build)
    lease = await host.acquire("pcm", pcm.retire)
    start = asyncio.create_task(pcm.start(lease))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        stop = pcm.retire()
        assert lease.revoked
        release.set()
        assert not await start
        assert await stop and stream.closed and not stream.started
        assert pcm.retirement_confirmed and host.owner is None
    finally:
        release.set()


async def test_pcm_graceful_drain_releases_only_after_native_finished_and_close():
    from Tests.Audio.test_streaming_sink import FakeStream

    host, streams = AudioOutputOwnership(), []
    events = []

    def build(**kwargs):
        stream = FakeStream(**kwargs)
        stream.auto_finish = False
        streams.append(stream)
        return stream

    pcm = runs.PcmOutputRun(24000, on_event=events.append, stream_factory=build)
    assert await pcm.start(await host.acquire("pcm", pcm.retire))
    assert pcm.sink.feed(bytes(960))
    pcm.sink.close()
    await asyncio.to_thread(streams[0].tick, 2)
    assert host.owner == "pcm" and not pcm.retirement_confirmed
    await asyncio.to_thread(streams[0].finish_output)
    await wait_until(lambda: pcm.retirement_confirmed)
    assert streams[0].closed and host.owner is None
    assert any(isinstance(event, runs.SinkDrained) for event in events)


async def test_pcm_retains_uncertain_close_until_checked_retry():
    host = AudioOutputOwnership()

    class Resistant(Stream):
        can_close = False

        def close(self):
            if not self.can_close:
                raise OSError("still open")
            super().close()

    stream = Resistant()
    pcm = runs.PcmOutputRun(
        24000, on_event=lambda _: None, stream_factory=lambda **_: stream
    )
    lease = await host.acquire("pcm", pcm.retire)
    assert await pcm.start(lease) and stream.started
    assert not await host.stop(lease) and host.blocked and not stream.closed
    stream.can_close = True
    assert await host.stop(lease) and pcm.retirement_confirmed and stream.closed


class Pipeline:
    def __init__(self):
        self.current = None
        self.seek_entered = threading.Event()
        self.allow_seek = threading.Event()
        self.expected = []
        self.partial = False
        self.can_stop = True
        self.controls = []

    def pause(self):
        self.controls.append(("pause", threading.get_ident()))

    def resume(self, *, start_authority):
        assert start_authority()
        self.controls.append(("resume", threading.get_ident()))

    def start(self, *, offset_seconds, start_authority):
        assert start_authority()
        self.current = PlayerRun(1, None, offset_seconds)
        if self.partial:
            raise OSError("partial spawn")
        return self.current

    def seek(self, offset, *, expected_run, start_authority):
        assert expected_run is self.current and start_authority()
        self.seek_entered.set()
        assert self.allow_seek.wait(4)
        self.current = PlayerRun(2, None, offset)
        return self.current

    def stop(self, *, expected_run=None):
        self.expected.append(expected_run)
        if not self.can_stop:
            raise OSError("still running")
        if expected_run is not None and expected_run is not self.current:
            return False
        self.current = None
        return True


def video(pipeline, *, audible):
    probe = PlayerProbe(16, 16, 2.0, audible)

    def build(source, actual_probe):
        assert actual_probe is probe and source == "clip.mp4"
        return pipeline

    return runs.MediaOutputRun("clip.mp4", probe, pipeline_factory=build)


async def test_silent_video_does_not_displace_audio_and_windows_refuses_before_acquire(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runs, "audio_pipe_supported", lambda: False)
    host, player = AudioOutputOwnership(), FilePlayer()
    file = await runs.FileOutputRun.prepare(tmp_path / "clip.wav", player=player)
    assert await file.start(await host.acquire("file", file.retire))
    with pytest.raises(RuntimeError, match="audio"):
        video(Pipeline(), audible=True)
    silent = video(Pipeline(), audible=False)
    assert not silent.audible and (await silent.start(None)).generation == 1
    assert await silent.stop() and host.owner == "file" and not player.stops
    assert await file.stop()


async def test_revoked_media_seek_joins_late_pair_without_publishing_it(monkeypatch):
    monkeypatch.setattr(runs, "audio_pipe_supported", lambda: True)
    host, pipeline = AudioOutputOwnership(), Pipeline()
    media = video(pipeline, audible=True)
    lease = await host.acquire("media", media.retire)
    old = await media.start(lease)
    seek = asyncio.create_task(media.seek(1.0))
    try:
        assert await asyncio.to_thread(pipeline.seek_entered.wait, 2)
        stop = media.retire()
        another_stop = media.retire()
        assert lease.revoked and media.current_run is old
        pipeline.allow_seek.set()
        with pytest.raises(RuntimeError):
            await seek
        assert await stop and await another_stop and media.current_run is None
        assert len(pipeline.expected) == 1 and pipeline.expected[0].generation == 2
        assert host.owner is None
    finally:
        pipeline.allow_seek.set()


async def test_private_media_transport_offloop_and_fenced_after_stop():
    pipeline = Pipeline()
    media = video(pipeline, audible=False)
    await media.start(None)
    assert await media.pause() and await media.resume()
    stop = media.stop()
    assert not await media.resume()
    assert await stop
    assert [action for action, _ in pipeline.controls] == ["pause", "resume"]
    assert all(worker != threading.get_ident() for _, worker in pipeline.controls)


async def test_failed_media_partial_start_retries_private_pipeline(monkeypatch):
    monkeypatch.setattr(runs, "audio_pipe_supported", lambda: True)
    host, pipeline = AudioOutputOwnership(), Pipeline()
    pipeline.partial = True
    pipeline.can_stop = False
    media = video(pipeline, audible=True)
    lease = await host.acquire("media", media.retire)
    with pytest.raises(OSError):
        await media.start(lease)
    assert not await host.stop(lease) and host.owner == "media"
    assert not media.retirement_confirmed and media.current_run is None
    pipeline.can_stop = True
    assert await host.stop(lease) and host.owner is None
    assert pipeline.current is None and pipeline.expected[-1] is None


@pytest.mark.parametrize(
    "leg,hold_demux", [("ffmpeg", False), ("ffplay", False), ("ffplay", True)]
)
async def test_posix_media_exec_refusal_admits_successor(monkeypatch, leg, hold_demux):
    from tldw_chatbook.Media_Playback import player_pipeline

    attempts, reaped = [], []

    def initialize(child, cmd, **kwargs):
        attempts.append(cmd[0])
        if cmd[0] == leg:
            raise FileNotFoundError("interpreter disappeared")
        child.stdout = None
        child.terminate = lambda: None
        child.kill = lambda: None

        def wait(timeout):
            if hold_demux:
                raise OSError("demux reap unconfirmed")
            reaped.append(cmd[0])
            return 0

        child.wait = wait

    monkeypatch.setattr(player_pipeline.subprocess.Popen, "__init__", initialize)
    monkeypatch.setattr(
        player_pipeline,
        "os",
        SimpleNamespace(name="posix", pipe=os.pipe, close=os.close),
    )
    monkeypatch.setattr(runs, "audio_pipe_supported", lambda: True)
    host = AudioOutputOwnership()
    media = runs.MediaOutputRun("clip.mp4", PlayerProbe(16, 16, 2.0, True))
    lease = await host.acquire("media", media.retire)
    with pytest.raises(
        OSError,
        match="demux reap unconfirmed" if hold_demux else "interpreter disappeared",
    ):
        await media.start(lease)
    if hold_demux:
        assert not await media.retire() and host.blocked and host.owner == "media"
        assert not reaped
        with pytest.raises(OutputBlocked):
            await host.acquire("next", media.retire)
        hold_demux = False
        assert await host.stop(lease)
    assert await media.retire()
    assert attempts == (["ffmpeg"] if leg == "ffmpeg" else ["ffmpeg", "ffplay"])
    assert reaped == ([] if leg == "ffmpeg" else ["ffmpeg"])
    assert host.owner is None and not host.blocked and host.handoff_epoch is None
    assert (await host.acquire("next", media.retire)).is_current()


@pytest.mark.parametrize("leg", ["ffmpeg", "ffplay"])
async def test_opaque_audio_process_birth_keeps_the_app_lease(monkeypatch, leg):
    from Tests.Media_Playback.test_player_pipeline import _SpawnRecorder
    from tldw_chatbook.Media_Playback import player_pipeline

    fake = _SpawnRecorder()

    def spawn(cmd, **kwargs):
        child = fake(cmd, **kwargs)
        if cmd[0] == leg:
            raise FileNotFoundError("after audio child birth")
        return child

    monkeypatch.setattr(runs, "audio_pipe_supported", lambda: True)
    monkeypatch.setattr(player_pipeline, "audio_pipe_supported", lambda: True)
    monkeypatch.setattr(
        player_pipeline,
        "os",
        SimpleNamespace(name="posix", pipe=os.pipe, close=os.close),
    )
    host = AudioOutputOwnership()
    media = runs.MediaOutputRun(
        "clip.mp4",
        PlayerProbe(16, 16, 2.0, True),
        pipeline_factory=lambda source, probe: player_pipeline.PlayerPipeline(
            source, probe, spawn=spawn
        ),
    )
    lease = await host.acquire("media", media.retire)
    with pytest.raises(OSError, match="after audio child birth"):
        await media.start(lease)
    assert not await host.stop(lease) and host.blocked and host.owner == "media"
    if leg == "ffplay":
        assert fake.calls[0].terminated
    assert not fake.calls[-1].terminated


@pytest.mark.parametrize("held_leg", ["ffplay", "ffmpeg"])
async def test_native_media_resume_rechecks_revocation_after_child_poll(
    monkeypatch, held_leg
):
    from Tests.Media_Playback.test_player_pipeline import _SpawnRecorder
    from tldw_chatbook.Media_Playback import player_pipeline

    entered, release = threading.Event(), threading.Event()
    children, signals = _SpawnRecorder(), []
    monkeypatch.setattr(runs, "audio_pipe_supported", lambda: True)
    monkeypatch.setattr(player_pipeline, "audio_pipe_supported", lambda: True)
    monkeypatch.setattr(player_pipeline.signal, "SIGCONT", 18, raising=False)
    monkeypatch.setattr(
        player_pipeline.os, "kill", lambda pid, signal: signals.append((pid, signal))
    )
    host = AudioOutputOwnership()
    media = runs.MediaOutputRun(
        "clip.mp4",
        PlayerProbe(16, 16, 2.0, True),
        pipeline_factory=lambda source, probe: player_pipeline.PlayerPipeline(
            source, probe, spawn=children
        ),
    )
    lease = await host.acquire("media", media.retire)
    await media.start(lease)
    assert await media.pause()
    signals.clear()
    demux, playback = children.calls

    def held_poll():
        entered.set()
        assert release.wait(4)
        return None

    (playback if held_leg == "ffplay" else demux).poll = held_poll
    try:
        resume = asyncio.create_task(media.resume())
        assert await asyncio.to_thread(entered.wait, 2)
        stop = media.retire()
        release.set()
        assert not await resume and await stop
        assert signals == ([] if held_leg == "ffplay" else [(playback.pid, 18)])
        assert demux.terminated and playback.terminated
    finally:
        release.set()
        await media.retire()


async def test_positive_retirement_observers_wait_for_exact_proof_even_after_timeout(
    monkeypatch,
):
    class BlockedRun(runs.PreparedOutputRun):
        def __init__(self):
            super().__init__()
            self.closed = asyncio.Event()
            self.confirm = False

        async def _close_native(self):
            await self.closed.wait()
            return self.confirm

    run = BlockedRun()
    proofs = []
    run.on_retired(lambda: proofs.append("original"))
    run.on_retired(lambda: 1 / 0)
    monkeypatch.setattr(runs, "_RETIRE_TIMEOUT", 0.01)
    assert not await run.retire()
    assert proofs == []
    monkeypatch.setattr(runs, "_RETIRE_TIMEOUT", 2)
    run.closed.set()
    assert not await run.retire()
    assert proofs == []
    run.confirm = True
    assert await run.retire()
    run.on_retired(lambda: proofs.append("immediate"))
    assert await run.retire()
    assert proofs == ["original", "immediate"]
