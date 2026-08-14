"""PlayerPipeline: probe parsing, lifecycle, and isolated run state."""

import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Media_Playback import player_pipeline as pp


_HAS_PROCESS_SUSPEND_SIGNALS = hasattr(pp.signal, "SIGSTOP") and hasattr(
    pp.signal, "SIGCONT"
)
_requires_process_suspend_signals = pytest.mark.skipif(
    not _HAS_PROCESS_SUSPEND_SIGNALS,
    reason="POSIX process suspend/resume signals are unavailable",
)


# -- tools / probe -------------------------------------------------------------


def test_playback_tools_available_present_and_missing(monkeypatch):
    monkeypatch.setattr(pp.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    ok, guidance = pp.playback_tools_available()
    assert ok and guidance == ""
    monkeypatch.setattr(
        pp.shutil,
        "which",
        lambda tool: None if tool == "ffplay" else f"/usr/bin/{tool}",
    )
    ok, guidance = pp.playback_tools_available()
    assert not ok and "ffplay" in guidance and "ffmpeg" in guidance


def test_parse_probe_json_video_and_audio():
    probe = pp.parse_probe_json(
        {
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "6.25"},
        },
        "clip.mp4",
    )
    assert probe.width == 1920 and probe.height == 1080
    assert probe.duration_seconds == 6.25
    assert probe.has_audio is True


def test_parse_probe_json_silent_stream_duration_fallback():
    probe = pp.parse_probe_json(
        {
            "streams": [
                {"codec_type": "video", "width": 64, "height": 48, "duration": "1.0"}
            ]
        },
        "silent.mp4",
    )
    assert probe.has_audio is False
    assert probe.duration_seconds == 1.0


def test_parse_probe_json_requires_video():
    with pytest.raises(RuntimeError, match="no video stream"):
        pp.parse_probe_json({"streams": [{"codec_type": "audio"}]}, "audio.mp3")


# -- fake process plumbing --------------------------------------------------------


class _FakeProc:
    _next_pid = 5000

    def __init__(
        self,
        cmd,
        *,
        fake_stdout=None,
        timeout_once=False,
        final_wait_error=None,
        **kwargs,
    ):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = _FakeProc._next_pid
        _FakeProc._next_pid += 1
        self.stdout = fake_stdout
        self.terminated = False
        self.killed = False
        self.signals: list[int] = []
        self.events: list[object] = []
        self._timeout_once = timeout_once
        self._final_wait_error = final_wait_error

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True
        self.events.append("terminate")

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        if timeout == 2 and self._timeout_once:
            self._timeout_once = False
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        if timeout is None and self._final_wait_error is not None:
            raise self._final_wait_error
        return 0

    def kill(self):
        self.killed = True
        self.events.append("kill")


class _SpawnRecorder:
    def __init__(
        self,
        stdout_factory=None,
        *,
        fail_at=None,
        timeout_at=None,
        final_wait_error_at=None,
    ):
        self.calls: list[_FakeProc] = []
        self._stdout_factory = stdout_factory
        self._fail_at = fail_at
        self._attempts = 0
        self._timeout_at = timeout_at
        self._final_wait_error_at = final_wait_error_at

    def __call__(self, cmd, **kwargs):
        self._attempts += 1
        if self._attempts == self._fail_at:
            raise OSError("spawn failed")
        stdout = None
        if kwargs.get("stdout") == subprocess.PIPE and self._stdout_factory:
            stdout = self._stdout_factory()
        proc = _FakeProc(
            cmd,
            fake_stdout=stdout,
            timeout_once=self._attempts == self._timeout_at,
            final_wait_error=(
                OSError("final wait failed")
                if self._attempts == self._final_wait_error_at
                else None
            ),
            **kwargs,
        )
        self.calls.append(proc)
        return proc


def _probe(width=64, height=48, duration=2.0, has_audio=True):
    return pp.PlayerProbe(
        width=width, height=height, duration_seconds=duration, has_audio=has_audio
    )


def _record_real_pipe_fds(monkeypatch):
    real_pipe = os.pipe
    real_close = os.close
    fds: list[int] = []
    closed: list[int] = []

    def recording_pipe():
        pair = real_pipe()
        fds.extend(pair)
        return pair

    def recording_close(fd):
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(pp.os, "pipe", recording_pipe)
    monkeypatch.setattr(pp.os, "close", recording_close)
    return fds, closed


def _assert_fds_closed(fds):
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_start_builds_single_demux_commands():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()
    assert len(spawn.calls) == 2
    ffmpeg_cmd, ffplay_cmd = spawn.calls[0].cmd, spawn.calls[1].cmd
    # AC6: one ffmpeg demuxes BOTH the video pipe and the audio PCM pipe.
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert "-re" in ffmpeg_cmd
    joined = " ".join(ffmpeg_cmd)
    assert "-map 0:v:0" in joined and "-map 0:a:0" in joined
    assert "pipe:1" in joined and "s16le" in joined
    # ffplay only plays the PCM pipe -- it never opens the source.
    assert ffplay_cmd[0] == "ffplay" and "pipe:0" in ffplay_cmd
    assert "clip.mp4" not in ffplay_cmd
    # The audio fd is inherited by ffmpeg only.
    assert spawn.calls[0].kwargs["pass_fds"]
    pipeline.stop()


def test_start_silent_source_skips_audio_branch():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("silent.mp4", _probe(has_audio=False), spawn=spawn)
    pipeline.start()
    assert len(spawn.calls) == 1  # no ffplay for a silent source
    assert "-map 0:a:0" not in " ".join(spawn.calls[0].cmd)
    pipeline.stop()


def test_silent_start_never_allocates_audio_pipe(monkeypatch):
    monkeypatch.setattr(
        pp.os,
        "pipe",
        lambda: pytest.fail("silent playback must not allocate an audio pipe"),
    )
    pipeline = pp.PlayerPipeline(
        "silent.mp4", _probe(has_audio=False), spawn=_SpawnRecorder()
    )
    pipeline.start()
    pipeline.stop()


def test_first_spawn_failure_closes_every_parent_pipe_fd(monkeypatch):
    fds, closed = _record_real_pipe_fds(monkeypatch)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=_SpawnRecorder(fail_at=1))

    with pytest.raises(OSError, match="spawn failed"):
        pipeline.start()

    assert sorted(closed) == sorted(fds)
    _assert_fds_closed(fds)
    assert pipeline._ffmpeg is None and pipeline._ffplay is None
    assert pipeline.current_run is None


def test_second_spawn_failure_reaps_ffmpeg_and_closes_stdout_and_pipe_fds(
    monkeypatch,
):
    fds, closed = _record_real_pipe_fds(monkeypatch)
    stdout = _FakeStdout(b"", chunk=1)
    spawn = _SpawnRecorder(stdout_factory=lambda: stdout, fail_at=2)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)

    with pytest.raises(OSError, match="spawn failed"):
        pipeline.start()

    ffmpeg = spawn.calls[0]
    assert ffmpeg.events == ["terminate", ("wait", 2)]
    assert stdout.close_calls == 1
    assert sorted(closed) == sorted(fds)
    _assert_fds_closed(fds)
    assert pipeline._ffmpeg is None and pipeline._ffplay is None
    assert pipeline.current_run is None


def test_audio_read_fd_close_failure_rolls_back_private_run_and_children(monkeypatch):
    real_pipe = os.pipe
    real_close = os.close
    fds: list[int] = []
    close_attempts: list[int] = []

    def recording_pipe():
        pair = real_pipe()
        fds.extend(pair)
        return pair

    def close_then_fail_on_audio_read(fd):
        close_attempts.append(fd)
        real_close(fd)
        if fds and fd == fds[0]:
            raise OSError("audio read fd close failed")

    monkeypatch.setattr(pp.os, "pipe", recording_pipe)
    monkeypatch.setattr(pp.os, "close", close_then_fail_on_audio_read)
    stdout = _FakeStdout(b"", chunk=1)
    spawn = _SpawnRecorder(stdout_factory=lambda: stdout)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)

    with pytest.raises(OSError, match="audio read fd close failed"):
        pipeline.start()

    assert sorted(close_attempts) == sorted(fds)
    _assert_fds_closed(fds)
    ffmpeg, ffplay = spawn.calls
    assert ffplay.events == ["terminate", ("wait", 2)]
    assert ffmpeg.events == ["terminate", ("wait", 2)]
    assert stdout.close_calls == 1
    assert pipeline._ffmpeg is None and pipeline._ffplay is None
    assert pipeline.current_run is None


def test_seek_offsets_and_clamps_to_duration():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(duration=2.0), spawn=spawn)
    pipeline.start()
    first_generation = pipeline._generation
    pipeline.seek(0.75)
    assert pipeline._generation == first_generation + 1
    assert "-ss 0.750" in " ".join(spawn.calls[-2].cmd)
    pipeline.seek(99.0)  # clamps to the 2.0s duration
    assert "-ss 2.000" in " ".join(spawn.calls[-2].cmd)
    pipeline.stop()


def test_sync_clock_math(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(pp.time, "monotonic", lambda: now[0])
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=_SpawnRecorder())
    run = pipeline.start()
    run.started_wall = 100.0
    now[0] = 101.0
    # offset 0 + 1.0s elapsed - 0 paused - audio lag
    assert pipeline.sync_clock(run) == pytest.approx(1.0 - pp.AUDIO_BUFFER_LAG_SECONDS)
    run.pause_started = 101.0
    now[0] = 103.0
    pipeline.resume()  # folds 2.0s of pause out of the clock
    assert pipeline.sync_clock(run) == pytest.approx(
        3.0 - 2.0 - pp.AUDIO_BUFFER_LAG_SECONDS
    )


def test_due_behind_and_stats(monkeypatch):
    monkeypatch.setattr(pp.time, "monotonic", lambda: 100.0)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=_SpawnRecorder())
    run = pipeline.start()
    run.started_wall = 100.0
    # clock ≈ -0.15 (lag): a 0.5s pts is not due yet
    assert not pipeline.frame_due(run, 0.5)
    # a -1.0s pts is way behind the (lag-adjusted) clock? clock is -0.15, so
    # -1.0 < -0.15 - 0.08 → behind
    assert pipeline.frames_behind(run, -1.0)
    pipeline.note_rendered(run, 0.5)
    pipeline.note_dropped(run, 0.4)
    assert pipeline.stats.rendered_frames == 1
    assert pipeline.stats.dropped_frames == 1
    assert pipeline.stats.position_seconds == 0.5


@_requires_process_suspend_signals
def test_pause_resume_signals(monkeypatch):
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(pp.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()
    pipeline.pause()
    pipeline.resume()
    assert len(killed) == 4  # SIGSTOP x2 + SIGCONT x2
    pipeline.stop()


def test_pause_resume_without_process_signals_updates_clock_without_kill(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(pp.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(pp, "signal", SimpleNamespace())
    monkeypatch.setattr(
        pp.os,
        "kill",
        lambda *_args: pytest.fail("pause/resume must not signal without support"),
    )
    pipeline = pp.PlayerPipeline(
        "silent.mp4", _probe(has_audio=False), spawn=_SpawnRecorder()
    )
    run = pipeline.start()

    pipeline.pause()
    assert run.pause_started == 100.0
    now[0] = 103.0
    pipeline.resume()

    assert run.pause_started is None
    assert run.paused_total == pytest.approx(3.0)
    pipeline.stop()


class _ObservedRLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.seek_attempted = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "seek-restart":
            self.seek_attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, *_args):
        self._lock.release()


@_requires_process_suspend_signals
@pytest.mark.parametrize(
    ("action", "signal_name"),
    [("pause", "SIGSTOP"), ("resume", "SIGCONT")],
)
def test_pause_and_resume_signal_only_captured_run_during_restart(
    monkeypatch, action, signal_name
):
    clock_entered = threading.Event()
    release_clock = threading.Event()
    killed: list[tuple[int, int]] = []

    def blocked_clock():
        clock_entered.set()
        assert release_clock.wait(timeout=2), "test did not release clock"
        return 100.0

    monkeypatch.setattr(pp.time, "monotonic", blocked_clock)
    monkeypatch.setattr(pp.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("silent.mp4", _probe(has_audio=False), spawn=spawn)
    old_run = pipeline.start()
    if action == "resume":
        old_run.pause_started = 99.0
    observed_lock = _ObservedRLock()
    pipeline._lifecycle_lock = observed_lock
    errors: list[Exception] = []

    def invoke_action():
        try:
            getattr(pipeline, action)()
        except Exception as exc:
            errors.append(exc)

    replacement: list[pp.PlayerRun] = []
    action_thread = threading.Thread(target=invoke_action, daemon=True)
    seek_thread = threading.Thread(
        target=lambda: replacement.append(pipeline.seek(1.0)),
        name="seek-restart",
        daemon=True,
    )
    action_thread.start()
    assert clock_entered.wait(timeout=2), f"{action} did not reach its clock update"
    seek_thread.start()
    assert observed_lock.seek_attempted.wait(timeout=2), "restart did not attempt lock"
    release_clock.set()
    action_thread.join(timeout=2)
    seek_thread.join(timeout=2)

    assert not action_thread.is_alive() and not seek_thread.is_alive()
    assert errors == []
    assert replacement
    expected_signal = getattr(pp.signal, signal_name)
    assert killed == [(spawn.calls[0].pid, expected_signal)]


# -- frame pump -----------------------------------------------------------------


class _FakeStdout:
    def __init__(self, payload: bytes, chunk: int, *, close_error=None):
        self._payload = payload
        self._chunk = chunk
        self._pos = 0
        self.read_calls = 0
        self.close_calls = 0
        self.close_error = close_error

    def read(self, size=-1):
        self.read_calls += 1
        if self._pos >= len(self._payload):
            return b""
        want = self._chunk if size < 0 else min(self._chunk, size)
        end = min(self._pos + want, len(self._payload))
        data = self._payload[self._pos : end]
        self._pos = end
        return data

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _RaisingStdout(_FakeStdout):
    def __init__(self, error):
        super().__init__(b"", chunk=1)
        self.error = error

    def read(self, size=-1):
        self.read_calls += 1
        raise self.error


class _BarrierRaisingStdout(_RaisingStdout):
    def __init__(self, error):
        super().__init__(error)
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def read(self, size=-1):
        self.read_started.set()
        assert self.release_read.wait(timeout=2), "test did not release pipe read"
        return super().read(size)


def test_iter_frames_exact_reads_and_pts_sequence():
    probe = _probe(width=4, height=2)  # 4*2*3 = 24 bytes per frame
    payload = b"\x00" * 24 * 3  # three frames
    spawn = _SpawnRecorder(stdout_factory=lambda: _FakeStdout(payload, chunk=7))
    pipeline = pp.PlayerPipeline("clip.mp4", probe, spawn=spawn)
    run = pipeline.start(offset_seconds=1.0)
    frames = list(pipeline.iter_frames(run))
    assert len(frames) == 3
    fps = pp.PLAYER_TARGET_FPS
    assert [pts for pts, _ in frames] == pytest.approx(
        [1.0, 1.0 + 1 / fps, 1.0 + 2 / fps]
    )
    assert all(len(data) == 24 for _, data in frames)
    assert run.eof


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_current_run_pipe_error_propagates_and_closes_stdout(error_type):
    stdout = _RaisingStdout(error_type("pipe read failed"))
    pipeline = pp.PlayerPipeline(
        "silent.mp4",
        _probe(has_audio=False),
        spawn=_SpawnRecorder(stdout_factory=lambda: stdout),
    )
    run = pipeline.start()

    with pytest.raises(error_type, match="pipe read failed"):
        list(pipeline.iter_frames(run))

    assert stdout.close_calls == 1


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_lifecycle_invalidated_run_treats_pipe_error_as_eof(error_type):
    stdout = _BarrierRaisingStdout(error_type("closed pipe"))
    pipeline = pp.PlayerPipeline(
        "silent.mp4",
        _probe(has_audio=False),
        spawn=_SpawnRecorder(stdout_factory=lambda: stdout),
    )
    run = pipeline.start()
    frames: list[tuple[float, bytes]] = []
    errors: list[Exception] = []

    def pump():
        try:
            frames.extend(pipeline.iter_frames(run))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    assert stdout.read_started.wait(timeout=2), "frame iterator did not start reading"
    pipeline.stop()
    stdout.release_read.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert frames == [] and errors == []
    assert run.eof is True
    assert stdout.close_calls == 1


def test_lazy_old_iterator_never_reads_replacement_stdout():
    frame_bytes = 2 * 1 * 3
    streams = iter(
        [
            _FakeStdout(b"o" * frame_bytes, chunk=frame_bytes),
            _FakeStdout(b"n" * frame_bytes, chunk=frame_bytes),
        ]
    )
    spawn = _SpawnRecorder(stdout_factory=lambda: next(streams))
    pipeline = pp.PlayerPipeline(
        "silent.mp4", _probe(width=2, height=1, has_audio=False), spawn=spawn
    )
    old_run = pipeline.start(offset_seconds=1.0)
    old_stdout = old_run.stdout
    old_iterator = pipeline.iter_frames(old_run)

    replacement = pipeline.seek(2.0)
    replacement_stdout = replacement.stdout
    with pytest.raises(StopIteration):
        next(old_iterator)

    assert old_stdout.read_calls == 0
    assert replacement_stdout.read_calls == 0
    replacement_pts, replacement_data = next(pipeline.iter_frames(replacement))
    assert replacement_pts == pytest.approx(2.0)
    assert replacement_data == b"n" * frame_bytes


class _BarrierStdout(_FakeStdout):
    def __init__(self, payload: bytes):
        super().__init__(payload, chunk=len(payload))
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def read(self, size=-1):
        self.read_started.set()
        assert self.release_read.wait(timeout=2), "test did not release blocked read"
        return super().read(size)


def test_released_old_frame_eof_and_stats_only_mutate_originating_run(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(pp.time, "monotonic", lambda: now[0])
    frame_bytes = 2 * 1 * 3
    old_stdout = _BarrierStdout(b"o" * frame_bytes)
    replacement_stdout = _FakeStdout(b"n" * frame_bytes, chunk=frame_bytes)
    streams = iter([old_stdout, replacement_stdout])
    pipeline = pp.PlayerPipeline(
        "silent.mp4",
        _probe(width=2, height=1, has_audio=False),
        spawn=_SpawnRecorder(stdout_factory=lambda: next(streams)),
    )
    old_run = pipeline.start(offset_seconds=1.0)
    old_frames: list[tuple[float, bytes]] = []
    pump = threading.Thread(
        target=lambda: old_frames.extend(pipeline.iter_frames(old_run)), daemon=True
    )
    pump.start()
    assert old_stdout.read_started.wait(timeout=2), "old run did not begin reading"

    replacement = pipeline.seek(2.0)
    assert replacement.frame_index == 0
    assert replacement.eof is False
    old_stdout.release_read.set()
    pump.join(timeout=2)
    assert not pump.is_alive()

    pipeline.note_rendered(old_run, old_frames[0][0])
    pipeline.note_dropped(old_run, old_frames[0][0])
    assert old_frames[0][0] == pytest.approx(1.0)
    assert old_run.frame_index == 1
    assert old_run.eof is True
    assert old_run.stats.rendered_frames == 1
    assert old_run.stats.dropped_frames == 1
    assert replacement.frame_index == 0
    assert replacement.eof is False
    assert replacement.stats == pp.SyncStats()
    replacement_pts, _ = next(pipeline.iter_frames(replacement))
    assert replacement_pts == pytest.approx(2.0)


def test_sync_and_stat_helpers_use_explicit_originating_run(monkeypatch):
    now = [20.0]
    monkeypatch.setattr(pp.time, "monotonic", lambda: now[0])
    pipeline = pp.PlayerPipeline(
        "silent.mp4", _probe(has_audio=False), spawn=_SpawnRecorder()
    )
    old_run = pipeline.start(offset_seconds=1.0)
    old_run.started_wall = 20.0
    replacement = pipeline.seek(2.0)
    replacement.started_wall = 10.0

    assert pipeline.sync_clock(old_run) == pytest.approx(1.0)
    assert pipeline.frame_due(old_run, 1.0)
    assert not pipeline.frames_behind(old_run, 1.0)
    pipeline.note_rendered(old_run, 1.0)
    pipeline.note_dropped(old_run, 1.0)

    assert old_run.stats.rendered_frames == 1
    assert old_run.stats.dropped_frames == 1
    assert replacement.stats == pp.SyncStats()


def test_natural_eof_then_repeated_stop_closes_stdout_exactly_once():
    frame_bytes = 2 * 1 * 3
    stdout = _FakeStdout(b"x" * frame_bytes, chunk=frame_bytes)
    pipeline = pp.PlayerPipeline(
        "silent.mp4",
        _probe(width=2, height=1, has_audio=False),
        spawn=_SpawnRecorder(stdout_factory=lambda: stdout),
    )
    run = pipeline.start()
    list(pipeline.iter_frames(run))

    pipeline.stop()
    pipeline.stop()

    assert stdout.close_calls == 1


def test_restart_then_active_old_iterator_close_closes_old_stdout_exactly_once():
    frame_bytes = 2 * 1 * 3
    streams = iter(
        [
            _FakeStdout(b"o" * frame_bytes, chunk=frame_bytes),
            _FakeStdout(b"n" * frame_bytes, chunk=frame_bytes),
        ]
    )
    pipeline = pp.PlayerPipeline(
        "silent.mp4",
        _probe(width=2, height=1, has_audio=False),
        spawn=_SpawnRecorder(stdout_factory=lambda: next(streams)),
    )
    old_run = pipeline.start()
    old_stdout = old_run.stdout
    old_iterator = pipeline.iter_frames(old_run)
    next(old_iterator)

    pipeline.seek(1.0)
    old_iterator.close()

    assert old_stdout.close_calls == 1


def test_restart_then_never_advanced_iterator_close_closes_old_stdout_exactly_once():
    frame_bytes = 2 * 1 * 3
    streams = iter(
        [
            _FakeStdout(b"o" * frame_bytes, chunk=frame_bytes),
            _FakeStdout(b"n" * frame_bytes, chunk=frame_bytes),
        ]
    )
    pipeline = pp.PlayerPipeline(
        "silent.mp4",
        _probe(width=2, height=1, has_audio=False),
        spawn=_SpawnRecorder(stdout_factory=lambda: next(streams)),
    )
    old_run = pipeline.start()
    old_stdout = old_run.stdout
    old_iterator = pipeline.iter_frames(old_run)

    pipeline.seek(1.0)
    old_iterator.close()

    assert old_stdout.close_calls == 1


def test_stop_force_kill_waits_for_process_after_terminate_timeout():
    stdout = _FakeStdout(b"", chunk=1)
    spawn = _SpawnRecorder(stdout_factory=lambda: stdout, timeout_at=1)
    pipeline = pp.PlayerPipeline("silent.mp4", _probe(has_audio=False), spawn=spawn)
    pipeline.start()

    pipeline.stop()

    assert spawn.calls[0].events == [
        "terminate",
        ("wait", 2),
        "kill",
        ("wait", None),
    ]
    assert stdout.close_calls == 1


def test_stop_reaps_ffplay_and_ffmpeg_for_audio_run():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()

    pipeline.stop()

    ffmpeg, ffplay = spawn.calls
    assert ffplay.events == ["terminate", ("wait", 2)]
    assert ffmpeg.events == ["terminate", ("wait", 2)]


def test_ffplay_timeout_force_kills_and_finally_reaps_before_ffmpeg():
    spawn = _SpawnRecorder(timeout_at=2)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()

    pipeline.stop()

    ffmpeg, ffplay = spawn.calls
    assert ffplay.events == [
        "terminate",
        ("wait", 2),
        "kill",
        ("wait", None),
    ]
    assert ffmpeg.events == ["terminate", ("wait", 2)]


def test_ffplay_terminal_cleanup_failure_does_not_skip_ffmpeg_or_stdout():
    stdout = _FakeStdout(b"", chunk=1)
    spawn = _SpawnRecorder(
        stdout_factory=lambda: stdout,
        timeout_at=2,
        final_wait_error_at=2,
    )
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()

    with pytest.raises(OSError, match="final wait failed"):
        pipeline.stop()

    ffmpeg, ffplay = spawn.calls
    assert ffplay.events == [
        "terminate",
        ("wait", 2),
        "kill",
        ("wait", None),
    ]
    assert ffmpeg.events == ["terminate", ("wait", 2)]
    assert stdout.close_calls == 1


def test_stdout_is_detached_even_when_native_close_fails():
    stdout = _FakeStdout(b"", chunk=1, close_error=OSError("close failed"))
    run = pp.PlayerRun(generation=1, stdout=stdout, offset_seconds=0.0)

    with pytest.raises(OSError, match="close failed"):
        run.close_stdout_once()

    assert run.stdout is None
    run.close_stdout_once()
    assert stdout.close_calls == 1


def test_stop_is_idempotent():
    stdout = _FakeStdout(b"", chunk=1)
    spawn = _SpawnRecorder(stdout_factory=lambda: stdout)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()
    pipeline.stop()
    pipeline.stop()  # no-op, not an error
    assert stdout.close_calls == 1
    assert spawn.calls[0].events == ["terminate", ("wait", 2)]


# -- streaming (task-3401.11) -----------------------------------------------------


def test_http_source_injects_reconnect_flags():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("https://cdn.example.net/v.mp4", _probe(), spawn=spawn)
    pipeline.start()
    joined = " ".join(spawn.calls[0].cmd)
    assert "-reconnect 1" in joined
    assert "-reconnect_streamed 1" in joined
    assert "-reconnect_delay_max 5" in joined
    assert "-rw_timeout 15000000" in joined
    pipeline.stop()


def test_file_source_has_no_reconnect_flags():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=spawn)
    pipeline.start()
    joined = " ".join(spawn.calls[0].cmd)
    assert "-reconnect" not in joined
    pipeline.stop()
