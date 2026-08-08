"""PlayerPipeline: probe parsing, command construction, sync math (task-3401.10)."""


import pytest

from tldw_chatbook.Media_Playback import player_pipeline as pp


# -- tools / probe -------------------------------------------------------------


def test_playback_tools_available_present_and_missing(monkeypatch):
    monkeypatch.setattr(pp.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    ok, guidance = pp.playback_tools_available()
    assert ok and guidance == ""
    monkeypatch.setattr(pp.shutil, "which", lambda tool: None if tool == "ffplay" else f"/usr/bin/{tool}")
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
        {"streams": [{"codec_type": "video", "width": 64, "height": 48, "duration": "1.0"}]},
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

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.pid = _FakeProc._next_pid
        _FakeProc._next_pid += 1
        self.stdout = None
        self.terminated = False
        self.killed = False
        self.signals: list[int] = []

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


class _SpawnRecorder:
    def __init__(self, stdout_factory=None):
        self.calls: list[_FakeProc] = []
        self._stdout_factory = stdout_factory

    def __call__(self, cmd, **kwargs):
        proc = _FakeProc(cmd, **kwargs)
        proc.stdout = self._stdout_factory() if self._stdout_factory else None
        self.calls.append(proc)
        return proc


def _probe(width=64, height=48, duration=2.0, has_audio=True):
    return pp.PlayerProbe(width=width, height=height, duration_seconds=duration, has_audio=has_audio)


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
    pipeline.start()
    pipeline._started_wall = 100.0
    now[0] = 101.0
    # offset 0 + 1.0s elapsed - 0 paused - audio lag
    assert pipeline.sync_clock == pytest.approx(1.0 - pp.AUDIO_BUFFER_LAG_SECONDS)
    pipeline._pause_started = 101.0
    now[0] = 103.0
    pipeline.resume()  # folds 2.0s of pause out of the clock
    assert pipeline.sync_clock == pytest.approx(3.0 - 2.0 - pp.AUDIO_BUFFER_LAG_SECONDS)


def test_due_behind_and_stats(monkeypatch):
    monkeypatch.setattr(pp.time, "monotonic", lambda: 100.0)
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=_SpawnRecorder())
    pipeline.start()
    pipeline._started_wall = 100.0
    # clock ≈ -0.15 (lag): a 0.5s pts is not due yet
    assert not pipeline.frame_due(0.5)
    # a -1.0s pts is way behind the (lag-adjusted) clock? clock is -0.15, so
    # -1.0 < -0.15 - 0.08 → behind
    assert pipeline.frames_behind(-1.0)
    pipeline.note_rendered(0.5)
    pipeline.note_dropped(0.4)
    assert pipeline.stats.rendered_frames == 1
    assert pipeline.stats.dropped_frames == 1
    assert pipeline.stats.position_seconds == 0.5


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


# -- frame pump -----------------------------------------------------------------


class _FakeStdout:
    def __init__(self, payload: bytes, chunk: int):
        self._payload = payload
        self._chunk = chunk
        self._pos = 0

    def read(self, size=-1):
        if self._pos >= len(self._payload):
            return b""
        want = self._chunk if size < 0 else min(self._chunk, size)
        end = min(self._pos + want, len(self._payload))
        data = self._payload[self._pos:end]
        self._pos = end
        return data


def test_iter_frames_exact_reads_and_pts_sequence():
    probe = _probe(width=4, height=2)  # 4*2*3 = 24 bytes per frame
    payload = b"\x00" * 24 * 3  # three frames
    spawn = _SpawnRecorder(stdout_factory=lambda: _FakeStdout(payload, chunk=7))
    pipeline = pp.PlayerPipeline("clip.mp4", probe, spawn=spawn)
    pipeline.start(offset_seconds=1.0)
    frames = list(pipeline.iter_frames())
    assert len(frames) == 3
    fps = pp.PLAYER_TARGET_FPS
    assert [pts for pts, _ in frames] == pytest.approx([1.0, 1.0 + 1 / fps, 1.0 + 2 / fps])
    assert all(len(data) == 24 for _, data in frames)
    assert pipeline.at_eof


def test_stop_is_idempotent():
    pipeline = pp.PlayerPipeline("clip.mp4", _probe(), spawn=_SpawnRecorder())
    pipeline.start()
    pipeline.stop()
    pipeline.stop()  # no-op, not an error


# -- streaming (task-3401.11) -----------------------------------------------------


def test_http_source_injects_reconnect_flags():
    spawn = _SpawnRecorder()
    pipeline = pp.PlayerPipeline(
        "https://cdn.example.net/v.mp4", _probe(), spawn=spawn
    )
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
