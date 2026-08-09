"""Real-ffmpeg pipeline integration (task-3401.10).

Builds tiny clips with ffmpeg itself (video-only and video+audio), then runs
the real demux pair headless (SDL_AUDIODRIVER=dummy so ffplay needs no
audio hardware). Marked integration: skipped when ffmpeg/ffplay are absent.
"""

import subprocess  # nosec B404 # test fixtures invoke probed system binaries

import pytest

from tldw_chatbook.Media_Playback.player_pipeline import (
    PlayerPipeline,
    playback_tools_available,
    probe_file,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not playback_tools_available()[0], reason="ffmpeg/ffplay not installed"
    ),
]


def _ffmpeg(*args: str) -> None:
    result = subprocess.run(  # nosec B603 # fixed argv fixture generation
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")[:400]


@pytest.fixture
def video_only_clip(tmp_path):
    path = tmp_path / "video_only.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=64x48:rate=10",
        "-c:v",
        "mpeg4",
        str(path),
    )
    return path


@pytest.fixture
def audio_video_clip(tmp_path):
    path = tmp_path / "audio_video.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=64x48:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    )
    return path


def test_probe_real_clip(video_only_clip):
    probe = probe_file(video_only_clip)
    assert probe.width == 64 and probe.height == 48
    assert probe.has_audio is False
    assert probe.duration_seconds == pytest.approx(1.0, abs=0.2)


def test_frames_flow_from_real_pipeline(video_only_clip):
    probe = probe_file(video_only_clip)
    pipeline = PlayerPipeline(str(video_only_clip), probe, target_fps=10.0)
    run = pipeline.start()
    frames = []
    for index, (pts, data) in enumerate(pipeline.iter_frames(run)):
        frames.append(pts)
        assert len(data) == 64 * 48 * 3
        if index >= 2:
            break
    pipeline.stop()
    assert frames == pytest.approx([0.0, 0.1, 0.2], abs=1e-6)


def test_seek_restarts_at_offset(video_only_clip):
    probe = probe_file(video_only_clip)
    pipeline = PlayerPipeline(str(video_only_clip), probe, target_fps=10.0)
    pipeline.start()
    run = pipeline.seek(0.5)
    pts, _data = next(pipeline.iter_frames(run))
    pipeline.stop()
    assert pts == pytest.approx(0.5, abs=1e-6)


def test_audio_branch_runs_headless(audio_video_clip, monkeypatch):
    # ffplay plays through SDL; the dummy driver needs no audio hardware.
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    probe = probe_file(audio_video_clip)
    assert probe.has_audio is True
    pipeline = PlayerPipeline(str(audio_video_clip), probe, target_fps=10.0)
    run = pipeline.start()
    assert pipeline._ffplay is not None
    ffmpeg = pipeline._ffmpeg
    ffplay = pipeline._ffplay
    assert ffmpeg is not None and ffplay is not None
    pts, data = next(pipeline.iter_frames(run))
    assert pts == pytest.approx(0.0, abs=1e-6)
    assert len(data) == 64 * 48 * 3
    pipeline.stop()
    # Both children are really gone.
    assert pipeline._ffmpeg is None and pipeline._ffplay is None
    assert ffmpeg.poll() is not None and ffplay.poll() is not None
