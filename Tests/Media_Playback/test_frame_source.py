"""AvFrameSource probe/iteration against a real generated clip (task-3401.9)."""

import pytest
from PIL import Image as PILImage

from tldw_chatbook.Media_Playback.frame_source import AvFrameSource

av = pytest.importorskip("av", reason="PyAV optional dependency not installed")


@pytest.fixture
def clip_path(tmp_path):
    """Write a tiny real mp4 (10 frames, 64x48, 10fps) using PyAV itself."""
    path = tmp_path / "fixture.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width, stream.height = 64, 48
    stream.pix_fmt = "yuv420p"
    for index in range(10):
        gray = index * 24 % 256
        image = PILImage.new("RGB", (64, 48), (gray, gray // 2, 255 - gray))
        frame = av.VideoFrame.from_image(image)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def test_probe_reports_shape(clip_path):
    source = AvFrameSource(clip_path)
    try:
        probe = source.probe()
        assert probe.width == 64 and probe.height == 48
        assert probe.duration_seconds is not None
        assert 0.5 <= probe.duration_seconds <= 2.0
    finally:
        source.close()


def test_check_eligible_within_caps(clip_path):
    source = AvFrameSource(clip_path)
    try:
        eligible, reason = source.check_eligible()
        assert eligible and reason == ""
    finally:
        source.close()


def test_iter_frames_yields_throttled_pil_frames(clip_path):
    source = AvFrameSource(clip_path)
    try:
        frames = list(source.iter_frames(target_fps=4.0))
        assert len(frames) >= 2
        timestamp, image = frames[0]
        assert isinstance(timestamp, float)
        assert image.size == (64, 48)
        # Timestamps are non-decreasing.
        stamps = [t for t, _ in frames]
        assert stamps == sorted(stamps)
    finally:
        source.close()


def test_close_is_idempotent(clip_path):
    source = AvFrameSource(clip_path)
    source.probe()
    source.close()
    source.close()  # second close is a no-op, not an error
