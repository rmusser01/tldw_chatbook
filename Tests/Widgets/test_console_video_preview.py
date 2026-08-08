"""ConsoleVideoPreview state machine + ConsoleVideoCard preview mount (task-3401.9)."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCard,
    ConsoleVideoCardSpec,
)
from tldw_chatbook.Widgets.Console.console_video_preview import (
    ConsoleVideoPreview,
    progress_line,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


# -- helpers ------------------------------------------------------------------


class _FakeTimer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeSource:
    """Stands in for AvFrameSource (no av, no file I/O)."""

    instances: list = []

    def __init__(self, path):
        self.path = path
        self.closed = False
        _FakeSource.instances.append(self)

    def check_eligible(self):
        return True, ""

    def iter_frames(self, *, target_fps=12.0):
        return iter(())

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate_registry():
    ConsoleVideoPreview._active = None
    _FakeSource.instances = []
    yield
    ConsoleVideoPreview._active = None


@pytest.fixture
def preview(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.av_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.AvFrameSource",
        _FakeSource,
    )
    widget = ConsoleVideoPreview("/tmp/fixture.mp4", duration_seconds=6.0)
    monkeypatch.setattr(widget, "run_worker", lambda *a, **k: None)
    monkeypatch.setattr(widget, "set_interval", lambda *a, **k: _FakeTimer())
    return widget


# -- progress line (pure) -------------------------------------------------------


def test_progress_line_bar_and_clock():
    line = progress_line(3.0, 6.0, width=10)
    assert line.startswith("▓▓▓▓▓░░░░░")
    assert "0:03 / 0:06" in line


def test_progress_line_unknown_duration():
    assert progress_line(3.0, None) == "0:03 / --:--"


# -- state machine ----------------------------------------------------------------


def test_default_state_is_poster(preview):
    # AC6: a freshly-mounted preview is paused -- no decode, no worker, no source.
    assert preview.state == "poster"
    assert preview._source is None
    assert "click to play" in preview._poster_text()


def test_play_then_pause_transitions(preview):
    preview.play()
    assert preview.state == "playing"
    assert len(_FakeSource.instances) == 1
    preview.pause()
    assert preview.state == "paused"
    assert _FakeSource.instances[0].closed
    assert "paused" in preview._poster_text()


def test_one_active_preview_rule(preview, monkeypatch):
    second = ConsoleVideoPreview("/tmp/other.mp4", duration_seconds=4.0)
    monkeypatch.setattr(second, "run_worker", lambda *a, **k: None)
    monkeypatch.setattr(second, "set_interval", lambda *a, **k: _FakeTimer())

    preview.play()
    assert ConsoleVideoPreview._active is preview
    second.play()
    # AC2: starting a second preview pauses the first.
    assert preview.state == "paused"
    assert second.state == "playing"
    assert ConsoleVideoPreview._active is second


def test_ineligible_preview_never_starts(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.av_available",
        lambda: True,
    )
    widget = ConsoleVideoPreview(
        "/tmp/long.mp4", duration_seconds=90.0, eligible=False,
        ineligible_reason="clip is 90s -- previews cap at 30s",
    )
    widget.play()
    assert widget.state == "poster"
    assert "previews cap" in widget._poster_text()


def test_missing_av_shows_install_guidance(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.av_available",
        lambda: False,
    )
    widget = ConsoleVideoPreview("/tmp/x.mp4", duration_seconds=5.0)
    widget.play()  # refused: not eligible
    assert widget.state == "poster"
    assert "video_playback" in widget._poster_text()
    assert ConsoleVideoPreview._active is None


# -- off-screen pause -----------------------------------------------------------


class _OffscreenPreview(ConsoleVideoPreview):
    def __init__(self, region, screen, **kwargs):
        super().__init__(**kwargs)
        self._fake_region = region
        self._fake_screen = screen

    @property
    def region(self):
        return self._fake_region

    @property
    def screen(self):
        return self._fake_screen


def _offscreen_widget(monkeypatch, bottom, top, height=24):
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.av_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.Console.console_video_preview.AvFrameSource",
        _FakeSource,
    )
    region = SimpleNamespace(bottom=bottom, top=top)
    screen = SimpleNamespace(size=SimpleNamespace(height=height))
    widget = _OffscreenPreview(region, screen, file_path="/tmp/x.mp4", duration_seconds=5.0)
    monkeypatch.setattr(widget, "run_worker", lambda *a, **k: None)
    monkeypatch.setattr(widget, "set_interval", lambda *a, **k: _FakeTimer())
    return widget


def test_offscreen_scroll_pauses(monkeypatch):
    widget = _offscreen_widget(monkeypatch, bottom=-10, top=-60)
    widget.play()
    assert widget.state == "playing"
    widget._pause_if_offscreen()
    assert widget.state == "paused"  # AC2: scrolled off the viewport


def test_onscreen_does_not_pause(monkeypatch):
    widget = _offscreen_widget(monkeypatch, bottom=10, top=0)
    widget.play()
    widget._pause_if_offscreen()
    assert widget.state == "playing"


# -- card mounts preview only when ready ----------------------------------------


def _card_spec(status="ready", **meta_overrides):
    meta_kwargs = {"name": "clip", "prompt": "p", "backend": "minimax", "duration_seconds": 6.0}
    meta_kwargs.update(meta_overrides)
    return ConsoleVideoCardSpec(
        message_id="m1",
        meta=VideoGenerationMetadata(**meta_kwargs),
        status=status,
        file_path="/tmp/clip.mp4" if status == "ready" else None,
    )


def test_ready_card_composes_preview():
    card = ConsoleVideoCard(_card_spec())
    children = list(card.compose())
    assert isinstance(children[0], ConsoleVideoPreview)
    assert children[0]._eligible


def test_expired_card_has_no_preview():
    card = ConsoleVideoCard(_card_spec(status="expired"))
    children = list(card.compose())
    assert not any(isinstance(child, ConsoleVideoPreview) for child in children)


def test_oversized_clip_preview_composes_ineligible():
    card = ConsoleVideoCard(_card_spec(duration_seconds=90.0))
    children = list(card.compose())
    preview = children[0]
    assert isinstance(preview, ConsoleVideoPreview)
    assert not preview._eligible
    assert "previews cap" in preview._ineligible_reason
