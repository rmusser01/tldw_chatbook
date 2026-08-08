"""VideoPlayerScreen bindings, hints, and compose branches (task-3401.10)."""

from textual.widgets import Static

from tldw_chatbook.UI.Screens.video_player_screen import (
    _HINTS,
    VideoPlayerScreen,
)


def test_bindings_follow_keybinding_conventions():
    forbidden_chords = {"ctrl+c", "ctrl+v", "ctrl+x", "ctrl+s", "ctrl+d", "ctrl+z", "ctrl+a", "ctrl+r", "ctrl+w"}
    keys = {binding.key for binding in VideoPlayerScreen.BINDINGS}
    assert keys == {"space", "s", "left", "right", "q"}
    assert not (keys & forbidden_chords)
    # decision 031: single-letter htop-style for letters.
    assert all(len(key) == 1 or key in {"space", "left", "right"} for key in keys)


def test_hints_line_names_only_implemented_actions():
    # AC2: the footer hints may only advertise implemented actions -- every
    # hinted key has a matching action method on the screen.
    hinted = {"space": "action_toggle_pause", "s": "action_stop_playback", "q": "action_close_player"}
    for key, method in hinted.items():
        assert key in _HINTS
        assert callable(getattr(VideoPlayerScreen, method))
    assert "←/→" in _HINTS
    assert callable(getattr(VideoPlayerScreen, "action_seek_back"))
    assert callable(getattr(VideoPlayerScreen, "action_seek_fwd"))


def test_compose_branches_per_mode():
    # halfcell / ascii / kitty: plain Static frame area.
    for mode in ("halfcell", "ascii", "kitty"):
        screen = VideoPlayerScreen("/tmp/x.mp4", render_mode=mode)
        children = list(screen.compose())
        assert isinstance(children[0], Static)
        assert len(children) == 3  # frame + status + hints
    # sixel: tries the sixel widget, degrades to Static when unavailable.
    screen = VideoPlayerScreen("/tmp/x.mp4", render_mode="sixel")
    children = list(screen.compose())
    assert len(children) == 3


def test_seek_relative_clamps_and_delegates():
    class _FakePipeline:
        def __init__(self):
            self.stats = type("S", (), {"position_seconds": 3.0})()
            self.seeked: list[float] = []

        def seek(self, target):
            self.seeked.append(target)

    screen = VideoPlayerScreen("/tmp/x.mp4")
    pipeline = _FakePipeline()
    screen._pipeline = pipeline
    screen._seek_relative(-5.0)
    screen._seek_relative(5.0)
    assert pipeline.seeked == [0.0, 8.0]  # clamps at 0, adds forward

    screen._finished = True
    screen._seek_relative(5.0)
    assert pipeline.seeked == [0.0, 8.0]  # no seek after EOF
