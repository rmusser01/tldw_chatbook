"""Render-mode detection + ASCII fallback renderer (task-3401.10)."""

import pytest

from tldw_chatbook.Media_Playback.render_mode import (
    ASCII_RAMP,
    detect_render_mode,
    frame_to_ascii,
)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"TERM": "dumb"}, "ascii"),
        ({"TERM": "linux"}, "ascii"),
        ({}, "ascii"),
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"KITTY_WINDOW_ID": "1", "TERM": "xterm-256color"}, "kitty"),
        ({"TERM_PROGRAM": "kitty", "TERM": "xterm-256color"}, "kitty"),
        ({"TERM_PROGRAM": "WezTerm", "TERM": "xterm-256color"}, "kitty"),
        ({"TERM_PROGRAM": "ghostty", "TERM": "xterm-256color"}, "kitty"),
        ({"TERM": "foot"}, "sixel"),
        ({"TERM": "mlterm"}, "sixel"),
        ({"TERM_PROGRAM": "contour", "TERM": "xterm"}, "sixel"),
        ({"TERM": "xterm-256color"}, "halfcell"),
        ({"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}, "halfcell"),
        ({"TERM": "xterm"}, "halfcell"),  # plain xterm: sixel usually NOT compiled in
        ({"TERM": "tmux-256color"}, "halfcell"),
    ],
)
def test_detect_render_mode(env, expected):
    assert detect_render_mode(env) == expected


def test_kitty_wins_over_sixel():
    env = {"TERM": "xterm-kitty", "TERM_PROGRAM": "contour"}
    assert detect_render_mode(env) == "kitty"


def _gradient_image(width=8, height=4):
    from PIL import Image as PILImage

    image = PILImage.new("L", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = int(255 * x / max(1, width - 1))
    return image


def test_frame_to_ascii_shape_and_ramp():
    text = frame_to_ascii(_gradient_image(), cols=8)
    lines = text.split("\n")
    assert len(lines) == 2  # 8x4 aspect with 2:1 cell correction at 8 cols
    assert all(len(line) == 8 for line in lines)
    assert set(text) <= set(ASCII_RAMP + "\n")


def test_frame_to_ascii_extremes():
    from PIL import Image as PILImage

    black = frame_to_ascii(PILImage.new("L", (4, 4), 0), cols=4)
    assert set(black) == {" ", "\n"}
    white = frame_to_ascii(PILImage.new("L", (4, 4), 255), cols=4)
    assert set(white) == {ASCII_RAMP[-1], "\n"}


def test_frame_to_ascii_guards():
    assert frame_to_ascii(_gradient_image(), cols=0)  # clamps to 1 col, no crash
