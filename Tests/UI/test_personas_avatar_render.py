"""Avatar pixels-fallback render-seam test (task-1532).

Characterizes that the non-graphics avatar path produces VISIBLE half-block
cells, not an empty renderable -- widget presence alone proved nothing when
the graphics path silently painted blanks under tmux (renderable != painted).
"""

import io

import pytest

PILImage = pytest.importorskip("PIL.Image")
from rich.console import Console


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def test_avatar_pixels_fallback_renders_visible_half_block_cells():
    """The personas fallback paints visible colored cells for a solid image."""
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    cache = ConsoleImageRenderCache()
    assert cache.prepare("avatar-test", _png_bytes((0, 119, 226)))

    pixels = PersonasScreen._build_avatar_pixels(cache, "avatar-test")

    assert pixels is not None
    console = Console(width=60, record=True, force_terminal=True)
    console.print(pixels)
    # A solid-color image bakes to background-painted cells; the color
    # carries in the styled stream (blank text output would mean nothing
    # painted -- the original task-1532 failure mode).
    styled = console.export_text(styles=True)
    assert "0;119;226" in styled


def _split_png_bytes() -> bytes:
    image = PILImage.new("RGB", (32, 32), (0, 0, 0))
    for x in range(16, 32):
        for y in range(32):
            image.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_avatar_fallback_resolves_left_right_split_within_a_cell():
    """The mosaic fallback doubles horizontal detail: a hard vertical edge
    must produce left/right-half (or quadrant) glyphs, which the old
    half-block renderer could never emit."""
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    cache = ConsoleImageRenderCache()
    assert cache.prepare("avatar-split", _split_png_bytes())

    renderable = PersonasScreen._build_avatar_pixels(cache, "avatar-split")

    assert renderable is not None
    console = Console(width=60, record=True, force_terminal=True)
    console.print(renderable)
    rendered = console.export_text(styles=False)
    assert any(g in rendered for g in ("▌", "▐", "▘", "▝", "▖", "▗"))


def test_console_rail_avatar_fallback_uses_mosaic():
    """The Console rail "Character" box shares the mosaic fallback: a hard
    vertical edge resolves inside single cells (quadrant glyphs)."""
    from tldw_chatbook.UI.Screens.chat_screen import (
        _character_avatar_fallback_renderable,
    )

    # The white region starts at an ODD source column so the hard edge lands
    # inside a cell's 2x2 block rather than exactly on a cell boundary.
    image = PILImage.new("RGB", (32, 32), (0, 0, 0))
    for x in range(15, 32):
        for y in range(32):
            image.putpixel((x, y), (255, 255, 255))

    renderable = _character_avatar_fallback_renderable(image)

    console = Console(width=40, record=True, force_terminal=True)
    console.print(renderable)
    rendered = console.export_text(styles=False)
    assert any(g in rendered for g in ("▌", "▐", "▘", "▝", "▖", "▗"))
