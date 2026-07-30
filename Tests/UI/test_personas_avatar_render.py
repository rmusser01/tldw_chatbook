"""Avatar pixels-fallback render-seam test (task-1532).

Characterizes that the non-graphics avatar path produces VISIBLE half-block
cells, not an empty renderable -- widget presence alone proved nothing when
the graphics path silently painted blanks under tmux (renderable != painted).
"""

import io

from PIL import Image as PILImage
from rich.console import Console


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def test_avatar_pixels_fallback_renders_visible_half_block_cells():
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache
    from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen

    cache = ConsoleImageRenderCache()
    assert cache.prepare("avatar-test", _png_bytes((0, 119, 226)))

    pixels = PersonasScreen._build_avatar_pixels(cache, "avatar-test")

    assert pixels is not None
    console = Console(width=60, record=True, force_terminal=True)
    console.print(pixels)
    rendered = console.export_text(styles=False)
    assert any(glyph in rendered for glyph in ("▄", "▀", "█"))
