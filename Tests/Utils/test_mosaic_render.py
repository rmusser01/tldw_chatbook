"""Quadrant-mosaic renderer tests (avatar/inline fallback sharpening).

Half-block rendering samples one pixel per column (1x2 subpixels per cell);
the quadrant mosaic samples 2x2 subpixels per cell using universal Block
Elements glyphs, doubling horizontal detail with no font risk. These tests
pin the glyph vocabulary, the resolution gain a half-block renderer cannot
express (a left/right split inside ONE cell), and box-fit dimensions.
"""

from PIL import Image as PILImage
from rich.console import Console

from tldw_chatbook.Utils.mosaic_render import QUADRANT_GLYPHS, mosaic_from_image


def _render_text(renderable, width: int = 80) -> str:
    console = Console(width=width, record=True, force_terminal=True)
    console.print(renderable)
    return console.export_text(styles=False)


def test_left_right_split_resolves_inside_one_cell():
    image = PILImage.new("RGB", (2, 2))
    image.putpixel((0, 0), (0, 0, 0))
    image.putpixel((0, 1), (0, 0, 0))
    image.putpixel((1, 0), (255, 255, 255))
    image.putpixel((1, 1), (255, 255, 255))

    rendered = _render_text(mosaic_from_image(image, 1, 1))

    assert "▌" in rendered or "▐" in rendered


def test_solid_image_renders_uniform_cells():
    image = PILImage.new("RGB", (16, 16), (200, 30, 30))

    rendered = _render_text(mosaic_from_image(image, 4, 2)).rstrip("\n")

    used = {ch for line in rendered.splitlines() for ch in line}
    assert used <= {" ", "█"}


def test_output_fits_requested_cell_box():
    image = PILImage.effect_noise((400, 100), 90).convert("RGB")

    rendered = _render_text(mosaic_from_image(image, 10, 5))

    lines = [line for line in rendered.splitlines() if line.strip()]
    assert 1 <= len(lines) <= 5
    assert all(len(line.rstrip()) <= 10 for line in lines)


def test_glyphs_stay_within_block_elements():
    image = PILImage.open_from = None  # guard against accidental reuse
    image = PILImage.effect_noise((32, 32), 64).convert("RGB")

    rendered = _render_text(mosaic_from_image(image, 8, 4))

    allowed = set(QUADRANT_GLYPHS) | {" ", "\n"}
    assert set(rendered) <= allowed


def test_rgba_input_is_composited_over_black():
    image = PILImage.new("RGBA", (8, 8), (255, 0, 0, 128))

    console = Console(width=20, record=True, force_terminal=True)
    console.print(mosaic_from_image(image, 4, 2))
    styled = console.export_text(styles=True)

    # 50% red over black composites to ~rgb(128,0,0) painted as cell
    # backgrounds; a crash or blank renderable would carry no color at all.
    assert "128;0;0" in styled or "127;0;0" in styled


def test_square_image_fills_square_cell_box_without_vertical_stretch():
    """Terminal cells are ~1:2 (w:h), so a 16x8-cell box is physically
    square. A square image must fill BOTH dimensions of it -- sampling that
    ignores the half-width of quadrant subpixels renders a square source as
    a tall ellipse occupying only half the columns."""
    image = PILImage.effect_noise((100, 100), 80).convert("RGB")

    rendered = _render_text(mosaic_from_image(image, 16, 8))

    lines = [line.rstrip() for line in rendered.splitlines() if line.strip()]
    assert len(lines) == 8
    assert max(len(line) for line in lines) == 16


def test_cover_fit_fills_entire_box_with_center_crop():
    """fit="cover" scales to FILL the box (cropping overflow) instead of
    letterboxing: a very wide image still paints every cell row. Geometry
    is asserted on the renderable's plain text (painted flat cells are
    styled SPACES, so console text export would miscount them)."""
    text = mosaic_from_image(
        PILImage.new("RGB", (400, 100), (10, 200, 60)), 10, 5, fit="cover"
    )

    lines = text.plain.split("\n")
    assert len(lines) == 5
    assert all(len(line) == 10 for line in lines)


def test_contain_stays_default():
    text = mosaic_from_image(PILImage.new("RGB", (400, 100), (10, 200, 60)), 10, 5)

    lines = text.plain.split("\n")
    assert len(lines) < 5
