"""Quadrant-mosaic rendering for terminal image fallbacks.

Half-block rendering (``rich_pixels``) samples one pixel per column and two
per line -- 1x2 subpixels per cell. This module samples 2x2 subpixels per
cell using quadrant glyphs from the universal Block Elements range, doubling
horizontal detail on every surface that can already show the half-block
fallback (tmux, textual-serve/browser, plain terminals) with no font risk.
Sextant/octant glyphs (Symbols for Legacy Computing) would sample finer
still, but their font coverage is spotty in browsers and stock macOS
terminal fonts, so they are deliberately not used.

Each cell carries at most two colors (foreground + background); the four
subpixels are split by luminance and each group is averaged. Flat-shaded art
(character card portraits) loses nothing; photographic gradients trade a
little color precision for the doubled spatial detail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image as PILImage
from rich.text import Text

if TYPE_CHECKING:  # pragma: no cover
    pass

# Index = 4-bit subpixel mask: bit 0 = top-left, 1 = top-right,
# 2 = bottom-left, 3 = bottom-right.
QUADRANT_GLYPHS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"

# Below this luminance spread a cell is treated as one flat color; the
# threshold split would otherwise amplify sensor/scaling noise into
# speckled glyphs.
_FLAT_CELL_SPREAD = 8.0


def _luminance(rgb: tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _mean(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    n = len(colors)
    return (
        sum(c[0] for c in colors) // n,
        sum(c[1] for c in colors) // n,
        sum(c[2] for c in colors) // n,
    )


def mosaic_from_image(
    image: "PILImage.Image", box_cols: int, box_lines: int
) -> Text:
    """Render ``image`` into a quadrant-glyph mosaic fitting a cell box.

    Mirrors ``scale_image_for_cell_box``'s fit convention (terminal cells
    are ~2x taller than wide), corrected for the mosaic's subpixel shape:
    a quadrant subpixel is HALF a cell wide but the same half-cell tall as
    a half-block subpixel, so the horizontal sampling density is doubled
    while the on-screen size stays the aspect-true fit -- ``scale = min(
    box_cols / width, box_lines * 2 / height)`` with a ``2 * width *
    scale`` x ``height * scale`` sampling grid.

    Args:
        image: Source PIL image; never modified. RGBA is composited over
            black before sampling.
        box_cols: Destination box width in character columns.
        box_lines: Destination box height in character lines.

    Returns:
        A non-wrapping Rich ``Text`` renderable, one line per cell row.
    """
    source = image
    if source.mode != "RGB":
        rgba = source.convert("RGBA")
        base = PILImage.new("RGBA", rgba.size, (0, 0, 0, 255))
        source = PILImage.alpha_composite(base, rgba).convert("RGB")
    src_w, src_h = source.size
    scale = min(max(1, box_cols) / src_w, max(1, box_lines) * 2 / src_h)
    grid_w = min(max(2, round(src_w * scale * 2)), max(1, box_cols) * 2)
    grid_h = min(max(1, round(src_h * scale)), max(1, box_lines) * 2)
    source = source.resize((grid_w, grid_h), PILImage.Resampling.LANCZOS)
    width, height = source.size
    cell_cols = max(1, (width + 1) // 2)
    cell_rows = max(1, (height + 1) // 2)
    pixels = source.load()

    def sample(x: int, y: int) -> tuple[int, int, int]:
        # Clamp so odd-sized thumbnails repeat their edge pixel instead of
        # reading out of bounds.
        return pixels[min(x, width - 1), min(y, height - 1)]

    text = Text(no_wrap=True, end="")
    for row in range(cell_rows):
        if row:
            text.append("\n")
        for col in range(cell_cols):
            cell = [
                sample(col * 2, row * 2),
                sample(col * 2 + 1, row * 2),
                sample(col * 2, row * 2 + 1),
                sample(col * 2 + 1, row * 2 + 1),
            ]
            lums = [_luminance(c) for c in cell]
            spread = max(lums) - min(lums)
            if spread < _FLAT_CELL_SPREAD:
                r, g, b = _mean(cell)
                text.append(" ", style=f"on rgb({r},{g},{b})")
                continue
            threshold = (max(lums) + min(lums)) / 2
            mask = 0
            fg_group: list[tuple[int, int, int]] = []
            bg_group: list[tuple[int, int, int]] = []
            for bit, (color, lum) in enumerate(zip(cell, lums)):
                if lum >= threshold:
                    mask |= 1 << bit
                    fg_group.append(color)
                else:
                    bg_group.append(color)
            fr, fg_, fb = _mean(fg_group)
            br, bg_, bb = _mean(bg_group)
            text.append(
                QUADRANT_GLYPHS[mask],
                style=f"rgb({fr},{fg_},{fb}) on rgb({br},{bg_},{bb})",
            )
    return text
