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

# Shade ramp for the monochrome path, darkest to lightest. The colour mosaic
# puts 100% of its information in BACKGROUND colour -- a flat cell is a space
# glyph with `on rgb(...)` -- so when colour is unavailable every cell becomes
# an ordinary space and the image does not degrade, it disappears entirely.
# Textual converts the whole app to monochrome when NO_COLOR is set
# (`textual/app.py` -> `filter.py`), which is one confirmed way a user sees no
# portrait at all. In that mode the GLYPH has to carry the luminance instead.
#
# Block Elements only (U+2591-2588), the same universal range the quadrant
# glyphs come from, so this adds no new font risk.
#
# No blank bucket, deliberately (Qodo review, PR #1183). An earlier ramp led
# with a space, so a sufficiently dark portrait mapped every cell to the
# darkest bucket and rendered as an all-space box -- invisible, reproducing
# the exact bug this path exists to fix. Character-card art on a dark theme
# is that case, so it was the common path rather than an edge one. The
# darkest ink still has to be ink.
_SHADE_RAMP = "░▒▓█"


def _shade_glyph(lum: float, lo: float = 0.0, hi: float = 255.0) -> str:
    """Return the shade glyph for a luminance, normalised to ``lo``..``hi``.

    The range is normalised PER IMAGE rather than mapped from absolute
    0-255. A character-card portrait on a dark theme can occupy a narrow
    dark band -- every cell then lands in one ramp bucket and renders as a
    uniform block: visible, but shapeless, which is not a portrait. Stretching
    the image's own range across the ramp keeps the subject distinguishable
    from its background.

    Args:
        lum: Mean luminance of the cell.
        lo: Darkest cell luminance in this image.
        hi: Brightest cell luminance in this image.

    Returns:
        One character from ``_SHADE_RAMP``; darker maps to a sparser glyph so
        the result reads like the image rather than its negative.
    """
    span = hi - lo
    # A genuinely flat image has nothing to stretch; render it mid-ramp
    # rather than dividing by ~0 and slamming everything to one end.
    if span < 1e-6:
        return _SHADE_RAMP[len(_SHADE_RAMP) // 2]
    scaled = (lum - lo) / span
    index = int(scaled * len(_SHADE_RAMP))
    return _SHADE_RAMP[min(max(index, 0), len(_SHADE_RAMP) - 1)]


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
    image: "PILImage.Image",
    box_cols: int,
    box_lines: int,
    *,
    fit: str = "contain",
    monochrome: bool = False,
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
        fit: "contain" (default) letterboxes the whole image inside the
            box; "cover" scales to FILL the box and center-crops the
            overflow (object-fit: cover) -- aspect is preserved either way.
        monochrome: Render luminance as SHADE GLYPHS instead of colour. The
            colour path stores the whole image in background colour, so a
            terminal without colour shows a box of blank spaces rather than
            a portrait; set this when colour is unavailable (Textual sets
            ``App.no_color`` from ``NO_COLOR``) so the glyph carries the
            image instead.

    Returns:
        A non-wrapping Rich ``Text`` renderable, one line per cell row.
    """
    source = image
    if source.mode != "RGB":
        rgba = source.convert("RGBA")
        base = PILImage.new("RGBA", rgba.size, (0, 0, 0, 255))
        source = PILImage.alpha_composite(base, rgba).convert("RGB")
    src_w, src_h = source.size
    box_w = max(1, box_cols)
    box_h = max(1, box_lines)
    if fit == "cover":
        scale = max(box_w / src_w, box_h * 2 / src_h)
        grid_w = max(box_w * 2, round(src_w * scale * 2))
        grid_h = max(box_h * 2, round(src_h * scale))
        source = source.resize((grid_w, grid_h), PILImage.Resampling.LANCZOS)
        left = (grid_w - box_w * 2) // 2
        top = (grid_h - box_h * 2) // 2
        source = source.crop((left, top, left + box_w * 2, top + box_h * 2))
    else:
        scale = min(box_w / src_w, box_h * 2 / src_h)
        grid_w = min(max(2, round(src_w * scale * 2)), box_w * 2)
        grid_h = min(max(1, round(src_h * scale)), box_h * 2)
        source = source.resize((grid_w, grid_h), PILImage.Resampling.LANCZOS)
    width, height = source.size
    cell_cols = max(1, (width + 1) // 2)
    cell_rows = max(1, (height + 1) // 2)
    pixels = source.load()

    def sample(x: int, y: int) -> tuple[int, int, int]:
        # Clamp so odd-sized thumbnails repeat their edge pixel instead of
        # reading out of bounds.
        return pixels[min(x, width - 1), min(y, height - 1)]

    mono_lo, mono_hi = 0.0, 255.0
    if monochrome:
        # One pass over the cell means to learn this image's actual range.
        cell_means = [
            sum(
                _luminance(sample(c * 2 + dx, r * 2 + dy))
                for dx in (0, 1)
                for dy in (0, 1)
            )
            / 4
            for r in range(cell_rows)
            for c in range(cell_cols)
        ]
        mono_lo, mono_hi = min(cell_means), max(cell_means)

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
            if monochrome:
                # One glyph per cell from the mean luminance: no colour is
                # consulted at all, so the result survives a monochrome
                # filter that would blank the coloured path entirely.
                # Normalised to this image's own range (see `_shade_glyph`)
                # so a dark portrait keeps its shape instead of flattening.
                text.append(_shade_glyph(sum(lums) / len(lums), mono_lo, mono_hi))
                continue
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
