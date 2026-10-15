#!/usr/bin/env python3
"""Generate the mobile PWA icon set with stdlib only (``struct`` + ``zlib``).

Writes, into ``tldw_chatbook/Web_Server/static/mobile/``:

- ``icon-192.png`` — 192x192, rounded square on a transparent field
- ``icon-512.png`` — 512x512, rounded square on a transparent field
- ``apple-touch-icon.png`` — 180x180, full-bleed opaque square (iOS masks
  its own corners onto opaque icons; a transparent field would render as
  black corners)

Design (ADR-234 Phase 1b): a single-accent rounded square with a subtle
top-light vertical shade of the same hue, carrying a white block "C" glyph
(an annulus with the right side opened ~50 degrees). No fonts, no Pillow:
each pixel is supersampled 4x4 and blended in Python, then emitted as a
PNG (8-bit RGBA, filter 0 scanlines, zlib IDAT).

Usage (run ONCE from the repo root, then commit the outputs)::

    python Helper_Scripts/generate_mobile_icons.py [--check]

Idempotent by construction: no timestamps, no randomness, deterministic
float math — rerunning overwrites the files with byte-identical output.
The committed PNGs are the source of truth; this script is NOT run in CI
(it is a maintainer regeneration hook). ``--check`` regenerates in memory
and reports whether the on-disk files match, exiting non-zero on drift.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import zlib
from pathlib import Path

__all__ = ["render_icon", "write_png", "main"]

#: Repository layout anchors (this script lives in ``Helper_Scripts/``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "tldw_chatbook" / "Web_Server" / "static" / "mobile"

#: The one accent hue (indigo) as RGB; the background shade runs from a
#: lighter tint of this hue at the top to the base tone at the bottom.
_ACCENT = (94, 82, 224)
_ACCENT_LIGHT = (120, 110, 240)
_GLYPH = (255, 255, 255, 255)

#: Rounded-square geometry (fractions of the icon size).
_CORNER_RADIUS = 0.225

#: Block "C" glyph geometry (fractions of the icon size): an annulus
#: centered slightly below middle, right side opened by this half-angle.
_C_OUTER_RADIUS = 0.290
_C_INNER_RADIUS = 0.175
_C_OPEN_HALF_ANGLE = 0.873  # radians (~50 degrees)

#: Supersampling grid per pixel edge (4x4 -> 16 samples per pixel).
_SS = 4

#: PNG signature + the deterministic chunks of a 8-bit RGBA, non-interlaced
#: image. Filter byte 0 (None) per scanline: flat-color icons compress
#: well enough and the writer stays trivially auditable.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _lerp(a: float, b: float, t: float) -> int:
    return int(round(a + (b - a) * t))


def _rounded_square_coverage(
    x: float, y: float, size: float, *, opaque: bool
) -> float:
    """Sample the background square's coverage at one point.

    ``opaque`` (the apple-touch variant) fills the whole canvas: iOS applies
    its own corner mask, so pre-rounding would double-round the icon.
    """
    if opaque:
        return 1.0
    radius = _CORNER_RADIUS * size
    half = size / 2.0
    dx = abs(x - half) - (half - radius)
    dy = abs(y - half) - (half - radius)
    if dx <= 0 or dy <= 0:
        return 1.0
    distance = (dx * dx + dy * dy) ** 0.5
    return 1.0 if distance <= radius else 0.0


def _glyph_coverage(x: float, y: float, size: float) -> float:
    """Sample the block "C" glyph's coverage at one point."""
    half = size / 2.0
    dx = x - half
    dy = half - y  # image y grows downward; the glyph opens to the RIGHT
    distance = (dx * dx + dy * dy) ** 0.5
    if not (_C_INNER_RADIUS * size <= distance <= _C_OUTER_RADIUS * size):
        return 0.0
    if dx > 0 and abs(math.atan2(dy, dx)) < _C_OPEN_HALF_ANGLE:
        return 0.0
    return 1.0


def render_icon(size: int, *, opaque: bool = False) -> bytes:
    """Render one icon as PNG bytes.

    Args:
        size: Canvas edge in pixels (square).
        opaque: Full-bleed background (apple-touch variant) instead of a
            rounded square on transparency.

    Returns:
        Complete PNG file bytes.
    """
    pixels = bytearray(size * size * 4)
    step = 1.0 / _SS
    for py in range(size):
        for px in range(size):
            bg_total = 0.0
            glyph_total = 0.0
            for sy in range(_SS):
                for sx in range(_SS):
                    x = px + (sx + 0.5) * step
                    y = py + (sy + 0.5) * step
                    if _glyph_coverage(x, y, size):
                        glyph_total += 1.0
                    elif _rounded_square_coverage(
                        x, y, size, opaque=opaque
                    ):
                        bg_total += 1.0
            samples = _SS * _SS
            glyph_a = glyph_total / samples
            bg_a = bg_total / samples
            # Vertical single-hue shade: lighter tint at the top.
            t = py / max(1, size - 1)
            r = _lerp(_ACCENT_LIGHT[0], _ACCENT[0], t)
            g = _lerp(_ACCENT_LIGHT[1], _ACCENT[1], t)
            b = _lerp(_ACCENT_LIGHT[2], _ACCENT[2], t)
            base = (py * size + px) * 4
            if glyph_a > 0:
                # White glyph blended over whatever is beneath it.
                pixels[base] = _blend_channel(glyph_a, r)
                pixels[base + 1] = _blend_channel(glyph_a, g)
                pixels[base + 2] = _blend_channel(glyph_a, b)
                pixels[base + 3] = 255
            else:
                pixels[base] = r
                pixels[base + 1] = g
                pixels[base + 2] = b
                pixels[base + 3] = int(round(bg_a * 255))
    return write_png(size, size, bytes(pixels))


def _blend_channel(coverage: float, underneath: int) -> int:
    """Blend the white glyph channel over ``underneath`` by ``coverage``."""
    return int(round(coverage * 255 + (1.0 - coverage) * underneath))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(width: int, height: int, rgba: bytes) -> bytes:
    """Serialize one 8-bit RGBA raster into PNG bytes (filter 0 scanlines)."""
    stride = width * 4
    raw = bytearray()
    for row in range(height):
        raw.append(0)
        raw.extend(rgba[row * stride : (row + 1) * stride])
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


#: (filename, edge, opaque) — the committed icon set (ADR-234 Phase 1b).
_ICONS = (
    ("icon-192.png", 192, False),
    ("icon-512.png", 512, False),
    ("apple-touch-icon.png", 180, True),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the on-disk icons against a fresh render; no writes",
    )
    args = parser.parse_args(argv)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    drifted: list[str] = []
    for name, size, opaque in _ICONS:
        fresh = render_icon(size, opaque=opaque)
        target = _OUTPUT_DIR / name
        if args.check:
            on_disk = (
                target.read_bytes() if target.exists() else None
            )
            if on_disk != fresh:
                drifted.append(name)
            print(f"{name}: {'OK' if on_disk == fresh else 'DRIFTED'}")
        else:
            target.write_bytes(fresh)
            print(f"wrote {target} ({len(fresh)} bytes)")
    if args.check and drifted:
        print(f"drift in: {', '.join(drifted)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
