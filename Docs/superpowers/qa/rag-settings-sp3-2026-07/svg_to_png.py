"""Convert this directory's captured SVGs to PNG via cairosvg.

Needs the cairo C library discoverable (macOS Homebrew build doesn't sit on
the default dyld search path):

    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
        .venv/bin/python3 Docs/superpowers/qa/rag-settings-sp3-2026-07/svg_to_png.py
"""

from pathlib import Path

import cairosvg

OUT = Path(__file__).resolve().parent

# cairosvg has no network access and this machine has no "Fira Code" font
# installed (Rich's SVG export hard-codes it), so the ▼/⚠ glyphs used by
# Collapsible titles and re-index warning labels render as tofu boxes.
# Route the monospace body font through macOS system fonts that DO carry
# those glyphs (Apple Symbols/Apple Color Emoji) as fallbacks -- cosmetic
# fix for THIS conversion step only; the real terminal app renders them
# fine already (confirmed in the original SP3 captures).
FONT_FALLBACK = '"Menlo", "Apple Symbols", "Apple Color Emoji", monospace'

for svg_path in sorted(OUT.glob("*.svg")):
    svg_text = svg_path.read_text()
    svg_text = svg_text.replace("Fira Code, monospace", FONT_FALLBACK)
    png_path = svg_path.with_suffix(".png")
    cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(png_path))
    print(f"{svg_path.name} -> {png_path.name}")
