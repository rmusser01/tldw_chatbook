# Tests/Chat/test_visual_renderer_decoupling.py
"""TASK-18606: the visual transcript renderer owns its own determinism.

The renderer used to borrow reproducibility from a `pillow==11.2.1` pin. That
did not work: `ImageFont.load_default()` is not a stable input, and on Pillow
12.1.1 an 82-character line measured 738px against 496px of usable canvas and
ran off the right edge, losing transcript text with no error.

These pin the three properties that replaced the pin, so none of them can be
quietly undone by a future edit.
"""

from __future__ import annotations

import re

import pytest
from PIL import Image, ImageDraw, ImageFont

from tldw_chatbook.Chat import console_visual_transcript as vt
from tldw_chatbook.Chat.console_visual_transcript import (
    CELL_WIDTH,
    render_visual_transcript,
    EVALUATION_RENDERER_PROFILES,
    LOGICAL_WIDTH,
    MARGIN_X,
    MAX_LINE_CHARACTERS,
    NATIVE_512_EVALUATION_PROFILE,
    PRODUCTION_RENDERER_PROFILE,
    VisualRendererFontError,
    renderer_font,
)


def _units(count: int) -> tuple:
    """Ordered durable units, in the shape the renderer actually takes."""
    from tldw_chatbook.Chat.console_context_compaction import (
        DurableConversationUnit,
        DurableMessageSnapshot,
    )

    return tuple(
        DurableConversationUnit(
            (
                DurableMessageSnapshot(f"user-{i}", 1, "user", f"line {i} " * 6),
                DurableMessageSnapshot(f"assistant-{i}", 1, "assistant", f"answer {i}"),
            )
        )
        for i in range(count)
    )


# -- 1. the font is fixed-cell, and its metrics are verified ----------------


def test_the_renderer_font_is_the_legacy_fixed_cell_font():
    """Not `load_default()`, which Pillow redefined mid-10.x to a
    proportional face."""
    font = renderer_font()
    assert type(font) is ImageFont.ImageFont, (
        "renderer must use the legacy fixed-cell bitmap font, not a "
        f"{type(font).__name__}"
    )


def test_a_full_width_line_fits_inside_the_canvas():
    """The regression that started this: text ran off the right edge and was
    silently lost."""
    font = renderer_font()
    image = Image.new("L", (LOGICAL_WIDTH, 64), 255)
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN_X, 8), "W" * MAX_LINE_CHARACTERS, fill=0, font=font)
    pixels = image.load()
    rightmost = max(
        (x for x in range(LOGICAL_WIDTH) for y in range(64) if pixels[x, y] < 128),
        default=0,
    )
    assert rightmost <= LOGICAL_WIDTH - MARGIN_X, (
        f"ink reaches x={rightmost}, past the usable edge "
        f"x={LOGICAL_WIDTH - MARGIN_X} -- transcript text is being clipped"
    )


def test_the_layout_constants_are_self_consistent():
    assert CELL_WIDTH * MAX_LINE_CHARACTERS <= LOGICAL_WIDTH - (2 * MARGIN_X)


def test_a_font_with_wrong_metrics_is_refused_loudly(monkeypatch):
    """The whole point of verifying: a changed cell size must raise, never
    silently clip. This is what would have caught the Pillow 12 break."""
    monkeypatch.setattr(vt, "_RENDERER_FONT", None)
    monkeypatch.setattr(
        ImageFont, "load_default_imagefont", lambda: ImageFont.load_default()
    )
    with pytest.raises(VisualRendererFontError, match="font metrics changed"):
        vt._load_renderer_font()


def test_a_pillow_without_the_legacy_font_is_refused(monkeypatch):
    monkeypatch.setattr(vt, "_RENDERER_FONT", None)
    monkeypatch.delattr(ImageFont, "load_default_imagefont", raising=False)
    with pytest.raises(VisualRendererFontError, match="load_default_imagefont"):
        vt._load_renderer_font()


# -- 2. identity excludes the PNG encoder ----------------------------------


def test_identity_hash_is_pixel_based_not_png_based():
    """An encoder change (compression level, chunk order) must not move the
    identity when every pixel is unchanged."""
    from hashlib import sha256
    from io import BytesIO

    image = Image.new("L", (64, 32), 255)
    for x in range(0, 64, 3):
        image.putpixel((x, 10), 0)

    def png_digest(compress: int) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG", optimize=False, compress_level=compress)
        return sha256(buf.getvalue()).hexdigest()

    assert png_digest(9) != png_digest(1), "fixture no longer varies the encoding"

    def pixel_digest() -> str:
        return sha256(
            f"{image.mode}:{image.width}x{image.height}:".encode("ascii")
            + image.tobytes()
        ).hexdigest()

    assert pixel_digest() == pixel_digest()


def test_pages_expose_both_digests_with_distinct_meanings():
    """Identity (pixels) and wire integrity (bytes) are different questions;
    conflating them is what put the encoder into the identity."""
    artifact = render_visual_transcript(_units(1), summarized_prefix_digest="d" * 64)
    page = artifact.pages[0]
    assert page.pixel_sha256 != page.png_sha256
    from hashlib import sha256

    assert page.png_sha256 == sha256(page.png_bytes).hexdigest()


# -- 3. identity no longer names the dependency ----------------------------


@pytest.mark.parametrize("profile", EVALUATION_RENDERER_PROFILES)
def test_renderer_version_does_not_embed_the_pillow_version(profile):
    """It was embedded AND drawn into every page footer, so the identity
    could not survive a dependency bump by construction."""
    assert "pillow" not in profile.renderer_version.lower()
    assert not re.search(r"\d+\.\d+\.\d+", profile.renderer_version)


def test_the_two_profiles_stay_distinguishable():
    assert (
        PRODUCTION_RENDERER_PROFILE.renderer_version
        != NATIVE_512_EVALUATION_PROFILE.renderer_version
    )


def test_rendering_is_stable_across_repeated_calls():
    """Determinism is now the renderer's own guarantee, so assert it directly."""
    units = _units(12)
    first = render_visual_transcript(units, summarized_prefix_digest="e" * 64)
    second = render_visual_transcript(units, summarized_prefix_digest="e" * 64)
    assert [p.pixel_sha256 for p in first.pages] == [
        p.pixel_sha256 for p in second.pages
    ]
