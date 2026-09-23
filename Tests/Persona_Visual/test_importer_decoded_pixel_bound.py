"""The untrusted Persona Visual import path bounds aggregate decoded pixels.

TASK-32901 (tier-2 S16 P2): of the three decode sites, only the *import* one
-- the one that reads an archive the user did not author -- was missing the
``width * height * frame_count`` bound its two siblings enforce
(``assets.py``, ``authoring_workspace.py``). ``_inspect_image`` bounded each
frame and the frame count (<=240), then decoded every frame in a loop:
4096 x 4096 x 240 = 4.03e9 pixels per asset, 60x the bound its siblings
enforce, from a compressed animation that fits easily inside the 100 MB
archive cap. Peak memory stays bounded because frames decode sequentially,
so it reads as safe -- it is CPU/wall-clock amplification, not a heap blowup.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from tldw_chatbook.Persona_Visual import importer
from tldw_chatbook.Persona_Visual.contracts import MAX_ASSET_DECODED_PIXELS


def _animation(frames: int, size: int = 32) -> BytesIO:
    images = []
    for index in range(frames):
        frame = Image.new("P", (size, size), color=0)
        # Distinct content per frame; GIF drops frames identical to their
        # predecessor, which would silently reduce the animation to one.
        frame.putpixel((index % size, index % size), 1 + index % 7)
        images.append(frame)
    buffer = BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=10,
    )
    buffer.seek(0)
    return buffer


def _record(buffer: BytesIO, size: int = 32) -> dict:
    return {
        "byte_count": len(buffer.getvalue()),
        "mime_type": "image/gif",
        "width": size,
        "height": size,
    }


def test_bound_is_shared_with_the_two_sibling_decode_sites():
    from tldw_chatbook.Persona_Visual import assets

    assert assets.MAX_ASSET_DECODED_PIXELS == MAX_ASSET_DECODED_PIXELS
    assert MAX_ASSET_DECODED_PIXELS == importer.MAX_ASSET_DECODED_PIXELS


def test_an_ordinary_animation_still_imports():
    buffer = _animation(4)
    frame_count, _ = importer._inspect_image(buffer, _record(buffer))
    assert frame_count == 4


def test_aggregate_decoded_pixels_are_refused_before_the_decode_loop(monkeypatch):
    buffer = _animation(4)
    # 4 frames x 32 x 32 = 4096 decoded pixels; bound it below that.
    monkeypatch.setattr(importer, "MAX_ASSET_DECODED_PIXELS", 1024)

    with pytest.raises(ValueError):
        importer._inspect_image(buffer, _record(buffer))
