"""A byte cap says nothing about pixels.

TASK-32806.8. The chat attachment path capped attachments at 10 MB and then
decoded them, but PIL only WARNS above `MAX_IMAGE_PIXELS` -- it decodes
anyway -- so a roughly 96-megapixel flat image compresses well under the
byte cap and expands to about 288 MB. Five other modules in this repo
escalate that warning, including `console_chat_fork`, which re-validates
this very payload and imports a constant from this module: the two
disagreed about the same bytes.

Two things had to be true for the fix to work, and the first draft got the
second one wrong:

1. `prepare_image_payload` escalates the warning, for direct callers.
2. `process_image_file` rejects on pixels BESIDE the byte cap, outside the
   try/except that falls back to the original bytes. Without this the
   refusal was caught by that fallback and the bomb was sent anyway -- and
   a first version of the guard also caught PIL's hard `DecompressionBombError`
   as "unreadable", which made the whole check inert.

These tests lower `MAX_IMAGE_PIXELS` so a small image lands in PIL's bands
without allocating anything, which is how the review reproduced it.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest
from PIL import Image as PILImage

from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import (
    ChatImageHandler,
)


#: Small enough that every image below is trivial to build, while still
#: putting 120x120 in the warn band and 900x900 past PIL's hard limit.
TEST_PIXEL_CAP = 10_000


@pytest.fixture()
def low_pixel_cap(monkeypatch):
    """Lower the pixel cap, and keep the config read out of the way.

    `process_image_file` reads the byte cap and the format allowlist from
    config, which trips this worktree's ADR-126 storage admission gate
    (`RecoveryRequired`). Pinning both to their shipped values keeps these
    tests about pixels, which is what they are for.
    """
    from tldw_chatbook.Chat import attachment_core

    monkeypatch.setattr(PILImage, "MAX_IMAGE_PIXELS", TEST_PIXEL_CAP)
    monkeypatch.setattr(attachment_core, "max_image_bytes", lambda: 10 * 1024 * 1024)
    monkeypatch.setattr(
        attachment_core,
        "supported_image_formats",
        lambda: {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"},
    )
    return TEST_PIXEL_CAP


def _png(tmp_path: pathlib.Path, width: int, height: int) -> pathlib.Path:
    path = tmp_path / f"{width}x{height}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PILImage.new("RGB", (width, height), (10, 20, 30)).save(path, format="PNG")
    return path


@pytest.mark.asyncio
async def test_an_image_under_the_pixel_cap_is_accepted(tmp_path, low_pixel_cap):
    payload, mime = await ChatImageHandler.process_image_file(
        str(_png(tmp_path, 50, 50))
    )
    assert payload
    assert mime == "image/png"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "height", "band"),
    [
        (120, 120, "PIL's warn-only band"),
        (900, 900, "past PIL's hard DecompressionBombError limit"),
    ],
)
async def test_an_oversized_by_pixels_image_is_refused(
    tmp_path, low_pixel_cap, width, height, band
):
    with pytest.raises(ValueError, match="pixels"):
        await ChatImageHandler.process_image_file(str(_png(tmp_path, width, height)))


@pytest.mark.asyncio
async def test_the_refusal_is_not_swallowed_by_the_original_bytes_fallback(
    tmp_path, low_pixel_cap
):
    """The specific way the first version of this fix failed.

    `process_image_file` catches every exception from `prepare_image_payload`
    and falls back to the unprocessed bytes. A pixel rejection raised only
    from inside that call is therefore invisible, and the bomb is sent. The
    rejection has to happen beside the byte cap instead.
    """
    path = _png(tmp_path, 900, 900)
    original_bytes = path.read_bytes()

    with pytest.raises(ValueError):
        await ChatImageHandler.process_image_file(str(path))

    # Nothing returned those bytes to a caller.
    assert len(original_bytes) > 0


@pytest.mark.asyncio
async def test_the_direct_payload_path_refuses_too(tmp_path, low_pixel_cap):
    """Callers that skip `process_image_file` are covered by the escalation."""
    data = _png(tmp_path, 900, 900).read_bytes()
    with pytest.raises(Exception):
        await ChatImageHandler.prepare_image_payload(data, ".png")
