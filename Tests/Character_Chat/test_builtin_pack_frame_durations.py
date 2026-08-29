"""Image inspection decodes frames only when there is a duration to find (TASK-24306).

`_inspect_image_bytes` computed every asset's animation duration by seeking and
`load()`-ing each frame, then threw the result away for still images on the very
next line (`duration_ms = decoded_duration_ms if is_animated else None`).

The bundled Samira pack is 31 STILL WebP files, and PIL routes even single-frame
WebP through `WebPAnimDecoder`, so a fresh profile paid 0.501 s of frame decoding
before first paint to produce 31 numbers it discarded. The fix guards the
computation on `is_animated`, which cannot change any returned value -- the
branch that no longer runs is exactly the branch whose result was dropped.

These tests pin both halves: stills must not decode frames, and animations must
still report their real duration.
"""

from __future__ import annotations

import io

import pytest

from tldw_chatbook.Character_Chat import visual_identity
from tldw_chatbook.Character_Chat.visual_identity import (
    _inspect_image_bytes,
    _load_samira_pack,
)

pytest.importorskip("PIL", reason="Pillow is required to decode image bytes")


def _still_webp() -> bytes:
    """One 8x8 non-animated WebP."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="WEBP")
    return buffer.getvalue()


def _animated_webp(durations_ms: list[int]) -> bytes:
    """An animated WebP whose frames carry the given durations."""
    from PIL import Image

    frames = [
        Image.new("RGB", (8, 8), (index * 20 % 256, 40, 60))
        for index in range(len(durations_ms))
    ]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
    )
    return buffer.getvalue()


def test_a_still_image_decodes_no_frames_for_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-animated image never enters the per-frame duration walk.

    Counting calls rather than timing: this is the whole 0.501 s, and a call
    count is the same on a loaded machine as an idle one.
    """
    calls = {"count": 0}
    real = visual_identity._image_duration_ms

    def counted(image, frame_count):
        calls["count"] += 1
        return real(image, frame_count)

    monkeypatch.setattr(visual_identity, "_image_duration_ms", counted)

    _format, _size, frame_count, is_animated, duration_ms, _pixels = (
        _inspect_image_bytes(_still_webp())
    )

    assert is_animated is False
    assert frame_count == 1
    assert duration_ms is None
    assert calls["count"] == 0, (
        "a still image walked its frames to compute a duration that is "
        "discarded for non-animated assets (TASK-24306)."
    )


def test_an_animated_image_still_reports_its_real_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must not cost animated assets their duration."""
    calls = {"count": 0}
    real = visual_identity._image_duration_ms

    def counted(image, frame_count):
        calls["count"] += 1
        return real(image, frame_count)

    monkeypatch.setattr(visual_identity, "_image_duration_ms", counted)

    durations = [40, 60, 80]
    _format, _size, frame_count, is_animated, duration_ms, _pixels = (
        _inspect_image_bytes(_animated_webp(durations))
    )

    assert is_animated is True
    assert frame_count == len(durations)
    assert calls["count"] == 1
    assert duration_ms == sum(durations), (
        f"animated duration came back as {duration_ms}, expected "
        f"{sum(durations)}; the guard changed a real result."
    )


def test_the_bundled_pack_is_entirely_still_images() -> None:
    """The saving is real for the pack that motivated it.

    Recorded as a fact about the shipped bundle, not an aspiration: if a future
    pack ships an animated expression it will decode again, and whoever sees
    first-run boot regress should find this test explaining why.
    """
    _card_json, _manifest, loaded_assets = _load_samira_pack(
        None, card_bytes=0, portrait_bytes=0
    )

    assert loaded_assets, "the bundled pack loaded no assets; the seed is broken"

    animated = [
        loaded.asset.storage_relpath
        for loaded in loaded_assets
        if loaded.asset.is_animated
    ]

    assert not animated, (
        "the bundled pack now ships animated assets: "
        f"{animated}. First-run boot will decode their frames again; that is "
        "correct behaviour, but the TASK-24306 saving no longer covers them."
    )
