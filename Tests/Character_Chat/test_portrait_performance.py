"""Six-pose normalization rejects unsafe artwork without retaining decoded buffers."""

from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat import portrait_performance as performance


def png(color="red", size=(8, 8), format="PNG"):
    with BytesIO() as out:
        image = Image.new("RGB", size, color)
        try:
            image.save(out, format=format)
            return out.getvalue()
        finally:
            image.close()


def test_six_roles_normalize_in_role_order_and_strip_metadata():
    inputs = {role: png(format="JPEG") for role in reversed(performance.FRAME_ROLES)}
    frames = performance.normalize_performance_frames(inputs)
    assert tuple(frame.role for frame in frames) == (
        "rest",
        "small",
        "open",
        "blink_rest",
        "blink_small",
        "blink_open",
    )
    assert all(frame.png.startswith(b"\x89PNG\r\n\x1a\n") for frame in frames)
    assert all((frame.width, frame.height) == (8, 8) for frame in frames)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "geometry", "corrupt", "animated", "pixels"]
)
def test_invalid_set_is_rejected(change):
    frames = {role: png() for role in performance.FRAME_ROLES}
    if change == "missing":
        frames.pop("open")
    elif change == "extra":
        frames["bogus"] = png()
    elif change == "geometry":
        frames["open"] = png(size=(7, 8))
    elif change == "corrupt":
        frames["open"] = b"not an image"
    elif change == "pixels":
        frames["open"] = png(size=(1025, 1024))
    else:
        with BytesIO() as out:
            a = Image.new("RGB", (8, 8), "red")
            b = Image.new("RGB", (8, 8), "blue")
            try:
                a.save(
                    out, format="PNG", save_all=True, append_images=[b], duration=100
                )
                frames["open"] = out.getvalue()
            finally:
                a.close()
                b.close()
    with pytest.raises(ValueError):
        performance.normalize_performance_frames(frames)


def test_encoded_limits_are_inclusive_and_aggregate_is_checked(monkeypatch):
    raw = png()
    normalized = performance.normalize_performance_frame("rest", raw)
    limit = len(normalized.png)
    monkeypatch.setattr(performance, "MAX_FRAME_BYTES", limit)
    monkeypatch.setattr(performance, "MAX_SET_BYTES", limit * 6)
    inputs = {role: raw for role in performance.FRAME_ROLES}
    assert len(performance.normalize_performance_frames(inputs)) == 6
    monkeypatch.setattr(performance, "MAX_SET_BYTES", limit * 6 - 1)
    with pytest.raises(ValueError):
        performance.normalize_performance_frames(inputs)
    monkeypatch.setattr(performance, "MAX_FRAME_BYTES", limit - 1)
    with pytest.raises(ValueError):
        performance.normalize_performance_frame("rest", raw)


def test_encoding_failure_closes_original_and_converted_images(monkeypatch):
    raw = png()
    opened = []
    converted = []
    original_open = Image.open
    original_convert = Image.Image.convert

    def track_open(*args, **kwargs):
        image = original_open(*args, **kwargs)
        opened.append(image)
        return image

    def track_convert(image, *args, **kwargs):
        result = original_convert(image, *args, **kwargs)
        converted.append(result)
        return result

    def fail_save(*args, **kwargs):
        raise OSError("encode failed")

    monkeypatch.setattr(Image, "open", track_open)
    monkeypatch.setattr(Image.Image, "convert", track_convert)
    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(ValueError):
        performance.normalize_performance_frame("rest", raw)
    assert opened and converted
    for image in opened + converted:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_historical_and_current_asset_share_canonical_source():
    from tldw_chatbook.Character_Chat.visual_identity import VisualIdentityResolution

    current = VisualIdentityResolution(
        actor_kind="character",
        actor_id="7",
        requested_expression_key="happy",
        manual_expression_key=None,
        resolved_expression_key="happy",
        pack_id=3,
        pack_version_id=4,
        asset_id=5,
        expression_id=None,
        storage_source="profile",
        storage_relpath="ignored.png",
        content_type="image/png",
        is_animated=False,
        resolution_source="active_pack",
        fallback_reason="",
        cache_identity=("current",),
        image_bytes=png(),
    )
    historical = replace(
        current, resolution_source="history_immutable", cache_identity=("history",)
    )
    assert performance.source_from_resolution(
        current
    ) == performance.source_from_resolution(historical)


def test_normalization_uses_and_releases_shared_preparation_budget(monkeypatch):
    from tldw_chatbook.Chat import character_expression_playback as playback

    before = playback.preparation_bytes()
    monkeypatch.setattr(playback, "MAX_PREPARATION_BYTES", before + 8 * 8 * 12 - 1)
    with pytest.raises(ValueError):
        performance.normalize_performance_frame("rest", png())
    assert playback.preparation_bytes() == before


def test_normalized_png_removes_embedded_profile_metadata():
    with BytesIO() as source:
        image = Image.new("RGB", (8, 8), "red")
        try:
            image.save(source, format="PNG", icc_profile=b"fixture profile")
            frame = performance.normalize_performance_frame("rest", source.getvalue())
        finally:
            image.close()
    with Image.open(BytesIO(frame.png)) as normalized:
        assert normalized.info == {}


def test_pixel_limit_is_inclusive_and_stricter_existing_limits_win(monkeypatch):
    from tldw_chatbook.Character_Chat import visual_identity

    assert (
        performance.normalize_performance_frame("rest", png(size=(1024, 1024))).width
        == 1024
    )
    monkeypatch.setattr(visual_identity, "MAX_EXPRESSION_IMAGE_DIMENSION", 7)
    with pytest.raises(ValueError):
        performance.normalize_performance_frame("rest", png())
