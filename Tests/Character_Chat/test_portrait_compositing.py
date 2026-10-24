"""Pixel oracles for source-preserving composition, not model output quality."""

import asyncio
import hashlib
import importlib
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image


def core():
    return importlib.import_module("tldw_chatbook.Character_Chat.portrait_compositing")


def png(color, size=(64, 64), *, level=6):
    with Image.new("RGBA", size, color) as image, BytesIO() as output:
        image.save(output, format="PNG", compress_level=level)
        return output.getvalue()


def pixels(raw):
    with Image.open(BytesIO(raw)) as image:
        rgba = image.convert("RGBA")
        try:
            return rgba.size, rgba.tobytes()
        finally:
            rgba.close()


def scene(size=(64, 64)):
    api = core()
    base = png((80, 100, 120, 180), size)
    recipe = api.PortraitCompositionRecipe(
        1,
        *size,
        api.prepared_base_pixel_sha256(base),
        api.PixelRect(0, 0, 10, 10),
        api.PixelRect(42, 0, 10, 10),
        api.PixelRect(22, 36, 20, 16),
        True,
    )
    components = {
        role: api.compose_component(base, recipe, role, png(color, size))
        for role, color in [
            ("eyelids", (5, 10, 15, 0)),
            ("small", (140, 150, 160, 255)),
            ("open", (200, 210, 220, 20)),
        ]
    }
    return base, recipe, components


def test_pixel_identity_ignores_encoding_but_not_rgba_or_dimensions():
    api = core()
    a = png((10, 20, 30, 0), level=0)
    b = png((10, 20, 30, 0), level=9)
    assert hashlib.sha256(a).digest() != hashlib.sha256(b).digest()
    assert api.prepared_base_pixel_sha256(a) == api.prepared_base_pixel_sha256(b)
    for changed in [
        png((11, 20, 30, 0)),
        png((10, 20, 30, 1)),
        png((10, 20, 30, 0), (32, 128)),
    ]:
        assert api.prepared_base_pixel_sha256(a) != api.prepared_base_pixel_sha256(
            changed
        )
    header = b"tldw-portrait-base-rgba-v1\0" + (64).to_bytes(4, "big") * 2
    assert (
        api.prepared_base_pixel_sha256(a)
        == hashlib.sha256(header + bytes((10, 20, 30, 0)) * 4096).hexdigest()
    )


@pytest.mark.parametrize("size", [(64, 64), (64, 80)])
def test_pairs_preserve_base_alpha_background_and_each_other(size):
    api = core()
    base, recipe, components = scene(size)
    frames = {f.role: f for f in api.assemble_components(base, recipe, components)}
    assert tuple(frames) == (
        "rest",
        "small",
        "open",
        "blink_rest",
        "blink_small",
        "blink_open",
    )
    assert pixels(frames["rest"].png) == pixels(base)
    with Image.open(BytesIO(base)) as original:
        images = {
            role: Image.open(BytesIO(frame.png)) for role, frame in frames.items()
        }
        try:
            for y in range(size[1]):
                for x in range(size[0]):
                    eye = (1 <= x <= 8 and 1 <= y <= 8) or (
                        43 <= x <= 50 and 1 <= y <= 8
                    )
                    mouth = 23 <= x <= 40 and 37 <= y <= 50
                    for image in images.values():
                        assert image.getpixel((x, y))[3] == 180
                        if not eye and not mouth:
                            assert image.getpixel((x, y)) == original.getpixel((x, y))
                    if mouth:
                        for role in ("rest", "small", "open"):
                            assert images[role].getpixel((x, y)) == images[
                                "blink_" + role
                            ].getpixel((x, y))
                    if eye:
                        assert (
                            len(
                                {
                                    images[r].getpixel((x, y))
                                    for r in ("rest", "small", "open")
                                }
                            )
                            == 1
                        )
                        assert (
                            len(
                                {
                                    images[r].getpixel((x, y))
                                    for r in ("blink_rest", "blink_small", "blink_open")
                                }
                            )
                            == 1
                        )
            assert images["small"].getpixel((30, 42)) == (140, 150, 160, 180)
            assert images["open"].getpixel((30, 42)) == (200, 210, 220, 180)
            assert images["blink_rest"].getpixel((4, 4)) == (5, 10, 15, 180)
            # Hand-checked inward feather: distance 1 / width 3 -> alpha 85.
            assert images["small"].getpixel((23, 42)) == (100, 117, 133, 180)
        finally:
            for image in images.values():
                image.close()


@pytest.mark.parametrize(
    "change",
    [
        {"left_eye": (0, 0, 2, 8)},
        {"left_eye": (-1, 0, 10, 10)},
        {"left_eye": (60, 0, 10, 10)},
        {"left_eye": (True, 0, 10, 10)},
        {"left_eye": (0.0, 0, 10, 10)},
        {"right_eye": (2, 2, 10, 10)},
        {"algorithm_version": 2},
        {"algorithm_version": True},
        {"neutral_base_confirmed": False},
        {"base_pixel_sha256": "0" * 64},
        {"width": 63},
    ],
)
def test_invalid_regions_identity_and_confirmation_are_refused(change):
    api = core()
    base, recipe, _ = scene()
    fields = {
        key: api.PixelRect(*value) if isinstance(value, tuple) else value
        for key, value in change.items()
    }
    with pytest.raises(ValueError):
        api.validate_recipe(base, replace(recipe, **fields))


def test_recovery_and_repeated_reuse_do_not_blend_finished_planes_twice():
    api = core()
    base, recipe, components = scene()
    frames = api.assemble_components(base, recipe, components)
    for _ in range(3):
        recovered = api.recover_components(base, recipe, frames)
        rebuilt = api.assemble_components(base, recipe, recovered)
        assert [pixels(f.png) for f in rebuilt] == [pixels(f.png) for f in frames]
        frames = rebuilt
    assert tuple(f.role for f in api.assemble_components(base, recipe, {})) == ("rest",)
    assert tuple(
        f.role
        for f in api.assemble_components(base, recipe, {"small": components["small"]})
    ) == ("rest", "small")
    changed = list(frames)
    from tldw_chatbook.Character_Chat.portrait_performance import (
        normalize_performance_frame,
    )

    changed[-1] = normalize_performance_frame("blink_open", png("red"))
    with pytest.raises(ValueError):
        api.recover_components(base, recipe, tuple(changed))
    with pytest.raises(ValueError):
        api.assemble_components(base, recipe, {"small": png("red")})


def test_bad_stills_roles_and_geometry_are_rejected():
    api = core()
    base, recipe, _ = scene()
    with (
        BytesIO() as output,
        Image.new("RGB", (64, 64), "red") as first,
        Image.new("RGB", (64, 64), "blue") as second,
    ):
        first.save(
            output, format="PNG", save_all=True, append_images=[second], duration=100
        )
        animated = output.getvalue()
    for raw in [
        b"invalid",
        b"",
        png("red", (65, 64)),
        animated,
        b"x" * (4 * 1024 * 1024 + 1),
        png("red", (1025, 1024)),
    ]:
        with pytest.raises(ValueError):
            api.compose_component(base, recipe, "small", raw)
    with pytest.raises(ValueError):
        api.compose_component(base, recipe, "wink", base)
    with pytest.raises(ValueError):
        api.assemble_components(base, recipe, {"unknown": base})


@pytest.mark.parametrize("failure", [None, OSError, asyncio.CancelledError])
def test_shared_budget_and_image_cores_are_released(monkeypatch, failure):
    api = core()
    base, recipe, _ = scene()
    from tldw_chatbook.Chat import character_expression_playback as playback

    before = playback.preparation_bytes()
    captured = []
    original_open = Image.open
    original_new = Image.new
    original_convert = Image.Image.convert

    def opened(*args, **kwargs):
        image = original_open(*args, **kwargs)
        captured.append(image)
        return image

    def created(*args, **kwargs):
        image = original_new(*args, **kwargs)
        captured.append(image)
        return image

    def converted(*args, **kwargs):
        image = original_convert(*args, **kwargs)
        captured.append(image)
        return image

    def failed_save(*args, **kwargs):
        raise failure("injected encode failure")

    monkeypatch.setattr(Image, "open", opened)
    monkeypatch.setattr(Image, "new", created)
    monkeypatch.setattr(Image.Image, "convert", converted)
    if failure is not None:
        monkeypatch.setattr(Image.Image, "save", failed_save)
        with pytest.raises(ValueError if failure is OSError else failure):
            api.compose_component(base, recipe, "small", base)
    else:
        assert api.compose_component(base, recipe, "small", base)
    assert playback.preparation_bytes() == before
    assert captured
    for image in captured:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_budget_refusal_and_set_failure_release_only_owned_memory(monkeypatch):
    api = core()
    base, recipe, components = scene()
    from tldw_chatbook.Chat import character_expression_playback as playback

    before = playback.preparation_bytes()
    held = playback.MAX_PREPARATION_BYTES - before
    playback._adjust_budget(held)
    try:
        with pytest.raises(ValueError, match="budget"):
            api.prepared_base_pixel_sha256(base)
        assert playback.preparation_bytes() == before + held
    finally:
        playback._adjust_budget(-held)
    monkeypatch.setattr(api, "MAX_SET_BYTES", 1)
    with pytest.raises(ValueError, match="set_bytes"):
        api.assemble_components(base, recipe, components)
    assert playback.preparation_bytes() == before


def test_invalid_recovery_recipe_is_refused_as_input():
    api = core()
    base, recipe, components = scene()
    frames = api.assemble_components(base, recipe, components)
    with pytest.raises(ValueError):
        api.recover_components(base, None, frames)


def test_maximum_canvas_fits_shared_budget_and_releases_after_recovery():
    api = core()
    from tldw_chatbook.Chat.character_expression_playback import preparation_bytes

    before = preparation_bytes()
    base, recipe, components = scene((1024, 1024))
    frames = api.assemble_components(base, recipe, components)
    assert len(frames) == 6 and all((f.width, f.height) == (1024, 1024) for f in frames)
    assert api.recover_components(base, recipe, frames) == {
        "eyelids": frames[3].png,
        "small": frames[1].png,
        "open": frames[2].png,
    }
    assert preparation_bytes() == before


def test_replacing_each_component_preserves_unaffected_encoded_frames():
    api = core()
    base, recipe, components = scene()
    before = {f.role: f.png for f in api.assemble_components(base, recipe, components)}
    for component, affected in [
        ("small", {"small", "blink_small"}),
        ("open", {"open", "blink_open"}),
        ("eyelids", {"blink_rest", "blink_small", "blink_open"}),
    ]:
        updated = dict(components)
        updated[component] = api.compose_component(
            base, recipe, component, png((40, 50, 60, 90))
        )
        after = {f.role: f.png for f in api.assemble_components(base, recipe, updated)}
        for role, raw in before.items():
            assert (raw != after[role]) == (role in affected)
