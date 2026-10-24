"""Original-frame native crop data, bounded sequential decode and mounted pair review."""

from dataclasses import replace
from io import BytesIO
from typing import ClassVar

import pytest
from PIL import Image
from textual.widgets import Button, Select, Static

from Tests.Character_Chat.test_portrait_compositing import core, scene
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.css.Themes.themes import ThemeVariableDefaultsMixin

pytestmark = pytest.mark.bootstrap_profile


def api():
    from tldw_chatbook.Widgets.Persona_Widgets import portrait_composition_review

    return portrait_composition_review


def fixture_frames():
    base, recipe, _ = scene()
    with Image.open(BytesIO(base)) as image, BytesIO() as output:
        image.putpixel((27, 40), (255, 0, 255, 180))
        image.save(output, format="PNG")
        base = output.getvalue()
    recipe = replace(recipe, base_pixel_sha256=core().prepared_base_pixel_sha256(base))
    components = {
        role: core().compose_component(base, recipe, role, base)
        for role in core().COMPONENT_ROLES
    }
    return recipe, core().assemble_components(base, recipe, components)


def test_native_detail_copies_all_six_same_source_boxes_and_preserves_one_pixel():
    recipe, frames = fixture_frames()
    raster = api().build_review_raster(
        frames, recipe, (36, 20), region="mouth", pan=(0, 0)
    )
    assert raster.width * raster.height <= 1_048_576
    assert raster.crop_box == (22, 36, 42, 52)
    assert raster.roles == (
        ("rest", "small", "open"),
        ("blink_rest", "blink_small", "blink_open"),
    )
    with Image.open(BytesIO(raster.png)) as image:
        for index, frame in enumerate(frames):
            with Image.open(BytesIO(frame.png)) as original:
                tile = image.crop(
                    (
                        (index % 3) * 20,
                        (index // 3) * 16,
                        (index % 3 + 1) * 20,
                        (index // 3 + 1) * 16,
                    )
                )
                expected = original.crop(raster.crop_box)
                try:
                    assert tile.tobytes() == expected.tobytes()
                finally:
                    tile.close()
                    expected.close()
        assert image.getpixel((5, 4)) == (255, 0, 255, 180)
    with Image.open(BytesIO(frames[0].png)) as original:
        thumbnail = original.resize((1, 1))
        try:
            assert thumbnail.getpixel((0, 0)) != (255, 0, 255, 180)
        finally:
            thumbnail.close()


def test_detail_pan_and_small_viewport_change_only_common_source_bounds():
    recipe, frames = fixture_frames()
    first = api().build_review_raster(
        frames, recipe, (12, 6), region="mouth", pan=(0, 0)
    )
    second = api().build_review_raster(
        frames, recipe, (12, 6), region="mouth", pan=(999, 999)
    )
    assert first.crop_box == (22, 36, 30, 42)
    assert second.crop_box == (34, 46, 42, 52)
    assert second.width * second.height <= 1_048_576


def test_review_sequentially_closes_sources_and_honors_shared_budget(monkeypatch):
    from tldw_chatbook.Chat import character_expression_playback as playback

    recipe, frames = fixture_frames()
    opened = []
    original = Image.open

    def capture(*args, **kwargs):
        if opened:
            with pytest.raises(ValueError):
                opened[-1].getpixel((0, 0))
        image = original(*args, **kwargs)
        opened.append(image)
        return image

    monkeypatch.setattr(Image, "open", capture)
    before = playback.preparation_bytes()
    api().build_review_raster(frames, recipe, (30, 20), region="left_eye", pan=(0, 0))
    assert len(opened) == 6 and playback.preparation_bytes() == before
    for image in opened:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    held = playback.MAX_PREPARATION_BYTES - before
    playback._adjust_budget(held)
    try:
        with pytest.raises(ValueError, match="budget"):
            api().build_review_raster(
                frames, recipe, (30, 20), region="mouth", pan=(0, 0)
            )
        assert playback.preparation_bytes() == before + held
    finally:
        playback._adjust_budget(-held)


class ReviewApp(ThemeVariableDefaultsMixin, ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [BUNDLED_STYLESHEET]


@pytest.mark.asyncio
async def test_mounted_static_review_pan_resize_and_close():
    recipe, frames = fixture_frames()
    app = ReviewApp()
    async with app.run_test(size=(80, 38)) as pilot:
        review = api().PortraitCompositionReview(frames, recipe, lambda: True)
        await app.push_screen(review)
        review.query_one("#portrait-pairs-region", Select).value = "mouth"
        await pilot.pause()
        await review.wait_for_render()
        assert review._raster.crop_box is not None
        assert review.query_one("#portrait-pairs-x").content_size.width >= 8
        assert review.query_one("#portrait-pairs-y").content_size.width >= 8
        assert "rest" in str(
            review.query_one("#portrait-pairs-labels", Static).renderable
        )
        await pilot.press("right", "down")
        await pilot.resize_terminal(48, 30)
        await pilot.pause()
        await review.wait_for_render()
        assert review._raster.width * review._raster.height <= 1_048_576
        review.query_one("#portrait-pairs-close", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert review._retired and review not in app.screen_stack
        assert tuple(frame.png for frame in frames) == tuple(
            frame.png for frame in review._frames
        )


@pytest.mark.parametrize("failure", [False, True])
def test_raster_encode_closes_matrix_and_releases_budget(monkeypatch, failure):
    from tldw_chatbook.Chat import character_expression_playback as playback

    recipe, frames = fixture_frames()
    created = []
    original = Image.new

    def capture(*args, **kwargs):
        image = original(*args, **kwargs)
        created.append(image)
        return image

    monkeypatch.setattr(Image, "new", capture)
    if failure:

        def failed(*args, **kwargs):
            raise OSError("encode fixture")

        monkeypatch.setattr(Image.Image, "save", failed)
    before = playback.preparation_bytes()
    if failure:
        with pytest.raises(OSError, match="encode fixture"):
            api().build_review_raster(frames, recipe, (30, 20), region="mouth")
    else:
        api().build_review_raster(frames, recipe, (30, 20), region="mouth")
    assert playback.preparation_bytes() == before and created
    for image in created:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


@pytest.mark.asyncio
async def test_closed_review_drops_held_crop_completion(monkeypatch):
    import asyncio
    from threading import Event

    recipe, frames = fixture_frames()
    ui = api()
    entered, release = Event(), Event()
    original = ui._render

    def held(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(ui, "_render", held)
    app = ReviewApp()
    async with app.run_test(size=(80, 38)) as pilot:
        review = ui.PortraitCompositionReview(frames, recipe, lambda: True)
        await app.push_screen(review)
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            canvas = review.query_one("#portrait-pairs-canvas", Static)
            review.retire()
            before = canvas.renderable
            release.set()
            await review.wait_for_render()
            await pilot.pause()
            assert review._raster is None and canvas.renderable is before
        finally:
            release.set()


def test_oversized_viewport_cannot_allocate_six_full_canvases():
    from Tests.Character_Chat.test_portrait_compositing import png
    from tldw_chatbook.Character_Chat.portrait_performance import (
        FRAME_ROLES,
        PerformanceFrame,
        normalize_performance_frame,
    )

    raw = png((80, 100, 120, 180), size=(1024, 1024))
    one = normalize_performance_frame("rest", raw)
    _, recipe, _ = scene()
    recipe = replace(
        recipe,
        width=1024,
        height=1024,
        base_pixel_sha256=core().prepared_base_pixel_sha256(one.png),
    )
    frames = tuple(
        PerformanceFrame(role, one.png, 1024, 1024, one.sha256) for role in FRAME_ROLES
    )
    raster = api().build_review_raster(frames, recipe, (100000, 100000))
    assert 1_000_000 < raster.width * raster.height <= 1_048_576
    assert max(raster.width, raster.height) <= 4096


@pytest.mark.asyncio
async def test_retirement_under_cover_keeps_cover_and_removes_stale_review_on_resume():
    from textual.screen import ModalScreen

    recipe, frames = fixture_frames()
    app = ReviewApp()
    async with app.run_test(size=(80, 38)) as pilot:
        review = api().PortraitCompositionReview(frames, recipe, lambda: True)
        await app.push_screen(review)
        await review.wait_for_render()
        cover = ModalScreen()
        await app.push_screen(cover)
        review.retire()
        assert (
            app.screen is cover
            and review._retired
            and review._safe_pending_close is not None
        )
        assert (
            str(review.query_one("#portrait-pairs-canvas", Static).renderable)
            == "Review retired."
        )
        cover.dismiss(None)
        await pilot.pause()
        await pilot.pause()
        assert review not in app.screen_stack
