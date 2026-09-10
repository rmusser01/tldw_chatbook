import threading
from io import BytesIO

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.character_expression_playback import preparation_bytes
from tldw_chatbook.Widgets.Console.character_expression_avatar import (
    CharacterExpressionAvatar,
)


def animation():
    first = Image.new("RGBA", (12, 12), "red")
    output = BytesIO()
    first.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGBA", (12, 12), "blue")],
        duration=[100, 400],
        loop=0,
    )
    return output.getvalue()


class AvatarApp(App):
    def __init__(self, *, animate=True, current=lambda: True):
        super().__init__()
        self.avatar = CharacterExpressionAvatar(
            animation(),
            box=(10, 6),
            animate=animate,
            is_current=current,
            monochrome=False,
            mode="pixels",
            id="avatar",
        )

    def compose(self) -> ComposeResult:
        yield self.avatar


async def wait_ready(avatar, pilot):
    for _ in range(100):
        if avatar.current_image is not None:
            return
        await pilot.pause(0.01)
    pytest.fail("avatar never painted")


@pytest.mark.asyncio
async def test_mounted_dynamic_changes_pixels_without_remount_and_releases():
    before = preparation_bytes()
    app = AvatarApp()
    async with app.run_test(size=(40, 16)) as pilot:
        await wait_ready(app.avatar, pilot)
        child = app.avatar.query_one(Static)
        app.avatar._elapsed_ms = 0
        app.avatar._last_tick = None
        await app.avatar._tick()
        red = child.render().spans
        app.avatar._elapsed_ms = 150
        app.avatar._last_tick = None
        await app.avatar._tick()
        blue = child.render().spans
        assert red != blue
        assert app.avatar.query_one(Static) is child
        assert app.avatar.current_image.getpixel((0, 0)) == (0, 0, 255, 255)
    assert preparation_bytes() == before


@pytest.mark.asyncio
async def test_static_and_hidden_time():
    app = AvatarApp(animate=False)
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)
        initial = app.avatar.current_image.tobytes()
        await pilot.pause(0.2)
        assert app.avatar.current_image.tobytes() == initial
        assert app.avatar._timer is None
    app = AvatarApp()
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)
        app.avatar.display = False
        await app.avatar._tick()
        elapsed = app.avatar._elapsed_ms
        await pilot.pause(0.2)
        assert app.avatar._elapsed_ms == elapsed
        app.avatar.display = True
        await app.avatar._tick()
        assert app.avatar._elapsed_ms == elapsed


@pytest.mark.asyncio
async def test_late_preparation_does_not_paint_removed_or_stale_widget(monkeypatch):
    import tldw_chatbook.Widgets.Console.character_expression_avatar as module

    started, release = threading.Event(), threading.Event()
    prepare = module.prepare_expression

    def slow(*args, **kwargs):
        started.set()
        release.wait(3)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_expression", slow)
    live = [True]
    before = preparation_bytes()
    app = AvatarApp(current=lambda: live[0])
    async with app.run_test() as pilot:
        for _ in range(100):
            if started.is_set():
                break
            await pilot.pause(0.01)
        assert started.is_set()
        live[0] = False
        release.set()
        await pilot.pause(0.2)
        assert app.avatar.current_image is None
    assert preparation_bytes() == before


@pytest.mark.asyncio
async def test_real_timer_paints_both_frames_and_stops_on_unmount():
    app = AvatarApp()
    seen = set()
    async with app.run_test() as pilot:
        for _ in range(100):
            await pilot.pause(0.02)
            if app.avatar.current_image is not None:
                seen.add(app.avatar.current_image.getpixel((0, 0)))
            if len(seen) == 2:
                break
        assert seen == {(255, 0, 0, 255), (0, 0, 255, 255)}
        await app.avatar.remove()
        assert app.avatar._timer is None
        assert app.avatar.current_image is None


@pytest.mark.asyncio
async def test_failed_animation_uses_bounded_neutral_portrait():
    app = AvatarApp()
    app.avatar._data = b"invalid"
    output = BytesIO()
    Image.new("RGBA", (8, 8), "green").save(output, format="PNG")
    app.avatar._fallback_data = output.getvalue()
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)
        assert app.avatar.current_image.getpixel((0, 0)) == (0, 128, 0, 255)
        assert "neutral portrait" in app.avatar.tooltip
        assert app.avatar._timer is None


@pytest.mark.asyncio
async def test_overlay_pauses_elapsed_time():
    app = AvatarApp()
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)
        overlay = Static("Overlay")
        overlay.styles.width = "100%"
        overlay.styles.height = "100%"
        overlay.styles.layer = "overlay"
        overlay.styles.position = "absolute"
        app.screen.styles.layers = ("base", "overlay")
        await app.mount(overlay)
        await pilot.pause(0.05)
        await app.avatar._tick()
        elapsed = app.avatar._elapsed_ms
        await pilot.pause(0.1)
        assert app.avatar._elapsed_ms == elapsed
        await overlay.remove()
        await pilot.pause(0.1)
        assert app.avatar._elapsed_ms > elapsed


@pytest.mark.asyncio
async def test_frame_renderer_failure_keeps_last_valid_frame(monkeypatch):
    from tldw_chatbook.UI.Console_Modules import character_avatar_layout

    app = AvatarApp()
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)

        def fail_render(*args, **kwargs):
            raise ValueError("renderer unavailable")

        monkeypatch.setattr(
            character_avatar_layout, "render_character_avatar_mosaic", fail_render
        )
        app.avatar._elapsed_ms = 150 if app.avatar._frame_index == 0 else 0
        app.avatar._last_tick = None
        previous = app.avatar.current_image.tobytes()
        await app.avatar._tick()
        assert app.avatar.current_image.tobytes() == previous
        assert app.avatar._timer is None
        assert "stopped" in app.avatar.tooltip


@pytest.mark.asyncio
async def test_graphics_widget_updates_image_without_remount():
    pytest.importorskip("textual_image.widget")
    app = AvatarApp()
    app.avatar._mode = "graphics"
    async with app.run_test() as pilot:
        await wait_ready(app.avatar, pilot)
        graphics = app.avatar._graphics
        assert graphics is not None
        app.avatar._elapsed_ms = 150 if app.avatar._frame_index == 0 else 0
        app.avatar._last_tick = None
        previous = app.avatar.current_image.tobytes()
        await app.avatar._tick()
        assert app.avatar.current_image.tobytes() != previous
        assert app.avatar._graphics is graphics
        assert graphics.image is app.avatar.current_image


@pytest.mark.asyncio
async def test_surface_removed_during_preparation_drops_result(monkeypatch):
    import tldw_chatbook.Widgets.Console.character_expression_avatar as module

    started, release = threading.Event(), threading.Event()
    original = module.prepare_expression

    def blocked(*args, **kwargs):
        started.set()
        release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_expression", blocked)
    app = AvatarApp()
    before = preparation_bytes()
    async with app.run_test() as pilot:
        for _ in range(100):
            if started.is_set():
                break
            await pilot.pause(0.01)
        assert started.is_set()
        await app.avatar.remove_children()
        release.set()
        await pilot.pause(0.1)
        assert app.avatar.current_image is None
    assert preparation_bytes() == before
