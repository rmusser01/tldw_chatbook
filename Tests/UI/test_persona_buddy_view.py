"""Painted Buddy contract: overlay budget, controls, frame safety and lifecycle."""

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Persona_Visual.contracts import PersonaVisualRegion
from tldw_chatbook.Persona_Visual.runtime import (
    PersonaVisualCacheIdentity,
    PersonaVisualResolution,
    PersonaVisualResolvedFrame,
)
from tldw_chatbook.Widgets.Persona_Widgets import persona_buddy as module


@dataclass(frozen=True)
class Preferences:
    enabled: bool = True
    local_persona_id: str = "p1"
    open: bool = True
    collapsed: bool = False
    x: int | None = None
    y: int | None = None
    width: int = 28
    height: int = 16


def resolution(*, animate=False, broken=False, region=None):
    image = Image.new("RGB", (8, 8), "red")
    image.paste("blue", (4, 0, 8, 8))
    output = BytesIO()
    image.save(output, format="PNG")
    frame = PersonaVisualResolvedFrame(
        1,
        "frame",
        "0" * 64,
        b"broken /private/image" if broken else output.getvalue(),
        40,
        region,
        0,
        0,
    )
    return PersonaVisualResolution(
        "sprite_frames",
        None,
        "idle",
        "idle",
        "idle",
        (frame, replace(frame, asset_id=2)) if animate else (frame,),
        25,
        True,
        None,
        animate,
        None,
        None,
        PersonaVisualCacheIdentity(None, "idle", "idle", "idle", False, ()),
    )


def snapshot(*, generation=1, name="Migu", animate=False):
    resolved = resolution(animate=animate)
    return SimpleNamespace(
        persona_id="p1",
        name=name,
        generation=generation,
        requested_state="idle",
        resolution=resolved,
        prepared=module.prepare_buddy_frames(resolved, 26, 11),
    )


class Host(ConsolidatedCSSApp):
    AUTO_FOCUS = "#composer"
    CSS_PATH = str(BUNDLED_STYLESHEET)
    CSS = """
    #navigation, #footer { height: 1; }
    #work { height: 1fr; }
    #composer { height: 3; }
    """

    def __init__(self, preferences=None, *, animate=False):
        super().__init__()
        self.preferences = preferences or Preferences()
        self.changes = []
        self.buddy = module.PersonaBuddyView(
            snapshot(animate=animate), self.preferences
        )

    def compose(self):
        yield Static("Navigation", id="navigation")
        with Vertical(id="work"):
            yield Input(id="composer")
        yield Static("Footer", id="footer")
        yield self.buddy

    def on_buddy_preferences_requested(self, event):
        assert event.control is self.buddy
        self.changes.append(event.changes)
        self.preferences = replace(self.preferences, **event.changes)
        self.buddy.apply_preferences(self.preferences)


def test_import_is_current_checkout():
    assert (
        Path(module.__file__).resolve().parents[3]
        == Path(__file__).resolve().parents[2]
    )


def test_preparation_crops_region_and_reports_path_free_failure():
    prepared = module.prepare_buddy_frames(
        resolution(region=PersonaVisualRegion(4, 0, 4, 8)), 4, 4
    )
    assert prepared.frames and prepared.reason is None
    spans = prepared.frames[0].renderable.spans
    assert spans and all("0,0,255" in str(span.style) for span in spans)
    failed = module.prepare_buddy_frames(resolution(broken=True), 4, 4)
    assert failed.reason == "persona_buddy_frame_unavailable"
    assert "private" not in str(failed)
    assert failed.frames == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 36), (160, 48), (80, 24)])
async def test_overlay_paints_without_consuming_sibling_flow(size):
    app = Host()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        buddy = app.buddy
        assert app.query_one("#work").region.height == size[1] - 2
        assert app.query_one("#footer").region.y == size[1] - 1
        assert buddy.region.right <= size[0] and buddy.region.bottom <= size[1] - 1
        assert buddy.region.y >= 1
        svg = app.export_screenshot()
        assert "Migu" in svg and "Close" in svg and "Collapse" in svg
        assert app.focused is app.query_one(Input)


@pytest.mark.asyncio
async def test_geometry_restores_desired_position_after_temporary_tiny_viewport():
    app = Host(Preferences(x=60, y=10, width=35, height=20))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.resize_terminal(18, 6)
        app.buddy.sync_geometry()
        await pilot.pause()
        assert app.buddy.region.width <= 18 and app.buddy.region.height <= 4
        assert "Buddy" in app.export_screenshot()
        await pilot.resize_terminal(120, 40)
        app.buddy.sync_geometry()
        await pilot.pause()
        assert app.buddy.region.x == 60 and app.buddy.region.y == 10
        assert app.buddy.region.width == 35 and app.buddy.region.height == 20
        assert app.changes == []


@pytest.mark.asyncio
async def test_keyboard_and_buttons_request_preferences_without_focus_steal():
    app = Host()
    async with app.run_test(size=(100, 36)) as pilot:
        buddy = app.buddy
        await pilot.pause()
        buddy.focus()
        await pilot.press("left", "shift+left")
        await pilot.pause()
        assert app.changes[0] == {"x": 71, "y": 19}
        assert app.changes[1] == {"width": 27, "height": 16}
        buddy.apply_snapshot(snapshot(generation=2, name="Updated"))
        await pilot.pause()
        assert app.focused is buddy
        await pilot.press("c")
        await pilot.pause()
        assert app.preferences.collapsed
        await pilot.press("r")
        await pilot.pause()
        assert app.preferences.x is None and app.preferences.width == 28
        await pilot.click("#persona-buddy-close")
        await pilot.pause()
        assert not app.preferences.open and not buddy.display


@pytest.mark.asyncio
async def test_animation_stops_when_collapsed_hidden_or_modal_and_rejects_stale_snapshot():
    app = Host(animate=True)
    async with app.run_test(size=(100, 36)) as pilot:
        buddy = app.buddy
        app.preferences = replace(app.preferences, collapsed=True)
        buddy.apply_preferences(app.preferences)
        index = buddy._frame_index
        buddy._advance_frame()
        assert buddy._frame_index == index
        app.preferences = replace(app.preferences, collapsed=False)
        buddy.apply_preferences(app.preferences)
        modal = ModalScreen()
        await app.push_screen(modal)
        index = buddy._frame_index
        buddy._advance_frame()
        assert buddy._frame_index == index
        await app.pop_screen()
        buddy.apply_snapshot(snapshot(generation=3, name="New"))
        buddy.apply_snapshot(snapshot(generation=2, name="Stale"))
        await pilot.pause()
        assert (
            "New" in app.export_screenshot() and "Stale" not in app.export_screenshot()
        )
        buddy.apply_preferences(replace(app.preferences, open=False))
        index = buddy._frame_index
        buddy._advance_frame()
        assert buddy._frame_index == index


@pytest.mark.asyncio
@pytest.mark.parametrize("handle,delta", [("name", (5, -3)), ("grip", (4, 2))])
async def test_native_terminal_mouse_codes_move_and_resize_buddy(handle, delta):
    from textual._xterm_parser import XTermParser

    app = Host(Preferences(x=20, y=8))
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        parser = XTermParser()
        region = app.query_one(f"#persona-buddy-{handle}").region
        x, y = region.x, region.y

        async def forward(buttons, px, py, suffix):
            event = parser.parse_mouse_code(
                f"\x1b[<{buttons};{px + 1};{py + 1}{suffix}"
            )
            assert event is not None and event.widget is None
            app.screen._forward_event(event)
            await pilot.pause()

        await forward(0, x, y, "M")
        assert app.mouse_captured is app.buddy
        await forward(32, x + delta[0] // 2, y + delta[1] // 2, "M")
        await forward(0, x + delta[0], y + delta[1], "m")
        assert app.mouse_captured is None
        if handle == "name":
            assert (app.preferences.x, app.preferences.y) == (25, 5)
        else:
            assert (app.preferences.width, app.preferences.height) == (32, 18)


@pytest.mark.asyncio
async def test_mouse_drag_resize_and_capture_release_on_hide():
    app = Host()
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        await pilot.mouse_down("#persona-buddy-name")
        assert app.mouse_captured is app.buddy
        await pilot.hover(offset=(65, 15))
        await pilot.mouse_up(offset=(65, 15))
        await pilot.pause()
        assert app.changes and "x" in app.changes[-1]
        assert app.mouse_captured is None
        old_width = app.buddy.region.width
        grip = app.query_one("#persona-buddy-grip").region
        await pilot.mouse_down("#persona-buddy-grip")
        await pilot.hover(offset=(grip.x + 4, grip.y + 1))
        await pilot.mouse_up(offset=(grip.x + 4, grip.y + 1))
        await pilot.pause()
        assert app.preferences.width == old_width + 4
        await pilot.mouse_down("#persona-buddy-name")
        app.buddy.apply_preferences(replace(app.preferences, open=False))
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_animation_advances_only_when_visible_active_and_motion_enabled():
    app = Host(animate=True)
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        buddy = app.buddy
        buddy._timer.pause()
        buddy._frame_index = 0
        buddy._last_frame = 0
        buddy._advance_frame()
        assert buddy._frame_index == 1
        for preferences in (
            replace(app.preferences, collapsed=True),
            replace(app.preferences, open=False),
        ):
            buddy.apply_preferences(preferences)
            buddy._last_frame = 0
            buddy._advance_frame()
            assert buddy._frame_index == 1
        buddy.apply_preferences(app.preferences)
        await app.push_screen(ModalScreen())
        buddy._last_frame = 0
        buddy._advance_frame()
        buddy.action_close()
        assert buddy._frame_index == 1 and app.changes == []
        await app.pop_screen()
        still = snapshot(generation=2, animate=True)
        still.resolution = replace(still.resolution, animate=False)
        buddy.apply_snapshot(still)
        buddy._last_frame = 0
        buddy._advance_frame()
        assert buddy._frame_index == 0


def test_selected_gif_frame_and_portrait_fallback_remain_visible():
    from tldw_chatbook.Persona_Visual.runtime import PersonaVisualPortrait

    output = BytesIO()
    first = Image.new("RGB", (4, 4), "red")
    first.save(
        output,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGB", (4, 4), "blue")],
        duration=100,
    )
    resolved = resolution()
    selected = replace(resolved.frames[0], data=output.getvalue(), selected_frame=1)
    prepared = module.prepare_buddy_frames(replace(resolved, frames=(selected,)), 4, 4)
    assert all(
        "0,0,255" in str(span.style) for span in prepared.frames[0].renderable.spans
    )
    portrait = PersonaVisualPortrait("p1", 0, "image/gif", "0" * 64, output.getvalue())
    failed = resolution(broken=True)
    fallback = module.prepare_buddy_frames(
        replace(failed, portrait=portrait), 4, 4, monochrome=True
    )
    assert fallback.frames and fallback.reason == "persona_buddy_frame_unavailable"
    assert fallback.frames[0].renderable.plain.strip()


@pytest.mark.asyncio
async def test_authority_refresh_preserves_long_frame_clock_and_position():
    app = Host(animate=True)
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        buddy = app.buddy
        buddy._timer.pause()
        buddy._frame_index = 1
        buddy._last_frame = 123.0
        buddy.apply_snapshot(snapshot(generation=2, animate=True))
        assert buddy._frame_index == 1
        assert buddy._last_frame == 123.0
        buddy._advance_frame()
        assert buddy._frame_index == 0


@pytest.mark.asyncio
async def test_edge_gestures_persist_bounded_geometry_through_real_controller(tmp_path):
    import json
    from dataclasses import asdict

    from textual import events

    from tldw_chatbook.Persona_Visual.buddy import BuddyController, BuddyPreferences

    initial = BuddyPreferences(enabled=True, local_persona_id="p1", x=0, y=1)
    profile = tmp_path / "profile"
    profile.mkdir(mode=0o700)
    saved = profile / "persona_buddy.json"
    saved.write_text(json.dumps(asdict(initial)))
    saved.chmod(0o600)
    controller = BuddyController(None, None, profile)
    await controller.load_preferences()

    class PersistedHost(Host):
        async def on_buddy_preferences_requested(self, event):
            assert event.control is self.buddy
            self.changes.append(event.changes)
            self.preferences = await controller.update_preferences(**event.changes)
            self.buddy.apply_preferences(self.preferences)

    app = PersistedHost(controller.preferences)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.buddy.focus()
        await pilot.press("left", "up")
        await pilot.pause()
        assert controller.preferences.x == 0 and controller.preferences.y == 1
        app.buddy.action_move(20000, 20000)
        await pilot.pause()
        assert controller.preferences.x == 52 and controller.preferences.y == 7
        app.buddy.action_resize(20000, 20000)
        await pilot.pause()
        assert (
            controller.preferences.width == 80 and controller.preferences.height == 22
        )
        app.buddy.action_reset()
        await pilot.pause()
        await pilot.mouse_down("#persona-buddy-name")
        app.buddy.post_message(
            events.MouseMove(
                app.buddy,
                0,
                0,
                0,
                0,
                1,
                False,
                False,
                False,
                screen_x=-20000,
                screen_y=-20000,
            )
        )
        await pilot.pause()
        await pilot.mouse_up()
        assert controller.preferences.x == 0 and controller.preferences.y == 1
        await pilot.mouse_down("#persona-buddy-grip")
        app.buddy.post_message(
            events.MouseMove(
                app.buddy,
                0,
                0,
                0,
                0,
                1,
                False,
                False,
                False,
                screen_x=20000,
                screen_y=20000,
            )
        )
        await pilot.pause()
        await pilot.mouse_up(offset=(79, 23))
        assert (
            controller.preferences.width == 80 and controller.preferences.height == 22
        )
        await pilot.resize_terminal(18, 6)
        app.buddy.sync_geometry()
        await pilot.pause()
        app.buddy.action_resize(20000, 20000)
        await pilot.pause()
        assert controller.preferences.width == 18 and controller.preferences.height == 4
        persisted = json.loads(saved.read_text())
        assert persisted["x"] == 0 and persisted["y"] == 1
        with pytest.raises(ValueError, match="persona_buddy_preferences_invalid"):
            await controller.update_preferences(x=-1)


@pytest.mark.asyncio
async def test_missing_frames_retain_only_same_authority_static_art():
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualIdentity

    app = Host(animate=True)
    identity = PersonaVisualIdentity("p1", 1, 1, 1, 1, 1, 1, 1, "0" * 64)
    app.buddy._snapshot.identity = identity
    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        buddy = app.buddy
        buddy._timer.pause()
        previous_art = buddy.query_one("#persona-buddy-image", Static).renderable
        missing = snapshot(animate=True)
        missing.identity = identity
        missing.resolution = replace(
            missing.resolution,
            source="unavailable",
            frames=(),
            animate=False,
            animation_id=None,
            cache_identity=replace(
                missing.resolution.cache_identity, animation_id=None
            ),
        )
        missing.prepared = module.PreparedBuddyFrames(
            (), "persona_buddy_frame_unavailable"
        )
        buddy.apply_snapshot(missing)
        await pilot.pause()
        assert (
            buddy.query_one("#persona-buddy-image", Static).renderable == previous_art
        )
        assert "Frame unavailable" in str(
            buddy.query_one("#persona-buddy-state", Static).renderable
        )
        buddy._last_frame = 0
        buddy._advance_frame()
        assert buddy._frame_index == 0
        replaced = snapshot(generation=2)
        replaced.identity = replace(identity, pack_version_id=2, version_number=2)
        replaced.resolution = missing.resolution
        replaced.prepared = missing.prepared
        buddy.apply_snapshot(replaced)
        await pilot.pause()
        assert "Portrait unavailable" in str(
            buddy.query_one("#persona-buddy-image", Static).renderable
        )
