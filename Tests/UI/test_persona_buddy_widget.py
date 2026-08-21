"""Paint, geometry, input, and focus contracts for Persona Buddy."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container
from textual.geometry import Offset
from textual.screen import ModalScreen, Screen
from textual.widgets import Input, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Persona_Buddy import (
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    PersonaBuddySnapshot,
    PersonaBuddyVisualSnapshot,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError("predicate did not become true before deadline")
        await asyncio.sleep(0.01)


class _FakeController:
    def __init__(
        self,
        *,
        size: tuple[int, int] = (28, 12),
        collapsed: bool = False,
        animate: bool = False,
        frames: tuple[str, ...] = ("BUDDY-A",),
    ) -> None:
        self.preferences = PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "persona-1"),
            collapsed=collapsed,
            geometry=PersonaBuddyGeometry(
                x=1_000_000,
                y=1_000_000,
                width=size[0],
                height=size[1],
            ),
        )
        self.generation = 1
        self.persisted: list[PersonaBuddyPreferences] = []
        prepared = tuple(
            SimpleNamespace(renderable=Text(label), duration_ms=20) for label in frames
        )
        self.visual = PersonaBuddyVisualSnapshot(
            available=True,
            reason=None,
            source="local",
            persona_id="persona-1",
            persona_revision=1,
            requested_state="idle",
            resolved_state="idle",
            animation_id="idle",
            graph_identity=None,
            cache_identity=None,
            frames=prepared,  # type: ignore[arg-type]
            frame_rate=50.0,
            loop=True,
            animate=animate,
        )

    def current_preferences(self) -> PersonaBuddyPreferences:
        return self.preferences

    def snapshot(self) -> PersonaBuddySnapshot:
        self.generation += 1
        return PersonaBuddySnapshot(
            generation=self.generation,
            selection=self.preferences.selection,
            state="idle",
            enabled=self.preferences.enabled,
            open=self.preferences.open,
            collapsed=self.preferences.collapsed,
            preferences_generation=self.generation,
            visual=self.visual,
        )

    async def update_preferences(
        self, preferences: PersonaBuddyPreferences
    ) -> PersonaBuddySnapshot:
        self.preferences = preferences
        self.persisted.append(preferences)
        return self.snapshot()


class _BuddyScreen(Screen):
    def __init__(self, controller: _FakeController) -> None:
        super().__init__()
        self.controller = controller
        self.buddy = PersonaBuddyWidget(
            controller=controller,
            view_generation=1,
            reconcile=lambda: None,
        )

    def compose(self) -> ComposeResult:
        with Container(id="buddy-flow-probe"):
            yield Input(id="focus-probe")
            yield Static("FLOW-END", id="flow-end")
        yield self.buddy


class _BuddyApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    CSS = """
    #buddy-flow-probe { height: 1fr; }
    #flow-end { dock: bottom; height: 1; }
    """

    def __init__(self, controller: _FakeController) -> None:
        super().__init__()
        self.buddy_screen = _BuddyScreen(controller)

    async def on_mount(self) -> None:
        await self.push_screen(self.buddy_screen)


def _compositor_text(screen: Screen) -> str:
    return "\n".join(strip.text for strip in screen._compositor.render_strips())


def _mouse(event_type, *, x: int, y: int, button: int = 1):
    return event_type(
        None,
        x,
        y,
        0,
        0,
        button,
        False,
        False,
        False,
        screen_x=x,
        screen_y=y,
    )


@pytest.mark.asyncio
async def test_overlay_paints_without_flow_or_fr_budget():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        flow = app.screen.query_one("#buddy-flow-probe")
        await _wait_until(lambda: buddy.region.width == 28)
        assert buddy.styles.position == "absolute"
        assert buddy.styles.overlay == "screen"
        assert flow.region.height == 24
        assert app.screen.query_one("#flow-end").region.bottom == 24
        assert "Persona Buddy" in _compositor_text(app.screen)
        assert "BUDDY-A" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_drag_and_resize_are_viewport_bounded_and_persist_once():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        initial_geometry = buddy._clamped_geometry(controller.preferences.geometry)
        assert initial_geometry.x + initial_geometry.width <= app.size.width
        assert initial_geometry.y + initial_geometry.height <= app.size.height

        buddy.on_mouse_down(
            _mouse(events.MouseDown, x=buddy.region.x + 2, y=buddy.region.y)
        )
        await asyncio.sleep(0)
        assert app.screen.focused is buddy
        buddy.on_mouse_move(_mouse(events.MouseMove, x=-50, y=-50))
        assert buddy.absolute_offset == Offset(0, 0)
        assert controller.persisted == []
        buddy.refresh_from_controller()
        assert buddy.absolute_offset == Offset(0, 0)
        buddy.on_mouse_up(_mouse(events.MouseUp, x=-50, y=-50))
        await _wait_until(lambda: len(controller.persisted) == 1)

        corner_x = buddy.region.right - 1
        corner_y = buddy.region.bottom - 1
        buddy.on_mouse_down(_mouse(events.MouseDown, x=corner_x, y=corner_y))
        buddy.on_mouse_move(_mouse(events.MouseMove, x=500, y=500))
        assert buddy.region.right <= app.size.width
        assert buddy.region.bottom <= app.size.height
        assert len(controller.persisted) == 1
        buddy.on_mouse_up(_mouse(events.MouseUp, x=500, y=500))
        await _wait_until(lambda: len(controller.persisted) == 2)
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_keyboard_move_resize_reset_collapse_close_exact_bindings():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(100, 30)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        buddy.focus(scroll_visible=False)
        start = buddy.absolute_offset
        await pilot.press("h")
        assert buddy.absolute_offset.x == max(0, start.x - 1)
        await pilot.press("j", "k", "l")
        await pilot.press("H", "J", "K", "L")
        await _wait_until(lambda: len(controller.persisted) >= 8)
        await pilot.press("0")
        assert buddy.absolute_offset == Offset(72, 18)
        await pilot.press("c")
        await _wait_until(lambda: controller.preferences.collapsed)
        await _wait_until(lambda: "Buddy" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy.region.height == 3)
        await pilot.press("x")
        await _wait_until(lambda: not controller.preferences.open)

        keys = {binding.key for binding in PersonaBuddyWidget.BINDINGS}
        assert keys == {"h", "j", "k", "l", "H", "J", "K", "L", "0", "c", "x"}
        assert not any(key.startswith("ctrl+") for key in keys)


@pytest.mark.asyncio
async def test_tiny_viewport_uses_labelled_compact_control():
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=(12, 4)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
        assert buddy.region.width <= 12
        assert buddy.region.height <= 4
        assert "Buddy" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_state_repaint_never_steals_focus():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        focus_probe = app.screen.query_one("#focus-probe", Input)
        focus_probe.focus(scroll_visible=False)
        assert app.screen.focused is focus_probe
        buddy = app.screen.query_one(PersonaBuddyWidget)
        buddy.refresh_from_controller()
        await asyncio.sleep(0)
        assert app.screen.focused is focus_probe


@pytest.mark.asyncio
async def test_animation_cells_change_then_freeze_hidden_collapsed():
    controller = _FakeController(animate=True, frames=("FRAME-A", "FRAME-B"))
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "FRAME-A" in _compositor_text(app.screen))
        buddy.advance_frame()
        await _wait_until(lambda: "FRAME-B" in _compositor_text(app.screen))

        controller.preferences = replace(controller.preferences, collapsed=True)
        buddy.refresh_from_controller()
        frozen = buddy.frame_index
        buddy.advance_frame()
        assert buddy.frame_index == frozen
        assert buddy.snapshot_polling_active

        buddy.display = False
        buddy.advance_frame()
        assert buddy.frame_index == frozen


@pytest.mark.asyncio
async def test_reduced_motion_paints_static_frame():
    controller = _FakeController(animate=False, frames=("STATIC-A", "STATIC-B"))
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "STATIC-A" in _compositor_text(app.screen))
        buddy.advance_frame()
        assert buddy.frame_index == 0
        assert "STATIC-A" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_modal_covers_buddy_hit_target():
    class _BlockingModal(ModalScreen):
        def compose(self) -> ComposeResult:
            yield Static("MODAL BLOCKER", id="modal-blocker")

    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        point = (buddy.region.x + 1, buddy.region.y + 1)
        await app.push_screen(_BlockingModal())
        target, _region = app.screen.get_widget_at(*point)
        assert not isinstance(target, PersonaBuddyWidget)
        assert "MODAL BLOCKER" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_capture_is_released_when_widget_unmounts_mid_drag():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        buddy.on_mouse_down(
            _mouse(events.MouseDown, x=buddy.region.x + 1, y=buddy.region.y)
        )
        assert app.mouse_captured is buddy
        await buddy.remove()
        assert app.mouse_captured is None
